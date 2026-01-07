"""Modal app for parallel Dwarkesh Podcast scraping.

This module defines Modal functions for:
1. Episode discovery from archive page
2. Parallel episode scraping
3. Volume-based intermediate storage

Usage:
    # Run full scrape pipeline
    modal run src/scrapers/dwarkesh/modal_app.py

    # Deploy as persistent app
    modal deploy src/scrapers/dwarkesh/modal_app.py
"""

import json
import re
from datetime import datetime
from pathlib import Path

import modal

# Modal app definition
app = modal.App("dwarkesh-scraper")

# Image with required dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "httpx>=0.25.0",
    "beautifulsoup4>=4.12.0",
    "lxml>=5.0.0",
    "pydantic>=2.5.0",
)

# Volume for intermediate transcript storage
volume = modal.Volume.from_name("dwarkesh-transcripts", create_if_missing=True)

# Constants
ARCHIVE_URL = "https://www.dwarkesh.com/podcast/archive?sort=new"
BASE_URL = "https://www.dwarkesh.com"
VOLUME_PATH = "/data/transcripts"


@app.function(image=image, timeout=300)
def discover_episodes() -> list[dict]:
    """
    Discover all episode URLs from the archive page.

    The page is JavaScript-rendered, so we extract slugs from script tags.

    Returns:
        List of episode metadata dicts sorted oldest to newest
    """
    import httpx
    from bs4 import BeautifulSoup

    client = httpx.Client(
        timeout=30.0,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; DwarkeshScraper/1.0)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )

    response = client.get(ARCHIVE_URL)
    response.raise_for_status()
    html = response.text
    client.close()

    soup = BeautifulSoup(html, "lxml")
    episodes = []
    seen_slugs: set[str] = set()

    # Page is JS-rendered, so extract slugs from script tags
    for script in soup.find_all("script"):
        text = script.get_text()
        if "/p/" in text:
            # Find all slugs in the script content
            slug_matches = re.findall(r'/p/([a-z0-9-]+)', text)
            for slug in slug_matches:
                # Skip non-episode slugs
                if slug in seen_slugs:
                    continue
                if slug in ("our-position-on-the-online-safety", "comments"):
                    continue

                seen_slugs.add(slug)
                url = f"{BASE_URL}/p/{slug}"
                title = slug.replace("-", " ").title()

                episodes.append({
                    "url": url,
                    "slug": slug,
                    "title": title,
                    "published_at": None,
                })

    # Also try anchor tags in case some are present
    for link in soup.find_all("a", href=True):
        href = str(link["href"])
        if "/p/" in href and not href.endswith("/comments"):
            slug = href.split("/p/")[1].split("/")[0].split("?")[0]
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            url = f"{BASE_URL}/p/{slug}"
            title = link.get_text().strip() or slug.replace("-", " ").title()

            episodes.append({
                "url": url,
                "slug": slug,
                "title": title,
                "published_at": None,
            })

    print(f"Discovered {len(episodes)} episodes")
    return episodes


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=600,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0),
)
def scrape_episode(episode_meta: dict) -> dict:
    """
    Scrape a single episode, parse it, and save to volume.

    Args:
        episode_meta: Dict with url, slug, title, published_at

    Returns:
        Dict with status and path
    """
    import httpx
    from bs4 import BeautifulSoup

    url = episode_meta["url"]
    slug = episode_meta["slug"]

    print(f"Scraping: {slug}")

    try:
        # Fetch HTML
        client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; DwarkeshScraper/1.0)",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        response = client.get(url)
        response.raise_for_status()
        html = response.text
        client.close()

        # Parse transcript
        soup = BeautifulSoup(html, "lxml")

        # Extract transcript content
        post_body = (
            soup.select_one(".body.markup")
            or soup.select_one(".post-content")
            or soup.select_one("article .body")
            or soup.select_one(".available-content")
        )

        if not post_body:
            for div in soup.find_all("div"):
                text = div.get_text()
                if "**Dwarkesh" in text or "_00:0" in text:
                    post_body = div
                    break

        if not post_body:
            return {
                "status": "error",
                "slug": slug,
                "error": "Could not find transcript content",
            }

        # Convert to text
        transcript_text = _html_to_markdown(post_body)

        # Parse turns and sections (handles transcripts and blog posts)
        turns, sections, doc_type = _parse_transcript(transcript_text)

        if not turns:
            return {
                "status": "skipped",
                "slug": slug,
                "error": "No content could be parsed (no dialogue turns or blog sections found)",
            }

        # Get guest name (only for transcripts)
        guest = None
        if doc_type == "transcript":
            for turn in turns:
                if turn["speaker"] != "Dwarkesh Patel":
                    guest = turn["speaker"]
                    break

        # Extract actual title from JSON-LD structured data
        title = episode_meta["title"]  # Fallback to slug-derived title
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                script_content = script.string
                if not script_content:
                    continue
                ld_data = json.loads(script_content)
                if isinstance(ld_data, dict) and ld_data.get("headline"):
                    title = ld_data["headline"]
                    break
            except (json.JSONDecodeError, TypeError):
                continue

        # Get published date from meta
        published_at = episode_meta.get("published_at")
        if not published_at:
            time_elem = soup.select_one("time")
            if time_elem and time_elem.get("datetime"):
                published_at = str(time_elem["datetime"])

        # Build episode data
        episode_data = {
            "url": url,
            "slug": slug,
            "title": title,
            "guest": guest,
            "published_at": published_at,
            "doc_type": doc_type,
            "turns": turns,
            "sections": sections,
            "raw_transcript": transcript_text,
            "scraped_at": datetime.now().isoformat(),
        }

        # Save to volume
        output_path = Path(VOLUME_PATH) / f"{slug}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(episode_data, indent=2))

        # Commit volume changes
        volume.commit()

        content_label = "sections" if doc_type == "blog" else "turns"
        print(f"  -> Saved {slug} ({doc_type}): {len(turns)} {content_label}")

        return {
            "status": "success",
            "slug": slug,
            "path": str(output_path),
            "turn_count": len(turns),
            "doc_type": doc_type,
        }

    except Exception as e:
        return {
            "status": "error",
            "slug": slug,
            "error": str(e),
        }


def _html_to_markdown(element) -> str:
    """Convert HTML element to markdown-like text."""
    lines: list[str] = []

    for child in element.descendants:
        if child.name == "p":
            text = _element_to_text(child)
            if text.strip():
                lines.append(text.strip())
                lines.append("")
        elif child.name in ("h1", "h2", "h3", "h4"):
            text = child.get_text().strip()
            if text:
                prefix = "#" * int(child.name[1])
                lines.append(f"{prefix} {text}")
                lines.append("")

    return "\n".join(lines)


def _element_to_text(element) -> str:
    """Convert element to text with markdown formatting."""
    result: list[str] = []

    for child in element.children:
        if isinstance(child, str):
            result.append(child)
        elif child.name in ("strong", "b"):
            result.append(f"**{child.get_text()}**")
        elif child.name in ("em", "i"):
            result.append(f"_{child.get_text()}_")
        elif child.name == "a":
            result.append(child.get_text())
        elif child.name == "br":
            result.append("\n")
        else:
            result.append(child.get_text())

    return "".join(result)


def _parse_transcript(content: str) -> tuple[list[dict], list[dict], str]:
    """Parse transcript into turns and sections.

    Handles three content formats:
    1. Format A (transcript): **Speaker** _HH:MM:SS_ (inline timestamp after speaker)
    2. Format B (transcript): **Speaker** alone, with timestamps in section headers like
       (HH:MM:SS) – Topic or ### HH:MM:SS – Topic
    3. Format C (blog): No speakers/timestamps - sections separated by headers become turns

    Returns:
        Tuple of (turns, sections, doc_type) where doc_type is 'transcript' or 'blog'
    """
    # Pattern for speaker with inline timestamp: **Name** _HH:MM:SS_
    speaker_with_ts_pattern = re.compile(
        r"\*\*(?P<speaker>[^*]+)\*\*\s*_(?P<hours>\d{1,2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})_"
    )

    # Pattern for speaker without inline timestamp: **Name** (not followed by _timestamp_)
    # Must be at start of line or after newline, followed by newline
    speaker_only_pattern = re.compile(
        r"^\*\*(?P<speaker>[^*]+)\*\*\s*$",
        re.MULTILINE
    )

    # Section patterns (multiple formats used across episodes)
    # Format: [(HH:MM:SS) – Title]
    section_bracket_pattern = re.compile(
        r"\[\((?P<hours>\d{1,2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})\)\s*[–\-]\s*(?P<title>[^\]]+)\]"
    )
    # Format: (HH:MM:SS) – Title (standalone, not in brackets)
    section_paren_pattern = re.compile(
        r"^\s*\((?P<hours>\d{1,2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})\)\s*[–\-]\s*(?P<title>.+)$",
        re.MULTILINE
    )
    # Format: ### HH:MM:SS – Title or ## (HH:MM:SS) – Title
    section_heading_pattern = re.compile(
        r"^#{2,4}\s*\(?(?P<hours>\d{1,2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})\)?\s*[–\-]\s*(?P<title>.+)$",
        re.MULTILINE
    )

    # Extract sections from all formats
    sections = []
    seen_timestamps = set()

    for pattern in [section_bracket_pattern, section_paren_pattern, section_heading_pattern]:
        for match in pattern.finditer(content):
            timestamp_seconds = (
                int(match.group("hours")) * 3600
                + int(match.group("minutes")) * 60
                + int(match.group("seconds"))
            )
            if timestamp_seconds not in seen_timestamps:
                seen_timestamps.add(timestamp_seconds)
                sections.append({
                    "title": match.group("title").strip(),
                    "timestamp_seconds": timestamp_seconds,
                })
    sections.sort(key=lambda s: s["timestamp_seconds"])

    # Try Format A first: speakers with inline timestamps
    markers_with_ts = list(speaker_with_ts_pattern.finditer(content))

    if markers_with_ts:
        # Use Format A parsing
        turns = []
        for i, match in enumerate(markers_with_ts):
            speaker = match.group("speaker").strip()
            timestamp_seconds = (
                int(match.group("hours")) * 3600
                + int(match.group("minutes")) * 60
                + int(match.group("seconds"))
            )

            start_pos = match.end()
            end_pos = markers_with_ts[i + 1].start() if i + 1 < len(markers_with_ts) else len(content)

            raw_text = content[start_pos:end_pos].strip()
            # Clean section markers from text
            cleaned_text = section_bracket_pattern.sub("", raw_text)
            cleaned_text = section_paren_pattern.sub("", cleaned_text)
            cleaned_text = section_heading_pattern.sub("", cleaned_text)
            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

            if not cleaned_text:
                continue

            # Find section
            section_title = None
            for section in sections:
                if section["timestamp_seconds"] <= timestamp_seconds:
                    section_title = section["title"]
                else:
                    break

            turns.append({
                "speaker": speaker,
                "start_time_seconds": timestamp_seconds,
                "text": cleaned_text,
                "section_title": section_title,
                "ord": len(turns),
            })

        return turns, sections, "transcript"

    # Try Format B: speakers without inline timestamps
    markers_only = list(speaker_only_pattern.finditer(content))

    if markers_only:
        turns = []
        # Build a list of (position, timestamp) from sections for timestamp lookup
        section_positions = []
        for pattern in [section_bracket_pattern, section_paren_pattern, section_heading_pattern]:
            for match in pattern.finditer(content):
                timestamp_seconds = (
                    int(match.group("hours")) * 3600
                    + int(match.group("minutes")) * 60
                    + int(match.group("seconds"))
                )
                section_positions.append((match.start(), timestamp_seconds, match.group("title").strip()))
        section_positions.sort(key=lambda x: x[0])

        for i, match in enumerate(markers_only):
            speaker = match.group("speaker").strip()

            # Find timestamp from nearest preceding section header
            timestamp_seconds = 0
            section_title = None
            for pos, ts, title in section_positions:
                if pos < match.start():
                    timestamp_seconds = ts
                    section_title = title
                else:
                    break

            start_pos = match.end()
            end_pos = markers_only[i + 1].start() if i + 1 < len(markers_only) else len(content)

            raw_text = content[start_pos:end_pos].strip()
            # Clean section markers from text
            cleaned_text = section_bracket_pattern.sub("", raw_text)
            cleaned_text = section_paren_pattern.sub("", cleaned_text)
            cleaned_text = section_heading_pattern.sub("", cleaned_text)
            # Also remove standalone **Speaker** patterns that might be nested
            cleaned_text = re.sub(r"^\*\*[^*]+\*\*\s*$", "", cleaned_text, flags=re.MULTILINE)
            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

            if not cleaned_text:
                continue

            turns.append({
                "speaker": speaker,
                "start_time_seconds": timestamp_seconds,
                "text": cleaned_text,
                "section_title": section_title,
                "ord": len(turns),
            })

        return turns, sections, "transcript"

    # Try Format C: blog post - parse sections by headers
    # Look for markdown headers (# Header, ## Header) or bold headers (**Header**)
    header_pattern = re.compile(
        r"^(?:#{1,4}\s+(?P<md_title>.+)|(?P<bold_title>\*\*[^*]+\*\*)\s*)$",
        re.MULTILINE
    )

    headers = list(header_pattern.finditer(content))

    if headers:
        turns = []
        for i, match in enumerate(headers):
            # Get header text
            header_title = match.group("md_title") or match.group("bold_title")
            if header_title:
                # Remove markdown bold markers if present
                header_title = re.sub(r"\*\*([^*]+)\*\*", r"\1", header_title).strip()

            # Get text from end of header to next header (or end)
            start_pos = match.end()
            end_pos = headers[i + 1].start() if i + 1 < len(headers) else len(content)

            raw_text = content[start_pos:end_pos].strip()
            # Clean up the text - remove nested headers and normalize whitespace
            cleaned_text = header_pattern.sub("", raw_text)
            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

            if not cleaned_text or len(cleaned_text) < 50:  # Skip very short sections
                continue

            turns.append({
                "speaker": "Dwarkesh Patel",
                "start_time_seconds": 0,  # No timestamps for blog posts
                "text": cleaned_text,
                "section_title": header_title,
                "ord": len(turns),
            })

        if turns:
            # Extract section info from headers for consistency
            blog_sections = [
                {"title": t["section_title"], "timestamp_seconds": 0}
                for t in turns if t["section_title"]
            ]
            return turns, blog_sections, "blog"

    # No content could be parsed
    return [], sections, "unknown"


@app.function(image=image, volumes={VOLUME_PATH: volume})
def list_scraped_episodes() -> list[str]:
    """List all scraped episode JSON files in the volume."""
    volume.reload()
    transcripts_dir = Path(VOLUME_PATH)

    if not transcripts_dir.exists():
        return []

    return [f.name for f in transcripts_dir.glob("*.json")]


@app.function(image=image, volumes={VOLUME_PATH: volume})
def read_episode(slug: str) -> dict | None:
    """Read a single episode from the volume."""
    volume.reload()
    path = Path(VOLUME_PATH) / f"{slug}.json"

    if not path.exists():
        return None

    return json.loads(path.read_text())


@app.function(image=image, volumes={VOLUME_PATH: volume})
def read_all_episodes() -> list[dict]:
    """Read all episodes from the volume."""
    volume.reload()
    transcripts_dir = Path(VOLUME_PATH)

    if not transcripts_dir.exists():
        return []

    episodes = []
    for path in sorted(transcripts_dir.glob("*.json")):
        try:
            episodes.append(json.loads(path.read_text()))
        except Exception as e:
            print(f"Error reading {path}: {e}")

    return episodes


@app.local_entrypoint()
def main(
    scrape_only: bool = False,
    limit: int = 0,
):
    """
    Main entrypoint for Modal CLI.

    Args:
        scrape_only: Only scrape, don't trigger ingestion
        limit: Limit number of episodes to scrape (0 = all)
    """
    print("=" * 60)
    print("DWARKESH PODCAST SCRAPER")
    print("=" * 60)

    # Step 1: Discover episodes
    print("\n[1/3] Discovering episodes...")
    episodes = discover_episodes.remote()
    print(f"Found {len(episodes)} episodes")

    if limit > 0:
        episodes = episodes[:limit]
        print(f"Limiting to {limit} episodes")

    # Step 2: Parallel scrape
    print(f"\n[2/3] Scraping {len(episodes)} episodes in parallel...")
    results = list(scrape_episode.map(episodes))

    # Summarize results
    success = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]

    # Break down success by doc_type
    transcripts = [r for r in success if r.get("doc_type") == "transcript"]
    blogs = [r for r in success if r.get("doc_type") == "blog"]

    print(f"\nScraping complete:")
    print(f"  Transcripts: {len(transcripts)}")
    print(f"  Blog posts: {len(blogs)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Errors: {len(errors)}")

    if skipped:
        print("\nSkipped (could not parse):")
        for s in skipped:
            print(f"  - {s['slug']}: {s.get('error', 'Unknown')}")

    if errors:
        print("\nFailed episodes:")
        for e in errors:
            print(f"  - {e['slug']}: {e.get('error', 'Unknown error')}")

    # Step 3: List what's in the volume
    print("\n[3/3] Volume contents:")
    files = list_scraped_episodes.remote()
    print(f"  {len(files)} episode files saved to volume")

    if not scrape_only:
        print("\n" + "=" * 60)
        print("Scraping complete! Run ingestion locally:")
        print("  python -m src.scrapers.dwarkesh.cli --ingest-only")
        print("=" * 60)

    return {
        "total": len(episodes),
        "success": len(success),
        "errors": len(errors),
        "files": files,
    }
