"""Parser for Dwarkesh Podcast transcripts.

Handles the markdown-style transcript format:
- Speaker turns: **Name** _HH:MM:SS_
- Section headers: [(HH:MM:SS) – Topic Title]
"""

import re
from bs4 import BeautifulSoup

from src.scrapers.dwarkesh.models import Episode, EpisodeMetadata, ParsedSection, ParsedTurn


# Regex patterns for transcript parsing
# Speaker with timestamp: **Name** _HH:MM:SS_ or **Name** _H:MM:SS_
SPEAKER_PATTERN = re.compile(
    r"\*\*(?P<speaker>[^*]+)\*\*\s*_(?P<hours>\d{1,2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})_"
)

# Section header: [(HH:MM:SS) – Topic Title] or [(HH:MM:SS) - Topic Title]
# Note: Uses en-dash (–) or hyphen (-)
SECTION_PATTERN = re.compile(
    r"\[\((?P<hours>\d{1,2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})\)\s*[–\-]\s*(?P<title>[^\]]+)\]"
)

# Alternative markdown heading section: ### (HH:MM:SS) – Topic
SECTION_HEADING_PATTERN = re.compile(
    r"^###?\s*\(?(?P<hours>\d{1,2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})\)?\s*[–\-]\s*(?P<title>.+)$",
    re.MULTILINE,
)


def parse_timestamp_to_seconds(hours: str, minutes: str, seconds: str) -> int:
    """Convert HH:MM:SS components to total seconds."""
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


class DwarkeshParser:
    """Parse Dwarkesh Podcast transcript content."""

    def parse_transcript(self, content: str) -> tuple[list[ParsedTurn], list[ParsedSection]]:
        """
        Parse transcript content into turns and sections.

        Args:
            content: Transcript text (markdown-formatted)

        Returns:
            Tuple of (turns, sections)
        """
        sections = self._extract_sections(content)
        turns = self._extract_turns(content, sections)
        return turns, sections

    def _extract_sections(self, content: str) -> list[ParsedSection]:
        """Extract section headers with timestamps."""
        sections = []

        # Try bracket format first: [(HH:MM:SS) – Topic]
        for match in SECTION_PATTERN.finditer(content):
            timestamp_seconds = parse_timestamp_to_seconds(
                match.group("hours"), match.group("minutes"), match.group("seconds")
            )
            sections.append(
                ParsedSection(
                    title=match.group("title").strip(),
                    timestamp_seconds=timestamp_seconds,
                )
            )

        # Also try heading format: ### (HH:MM:SS) – Topic
        for match in SECTION_HEADING_PATTERN.finditer(content):
            timestamp_seconds = parse_timestamp_to_seconds(
                match.group("hours"), match.group("minutes"), match.group("seconds")
            )
            # Avoid duplicates
            if not any(s.timestamp_seconds == timestamp_seconds for s in sections):
                sections.append(
                    ParsedSection(
                        title=match.group("title").strip(),
                        timestamp_seconds=timestamp_seconds,
                    )
                )

        # Sort by timestamp
        sections.sort(key=lambda s: s.timestamp_seconds)
        return sections

    def _extract_turns(
        self, content: str, sections: list[ParsedSection]
    ) -> list[ParsedTurn]:
        """
        Extract speaker turns from transcript content.

        Each turn is identified by **Speaker** _HH:MM:SS_ pattern.
        Text following the pattern until the next speaker is the turn content.
        """
        turns = []

        # Find all speaker markers with their positions
        markers = list(SPEAKER_PATTERN.finditer(content))

        if not markers:
            return turns

        for i, match in enumerate(markers):
            speaker = match.group("speaker").strip()
            timestamp_seconds = parse_timestamp_to_seconds(
                match.group("hours"), match.group("minutes"), match.group("seconds")
            )

            # Extract text from end of this marker to start of next marker (or end of content)
            start_pos = match.end()
            end_pos = markers[i + 1].start() if i + 1 < len(markers) else len(content)

            # Get the raw text and clean it
            raw_text = content[start_pos:end_pos].strip()
            cleaned_text = self._clean_turn_text(raw_text)

            # Skip empty turns
            if not cleaned_text:
                continue

            # Find which section this turn belongs to
            section_title = self._find_section_for_timestamp(timestamp_seconds, sections)

            turns.append(
                ParsedTurn(
                    speaker=speaker,
                    start_time_seconds=timestamp_seconds,
                    text=cleaned_text,
                    section_title=section_title,
                    ord=len(turns),
                )
            )

        return turns

    def _clean_turn_text(self, text: str) -> str:
        """Clean turn text by removing section headers and extra whitespace."""
        # Remove section headers
        text = SECTION_PATTERN.sub("", text)
        text = SECTION_HEADING_PATTERN.sub("", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def _find_section_for_timestamp(
        self, timestamp_seconds: int, sections: list[ParsedSection]
    ) -> str | None:
        """Find the section title for a given timestamp."""
        if not sections:
            return None

        # Find the most recent section that started before or at this timestamp
        current_section = None
        for section in sections:
            if section.timestamp_seconds <= timestamp_seconds:
                current_section = section.title
            else:
                break

        return current_section

    def extract_transcript_from_html(self, html: str) -> str:
        """
        Extract transcript markdown from Substack HTML page.

        Args:
            html: Raw HTML from episode page

        Returns:
            Transcript content as markdown-like text
        """
        soup = BeautifulSoup(html, "lxml")

        # Find the main post content
        # Substack uses various class names, try common ones
        post_body = (
            soup.select_one(".body.markup")
            or soup.select_one(".post-content")
            or soup.select_one("article .body")
            or soup.select_one(".available-content")
        )

        if not post_body:
            # Fallback: try to find any div with transcript markers
            for div in soup.find_all("div"):
                text = div.get_text()
                if "**Dwarkesh" in text or "_00:0" in text:
                    post_body = div
                    break

        if not post_body:
            raise ValueError("Could not find transcript content in HTML")

        # Convert HTML to markdown-like text
        return self._html_to_markdown(post_body)

    def _html_to_markdown(self, element) -> str:
        """Convert HTML element to markdown-like text preserving formatting."""
        lines: list[str] = []

        for child in element.descendants:
            if child.name == "p":
                text = self._element_to_text(child)
                if text.strip():
                    lines.append(text.strip())
                    lines.append("")  # Add blank line after paragraph
            elif child.name in ("h1", "h2", "h3", "h4"):
                text = child.get_text().strip()
                if text:
                    prefix = "#" * int(child.name[1])
                    lines.append(f"{prefix} {text}")
                    lines.append("")

        return "\n".join(lines)

    def _element_to_text(self, element) -> str:
        """Convert single element to text with markdown formatting."""
        result: list[str] = []

        for child in element.children:
            if isinstance(child, str):
                result.append(child)
            elif child.name == "strong" or child.name == "b":
                result.append(f"**{child.get_text()}**")
            elif child.name == "em" or child.name == "i":
                result.append(f"_{child.get_text()}_")
            elif child.name == "a":
                result.append(child.get_text())
            elif child.name == "br":
                result.append("\n")
            else:
                result.append(child.get_text())

        return "".join(result)

    def parse_episode_metadata_from_html(self, html: str, url: str) -> EpisodeMetadata:
        """
        Extract episode metadata from HTML page.

        Args:
            html: Raw HTML from episode page
            url: Episode URL

        Returns:
            EpisodeMetadata with title, slug, etc.
        """
        soup = BeautifulSoup(html, "lxml")

        # Extract title
        title_elem = soup.select_one("h1.post-title") or soup.select_one("h1") or soup.find("title")
        title = title_elem.get_text().strip() if title_elem else "Unknown Episode"

        # Clean up title (remove site name suffix if present)
        if " - " in title:
            title = title.split(" - ")[0].strip()

        # Extract slug from URL
        slug = url.rstrip("/").split("/")[-1]

        # Try to extract published date
        published_at = None
        time_elem = soup.select_one("time")
        if time_elem and time_elem.get("datetime"):
            try:
                from datetime import datetime

                dt_str = str(time_elem["datetime"])
                published_at = datetime.fromisoformat(
                    dt_str.replace("Z", "+00:00")
                )
            except (ValueError, KeyError):
                pass

        return EpisodeMetadata(
            url=url,
            slug=slug,
            title=title,
            published_at=published_at,
        )

    def parse_full_episode(self, html: str, url: str) -> Episode:
        """
        Parse full episode from HTML.

        Args:
            html: Raw HTML from episode page
            url: Episode URL

        Returns:
            Complete Episode with metadata and turns
        """
        metadata = self.parse_episode_metadata_from_html(html, url)
        transcript_content = self.extract_transcript_from_html(html)
        turns, sections = self.parse_transcript(transcript_content)

        # Try to identify guest (first non-Dwarkesh speaker)
        guest = None
        for turn in turns:
            if turn.speaker != "Dwarkesh Patel":
                guest = turn.speaker
                break

        return Episode(
            url=metadata.url,
            slug=metadata.slug,
            title=metadata.title,
            guest=guest,
            published_at=metadata.published_at,
            turns=turns,
            sections=sections,
            raw_transcript=transcript_content,
        )
