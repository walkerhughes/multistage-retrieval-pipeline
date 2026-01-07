"""Orchestration pipeline for Dwarkesh Podcast ingestion.

Coordinates:
1. Modal scraping (parallel)
2. Reading from Modal volume
3. Local database ingestion
"""

import json
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.database.connection import close_db_pool, execute_query, get_db_connection, init_db_pool
from src.ingestion.pipeline import IngestionPipeline


def ingest_from_modal_volume(
    generate_embeddings: bool = True,
    skip_existing: bool = True,
) -> dict:
    """
    Ingest episodes from Modal volume into local database.

    This function:
    1. Reads all episode JSON files from the Modal volume
    2. Checks for existing episodes (by URL) if skip_existing=True
    3. Ingests new episodes using ingest_with_turns()

    Args:
        generate_embeddings: Whether to generate embeddings during ingestion
        skip_existing: Skip episodes that are already in the database

    Returns:
        Dict with ingestion statistics
    """
    import modal

    print("=" * 60)
    print("DWARKESH PODCAST INGESTION")
    print("=" * 60)

    # Read all episodes from Modal volume directly using Volume API
    print("\n[1/3] Reading episodes from Modal volume...")
    try:
        volume = modal.Volume.from_name("dwarkesh-transcripts")
        print("  Connected to Modal volume 'dwarkesh-transcripts'")

        # List all JSON files in the volume root
        # Files are saved to /data/transcripts/ in Modal, which maps to the volume root
        episodes = []

        # Try different paths that might contain the files
        for dir_path in ["/", "/transcripts", "/"]:
            try:
                entries = list(volume.listdir(dir_path))
                if entries:
                    print(f"  Found {len(entries)} entries at {dir_path}")
                    for entry in entries:
                        if entry.path.endswith(".json"):
                            try:
                                # Construct full path
                                full_path = f"{dir_path.rstrip('/')}/{entry.path}" if dir_path != "/" else entry.path
                                file_bytes = b"".join(volume.read_file(full_path))
                                episode_data = json.loads(file_bytes.decode("utf-8"))
                                episodes.append(episode_data)
                                print(f"    Loaded: {entry.path}")
                            except Exception as file_err:
                                print(f"    Warning: Could not read {entry.path}: {file_err}")
                    if episodes:
                        break
            except Exception as list_err:
                print(f"  Could not list {dir_path}: {list_err}")
                continue

    except Exception as e:
        error_msg = str(e)
        if "not found" in error_msg.lower() or "NotFoundError" in str(type(e)):
            print("Error: Modal volume 'dwarkesh-transcripts' not found.")
        else:
            print(f"Error reading from Modal volume: {e}")
        print("Make sure you ran the scraper first: modal run src/scrapers/dwarkesh/modal_app.py --scrape-only")
        return {"total": 0, "ingested": 0, "skipped": 0, "failed": 0, "error": str(e)}

    print(f"Found {len(episodes)} episodes in volume")

    if not episodes:
        print("No episodes to ingest!")
        return {"total": 0, "ingested": 0, "skipped": 0, "failed": 0}

    # Sort by published_at (oldest first)
    episodes.sort(
        key=lambda e: (
            e.get("published_at") is None,
            e.get("published_at") or "9999",
        )
    )

    # Initialize database
    print("\n[2/3] Initializing database...")
    init_db_pool()

    # Get existing URLs if skipping
    existing_urls = set()
    if skip_existing:
        try:
            rows = execute_query(
                "SELECT url FROM docs WHERE source = 'dwarkesh'",
                {},
            )
            existing_urls = {row["url"] for row in rows}
            print(f"Found {len(existing_urls)} existing episodes in database")
        except Exception as e:
            print(f"Warning: Could not check existing episodes: {e}")

    # Initialize pipeline
    pipeline = IngestionPipeline(generate_embeddings=generate_embeddings)

    # Ingest episodes
    print(f"\n[3/3] Ingesting {len(episodes)} episodes...")
    results = {"total": len(episodes), "ingested": 0, "skipped": 0, "failed": 0}

    for episode in tqdm(episodes, desc="Ingesting"):
        url = episode["url"]
        slug = episode["slug"]

        # Skip if already exists
        if url in existing_urls:
            results["skipped"] += 1
            continue

        try:
            # Parse published_at
            published_at = None
            if episode.get("published_at"):
                try:
                    published_at = datetime.fromisoformat(
                        episode["published_at"].replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    pass

            # Ingest with turns
            doc_type = episode.get("doc_type", "transcript")
            result = pipeline.ingest_with_turns(
                turns=episode["turns"],
                title=episode["title"],
                url=url,
                published_at=published_at,
                metadata={
                    "guest": episode.get("guest"),
                    "slug": slug,
                    "scraped_at": episode.get("scraped_at"),
                },
                doc_type=doc_type,
            )

            results["ingested"] += 1
            content_label = "sections" if doc_type == "blog" else "turns"
            tqdm.write(f"  ✓ {slug} ({doc_type}): doc_id={result['doc_id']}, {content_label}={result['turn_count']}, chunks={result['chunk_count']}")

        except Exception as e:
            results["failed"] += 1
            tqdm.write(f"  ✗ {slug}: {e}")

    # Cleanup
    close_db_pool()

    # Summary
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Total episodes: {results['total']}")
    print(f"Ingested: {results['ingested']}")
    print(f"Skipped (existing): {results['skipped']}")
    print(f"Failed: {results['failed']}")

    return results


def ingest_from_json_files(
    json_dir: str | Path,
    generate_embeddings: bool = True,
    skip_existing: bool = True,
) -> dict:
    """
    Ingest episodes from local JSON files.

    This is useful for testing or when episodes were saved locally
    instead of to a Modal volume.

    Args:
        json_dir: Directory containing episode JSON files
        generate_embeddings: Whether to generate embeddings
        skip_existing: Skip episodes already in database

    Returns:
        Dict with ingestion statistics
    """
    json_dir = Path(json_dir)

    if not json_dir.exists():
        raise ValueError(f"Directory not found: {json_dir}")

    # Load all JSON files
    episodes = []
    for path in sorted(json_dir.glob("*.json")):
        try:
            episodes.append(json.loads(path.read_text()))
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")

    if not episodes:
        print("No episode JSON files found!")
        return {"total": 0, "ingested": 0, "skipped": 0, "failed": 0}

    # Sort by published_at (oldest first)
    episodes.sort(
        key=lambda e: (
            e.get("published_at") is None,
            e.get("published_at") or "9999",
        )
    )

    print(f"Found {len(episodes)} episode files")

    # Initialize database
    init_db_pool()

    # Get existing URLs
    existing_urls = set()
    if skip_existing:
        try:
            rows = execute_query(
                "SELECT url FROM docs WHERE source = 'dwarkesh'",
                {},
            )
            existing_urls = {row["url"] for row in rows}
        except Exception:
            pass

    # Initialize pipeline
    pipeline = IngestionPipeline(generate_embeddings=generate_embeddings)

    # Ingest
    results = {"total": len(episodes), "ingested": 0, "skipped": 0, "failed": 0}

    for episode in tqdm(episodes, desc="Ingesting"):
        url = episode["url"]

        if url in existing_urls:
            results["skipped"] += 1
            continue

        try:
            published_at = None
            if episode.get("published_at"):
                try:
                    published_at = datetime.fromisoformat(
                        episode["published_at"].replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    pass

            pipeline.ingest_with_turns(
                turns=episode["turns"],
                title=episode["title"],
                url=url,
                published_at=published_at,
                metadata={
                    "guest": episode.get("guest"),
                    "slug": episode["slug"],
                },
            )
            results["ingested"] += 1

        except Exception as e:
            results["failed"] += 1
            tqdm.write(f"Failed {episode['slug']}: {e}")

    close_db_pool()

    return results


def clear_dwarkesh_data() -> None:
    """Clear all Dwarkesh podcast data from the database."""
    init_db_pool()

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get doc_ids first
                cur.execute(
                    "SELECT id FROM docs WHERE source = 'dwarkesh'"
                )
                doc_ids = [row["id"] for row in cur.fetchall()]

                if not doc_ids:
                    print("No Dwarkesh data to clear")
                    return

                # Delete in order (respecting FKs)
                cur.execute(
                    """
                    DELETE FROM chunk_embeddings
                    WHERE chunk_id IN (
                        SELECT id FROM chunks WHERE doc_id = ANY(%(doc_ids)s)
                    )
                    """,
                    {"doc_ids": doc_ids},
                )
                cur.execute(
                    "DELETE FROM chunks WHERE doc_id = ANY(%(doc_ids)s)",
                    {"doc_ids": doc_ids},
                )
                cur.execute(
                    "DELETE FROM turns WHERE doc_id = ANY(%(doc_ids)s)",
                    {"doc_ids": doc_ids},
                )
                cur.execute(
                    "DELETE FROM docs WHERE id = ANY(%(doc_ids)s)",
                    {"doc_ids": doc_ids},
                )
                conn.commit()

                print(f"Cleared {len(doc_ids)} Dwarkesh episodes from database")

    finally:
        close_db_pool()
