#!/usr/bin/env python
"""Ingest Dwarkesh Podcast transcripts using Modal scraper.

This script orchestrates the full ingestion pipeline:
1. Discover episodes from the podcast archive
2. Scrape transcripts in parallel using Modal
3. Ingest into local database with speaker turn structure

Usage:
    # Full pipeline: scrape with Modal, then ingest locally
    python -m src.scrapers.dwarkesh.cli

    # Scrape only (save to Modal volume, don't ingest)
    python -m src.scrapers.dwarkesh.cli --scrape-only

    # Ingest only (read from Modal volume)
    python -m src.scrapers.dwarkesh.cli --ingest-only

    # Limit number of episodes (for testing)
    python -m src.scrapers.dwarkesh.cli --limit 5

    # Skip embedding generation
    python -m src.scrapers.dwarkesh.cli --no-embeddings

    # Re-ingest all (don't skip existing)
    python -m src.scrapers.dwarkesh.cli --no-skip-existing

    # Clear all Dwarkesh data
    python -m src.scrapers.dwarkesh.cli --clear
"""

import argparse
import subprocess
import sys
from pathlib import Path


def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent


def run_modal_scrape(limit: int = 0) -> bool:
    """Run Modal scraper to fetch all episodes."""
    print("=" * 60)
    print("MODAL SCRAPING PHASE")
    print("=" * 60)

    # Build command
    cmd = [
        "modal", "run",
        "src/scrapers/dwarkesh/modal_app.py",
        "--scrape-only",
    ]

    if limit > 0:
        cmd.extend(["--limit", str(limit)])

    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=_get_project_root())
    return result.returncode == 0


def run_ingestion(
    generate_embeddings: bool = True,
    skip_existing: bool = True,
) -> dict:
    """Run local ingestion from Modal volume."""
    from src.scrapers.dwarkesh.pipeline import ingest_from_modal_volume

    return ingest_from_modal_volume(
        generate_embeddings=generate_embeddings,
        skip_existing=skip_existing,
    )


def clear_data() -> None:
    """Clear all Dwarkesh data from database."""
    from src.scrapers.dwarkesh.pipeline import clear_dwarkesh_data

    print("Clearing all Dwarkesh podcast data...")
    clear_dwarkesh_data()
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Dwarkesh Podcast transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--scrape-only",
        action="store_true",
        help="Only run Modal scraper, don't ingest to database",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only ingest from Modal volume, don't scrape",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of episodes to scrape (0 = all)",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding generation during ingestion",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-ingest episodes even if they exist in database",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all Dwarkesh data from database",
    )

    args = parser.parse_args()

    # Handle clear
    if args.clear:
        clear_data()
        return

    # Validate args
    if args.scrape_only and args.ingest_only:
        print("Error: Cannot use --scrape-only and --ingest-only together")
        sys.exit(1)

    # Run phases
    if not args.ingest_only:
        # Run Modal scraping
        success = run_modal_scrape(limit=args.limit)
        if not success:
            print("Modal scraping failed!")
            sys.exit(1)

    if not args.scrape_only:
        # Run ingestion
        results = run_ingestion(
            generate_embeddings=not args.no_embeddings,
            skip_existing=not args.no_skip_existing,
        )

        if results["failed"] > 0:
            print(f"\nWarning: {results['failed']} episodes failed to ingest")
            sys.exit(1)

    print("\nAll done!")


if __name__ == "__main__":
    main()
