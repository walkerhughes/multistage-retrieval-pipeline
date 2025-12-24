#!/usr/bin/env python3
"""Test script for ingestion pipeline."""

import sys
from src.ingestion.youtube_loader import YouTubeTranscriptFetcher
from src.ingestion.chunker import TokenBasedChunker

def test_youtube_loader():
    """Test YouTube transcript fetching."""
    print("Testing YouTube transcript loader...")

    # Try a few different videos
    test_urls = [
        "https://www.youtube.com/watch?v=aircAruvnKk",  # 3Blue1Brown neural networks
        "https://www.youtube.com/watch?v=RF5_MPSNAtU",  # Computerphile
    ]

    fetcher = YouTubeTranscriptFetcher()

    for url in test_urls:
        print(f"\nTrying: {url}")
        try:
            doc = fetcher.fetch(url)
            print(f"✓ Success!")
            print(f"  Title: {doc.title}")
            print(f"  Author: {doc.author}")
            print(f"  Text length: {len(doc.text)} chars")
            print(f"  Published: {doc.published_at}")

            # Test chunking
            print("\nTesting chunker...")
            chunker = TokenBasedChunker()
            chunks = chunker.chunk(doc.text)
            print(f"✓ Created {len(chunks)} chunks")
            if chunks:
                print(f"  First chunk: {chunks[0].token_count} tokens")
                print(f"  Last chunk: {chunks[-1].token_count} tokens")

            return True

        except Exception as e:
            print(f"✗ Error: {str(e)}")
            print(f"  Type: {type(e).__name__}")
            continue

    return False

if __name__ == "__main__":
    success = test_youtube_loader()
    sys.exit(0 if success else 1)
