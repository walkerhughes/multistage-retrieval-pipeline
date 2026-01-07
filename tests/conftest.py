"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient

from src.agents.helpers import flush_traces, initialize_tracing, shutdown_tracing
from src.database.connection import close_db_pool, get_db_connection, init_db_pool
from src.main import app


@pytest.fixture(scope="session")
def db_pool():
    """Initialize database connection pool for the test session."""
    init_db_pool()
    initialize_tracing()
    yield
    shutdown_tracing()
    close_db_pool()


@pytest.fixture(scope="function", autouse=True)
def flush_traces_after_test():
    """Flush LangSmith traces after each test to ensure they're captured."""
    yield
    flush_traces()


@pytest.fixture(scope="function")
def db_connection(db_pool):
    """Provide a database connection for each test function."""
    with get_db_connection() as conn:
        yield conn


@pytest.fixture(scope="function")
def clean_db(db_connection):
    """Clean database before and after each test."""
    # Clean before test
    with db_connection.cursor() as cur:
        cur.execute("TRUNCATE chunk_embeddings CASCADE")
        cur.execute("TRUNCATE chunks CASCADE")
        cur.execute("TRUNCATE turns CASCADE")
        cur.execute("TRUNCATE docs CASCADE")
        db_connection.commit()

    yield db_connection

    # Clean after test
    with db_connection.cursor() as cur:
        cur.execute("TRUNCATE chunk_embeddings CASCADE")
        cur.execute("TRUNCATE chunks CASCADE")
        cur.execute("TRUNCATE turns CASCADE")
        cur.execute("TRUNCATE docs CASCADE")
        db_connection.commit()


@pytest.fixture(scope="module")
def test_client():
    """Provide FastAPI test client."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_text():
    """Provide sample text for testing."""
    return """
    Machine learning is a subset of artificial intelligence that focuses on the development
    of algorithms and statistical models. These models enable computers to improve their
    performance on a specific task through experience without being explicitly programmed.

    The fundamental concept behind machine learning is pattern recognition. By analyzing
    large amounts of data, machine learning systems can identify patterns and make predictions
    or decisions based on those patterns. This capability has revolutionized many fields
    including natural language processing, computer vision, and data analytics.
    """


@pytest.fixture
def sample_short_text():
    """Provide short sample text for testing."""
    return "Machine learning is a cool technology that helps computers learn from data."


@pytest.fixture
def sample_podcast_with_turns(clean_db):
    """Create a sample podcast document with turns and chunks for testing expansion endpoints.

    Returns dict with doc_id, turn_ids, and chunk_ids for use in tests.
    """
    with clean_db.cursor() as cur:
        # Create document
        cur.execute(
            """
            INSERT INTO docs (source, url, title, doc_type, raw_text, metadata)
            VALUES ('dwarkesh', 'https://example.com/podcast', 'Test Podcast Episode', 'transcript',
                    'Full transcript text here', '{"guest": "Test Guest"}')
            RETURNING id
            """
        )
        doc_id = cur.fetchone()["id"]

        # Create turns (simulating a podcast conversation)
        turns_data = [
            (0, "Dwarkesh Patel", 0, "What are your thoughts on AI safety?", "Introduction"),
            (1, "Test Guest", 60, "AI safety is crucial for several reasons. First, we need to consider alignment.", "AI Safety"),
            (2, "Dwarkesh Patel", 120, "Can you elaborate on alignment specifically?", "AI Safety"),
            (3, "Test Guest", 180, "Alignment is about ensuring AI systems do what we intend. This involves technical research.", "AI Safety"),
            (4, "Dwarkesh Patel", 240, "What about governance and policy?", "Governance"),
            (5, "Test Guest", 300, "Governance frameworks are essential. We need international cooperation.", "Governance"),
        ]

        turn_ids = []
        for ord_val, speaker, start_time, text, section in turns_data:
            cur.execute(
                """
                INSERT INTO turns (doc_id, ord, speaker, start_time_seconds, text, section_title, token_count, metadata)
                VALUES (%(doc_id)s, %(ord)s, %(speaker)s, %(start_time)s, %(text)s, %(section)s, %(tokens)s, '{}')
                RETURNING id
                """,
                {
                    "doc_id": doc_id,
                    "ord": ord_val,
                    "speaker": speaker,
                    "start_time": start_time,
                    "text": text,
                    "section": section,
                    "tokens": len(text.split()) * 2,  # Approximate token count
                }
            )
            turn_ids.append(cur.fetchone()["id"])

        # Create chunks linked to turns
        # Turn 1 (guest's first answer) is long enough to have 2 chunks
        chunks_data = [
            (0, turn_ids[0], "What are your thoughts on AI safety?"),
            (1, turn_ids[1], "AI safety is crucial for several reasons."),
            (2, turn_ids[1], "First, we need to consider alignment."),  # Same turn, different chunk
            (3, turn_ids[2], "Can you elaborate on alignment specifically?"),
            (4, turn_ids[3], "Alignment is about ensuring AI systems do what we intend."),
            (5, turn_ids[3], "This involves technical research."),  # Same turn
            (6, turn_ids[4], "What about governance and policy?"),
            (7, turn_ids[5], "Governance frameworks are essential. We need international cooperation."),
        ]

        chunk_ids = []
        for ord_val, turn_id, text in chunks_data:
            cur.execute(
                """
                INSERT INTO chunks (doc_id, turn_id, ord, text, token_count)
                VALUES (%(doc_id)s, %(turn_id)s, %(ord)s, %(text)s, %(tokens)s)
                RETURNING id
                """,
                {
                    "doc_id": doc_id,
                    "turn_id": turn_id,
                    "ord": ord_val,
                    "text": text,
                    "tokens": len(text.split()) * 2,
                }
            )
            chunk_ids.append(cur.fetchone()["id"])

        clean_db.commit()

        return {
            "doc_id": doc_id,
            "turn_ids": turn_ids,
            "chunk_ids": chunk_ids,
            "turns_data": turns_data,
        }
