"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient

from src.database.connection import close_db_pool, get_db_connection, init_db_pool
from src.main import app


@pytest.fixture(scope="session")
def db_pool():
    """Initialize database connection pool for the test session."""
    init_db_pool()
    yield
    close_db_pool()


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
        cur.execute("TRUNCATE docs CASCADE")
        db_connection.commit()

    yield db_connection

    # Clean after test
    with db_connection.cursor() as cur:
        cur.execute("TRUNCATE chunk_embeddings CASCADE")
        cur.execute("TRUNCATE chunks CASCADE")
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
