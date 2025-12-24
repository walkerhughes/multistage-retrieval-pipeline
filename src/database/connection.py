from contextlib import contextmanager
from typing import Generator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.config import settings

# Global connection pool
_pool: ConnectionPool | None = None


def init_db_pool() -> None:
    """Initialize the connection pool."""
    global _pool
    _pool = ConnectionPool(
        conninfo=settings.database_url,
        min_size=2,
        max_size=10,
        open=True,
    )


def close_db_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool:
        _pool.close()
        _pool = None


@contextmanager
def get_db_connection() -> Generator[psycopg.Connection, None, None]:
    """Context manager for database connections."""
    if not _pool:
        raise RuntimeError("Database pool not initialized")

    with _pool.connection() as conn:
        conn.row_factory = dict_row
        yield conn


def execute_query(query: str, params: dict | None = None) -> list[dict]:
    """Execute a query and return results as list of dicts."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or {})
            return cur.fetchall()


def execute_insert(query: str, params: dict | None = None) -> int:
    """Execute an INSERT and return the inserted ID."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or {})
            result = cur.fetchone()
            conn.commit()
            return result["id"] if result else None
