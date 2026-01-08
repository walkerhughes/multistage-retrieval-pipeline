# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Postgres-first, high-performance retrieval system for podcast transcripts. The system emphasizes **fast, index-backed retrieval** (milliseconds) with evaluation-driven quality improvements.

**Core Principle:** Retrieve broadly and cheaply → reason narrowly and expensively

## Quick Start Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Copy environment template (if needed)
cp .env.example .env
```

### Database Operations
```bash
# Start new Postgres with schema auto-initialization
make service

# Stop database/api
docker compose down
```

### Running the Application
```bash
# Start API server (with hot reload)
make api
```

### API Testing
```bash
# Access interactive API docs
open http://localhost:8000/docs

# Quick health check
curl http://localhost:8000/api/health
```

## Architecture Overview

### System Flow
```
Text/Transcript → Token Chunker → Postgres (FTS + tsvector) → FastAPI
```

### Key Components

**1. Ingestion Pipeline** (`src/ingestion/`)
- `chunker.py`: Token-based chunking (400-800 tokens using tiktoken cl100k_base)
- `pipeline.py`: Orchestrates chunk → embed → store workflow

**2. Database Layer** (`src/database/`)
- `connection.py`: psycopg3 connection pool (2-10 connections)
- `schema.sql`: Tables with generated tsvector columns and GIN indexes
- Tables: `docs` (document metadata), `chunks` (text with FTS), `chunk_embeddings` (embeddings)

**3. Retrieval System** (`src/retrieval/`)
- `fts.py`: Full-Text Search using `websearch_to_tsquery` and `ts_rank`
- Supports metadata filters (date ranges, doc_type, source)
- Returns EXPLAIN ANALYZE for performance verification

**4. API Layer** (`src/api/`)
- `routes.py`: Core endpoints + health check
- `schemas.py`: Pydantic models for request/response validation
- FastAPI with CORS middleware

**5. Configuration** (`src/config.py`)
- Pydantic Settings with `.env` support
- Centralizes DB connection, chunking params, and API config

### Database Schema

**docs table:**
- Stores document metadata (url, title, published_at)
- JSONB metadata field for extensibility
- Indexed on: published_at, doc_type, source, metadata (GIN), url

**chunks table:**
- Text chunks with `ord` (order within document)
- Generated `tsv` column (tsvector) for FTS
- Indexed on: tsvector (GIN - critical for performance), (doc_id, ord) unique

**chunk_embeddings table:**
- pgvector extension enabled and populates as new chunks added

### Performance Characteristics

**Critical Indexes:**
- `chunks_tsv_gin`: GIN index on tsvector (enables <50ms retrieval)
- `chunks_doc_ord_uq`: Unique index for chunk ordering
- Always verify index usage with `/api/bench/retrieval` endpoint

**Connection Pooling:**
- psycopg3 with 2-10 connection pool
- Synchronous with context managers
- All queries use `dict_row` factory

## API Endpoints

### POST /api/ingest/text
Ingest raw text directly.

Request:
```json
{
  "text": "Your text content here...",
  "title": "Optional document title",
  "metadata": {"author": "John Doe", "source": "manual"}
}
```

**Optional Fields:**
- `title`: Document title
- `metadata`: Custom metadata as key-value pairs

Returns: `doc_id`, `title`, `chunk_count`, `total_tokens`, `ingestion_time_ms`, `embeddings_generated`

### POST /api/retrieval/query
Retrieve chunks using FTS.

Request:
```json
{
  "q": "query string",
  "n": 50,
  "filters": {
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "doc_type": "transcript",
    "source": "dwarkesh"
  }
}
```

Returns: Ranked chunks with timing breakdown and metadata

### GET /api/retrieval/bench
Performance benchmark with EXPLAIN ANALYZE output. Use this to verify GIN index usage and detect regressions.

Query params: `?query=search+terms&mode=fts`

Returns: `query_time_ms`, `rows_returned`, `explain` (full EXPLAIN ANALYZE output)

## Development Guidelines

### When Modifying Retrieval Logic
1. Always test with `/api/retrieval/bench` to verify index usage
2. Check EXPLAIN output confirms "Bitmap Index Scan on chunks_tsv_gin"
3. Target: p95 < 50ms retrieval time
4. Never introduce table scans on the chunks table

### When Changing Database Schema
1. Modify `src/database/schema.sql` (source of truth)
2. Schema is auto-applied via docker-compose volume mount
3. For existing containers: `docker compose down -v && docker compose up -d`
4. Test with `seed_test_data.py` after schema changes

### Token Chunking Configuration
- Uses tiktoken's `cl100k_base` encoding (GPT-4 tokenizer)
- Configured in `src/config.py`: min=400, max=800, overlap=50 tokens
- Chunker preserves document order via `ord` field
- Each chunk stores `token_count` for budgeting

### Connection Management
- Use `get_db_connection()` context manager for all DB operations
- Pool is initialized in FastAPI lifespan handler
- For read-only queries: use `execute_query()`
- For inserts: use `execute_insert()` or explicit transactions

## Files of Interest

### Configuration & Setup
- `pyproject.toml`: Dependencies and project metadata
- `docker-compose.yml`: Postgres + pgvector container setup
- `.env.example`: Environment variable template

### Core Implementation
- `src/main.py`: FastAPI app with lifespan management
- `src/ingestion/pipeline.py`: End-to-end ingestion orchestration
- `src/retrieval/`: Retrieval implementations
- `src/database/schema.sql`: Database schema with indexes

## Important Implementation Notes

### FTS Query Building
- Uses `websearch_to_tsquery` for user-friendly query parsing (supports phrases, operators)
- Ranking via `ts_rank` on tsvector column
- Metadata filters are applied in WHERE clauses before ranking
- Always use parameterized queries (psycopg3 dict params)

### Error Handling
- Ingestion failures should return 400 with descriptive messages
- Database errors should log and return 500 with sanitized messages

### Performance Monitoring
- All endpoints track timing with `Timer` utility class
- Retrieval timing is separate from total request timing
- Benchmark endpoint provides EXPLAIN ANALYZE for query optimization
- Use BUFFERS option in EXPLAIN for I/O analysis

### Testing
New code must be tested with proper unit & integration tests as necessary in the `tests/` directory. These tests are run as part of the CI pipeline to avoid regressions.

### Type-Safety
Use the command `pyrefly check` to identify errors and address them before any commits.
