# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Postgres-first, high-performance retrieval system for YouTube video transcripts. The system emphasizes **fast, index-backed retrieval** (milliseconds) with evaluation-driven quality improvements. Currently at **M1 (Retrieval Baseline)** with Full-Text Search.

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
# Start Postgres with schema auto-initialization
docker compose up -d

# Verify database health
docker exec retrieval-evals-db pg_isready -U retrieval_user -d retrieval_db

# Connect to database for debugging
docker exec -it retrieval-evals-db psql -U retrieval_user -d retrieval_db

# Stop database
docker compose down
```

### Running the Application
```bash
# Start API server (with hot reload)
.venv/bin/python -m src.main

# Seed test data (YouTube API can be unreliable)
.venv/bin/python seed_test_data.py

# Test ingestion pipeline
.venv/bin/python test_ingestion.py
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
YouTube URL → LangChain Loader → Token Chunker → Postgres (FTS + tsvector) → FastAPI
```

### Key Components

**1. Ingestion Pipeline** (`src/ingestion/`)
- `youtube_loader.py`: Fetches transcripts using LangChain's YouTube loader
- `chunker.py`: Token-based chunking (400-800 tokens using tiktoken cl100k_base)
- `pipeline.py`: Orchestrates fetch → chunk → store workflow

**2. Database Layer** (`src/database/`)
- `connection.py`: psycopg3 connection pool (2-10 connections)
- `schema.sql`: Tables with generated tsvector columns and GIN indexes
- Tables: `docs` (video metadata), `chunks` (text with FTS), `chunk_embeddings` (future)

**3. Retrieval System** (`src/retrieval/`)
- `fts.py`: Full-Text Search using `websearch_to_tsquery` and `ts_rank`
- Supports metadata filters (date ranges, doc_type, source)
- Returns EXPLAIN ANALYZE for performance verification

**4. API Layer** (`src/api/`)
- `routes.py`: Three core endpoints + health check
- `schemas.py`: Pydantic models for request/response validation
- FastAPI with CORS middleware

**5. Configuration** (`src/config.py`)
- Pydantic Settings with `.env` support
- Centralizes DB connection, chunking params, and API config

### Database Schema

**docs table:**
- Stores YouTube video metadata (url, title, published_at)
- JSONB metadata field for extensibility
- Indexed on: published_at, doc_type, source, metadata (GIN), url

**chunks table:**
- Text chunks with `ord` (order within document)
- Generated `tsv` column (tsvector) for FTS
- Indexed on: tsvector (GIN - critical for performance), (doc_id, ord) unique

**chunk_embeddings table:**
- Created for M5 (hybrid retrieval)
- pgvector extension enabled but not yet used

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

### POST /api/ingest
Ingest YouTube transcript. **Note:** YouTube may block automated requests (HTTP 400). Use `seed_test_data.py` for testing.

Request:
```json
{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}
```

Returns: `doc_id`, `chunk_count`, `total_tokens`, `ingestion_time_ms`

### POST /api/query
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
    "source": "youtube"
  }
}
```

Returns: Ranked chunks with timing breakdown and metadata

### GET /api/bench/retrieval
Performance benchmark with EXPLAIN ANALYZE output. Use this to verify GIN index usage and detect regressions.

Query params: `?q=query&mode=fts`

Returns: `query_time_ms`, `rows_returned`, `explain` (full EXPLAIN ANALYZE output)

## Development Guidelines

### When Modifying Retrieval Logic
1. Always test with `/api/bench/retrieval` to verify index usage
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

### YouTube Transcript Limitations
- YouTube may block transcript requests (anti-bot measures)
- Not a bug in the system - use `seed_test_data.py` for local testing
- Production systems should implement retry logic and fallbacks

## Future Milestones (Roadmap)

- **M2**: Reranking agent with citation-enforced answers (K=8 selected from N=50)
- **M3**: LangSmith eval harness (retrieval-only vs reranked comparison)
- **M4**: Recency-aware scoring, domain allowlists, date filters
- **M5**: Hybrid retrieval (FTS + pgvector embeddings)

## Files of Interest

### Configuration & Setup
- `pyproject.toml`: Dependencies and project metadata
- `docker-compose.yml`: Postgres + pgvector container setup
- `.env.example`: Environment variable template

### Core Implementation
- `src/main.py`: FastAPI app with lifespan management
- `src/ingestion/pipeline.py`: End-to-end ingestion orchestration
- `src/retrieval/fts.py`: Full-Text Search implementation
- `src/database/schema.sql`: Database schema with indexes

### Testing & Utilities
- `seed_test_data.py`: Test data seeding script
- `test_ingestion.py`: Ingestion pipeline testing
- `src/utils/timing.py`: Performance timing utilities

## Important Implementation Notes

### FTS Query Building
- Uses `websearch_to_tsquery` for user-friendly query parsing (supports phrases, operators)
- Ranking via `ts_rank` on tsvector column
- Metadata filters are applied in WHERE clauses before ranking
- Always use parameterized queries (psycopg3 dict params)

### Error Handling
- YouTube ingestion failures should return 400 with descriptive messages
- Database errors should log and return 500 with sanitized messages
- Always validate URLs before attempting transcript fetch

### Performance Monitoring
- All endpoints track timing with `Timer` utility class
- Retrieval timing is separate from total request timing
- Benchmark endpoint provides EXPLAIN ANALYZE for query optimization
- Use BUFFERS option in EXPLAIN for I/O analysis
