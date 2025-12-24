# Retrieval Evals - YouTube Transcript Retrieval System

**M1 (Retrieval Baseline)** implementation complete.

A Postgres-first, fast, eval-driven retrieval system for YouTube video transcripts with Full-Text Search.

## Architecture

```
YouTube URL → LangChain Loader → Token Chunker → Postgres (FTS) → FastAPI
```

**Key Features:**
- FastAPI server with 3 endpoints: `/ingest`, `/query`, `/bench/retrieval`
- Postgres with FTS (tsvector + GIN indexes)
- Token-based chunking (400-800 tokens using tiktoken)
- Connection pooling for performance

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Start Postgres

```bash
docker compose up -d
```

The database will automatically initialize with the schema from `src/database/schema.sql`.

### 3. Start API Server

```bash
.venv/bin/python -m src.main
```

Server runs on http://localhost:8000

### 4. Seed Test Data (Optional)

Since YouTube transcript API can be blocked:

```bash
.venv/bin/python seed_test_data.py
```

## API Endpoints

### POST /api/ingest

Ingest a YouTube video transcript.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/ingest" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

**Response:**
```json
{
  "status": "accepted",
  "doc_id": 1,
  "url": "...",
  "title": "Video Title",
  "chunk_count": 42,
  "total_tokens": 25000,
  "ingestion_time_ms": 3200
}
```

**Note:** YouTube's API sometimes blocks automated requests. Use `seed_test_data.py` for testing.

### POST /api/query

Retrieve chunks using Full-Text Search.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"q": "neural networks deep learning", "n": 10}'
```

**Response:**
```json
{
  "chunks": [
    {
      "chunk_id": 1,
      "doc_id": 1,
      "score": 0.82,
      "text": "...",
      "metadata": {"url": "...", "title": "...", "published_at": "..."},
      "ord": 0
    }
  ],
  "timing_ms": {"retrieval": 10.26, "total": 10.26},
  "query_info": {
    "query": "neural networks deep learning",
    "n": 10,
    "results_returned": 1
  }
}
```

### GET /api/bench/retrieval

Benchmark retrieval performance with EXPLAIN output.

**Request:**
```bash
curl "http://localhost:8000/api/bench/retrieval?q=machine+learning&mode=fts"
```

**Response:**
```json
{
  "query_time_ms": 6.22,
  "rows_returned": 1,
  "explain": "EXPLAIN (ANALYZE, BUFFERS) ...",
  "query": "machine learning"
}
```

## Project Structure

```
retrieval-evals/
├── docker-compose.yml          # Postgres + pgvector
├── pyproject.toml              # Dependencies
├── .env.example                # Config template
├── SPEC.md                     # System specification
├── README.md                   # This file
├── seed_test_data.py          # Test data seeder
├── test_ingestion.py          # Ingestion testing
├── src/
│   ├── main.py                # FastAPI application
│   ├── config.py              # Configuration
│   ├── database/
│   │   ├── connection.py      # DB pooling
│   │   └── schema.sql         # Tables + indexes
│   ├── ingestion/
│   │   ├── youtube_loader.py  # Transcript fetching
│   │   ├── chunker.py         # Token-based chunking
│   │   └── pipeline.py        # End-to-end ingestion
│   ├── retrieval/
│   │   └── fts.py             # Full-Text Search
│   ├── api/
│   │   ├── schemas.py         # Pydantic models
│   │   └── routes.py          # FastAPI routes
│   └── utils/
│       └── timing.py          # Performance timing
```

## Database Schema

### Tables

- **docs** - YouTube video metadata (url, title, published_at, raw_text, metadata jsonb)
- **chunks** - Text chunks with generated tsvector column, token_count, ord (order)
- **chunk_embeddings** - (Optional, for M5) Vector embeddings

### Indexes

- `chunks_tsv_gin` - GIN index on tsvector (critical for FTS performance)
- `chunks_doc_ord_uq` - Unique index on (doc_id, ord)
- `docs_published_at_desc`, `docs_metadata_gin` - Filter indexes

## Performance Targets (M1)

| Metric | Target | Actual |
|--------|--------|--------|
| Retrieval p95 | < 50ms | ~10ms ✅ |
| Index usage | GIN index scan | Verified with EXPLAIN ✅ |
| Chunk size | 400-800 tokens | Configured ✅ |

## Technical Decisions

### LangChain YouTube Loader
- Simple API, no authentication needed
- Handles captions automatically
- Note: YouTube may block automated requests

### psycopg3 with Connection Pooling
- Modern async support
- Built-in pooling (2-10 connections)
- Better performance than psycopg2

### tiktoken for Token Counting
- Same tokenizer as GPT-4
- Accurate token budgets
- cl100k_base encoding

### websearch_to_tsquery
- User-friendly query parsing
- Supports phrases and operators automatically

## Next Steps (Future Milestones)

- **M2**: Reranking agent with citation-enforced answers
- **M3**: LangSmith eval harness
- **M4**: Recency-aware scoring, domain allowlists
- **M5**: Hybrid retrieval (FTS + vector embeddings)

## Development Commands

```bash
# Start database
docker compose up -d

# Run API
.venv/bin/python -m src.main

# Seed test data
.venv/bin/python seed_test_data.py

# Access API docs
open http://localhost:8000/docs

# Check database
docker exec retrieval-evals-db psql -U retrieval_user -d retrieval_db
```

## Configuration

Copy `.env.example` to `.env` and modify as needed:

```bash
# Database
POSTGRES_PORT=5433
POSTGRES_DB=retrieval_db

# Chunking
CHUNK_MIN_TOKENS=400
CHUNK_MAX_TOKENS=800
CHUNK_OVERLAP_TOKENS=50

# API
API_PORT=8000
```
