# Multistage RAG Pipeline for YouTube Video Q&A

A production-grade, Postgres-first retrieval and response generation system for YouTube video transcripts. Built on Postgres Full-Text Search with semantic re-ranking. Features eval-driven quality improvements (to come).

**Key Features:**
- FastAPI server with ingestion, retrieval, and benchmarking endpoints
- Postgres Full-Text Search with tsvector + GIN indexes for sub-50ms retrieval
- Token-based chunking (400-800 tokens using tiktoken cl100k_base)
- Automatic text cleaning (removes newlines and backslashes)
- pgvector support for hybrid retrieval with embedding-based re-ranking

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Modify `.env` if needed. Default configuration works out of the box.

### 3. Start Postgres with Schema

```bash
make service  # or: docker compose up -d
```

The database automatically initializes with schema from [src/database/schema.sql](src/database/schema.sql).

### 4. Start API Server

```bash
make api  # or: .venv/bin/python -m src.main
```

Server runs on http://localhost:8000. Access interactive docs at http://localhost:8000/docs

### 5. Seed Test Data (Recommended)

Since YouTube may block automated transcript requests:

```bash
.venv/bin/python seed_test_data.py
```

This populates the database with sample transcript data for testing.

## API Endpoints

### POST /api/ingest/youtube

Ingest a YouTube video transcript with automatic text cleaning.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/ingest/youtube" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "title": "Optional title override",
    "metadata": {"category": "tutorial"}
  }'
```

**Response:**
```json
{
  "doc_id": 1,
  "title": "Video Title",
  "chunk_count": 42,
  "total_tokens": 25000,
  "ingestion_time_ms": 3200,
  "embeddings_generated": false
}
```

**Note:** YouTube may block automated requests (HTTP 400). Use `seed_test_data.py` for reliable testing.

### POST /api/ingest/text

Ingest raw text directly without fetching from YouTube.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/ingest/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text content here...",
    "title": "Document Title",
    "metadata": {"author": "John Doe"}
  }'
```

### POST /api/retrieval/query

Retrieve chunks using Full-Text Search with optional metadata filters.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/retrieval/query" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "neural networks deep learning",
    "n": 50,
    "filters": {
      "start_date": "2023-01-01",
      "end_date": "2024-01-01",
      "doc_type": "transcript"
    }
  }'
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
      "ord": 0,
      "token_count": 512
    }
  ],
  "timing_ms": {"retrieval": 8.5, "total": 10.2},
  "query_info": {
    "query": "neural networks deep learning",
    "n": 50,
    "results_returned": 15
  }
}
```

### GET /api/retrieval/bench

Performance benchmark with EXPLAIN ANALYZE output. Use this to verify GIN index usage.

**Request:**
```bash
curl "http://localhost:8000/api/retrieval/bench?query=machine+learning&mode=fts"
```

**Response:**
```json
{
  "query_time_ms": 6.2,
  "rows_returned": 15,
  "explain": "Bitmap Heap Scan on chunks...\n  -> Bitmap Index Scan on chunks_tsv_gin...",
  "query": "machine learning"
}
```

Check that the EXPLAIN output shows "Bitmap Index Scan on chunks_tsv_gin" to confirm optimal performance.

## Project Structure

```
retrieval-evals/
├── docker-compose.yml          # Postgres + pgvector container
├── Makefile                    # Convenience commands (make api, make service)
├── pyproject.toml              # Dependencies (managed by uv)
├── .env.example                # Environment configuration template
├── CLAUDE.md                   # Development instructions for Claude Code
├── README.md                   # This file
├── seed_test_data.py           # Test data seeder script
├── src/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration (Pydantic Settings)
│   ├── database/
│   │   ├── connection.py       # psycopg3 connection pooling
│   │   └── schema.sql          # Database schema (source of truth)
│   ├── ingestion/
│   │   ├── youtube_loader.py   # YouTube transcript fetching (LangChain)
│   │   ├── text_cleaner.py     # Transcript text cleaning
│   │   ├── chunker.py          # Token-based chunking (tiktoken)
│   │   └── pipeline.py         # Orchestrates fetch → clean → chunk → store
│   ├── retrieval/
│   │   └── fts.py              # Full-Text Search implementation
│   ├── api/
│   │   ├── schemas.py          # Pydantic request/response models
│   │   └── routes.py           # FastAPI route handlers
│   └── utils/
│       └── timing.py           # Performance timing utility
└── tests/
    ├── unit/                   # Unit tests
    └── integration/            # Integration tests (CI pipeline)
```

## Database Schema

### Tables

**docs**
- Stores YouTube video metadata: `url`, `title`, `published_at`, `raw_text`
- `metadata` JSONB field for extensibility
- Indexed on: `published_at`, `doc_type`, `source`, `metadata` (GIN), `url`

**chunks**
- Text chunks with `ord` (order within document) and `token_count`
- Generated `tsv` column (tsvector) updated automatically via trigger
- Indexed on: `tsv` (GIN - critical for performance), `(doc_id, ord)` unique constraint

**chunk_embeddings**
- pgvector embeddings with HNSW index
- Supports future hybrid retrieval (FTS + vector similarity)

### Critical Indexes

- `chunks_tsv_gin` - GIN index on tsvector (enables sub-50ms retrieval)
- `chunks_doc_ord_uq` - Unique index ensuring chunk ordering
- `docs_published_at_desc`, `docs_metadata_gin` - Metadata filtering

Always verify index usage with the `/api/retrieval/bench` endpoint to detect performance regressions.

## Performance Characteristics

| Metric | Target | Status |
|--------|--------|--------|
| Retrieval p95 | < 50ms | ✅ ~8-10ms |
| Index usage | GIN bitmap scan | ✅ Verified with EXPLAIN |
| Chunk size | 400-800 tokens | ✅ Configurable via .env |
| Connection pool | 2-10 connections | ✅ psycopg3 pooling |

## Technical Decisions

### Postgres Full-Text Search
- **websearch_to_tsquery**: User-friendly query parsing with automatic phrase/operator support
- **ts_rank**: Relevance scoring on tsvector columns
- **GIN indexes**: Enable bitmap index scans for fast retrieval (typically 5-15ms)
- Trade-off: Simpler than vector embeddings but sufficient for exact keyword matching

### LangChain YouTube Loader
- Simple API requiring no authentication
- Automatic caption/transcript handling
- Limitation: YouTube may block automated requests; use `seed_test_data.py` for testing

### psycopg3 with Connection Pooling
- Modern PostgreSQL driver with built-in connection pooling
- 2-10 connection pool for balanced performance/resource usage
- Synchronous design with `dict_row` factory for ergonomic query results

### tiktoken for Token Counting
- Uses GPT-4's `cl100k_base` tokenizer for accurate token budgets
- Chunking: 400-800 tokens with 50-token overlap to preserve context
- Order preserved via `ord` field in chunks table

### Text Cleaning
- Removes newlines (`\n`) and backslashes (`\`) from transcripts
- Improves FTS query matching and readability
- Applied automatically during ingestion pipeline

## Development

### Useful Commands

```bash
# Start services
make service           # Start Postgres + pgvector
make api              # Start API server with hot reload

# Database access
docker exec retrieval-evals-db psql -U retrieval_user -d retrieval_db

# Testing
.venv/bin/pytest tests/              # Run all tests
pyrefly check                         # Type-check codebase

# API access
open http://localhost:8000/docs      # Interactive API docs
curl http://localhost:8000/api/health # Health check
```

### Type Safety

Run `pyrefly check` before commits to identify type errors. All new code must pass type checking.

### Testing

```bash
# Run all tests
.venv/bin/pytest tests/

# Run integration tests only
.venv/bin/pytest tests/integration/

# Run with verbose output
.venv/bin/pytest tests/ -v
```

Tests run automatically in the CI pipeline on push/PR to prevent regressions.

## Configuration

Environment variables are loaded from `.env` (copy from `.env.example`):

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=retrieval_db
POSTGRES_USER=retrieval_user
POSTGRES_PASSWORD=retrieval_pass

# Chunking (tiktoken cl100k_base)
CHUNK_MIN_TOKENS=400
CHUNK_MAX_TOKENS=800
CHUNK_OVERLAP_TOKENS=50

# API
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true
```

All configuration is managed via Pydantic Settings in [src/config.py](src/config.py).

## Contributing

1. Create a feature branch from `main`
2. Make changes and add tests
3. Run `pyrefly check` to verify type safety
4. Run tests with `.venv/bin/pytest tests/`
5. Submit a pull request

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines.
