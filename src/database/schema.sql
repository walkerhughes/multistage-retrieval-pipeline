-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- ========================================
-- TABLES
-- ========================================

-- Documents table (YouTube videos)
CREATE TABLE docs (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT,
    doc_type TEXT NOT NULL DEFAULT 'transcript',
    published_at TIMESTAMPTZ,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw_text TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB
);

-- Chunks table with generated tsvector column
CREATE TABLE chunks (
    id BIGSERIAL PRIMARY KEY,
    doc_id BIGINT NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    ord INT NOT NULL,
    text TEXT NOT NULL,
    token_count INT NOT NULL,
    tsv TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', COALESCE(text, '')), 'A')
    ) STORED
);

-- Optional: Embeddings table (create now, populate in M5)
CREATE TABLE chunk_embeddings (
    chunk_id BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    embedding VECTOR(1536) NOT NULL
);

-- ========================================
-- INDEXES
-- ========================================

-- FTS index (critical for performance)
CREATE INDEX chunks_tsv_gin ON chunks USING GIN(tsv);

-- Chunk ordering and joins
CREATE UNIQUE INDEX chunks_doc_ord_uq ON chunks(doc_id, ord);
CREATE INDEX chunks_doc_id_idx ON chunks(doc_id);

-- Document filters
CREATE INDEX docs_published_at_desc ON docs(published_at DESC);
CREATE INDEX docs_doc_type_idx ON docs(doc_type);
CREATE INDEX docs_source_idx ON docs(source);
CREATE INDEX docs_metadata_gin ON docs USING GIN(metadata);
CREATE INDEX docs_url_idx ON docs(url);

-- Vector similarity index (M5) - HNSW for fast approximate nearest neighbor search
CREATE INDEX chunk_embeddings_hnsw ON chunk_embeddings
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- ========================================
-- HELPER FUNCTIONS
-- ========================================

-- Function to get document stats
CREATE OR REPLACE FUNCTION get_doc_stats()
RETURNS TABLE(
    total_docs BIGINT,
    total_chunks BIGINT,
    avg_chunks_per_doc NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(DISTINCT d.id) AS total_docs,
        COUNT(c.id) AS total_chunks,
        ROUND(COUNT(c.id)::NUMERIC / NULLIF(COUNT(DISTINCT d.id), 0), 2) AS avg_chunks_per_doc
    FROM docs d
    LEFT JOIN chunks c ON d.id = c.doc_id;
END;
$$ LANGUAGE plpgsql;
