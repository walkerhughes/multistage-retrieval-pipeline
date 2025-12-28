#!/bin/bash
# Orchestrate full YouTube channel transcript ingestion pipeline
#
# Setup:
#   1. direnv allow . (one-time)
#   2. ./getting_started.sh <channel_name>
#
# Example:
#   ./getting_started.sh DwarkeshPatel
#
# This script:
#   1. Creates a temporary Modal Volume
#   2. Runs Modal pipeline to fetch transcripts
#   3. Downloads transcripts locally
#   4. Ingests to Postgres via local API
#   5. Cleans up (deletes volume, removes temp files)

set -e

# ============================================================================
# Configuration
# ============================================================================

if [ -z "$1" ]; then
    echo "Usage: ./getting_started.sh <channel_name>"
    echo "Example: ./getting_started.sh DwarkeshPatel"
    exit 1
fi

CHANNEL="$1"
VOLUME_NAME="transcripts"
TMP_DIR="./tmp/transcripts"
API_URL="${API_URL:-http://localhost:8000}"

# ============================================================================
# Helper Functions
# ============================================================================

log_step() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║ $1"
    echo "╚════════════════════════════════════════════════════════════╝"
}

log_error() {
    echo ""
    echo "❌ ERROR: $1"
    echo ""
}

check_api_health() {
    local max_retries=5
    local retry=0

    while [ $retry -lt $max_retries ]; do
        if curl -s -o /dev/null -w "%{http_code}" "$API_URL/api/health" | grep -q "200"; then
            echo "✓ API is healthy"
            return 0
        fi
        retry=$((retry + 1))
        if [ $retry -lt $max_retries ]; then
            echo "  Retrying in 2 seconds... ($retry/$max_retries)"
            sleep 2
        fi
    done

    log_error "Local API is not responding at $API_URL"
    echo "Please ensure:"
    echo "  1. Local API is running: python -m src.main"
    echo "  2. Database is running: docker compose up -d"
    exit 1
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Command not found: $1"
        echo "Please install: $2"
        exit 1
    fi
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

log_step "Pre-flight Checks"

check_command "modal" "Modal: pip install modal"
check_command "python" "Python 3.11+"
check_command "curl" "curl"

echo "✓ All required commands found"

# ============================================================================
# Step 1: Verify Local API is Running
# ============================================================================

log_step "Step 1: Verify Local API"

echo "Checking API health at $API_URL..."
check_api_health

# ============================================================================
# Step 2: Create Modal Volume
# ============================================================================

log_step "Step 2: Create Modal Volume"

echo "Creating volume: $VOLUME_NAME"
if modal volume create "$VOLUME_NAME"; then
    echo "✓ Volume created successfully"
else
    log_error "Failed to create Modal Volume"
    echo "Make sure you've authenticated with Modal: modal token set"
    exit 1
fi

# ============================================================================
# Step 3: Run Modal Pipeline
# ============================================================================

log_step "Step 3: Run Modal Pipeline"

echo "Discovering and fetching transcripts for: $CHANNEL"
if modal run src/data/modal_ingestion.py --channel "$CHANNEL" --volume "$VOLUME_NAME"; then
    echo "✓ Modal pipeline completed"
else
    log_error "Modal pipeline failed"
    echo "Cleaning up volume: $VOLUME_NAME"
    modal volume delete "$VOLUME_NAME" || true
    exit 1
fi

# ============================================================================
# Step 4: Download Transcripts from Volume
# ============================================================================

log_step "Step 4: Download Transcripts"

echo "Downloading transcripts from volume..."
mkdir -p "$TMP_DIR"

if modal volume get "$VOLUME_NAME" "$TMP_DIR"; then
    transcript_count=$(find "$TMP_DIR" -name "*.json" ! -name "*_error.json" | wc -l)
    echo "✓ Downloaded $transcript_count transcripts"
else
    log_error "Failed to download transcripts from Modal Volume"
    echo "Cleaning up volume: $VOLUME_NAME"
    modal volume delete "$VOLUME_NAME" || true
    exit 1
fi

# ============================================================================
# Step 5: Ingest to Local Postgres
# ============================================================================

log_step "Step 5: Ingest to Postgres"

echo "Ingesting transcripts via: $API_URL/api/ingest/text"
if python scripts/ingest_from_volume.py --dir "$TMP_DIR" --api-url "$API_URL"; then
    echo "✓ Ingestion completed"
else
    log_error "Ingestion failed"
    echo "Transcript files remain in: $TMP_DIR"
    echo "Volume remains: $VOLUME_NAME"
    echo "You can retry ingestion with:"
    echo "  python scripts/ingest_from_volume.py --dir $TMP_DIR --api-url $API_URL"
    exit 1
fi

# ============================================================================
# Step 6: Cleanup
# ============================================================================

log_step "Step 6: Cleanup"

echo "Deleting Modal Volume: $VOLUME_NAME"
if modal volume delete "$VOLUME_NAME"; then
    echo "✓ Volume deleted"
else
    echo "⚠ Warning: Failed to delete volume (may need manual cleanup)"
fi

echo "Removing local temp files: $TMP_DIR"
rm -rf "$TMP_DIR"
echo "✓ Cleaned up"

# ============================================================================
# Complete
# ============================================================================

log_step "✓ Complete!"

echo "All transcripts have been ingested into Postgres."
echo ""
echo "Next steps:"
echo "  1. Verify data in Postgres:"
echo "     docker exec -it retrieval-evals-db psql -U retrieval_user -d retrieval_db -c 'SELECT COUNT(*) FROM chunks;'"
echo "  2. Test retrieval:"
echo "     curl -X POST http://localhost:8000/api/retrieval/query -H 'Content-Type: application/json' -d '{\"q\": \"your search query\"}'"
