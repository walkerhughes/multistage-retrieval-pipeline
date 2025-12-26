#!/bin/bash

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default channel
CHANNEL="${1:-DwarkeshPatel}"

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}    YouTube Channel Data Loader${NC}"
echo -e "${BLUE}=================================================${NC}\n"

echo -e "${YELLOW}Channel: ${CHANNEL}${NC}"

# Check if Docker is running
echo -e "\n${BLUE}Checking Docker...${NC}"
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker is running${NC}"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠ .env file not found. Checking for .env.example...${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ Created .env from .env.example${NC}"
    else
        echo -e "${RED}❌ Neither .env nor .env.example found.${NC}"
        exit 1
    fi
fi

# Start database
echo -e "\n${BLUE}Starting Postgres database...${NC}"
docker compose down -v 2>/dev/null || true
docker compose up -d
sleep 5

# Verify database is ready
echo -e "${BLUE}Verifying database connection...${NC}"
RETRIES=10
while [ $RETRIES -gt 0 ]; do
    if docker exec retrieval-evals-db pg_isready -U retrieval_user -d retrieval_db > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Database is ready${NC}"
        break
    fi
    RETRIES=$((RETRIES - 1))
    if [ $RETRIES -eq 0 ]; then
        echo -e "${RED}❌ Failed to connect to database after multiple attempts${NC}"
        exit 1
    fi
    echo "Waiting for database..."
    sleep 1
done

# Install dependencies
echo -e "\n${BLUE}Installing dependencies...${NC}"
uv sync
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Load channel videos
echo -e "\n${BLUE}Loading videos from @${CHANNEL}...${NC}"
START_TIME=$(date +%s)

python -c "
from src.data.data_loader import ChannelDataLoader
from src.database.connection import init_db_pool

# Initialize database connection
init_db_pool()

# Load channel
loader = ChannelDataLoader(generate_embeddings=False)
result = loader.load_channel(channel='${CHANNEL}')

if result['success']:
    print(f'\n\n{\"=\"*50}')
    print(f'✅ Channel loading completed!')
    print(f'{\"=\"*50}')
    print(f'📊 Statistics:')
    print(f'   Videos loaded: {result[\"total_docs\"]}')
    print(f'   Chunks created: {result[\"total_chunks\"]}')
    print(f'   Failed videos: {result[\"failed_count\"]}')
    print(f'   Total time: {result[\"duration_seconds\"]} seconds')
    print(f'{\"=\"*50}\n')

    if result.get('failed_urls'):
        print(f'⚠️ Failed videos:')
        for failed in result['failed_urls']:
            print(f'   - {failed[\"url\"]}: {failed[\"error\"]}')
else:
    print(f'❌ Error: {result[\"error\"]}')
    exit(1)
"

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo -e "${BLUE}\n=================================================${NC}"
echo -e "${GREEN}✓ Setup complete in ${TOTAL_DURATION} seconds${NC}"
echo -e "${BLUE}=================================================${NC}\n"

echo -e "${GREEN}Next steps:${NC}"
echo -e "  1. Start the API server: ${YELLOW}.venv/bin/python -m src.main${NC}"
echo -e "  2. View API docs: ${YELLOW}open http://localhost:8000/docs${NC}"
echo -e "  3. Query the retrieved data: ${YELLOW}curl http://localhost:8000/api/retrieval/query?q=your+search+term${NC}\n"
