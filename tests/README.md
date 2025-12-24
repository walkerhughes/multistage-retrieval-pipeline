# Tests

## Test Structure

```
tests/
├── conftest.py                            # Pytest fixtures and configuration
└── unit/                                  # Unit tests
│   └── test_ingestion.py                  # Legacy ingestion tests (manual)
└── integration/                           # Integration tests
    └── test_text_ingestion_integration.py
```

## Running Tests

### Prerequisites

1. **Install dev dependencies:**
   ```bash
   uv sync --extra dev
   ```

2. **Start the database:**
   ```bash
   docker compose up -d postgres
   ```

3. **Set OpenAI API key (for embedding tests):**
   ```bash
   export OPENAI_API_KEY=your-key-here
   ```

### Run All Tests

```bash
pytest
```

### Run by Marker

```bash
# Run only integration tests
pytest -m integration

# Run only unit tests
pytest -m unit

# Run tests that don't require OpenAI
pytest -m "not requires_openai"

# Run tests that require OpenAI (if key is set)
pytest -m requires_openai
```

### Run Specific Test File

```bash
pytest tests/test_text_ingestion_integration.py
```

### Run Specific Test

```bash
pytest tests/test_text_ingestion_integration.py::TestTextIngestionEndpoint::test_ingest_text_basic
```

### Verbose Output

```bash
pytest -v
```

### Show Print Statements

```bash
pytest -s
```

### Stop on First Failure

```bash
pytest -x
```

## Test Markers

- `@pytest.mark.unit` - Fast unit tests with no external dependencies
- `@pytest.mark.integration` - Integration tests requiring database/API
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.requires_openai` - Tests requiring OpenAI API key

## Fixtures

### Database Fixtures

- `db_pool` - Database connection pool (session scope)
- `db_connection` - Database connection (function scope)
- `clean_db` - Clean database before/after each test

### API Fixtures

- `test_client` - FastAPI TestClient for API testing

### Data Fixtures

- `sample_text` - Long sample text for testing
- `sample_short_text` - Short sample text for testing

## Writing New Tests

### Integration Test Example

```python
import pytest

@pytest.mark.integration
def test_my_feature(test_client, clean_db):
    # Arrange
    payload = {"text": "test data"}

    # Act
    response = test_client.post("/api/ingest/text", json=payload)

    # Assert
    assert response.status_code == 200
```

### Unit Test Example

```python
import pytest

@pytest.mark.unit
def test_my_function():
    # Arrange
    input_data = "test"

    # Act
    result = my_function(input_data)

    # Assert
    assert result == "expected"
```

## Continuous Integration

Tests are designed to run in CI environments. Ensure:
1. Database is available at `localhost:5433`
2. Environment variables are set
3. OpenAI-dependent tests are skipped if no API key

## Troubleshooting

### Tests Fail with Connection Error

Ensure database is running:
```bash
docker compose up -d postgres
docker compose ps
```

### Tests Fail with "No Module Named..."

Install dependencies:
```bash
uv sync --extra dev
```

### Embedding Tests Skipped

Set OpenAI API key:
```bash
export OPENAI_API_KEY=your-key-here
```
