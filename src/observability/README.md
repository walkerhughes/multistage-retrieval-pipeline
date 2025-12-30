# Observability Module

This module provides LangSmith tracing integration for monitoring RAG agent performance, costs, and execution traces.

## Features

LangSmith automatically tracks:
- **Token usage**: Input and output tokens for each LLM call
- **Cost tracking**: Automatic cost calculation based on model pricing
- **Latency**: Execution time for each traced function
- **Full traces**: Complete execution flow with nested function calls

## Usage

### Basic Tracing

Add the `@traceable` decorator to any function you want to trace:

```python
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI

# Wrap OpenAI client for automatic cost tracking
client = wrap_openai(OpenAI())

@traceable(name="retrieval")
async def query_database(query: str):
    # Your retrieval logic here
    ...

@traceable(name="vanilla_rag")
async def rag(question: str):
    docs = await query_database(query=question)
    # Generate response
    resp = client.chat.completions.create(...)
    return str(resp.choices[0].message.content)
```

### Configuration

Set the following environment variables in your `.env` file:

```bash
# Required
LANGSMITH_API_KEY=lsv2_pt_...

# Optional (defaults shown)
LANGSMITH_PROJECT=retrieval-evals
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_TRACING=true
```

### Viewing Traces

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Select your project (default: `retrieval-evals`)
3. View traces with:
   - Execution timeline
   - Token usage and costs
   - Input/output for each step
   - Error details if failures occur

## API Reference

### `get_langsmith_config()`

Returns LangSmith configuration from environment variables.

**Returns:**
- `dict`: Configuration with keys:
  - `api_key`: LangSmith API key
  - `project`: Project name
  - `endpoint`: LangSmith API endpoint
  - `tracing_enabled`: Whether tracing is enabled

**Raises:**
- `ValueError`: If `LANGSMITH_API_KEY` is not set

**Example:**
```python
from src.observability import get_langsmith_config

config = get_langsmith_config()
print(f"Tracing to project: {config['project']}")
```

## Implementation Details

### Cost Tracking

`wrap_openai()` automatically intercepts OpenAI API calls and:
1. Extracts token usage from response
2. Calculates cost based on model pricing
3. Adds cost metadata to trace

No manual cost calculation needed!

### Trace Hierarchy

Traces are nested to show execution flow:

```
vanilla_rag (parent)
├── retrieval (child)
│   ├── latency: 450ms
│   └── chunks_returned: 10
└── generation (child)
    ├── model: gpt-4o
    ├── input_tokens: 1500
    ├── output_tokens: 200
    ├── cost: $0.0065
    └── latency: 2100ms
```

## Testing

Run tests with:
```bash
pytest tests/unit/test_observability.py -v
```

## See Also

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangSmith Python SDK](https://github.com/langchain-ai/langsmith-sdk)
