"""LangSmith tracing utilities for observability.

LangSmith automatically provides:
- Token usage tracking (input/output tokens)
- Cost tracking (via wrap_openai)
- Latency measurements
- Full execution traces

Simply use:
    from langsmith import traceable
    from langsmith.wrappers import wrap_openai

    client = wrap_openai(OpenAI())

    @traceable(name="my_function")
    def my_function():
        ...
"""

import os
from typing import Any, Dict


def get_langsmith_config() -> Dict[str, Any]:
    """Get LangSmith configuration from environment variables.

    Returns:
        Dict with LangSmith configuration

    Raises:
        ValueError: If required environment variables are missing
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        raise ValueError(
            "LANGSMITH_API_KEY not found in environment. "
            "Please set it in .env file or environment variables."
        )

    return {
        "api_key": api_key,
        "project": os.getenv("LANGSMITH_PROJECT", "retrieval-evals"),
        "endpoint": os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        "tracing_enabled": os.getenv("LANGSMITH_TRACING", "true").lower() == "true",
    }
