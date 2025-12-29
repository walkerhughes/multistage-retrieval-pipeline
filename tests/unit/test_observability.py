"""Unit tests for observability module."""

import os
import pytest
from src.observability.tracing import get_langsmith_config


class TestLangSmithConfig:
    """Test suite for LangSmith configuration."""

    def test_get_langsmith_config_with_api_key(self, monkeypatch):
        """Test that config is returned when API key is set."""
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-api-key")
        monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://test.endpoint.com")
        monkeypatch.setenv("LANGSMITH_TRACING", "true")

        config = get_langsmith_config()

        assert config["api_key"] == "test-api-key"
        assert config["project"] == "test-project"
        assert config["endpoint"] == "https://test.endpoint.com"
        assert config["tracing_enabled"] is True

    def test_get_langsmith_config_with_defaults(self, monkeypatch):
        """Test that defaults are used when optional vars are missing."""
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-api-key")
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
        monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

        config = get_langsmith_config()

        assert config["api_key"] == "test-api-key"
        assert config["project"] == "retrieval-evals"
        assert config["endpoint"] == "https://api.smith.langchain.com"
        assert config["tracing_enabled"] is True

    def test_get_langsmith_config_tracing_disabled(self, monkeypatch):
        """Test that tracing can be disabled."""
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-api-key")
        monkeypatch.setenv("LANGSMITH_TRACING", "false")

        config = get_langsmith_config()

        assert config["tracing_enabled"] is False

    def test_get_langsmith_config_missing_api_key(self, monkeypatch):
        """Test that ValueError is raised when API key is missing."""
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

        with pytest.raises(ValueError, match="LANGSMITH_API_KEY not found"):
            get_langsmith_config()
