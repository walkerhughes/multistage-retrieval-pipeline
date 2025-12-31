from dotenv import load_dotenv
from openai import OpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Database
    postgres_user: str
    postgres_password: str
    postgres_host: str
    postgres_port: int
    postgres_db: str

    # Chunking
    chunk_min_tokens: int = 400
    chunk_max_tokens: int = 800
    chunk_overlap_tokens: int = 50

    # Retrieval
    default_retrieval_n: int = 50
    default_rerank_k: int = 8  # Number of chunks to return after reranking

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    # OpenAI
    openai_api_key: str | None = None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # LangSmith (optional - for agent tracing)
    langsmith_api_key: str | None = None
    langsmith_project: str = "retrieval-evals"
    langsmith_tracing: bool = False  # Maps to LANGSMITH_TRACING env var

    @property
    def client(self):
        """Lazy-initialized OpenAI client."""
        if not hasattr(self, "_client"):
            self._client = OpenAI(api_key=self.openai_api_key)
        return self._client

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
