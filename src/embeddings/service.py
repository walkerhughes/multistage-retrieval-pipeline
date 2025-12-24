from openai import OpenAI

from src.config import settings


class EmbeddingService:
    """Service for generating embeddings using OpenAI API."""

    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY not configured. Set it in .env file or environment."
            )
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text string.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        response = self.client.embeddings.create(input=text, model=self.model)
        embedding = response.data[0].embedding

        # Verify dimension matches expected
        if len(embedding) != self.dimensions:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.dimensions}, "
                f"got {len(embedding)}"
            )

        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in a single API call.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        response = self.client.embeddings.create(input=texts, model=self.model)

        # Extract embeddings in order
        embeddings = [item.embedding for item in response.data]

        # Verify all dimensions match
        for i, embedding in enumerate(embeddings):
            if len(embedding) != self.dimensions:
                raise ValueError(
                    f"Embedding dimension mismatch at index {i}. "
                    f"Expected {self.dimensions}, got {len(embedding)}"
                )

        return embeddings
