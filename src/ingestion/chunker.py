from dataclasses import dataclass

import tiktoken

from src.config import settings


@dataclass
class Chunk:
    text: str
    token_count: int
    ord: int  # Order within document


class TokenBasedChunker:
    """
    Chunks text into token-based segments with optional overlap.
    Uses tiktoken (OpenAI tokenizer) for accurate token counting.
    """

    def __init__(
        self,
        min_tokens: int = settings.chunk_min_tokens,
        max_tokens: int = settings.chunk_max_tokens,
        overlap_tokens: int = settings.chunk_overlap_tokens,
        encoding_name: str = "cl100k_base",  # GPT-4 encoding
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk(self, text: str) -> list[Chunk]:
        """
        Chunk text into token-based segments.

        Strategy:
        1. Tokenize entire document
        2. Create chunks of max_tokens size
        3. Add overlap_tokens from previous chunk (if not first chunk)
        4. Decode tokens back to text

        Args:
            text: Full document text

        Returns:
            List of Chunk objects with text, token count, and order
        """
        # Encode full text to tokens
        tokens = self.encoding.encode(text)

        if len(tokens) == 0:
            return []

        chunks = []
        start_idx = 0
        chunk_ord = 0

        while start_idx < len(tokens):
            # Determine end index for this chunk
            end_idx = min(start_idx + self.max_tokens, len(tokens))

            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)

            # Skip if chunk is too small (except for last chunk)
            if len(chunk_tokens) >= self.min_tokens or end_idx == len(tokens):
                chunks.append(
                    Chunk(
                        text=chunk_text.strip(),
                        token_count=len(chunk_tokens),
                        ord=chunk_ord,
                    )
                )
                chunk_ord += 1

            # Move start index, accounting for overlap
            if end_idx == len(tokens):
                break

            start_idx = end_idx - self.overlap_tokens

        return chunks
