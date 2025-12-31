"""Pydantic schemas for evaluation examples."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """Types of evaluation questions.

    - factual: Direct fact recall from transcript (e.g., "What did X say about Y?")
    - analytical: Requires reasoning across transcript content (e.g., "How does X compare to Y?")
    - opinion: Asks about subjective views expressed (e.g., "What is X's opinion on Y?")
    """

    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    OPINION = "opinion"


class DifficultyLevel(str, Enum):
    """Question difficulty levels.

    - easy: Single-chunk, direct answer (fact lookup)
    - medium: Multi-chunk or requires simple reasoning
    - hard: Complex reasoning, synthesis across multiple sections
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EvalTask(BaseModel):
    """A single evaluation question with ground truth.

    Used to evaluate retrieval system quality by comparing retrieved chunks
    against expected sections and generated answers against reference answers.
    """

    id: str = Field(
        ...,
        description="Unique identifier for this eval example (e.g., 'eval_001')",
        examples=["eval_001", "eval_002"],
    )
    question: str = Field(
        ...,
        description="The question to ask the retrieval system",
        min_length=10,
        examples=["What did Andrej Karpathy say about the decade of agents?"],
    )
    reference_answer: str = Field(
        ...,
        description="Ground truth answer (1-3 sentences) for evaluation",
        min_length=10,
        examples=[
            "Karpathy stated that while people predicted this would be the year of agents, "
            "he believes it's more accurately the decade of agents due to the remaining work needed."
        ],
    )
    expected_sections: list[str] = Field(
        ...,
        description="Key topics/phrases that should appear in retrieved chunks (2-4 items)",
        min_length=1,
        examples=[["decade of agents", "bottlenecks", "continual learning"]],
    )
    difficulty_level: DifficultyLevel = Field(
        ...,
        description="Question difficulty: easy (single chunk), medium (multi-chunk), hard (synthesis)",
        examples=[DifficultyLevel.MEDIUM],
    )
    source_chunk_ids: Optional[list[int]] = Field(
        None,
        description="Database chunk IDs containing the answer (populated after ingestion)",
        examples=[[123, 124, 125]],
    )
    question_type: QuestionType = Field(
        ...,
        description="Category of question: factual, analytical, or opinion",
        examples=[QuestionType.FACTUAL],
    )
    transcript_source: Optional[str] = Field(
        None,
        description="Filename of source transcript (e.g., 'ilya-sutskever-–-...')",
        examples=["andrej-karpathy-—-were-summoning-ghosts-not-building-animals.md"],
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata (topic, tags, etc.)",
        examples=[{"topic": "ai-timelines", "speaker": "Andrej Karpathy"}],
    )


class EvalDataset(BaseModel):
    """Collection of evaluation examples with versioning.

    Used to load and validate the eval_questions.json file.
    """

    version: str = Field(
        ...,
        description="Dataset version (semantic versioning recommended)",
        examples=["1.0.0"],
    )
    description: str = Field(
        ...,
        description="Brief description of the dataset purpose and scope",
        examples=["Curated evaluation questions from AI podcast transcripts"],
    )
    created_at: str = Field(
        ...,
        description="Creation date in ISO 8601 format",
        examples=["2025-12-30"],
    )
    examples: list[EvalTask] = Field(
        ...,
        description="List of evaluation questions",
        min_length=1,
    )

    @property
    def count(self) -> int:
        """Number of examples in the dataset."""
        return len(self.examples)

    def by_difficulty(self, level: DifficultyLevel) -> list[EvalTask]:
        """Filter examples by difficulty level."""
        return [ex for ex in self.examples if ex.difficulty_level == level]

    def by_type(self, question_type: QuestionType) -> list[EvalTask]:
        """Filter examples by question type."""
        return [ex for ex in self.examples if ex.question_type == question_type]
