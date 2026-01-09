"""Evaluation dataset definitions for agent tool parameter extraction.

This module defines the eval cases that test whether an agent correctly
extracts and applies filters from natural language queries.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExpectedFilters:
    """Expected filter parameters that should be extracted from a query.

    All fields are optional - None means the filter should NOT be applied.
    An empty string means the field was mentioned but value couldn't be extracted.
    """

    speaker: Optional[str] = None
    source: Optional[str] = None
    doc_type: Optional[str] = None
    start_date: Optional[str] = None  # ISO format string
    end_date: Optional[str] = None  # ISO format string


@dataclass
class EvalCase:
    """A single evaluation test case.

    Attributes:
        id: Unique identifier for the test case
        query: Natural language query to send to the agent
        expected_filters: Filters that should be extracted and applied
        description: Human-readable description of what this tests
        category: Category for grouping (e.g., "speaker_filter", "date_filter")
    """

    id: str
    query: str
    expected_filters: ExpectedFilters
    description: str
    category: str = "general"


# Pre-defined evaluation dataset
EVAL_CASES: list[EvalCase] = [
    # ========================================
    # Speaker/Name Filter Tests
    # ========================================
    EvalCase(
        id="speaker_001",
        query="What has Elon Musk said about artificial intelligence?",
        expected_filters=ExpectedFilters(speaker="Elon Musk"),
        description="Should extract speaker name from 'has X said' pattern",
        category="speaker_filter",
    ),
    EvalCase(
        id="speaker_002",
        query="What are John Carmack's views on AGI timelines?",
        expected_filters=ExpectedFilters(speaker="John Carmack"),
        description="Should extract speaker name from possessive pattern",
        category="speaker_filter",
    ),
    EvalCase(
        id="speaker_003",
        query="Tell me about Yann LeCun's opinions on large language models",
        expected_filters=ExpectedFilters(speaker="Yann LeCun"),
        description="Should extract speaker name from possessive with 'opinions'",
        category="speaker_filter",
    ),
    EvalCase(
        id="speaker_004",
        query="According to Sam Altman, when will AGI arrive?",
        expected_filters=ExpectedFilters(speaker="Sam Altman"),
        description="Should extract speaker from 'according to X' pattern",
        category="speaker_filter",
    ),
    EvalCase(
        id="speaker_005",
        query="What did Demis Hassabis mention about AlphaFold?",
        expected_filters=ExpectedFilters(speaker="Demis Hassabis"),
        description="Should extract speaker from 'did X mention' pattern",
        category="speaker_filter",
    ),
    EvalCase(
        id="speaker_006",
        query="I want to hear what Patrick Collison thinks about progress studies",
        expected_filters=ExpectedFilters(speaker="Patrick Collison"),
        description="Should extract speaker from 'what X thinks' pattern",
        category="speaker_filter",
    ),
    EvalCase(
        id="speaker_007",
        query="Show me Tyler Cowen's discussion about economic growth",
        expected_filters=ExpectedFilters(speaker="Tyler Cowen"),
        description="Should extract speaker from possessive with 'discussion'",
        category="speaker_filter",
    ),
    EvalCase(
        id="speaker_008",
        query="What has the host Dwarkesh Patel asked about consciousness?",
        expected_filters=ExpectedFilters(speaker="Dwarkesh Patel"),
        description="Should extract host name when explicitly mentioned",
        category="speaker_filter",
    ),

    # ========================================
    # No Speaker Filter Tests (negative cases)
    # ========================================
    EvalCase(
        id="no_speaker_001",
        query="What are the main arguments for AGI timelines?",
        expected_filters=ExpectedFilters(speaker=None),
        description="No speaker mentioned - should NOT apply speaker filter",
        category="no_speaker_filter",
    ),
    EvalCase(
        id="no_speaker_002",
        query="Explain the concept of scaling laws in AI",
        expected_filters=ExpectedFilters(speaker=None),
        description="Technical question with no speaker reference",
        category="no_speaker_filter",
    ),
    EvalCase(
        id="no_speaker_003",
        query="What topics have been discussed about nuclear energy?",
        expected_filters=ExpectedFilters(speaker=None),
        description="Topic query without speaker attribution",
        category="no_speaker_filter",
    ),

    # ========================================
    # Date/Time Filter Tests
    # ========================================
    EvalCase(
        id="date_001",
        query="What has been said about AI safety in 2024?",
        expected_filters=ExpectedFilters(
            start_date="2024-01-01",
            end_date="2024-12-31",
        ),
        description="Should extract year as date range",
        category="date_filter",
    ),
    EvalCase(
        id="date_002",
        query="What were the discussions about GPT-4 after March 2023?",
        expected_filters=ExpectedFilters(start_date="2023-03-01"),
        description="Should extract 'after' as start_date",
        category="date_filter",
    ),
    EvalCase(
        id="date_003",
        query="Show me conversations from before 2023 about transformers",
        expected_filters=ExpectedFilters(end_date="2022-12-31"),
        description="Should extract 'before year' as end_date",
        category="date_filter",
    ),
    EvalCase(
        id="date_004",
        query="What did guests say about crypto between 2021 and 2022?",
        expected_filters=ExpectedFilters(
            start_date="2021-01-01",
            end_date="2022-12-31",
        ),
        description="Should extract 'between X and Y' as date range",
        category="date_filter",
    ),
    EvalCase(
        id="date_005",
        query="Recent discussions about quantum computing",
        expected_filters=ExpectedFilters(speaker=None),  # 'recent' is ambiguous
        description="Ambiguous time reference should not create specific filter",
        category="date_filter",
    ),

    # ========================================
    # Combined Filter Tests
    # ========================================
    EvalCase(
        id="combined_001",
        query="What did Elon Musk say about Mars in 2023?",
        expected_filters=ExpectedFilters(
            speaker="Elon Musk",
            start_date="2023-01-01",
            end_date="2023-12-31",
        ),
        description="Should extract both speaker and date filters",
        category="combined_filters",
    ),
    EvalCase(
        id="combined_002",
        query="Show me what Sam Altman said about GPT-5 after January 2024",
        expected_filters=ExpectedFilters(
            speaker="Sam Altman",
            start_date="2024-01-01",
        ),
        description="Should extract speaker and start_date",
        category="combined_filters",
    ),
    EvalCase(
        id="combined_003",
        query="According to Patrick Collison in his 2022 interview, what drives innovation?",
        expected_filters=ExpectedFilters(
            speaker="Patrick Collison",
            start_date="2022-01-01",
            end_date="2022-12-31",
        ),
        description="Should extract speaker from 'according to' and year",
        category="combined_filters",
    ),

    # ========================================
    # Edge Cases and Tricky Queries
    # ========================================
    EvalCase(
        id="edge_001",
        query="What do people think about what Elon Musk said?",
        expected_filters=ExpectedFilters(speaker="Elon Musk"),
        description="Should still extract the relevant speaker despite indirection",
        category="edge_cases",
    ),
    EvalCase(
        id="edge_002",
        query="Compare what Sam Altman and Demis Hassabis said about AGI",
        expected_filters=ExpectedFilters(speaker=None),  # Multiple speakers - tricky
        description="Multiple speakers mentioned - behavior may vary",
        category="edge_cases",
    ),
    EvalCase(
        id="edge_003",
        query="Who talked about the paperclip problem?",
        expected_filters=ExpectedFilters(speaker=None),
        description="Asking WHO said something, not filtering BY speaker",
        category="edge_cases",
    ),
    EvalCase(
        id="edge_004",
        query="john smith's thoughts on robotics",
        expected_filters=ExpectedFilters(speaker="john smith"),
        description="Lowercase speaker name should still be extracted",
        category="edge_cases",
    ),
    EvalCase(
        id="edge_005",
        query="What has Dr. Fei-Fei Li discussed about computer vision?",
        expected_filters=ExpectedFilters(speaker="Fei-Fei Li"),
        description="Should handle titles (Dr.) and hyphenated names",
        category="edge_cases",
    ),
]


class EvalDataset:
    """Container for evaluation test cases with filtering and iteration."""

    def __init__(self, cases: Optional[list[EvalCase]] = None):
        """Initialize with optional custom cases, defaults to EVAL_CASES."""
        self.cases = cases if cases is not None else EVAL_CASES

    def __iter__(self):
        return iter(self.cases)

    def __len__(self):
        return len(self.cases)

    def by_category(self, category: str) -> list[EvalCase]:
        """Filter cases by category."""
        return [c for c in self.cases if c.category == category]

    def categories(self) -> list[str]:
        """Get list of unique categories."""
        return list(set(c.category for c in self.cases))

    def get_by_id(self, case_id: str) -> Optional[EvalCase]:
        """Get a specific case by ID."""
        for case in self.cases:
            if case.id == case_id:
                return case
        return None
