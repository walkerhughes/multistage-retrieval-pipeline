"""Tests for agent tool parameter extraction.

These tests evaluate whether the agent correctly extracts and uses
filter parameters (speaker, date, etc.) from natural language queries.

Run with: pytest tests/unit/test_tool_params_evals.py -v --tb=short
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from evals.tool_params.dataset import (
    EVAL_CASES,
    EvalCase,
    ExpectedFilters,
    ToolParamsDataset,
)
from evals.tool_params.metrics import (
    ToolParamsMetrics,
    compute_tool_params_metrics,
    format_detailed_results,
    format_metrics_report,
)


# Local definition of ToolParamsEvalResult for tests
# This avoids importing from runner.py which has heavy dependencies
@dataclass
class ToolParamsEvalResult:
    """Result of running a single tool parameter evaluation case."""

    case_id: str
    query: str
    expected_filters: ExpectedFilters
    actual_filters: dict[str, Any]
    tool_calls: list[Any]
    filter_matches: dict[str, bool]
    overall_match: bool
    answer: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None


class TestToolParamsDataset:
    """Tests for the tool parameter evaluation dataset."""

    def test_dataset_has_cases(self):
        """Dataset should contain evaluation cases."""
        dataset = ToolParamsDataset()
        assert len(dataset) > 0

    def test_dataset_iteration(self):
        """Should be able to iterate over cases."""
        dataset = ToolParamsDataset()
        cases = list(dataset)
        assert len(cases) == len(EVAL_CASES)

    def test_filter_by_category(self):
        """Should filter cases by category."""
        dataset = ToolParamsDataset()
        speaker_cases = dataset.by_category("speaker_filter")
        assert len(speaker_cases) > 0
        for case in speaker_cases:
            assert case.category == "speaker_filter"

    def test_get_by_id(self):
        """Should retrieve case by ID."""
        dataset = ToolParamsDataset()
        case = dataset.get_by_id("speaker_001")
        assert case is not None
        assert case.id == "speaker_001"

    def test_get_by_id_not_found(self):
        """Should return None for unknown ID."""
        dataset = ToolParamsDataset()
        case = dataset.get_by_id("nonexistent_case")
        assert case is None

    def test_all_cases_have_required_fields(self):
        """All cases should have required fields."""
        dataset = ToolParamsDataset()
        for case in dataset:
            assert case.id, "Case must have an id"
            assert case.query, "Case must have a query"
            assert case.expected_filters is not None, "Case must have expected_filters"
            assert case.description, "Case must have a description"

    def test_categories_list(self):
        """Should return list of unique categories."""
        dataset = ToolParamsDataset()
        categories = dataset.categories()
        assert len(categories) > 0
        assert "speaker_filter" in categories
        assert "no_speaker_filter" in categories

    def test_count_property(self):
        """Should return count of cases."""
        dataset = ToolParamsDataset()
        assert dataset.count == len(EVAL_CASES)


class TestExpectedFilters:
    """Tests for ExpectedFilters dataclass."""

    def test_default_values(self):
        """All filters should default to None."""
        filters = ExpectedFilters()
        assert filters.speaker is None
        assert filters.source is None
        assert filters.doc_type is None
        assert filters.start_date is None
        assert filters.end_date is None

    def test_partial_initialization(self):
        """Should allow partial initialization."""
        filters = ExpectedFilters(speaker="John Smith")
        assert filters.speaker == "John Smith"
        assert filters.source is None

    def test_full_initialization(self):
        """Should allow full initialization."""
        filters = ExpectedFilters(
            speaker="Test Speaker",
            source="youtube",
            doc_type="transcript",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        assert filters.speaker == "Test Speaker"
        assert filters.source == "youtube"
        assert filters.doc_type == "transcript"
        assert filters.start_date == "2024-01-01"
        assert filters.end_date == "2024-12-31"


class TestMetricsComputation:
    """Tests for metrics computation."""

    def test_compute_metrics_empty_results(self):
        """Should handle empty results list."""
        metrics = compute_tool_params_metrics([])
        assert metrics.total_cases == 0
        assert metrics.overall_accuracy == 0.0

    def test_compute_metrics_all_passed(self):
        """Should calculate 100% accuracy when all pass."""
        results = [
            ToolParamsEvalResult(
                case_id="test_1",
                query="test query 1",
                expected_filters=ExpectedFilters(speaker="John"),
                actual_filters={"speaker": "John"},
                tool_calls=[],
                filter_matches={"speaker": True},
                overall_match=True,
            ),
            ToolParamsEvalResult(
                case_id="test_2",
                query="test query 2",
                expected_filters=ExpectedFilters(speaker="Jane"),
                actual_filters={"speaker": "Jane"},
                tool_calls=[],
                filter_matches={"speaker": True},
                overall_match=True,
            ),
        ]
        metrics = compute_tool_params_metrics(results)
        assert metrics.total_cases == 2
        assert metrics.passed == 2
        assert metrics.failed == 0
        assert metrics.overall_accuracy == 1.0

    def test_compute_metrics_all_failed(self):
        """Should calculate 0% accuracy when all fail."""
        results = [
            ToolParamsEvalResult(
                case_id="test_1",
                query="test query 1",
                expected_filters=ExpectedFilters(speaker="John"),
                actual_filters={},  # No speaker extracted
                tool_calls=[],
                filter_matches={"speaker": False},
                overall_match=False,
            ),
        ]
        metrics = compute_tool_params_metrics(results)
        assert metrics.total_cases == 1
        assert metrics.passed == 0
        assert metrics.failed == 1
        assert metrics.overall_accuracy == 0.0

    def test_compute_metrics_with_errors(self):
        """Should track error cases separately."""
        results = [
            ToolParamsEvalResult(
                case_id="test_1",
                query="test query",
                expected_filters=ExpectedFilters(),
                actual_filters={},
                tool_calls=[],
                filter_matches={},
                overall_match=False,
                error="Test error",
            ),
        ]
        metrics = compute_tool_params_metrics(results)
        assert metrics.errors == 1
        # Errors shouldn't count toward pass/fail
        assert metrics.passed == 0
        assert metrics.failed == 0

    def test_filter_metrics_precision_recall(self):
        """Should calculate precision and recall correctly."""
        # Create results with known TP, FP, FN, TN for speaker filter
        results = [
            # True Positive: expected speaker, got correct speaker
            ToolParamsEvalResult(
                case_id="tp_1",
                query="what did John say",
                expected_filters=ExpectedFilters(speaker="John"),
                actual_filters={"speaker": "John"},
                tool_calls=[],
                filter_matches={"speaker": True},
                overall_match=True,
            ),
            # False Negative: expected speaker, didn't get it
            ToolParamsEvalResult(
                case_id="fn_1",
                query="what did Jane say",
                expected_filters=ExpectedFilters(speaker="Jane"),
                actual_filters={},
                tool_calls=[],
                filter_matches={"speaker": False},
                overall_match=False,
            ),
            # False Positive: didn't expect speaker, got one
            ToolParamsEvalResult(
                case_id="fp_1",
                query="what is AI",
                expected_filters=ExpectedFilters(speaker=None),
                actual_filters={"speaker": "Random"},
                tool_calls=[],
                filter_matches={"speaker": False},
                overall_match=False,
            ),
            # True Negative: didn't expect speaker, didn't get one
            ToolParamsEvalResult(
                case_id="tn_1",
                query="explain transformers",
                expected_filters=ExpectedFilters(speaker=None),
                actual_filters={},
                tool_calls=[],
                filter_matches={"speaker": True},
                overall_match=True,
            ),
        ]

        metrics = compute_tool_params_metrics(results)
        speaker_metrics = metrics.filter_metrics["speaker"]

        # TP=1, FP=1, FN=1, TN=1
        assert speaker_metrics.true_positives == 1
        assert speaker_metrics.false_positives == 1
        assert speaker_metrics.false_negatives == 1
        assert speaker_metrics.true_negatives == 1

        # Precision = TP / (TP + FP) = 1/2 = 0.5
        assert speaker_metrics.precision == 0.5

        # Recall = TP / (TP + FN) = 1/2 = 0.5
        assert speaker_metrics.recall == 0.5

        # Accuracy = (TP + TN) / Total = 2/4 = 0.5
        assert speaker_metrics.accuracy == 0.5


class TestMetricsFormatting:
    """Tests for metrics report formatting."""

    def test_format_metrics_report(self):
        """Should format metrics as readable report."""
        metrics = ToolParamsMetrics(
            total_cases=10,
            passed=8,
            failed=2,
            errors=0,
            overall_accuracy=0.8,
            avg_latency_ms=150.5,
        )
        report = format_metrics_report(metrics)

        assert "EVALUATION REPORT" in report
        assert "Total Cases:" in report
        assert "10" in report
        assert "80.0%" in report

    def test_format_detailed_results(self):
        """Should format detailed results."""
        results = [
            ToolParamsEvalResult(
                case_id="test_1",
                query="What has John said about AI?",
                expected_filters=ExpectedFilters(speaker="John"),
                actual_filters={"speaker": "John"},
                tool_calls=[],
                filter_matches={"speaker": True},
                overall_match=True,
                latency_ms=100.0,
            ),
        ]
        report = format_detailed_results(results)

        assert "test_1" in report
        assert "PASS" in report
        assert "John" in report


class TestEvalCaseCoverage:
    """Tests to verify eval case coverage."""

    def test_speaker_filter_cases_have_speakers(self):
        """Speaker filter cases should expect a speaker."""
        dataset = ToolParamsDataset()
        for case in dataset.by_category("speaker_filter"):
            assert case.expected_filters.speaker is not None, (
                f"Case {case.id} in speaker_filter category should expect a speaker"
            )

    def test_no_speaker_cases_have_no_speaker(self):
        """No speaker filter cases should not expect a speaker."""
        dataset = ToolParamsDataset()
        for case in dataset.by_category("no_speaker_filter"):
            assert case.expected_filters.speaker is None, (
                f"Case {case.id} in no_speaker_filter category should not expect a speaker"
            )

    def test_case_ids_are_unique(self):
        """All case IDs should be unique."""
        dataset = ToolParamsDataset()
        ids = [case.id for case in dataset]
        assert len(ids) == len(set(ids)), "Case IDs must be unique"

    def test_minimum_cases_per_category(self):
        """Each category should have at least 2 cases."""
        dataset = ToolParamsDataset()
        for category in dataset.categories():
            cases = dataset.by_category(category)
            assert len(cases) >= 2, (
                f"Category {category} should have at least 2 cases, has {len(cases)}"
            )
