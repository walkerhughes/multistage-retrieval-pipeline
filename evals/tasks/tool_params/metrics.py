"""Metrics calculation for agent tool parameter evaluations.

This module provides functions to compute aggregate metrics from
evaluation results, including precision, recall, accuracy, and
per-category breakdowns.
"""

from dataclasses import dataclass, field
from typing import Optional

from evals.tasks.tool_params.types import ToolParamsEvalResult


@dataclass
class FilterMetrics:
    """Metrics for a specific filter type.

    Attributes:
        filter_name: Name of the filter (speaker, start_date, etc.)
        true_positives: Correctly applied when expected
        true_negatives: Correctly NOT applied when not expected
        false_positives: Applied when should not have been
        false_negatives: Not applied when should have been
        precision: TP / (TP + FP)
        recall: TP / (TP + FN)
        accuracy: (TP + TN) / Total
        f1_score: Harmonic mean of precision and recall
    """

    filter_name: str
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    accuracy: float = 0.0
    f1_score: float = 0.0


@dataclass
class CategoryMetrics:
    """Metrics for a specific test category.

    Attributes:
        category: Category name
        total_cases: Number of cases in this category
        passed: Number of cases where all filters matched
        failed: Number of cases with at least one filter mismatch
        pass_rate: Proportion of passed cases
        errors: Number of cases that encountered errors
    """

    category: str
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    errors: int = 0


@dataclass
class ToolParamsMetrics:
    """Aggregate metrics from a tool parameter evaluation run.

    Attributes:
        total_cases: Total number of eval cases run
        passed: Number of cases with all filters correct
        failed: Number of cases with filter mismatches
        errors: Number of cases that encountered errors
        overall_accuracy: Proportion of fully correct cases
        filter_metrics: Per-filter breakdown
        category_metrics: Per-category breakdown
        avg_latency_ms: Average latency across all cases
    """

    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    overall_accuracy: float = 0.0
    filter_metrics: dict[str, FilterMetrics] = field(default_factory=dict)
    category_metrics: dict[str, CategoryMetrics] = field(default_factory=dict)
    avg_latency_ms: float = 0.0


def _compute_filter_metrics(
    filter_name: str,
    results: list[ToolParamsEvalResult],
) -> FilterMetrics:
    """Compute metrics for a single filter type across all results.

    Args:
        filter_name: Name of the filter to analyze
        results: List of evaluation results

    Returns:
        FilterMetrics for the specified filter
    """
    metrics = FilterMetrics(filter_name=filter_name)

    for result in results:
        if result.error:
            continue  # Skip errored cases

        # Get expected and actual values
        expected = getattr(result.expected_filters, filter_name, None)
        actual = result.actual_filters.get(filter_name)

        expected_applied = expected is not None
        actual_applied = actual is not None

        if expected_applied and actual_applied:
            # Check if the match was correct
            if result.filter_matches.get(filter_name, False):
                metrics.true_positives += 1
            else:
                # Applied but wrong value - count as FP for value accuracy
                metrics.false_positives += 1
        elif not expected_applied and not actual_applied:
            metrics.true_negatives += 1
        elif expected_applied and not actual_applied:
            metrics.false_negatives += 1
        else:  # not expected_applied and actual_applied
            metrics.false_positives += 1

    # Calculate derived metrics
    total = (
        metrics.true_positives
        + metrics.true_negatives
        + metrics.false_positives
        + metrics.false_negatives
    )

    if total > 0:
        metrics.accuracy = (metrics.true_positives + metrics.true_negatives) / total

    tp_fp = metrics.true_positives + metrics.false_positives
    if tp_fp > 0:
        metrics.precision = metrics.true_positives / tp_fp

    tp_fn = metrics.true_positives + metrics.false_negatives
    if tp_fn > 0:
        metrics.recall = metrics.true_positives / tp_fn

    if metrics.precision + metrics.recall > 0:
        metrics.f1_score = (
            2 * metrics.precision * metrics.recall
        ) / (metrics.precision + metrics.recall)

    return metrics


def _compute_category_metrics(
    category: str,
    results: list[ToolParamsEvalResult],
) -> CategoryMetrics:
    """Compute metrics for a specific test category.

    Args:
        category: Category name to filter by (matches case.category)
        results: List of evaluation results (should already be filtered by category)

    Returns:
        CategoryMetrics for the specified category
    """
    metrics = CategoryMetrics(category=category)
    metrics.total_cases = len(results)

    for result in results:
        if result.error:
            metrics.errors += 1
        elif result.overall_match:
            metrics.passed += 1
        else:
            metrics.failed += 1

    if metrics.total_cases > 0:
        metrics.pass_rate = metrics.passed / metrics.total_cases

    return metrics


def compute_tool_params_metrics(
    results: list[ToolParamsEvalResult],
    case_categories: Optional[dict[str, str]] = None,
) -> ToolParamsMetrics:
    """Compute aggregate metrics from tool parameter evaluation results.

    Args:
        results: List of evaluation results
        case_categories: Optional mapping of case_id to category name.
                        If not provided, categories are inferred from results.

    Returns:
        ToolParamsMetrics with overall and per-filter/category breakdowns
    """
    metrics = ToolParamsMetrics()
    metrics.total_cases = len(results)

    if not results:
        return metrics

    # Count overall results
    total_latency = 0.0
    for result in results:
        if result.error:
            metrics.errors += 1
        elif result.overall_match:
            metrics.passed += 1
        else:
            metrics.failed += 1
        total_latency += result.latency_ms

    # Calculate overall accuracy
    non_error_cases = metrics.total_cases - metrics.errors
    if non_error_cases > 0:
        metrics.overall_accuracy = metrics.passed / non_error_cases

    # Calculate average latency
    if metrics.total_cases > 0:
        metrics.avg_latency_ms = total_latency / metrics.total_cases

    # Compute per-filter metrics
    filter_names = ["speaker", "start_date", "end_date", "source", "doc_type"]
    for filter_name in filter_names:
        metrics.filter_metrics[filter_name] = _compute_filter_metrics(
            filter_name, results
        )

    # Compute per-category metrics
    # Group results by category (from case_categories or infer from result patterns)
    categories: dict[str, list[ToolParamsEvalResult]] = {}

    if case_categories:
        for result in results:
            cat = case_categories.get(result.case_id, "unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
    else:
        # Infer category from case_id prefix (e.g., "speaker_001" -> "speaker_filter")
        for result in results:
            # Default categorization based on case_id patterns
            if result.case_id.startswith("speaker_"):
                cat = "speaker_filter"
            elif result.case_id.startswith("no_speaker_"):
                cat = "no_speaker_filter"
            elif result.case_id.startswith("date_"):
                cat = "date_filter"
            elif result.case_id.startswith("combined_"):
                cat = "combined_filters"
            elif result.case_id.startswith("edge_"):
                cat = "edge_cases"
            else:
                cat = "general"

            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)

    for cat, cat_results in categories.items():
        metrics.category_metrics[cat] = _compute_category_metrics(cat, cat_results)

    return metrics


def format_metrics_report(metrics: ToolParamsMetrics) -> str:
    """Format metrics as a human-readable report.

    Args:
        metrics: Computed evaluation metrics

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("AGENT TOOL PARAMETER EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Overall summary
    lines.append("OVERALL RESULTS")
    lines.append("-" * 40)
    lines.append(f"Total Cases:     {metrics.total_cases}")
    lines.append(f"Passed:          {metrics.passed}")
    lines.append(f"Failed:          {metrics.failed}")
    lines.append(f"Errors:          {metrics.errors}")
    lines.append(f"Overall Accuracy: {metrics.overall_accuracy:.1%}")
    lines.append(f"Avg Latency:     {metrics.avg_latency_ms:.1f}ms")
    lines.append("")

    # Per-filter metrics
    lines.append("FILTER-LEVEL METRICS")
    lines.append("-" * 40)
    lines.append(f"{'Filter':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
    lines.append("-" * 52)

    for name, fm in metrics.filter_metrics.items():
        lines.append(
            f"{name:<12} {fm.precision:>10.1%} {fm.recall:>10.1%} "
            f"{fm.f1_score:>10.1%} {fm.accuracy:>10.1%}"
        )
    lines.append("")

    # Per-category metrics
    lines.append("CATEGORY BREAKDOWN")
    lines.append("-" * 40)
    lines.append(f"{'Category':<20} {'Total':>8} {'Pass':>8} {'Fail':>8} {'Rate':>10}")
    lines.append("-" * 54)

    for cat, cm in sorted(metrics.category_metrics.items()):
        lines.append(
            f"{cat:<20} {cm.total_cases:>8} {cm.passed:>8} "
            f"{cm.failed:>8} {cm.pass_rate:>10.1%}"
        )
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_detailed_results(results: list[ToolParamsEvalResult]) -> str:
    """Format detailed per-case results.

    Args:
        results: List of evaluation results

    Returns:
        Formatted string with details for each case
    """
    lines = []
    lines.append("DETAILED RESULTS")
    lines.append("=" * 60)

    for result in results:
        status = "PASS" if result.overall_match else "FAIL"
        if result.error:
            status = "ERROR"

        lines.append("")
        lines.append(f"Case: {result.case_id} [{status}]")
        lines.append(f"Query: {result.query}")
        lines.append(f"Latency: {result.latency_ms:.1f}ms")

        if result.error:
            lines.append(f"Error: {result.error}")
            continue

        # Expected vs Actual filters
        lines.append("Expected Filters:")
        exp = result.expected_filters
        for field_name in ["speaker", "start_date", "end_date", "source", "doc_type"]:
            val = getattr(exp, field_name, None)
            if val is not None:
                lines.append(f"  {field_name}: {val}")

        lines.append("Actual Filters:")
        if result.actual_filters:
            for k, v in result.actual_filters.items():
                match_status = "OK" if result.filter_matches.get(k, False) else "MISMATCH"
                lines.append(f"  {k}: {v} [{match_status}]")
        else:
            lines.append("  (none)")

        # Tool calls
        if result.tool_calls:
            lines.append(f"Tool Calls: {len(result.tool_calls)}")
            for tc in result.tool_calls:
                lines.append(f"  - query: {tc.query[:50]}...")
                if tc.filters:
                    lines.append(f"    filters: {tc.filters}")

        lines.append("-" * 40)

    return "\n".join(lines)
