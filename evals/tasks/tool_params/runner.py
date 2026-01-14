#!/usr/bin/env python
"""Evaluation runner for agent tool parameter extraction.

This script runs evaluations to test whether an agent correctly extracts
and applies filter parameters (speaker, date, etc.) from natural language queries.

Usage:
    python -m evals.tasks.tool_params.runner --category speaker_filter
    python -m evals.tasks.tool_params.runner --case-id speaker_001 --verbose
    python -m evals.tasks.tool_params.runner --help

Examples:
    # Run all evals with FTS mode
    python -m evals.tasks.tool_params.runner --mode fts

    # Run speaker filter tests only
    python -m evals.tasks.tool_params.runner --category speaker_filter

    # Run with verbose output
    python -m evals.tasks.tool_params.runner --verbose --num-samples 5
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from agents import Agent, Runner, function_tool  # pyrefly: ignore

from evals.tasks.tool_params.dataset import EvalCase, ExpectedFilters, ToolParamsDataset
from evals.tasks.tool_params.metrics import (
    ToolParamsMetrics,
    compute_tool_params_metrics,
    format_detailed_results,
    format_metrics_report,
)
from evals.tasks.tool_params.types import ToolCallCapture, ToolParamsEvalResult
from src.agents.helpers import get_trace_id, initialize_tracing, retrieve_chunks
from src.config import settings
from src.utils.timing import Timer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _normalize_filter_value(value: Any) -> Optional[str]:
    """Normalize filter values for comparison.

    Handles case-insensitive comparison and whitespace normalization.
    """
    if value is None:
        return None
    return str(value).strip().lower()


def _compare_filters(
    expected: ExpectedFilters,
    actual: dict[str, Any],
) -> tuple[dict[str, bool], bool]:
    """Compare expected filters against actual filters.

    Args:
        expected: Expected filter values from eval case
        actual: Actual filters applied by the agent

    Returns:
        Tuple of (filter_matches dict, overall_match bool)
    """
    matches = {}

    # Check speaker filter
    expected_speaker = _normalize_filter_value(expected.speaker)
    actual_speaker = _normalize_filter_value(actual.get("speaker"))

    if expected_speaker is None and actual_speaker is None:
        matches["speaker"] = True  # Both None = correct
    elif expected_speaker is None and actual_speaker is not None:
        matches["speaker"] = False  # Applied filter when shouldn't
    elif expected_speaker is not None and actual_speaker is None:
        matches["speaker"] = False  # Didn't apply filter when should
    else:
        # Both have values - check if actual contains expected (partial match OK)
        matches["speaker"] = (
            expected_speaker in actual_speaker or actual_speaker in expected_speaker  # type: ignore[operator]
        )

    # Check date filters (more lenient - just check if applied correctly)
    for date_field in ["start_date", "end_date"]:
        expected_date = getattr(expected, date_field)
        actual_date = actual.get(date_field)

        if expected_date is None and actual_date is None:
            matches[date_field] = True
        elif expected_date is None and actual_date is not None:
            matches[date_field] = False  # Applied when shouldn't
        elif expected_date is not None and actual_date is None:
            matches[date_field] = False  # Didn't apply when should
        else:
            # Both have values - check year matches at minimum
            try:
                expected_year = expected_date[:4] if expected_date else None
                actual_year = str(actual_date)[:4] if actual_date else None
                matches[date_field] = expected_year == actual_year
            except (IndexError, TypeError):
                matches[date_field] = False

    # Check source filter
    expected_source = _normalize_filter_value(expected.source)
    actual_source = _normalize_filter_value(actual.get("source"))
    if expected_source is None and actual_source is None:
        matches["source"] = True
    elif expected_source is None or actual_source is None:
        matches["source"] = expected_source == actual_source
    else:
        matches["source"] = expected_source in actual_source

    # Check doc_type filter
    expected_doc_type = _normalize_filter_value(expected.doc_type)
    actual_doc_type = _normalize_filter_value(actual.get("doc_type"))
    if expected_doc_type is None and actual_doc_type is None:
        matches["doc_type"] = True
    elif expected_doc_type is None or actual_doc_type is None:
        matches["doc_type"] = expected_doc_type == actual_doc_type
    else:
        matches["doc_type"] = expected_doc_type in actual_doc_type

    overall = all(matches.values())
    return matches, overall


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run agent tool parameter evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all evals with defaults
  python -m evals.tasks.tool_params.runner

  # Run specific category
  python -m evals.tasks.tool_params.runner --category speaker_filter

  # Run specific case with verbose output
  python -m evals.tasks.tool_params.runner --case-id speaker_001 --verbose
        """,
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Run only cases in specified category",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        help="Run only a specific case by ID",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of cases to run (default: all)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed results for each case",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="evals/results",
        help="Output directory for results (default: evals/results)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fts",
        choices=["fts", "vector", "hybrid"],
        help="Retrieval mode (default: fts)",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available categories and exit",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List all cases and exit",
    )

    return parser.parse_args()


class ToolParamsHarness:
    """Harness for running agent tool parameter evaluations.

    This creates an agent with a filter-aware retrieval tool and captures
    the tool calls to evaluate parameter extraction accuracy.
    """

    def __init__(self, retrieval_params: Optional[dict[str, Any]] = None):
        """Initialize the eval harness.

        Args:
            retrieval_params: Base retrieval parameters (mode, max_returned, etc.)
        """
        self.retrieval_params = retrieval_params or {
            "mode": "fts",
            "operator": "or",
            "fts_candidates": 100,
            "max_returned": 5,
        }
        initialize_tracing()

    async def run_case(self, case: EvalCase) -> ToolParamsEvalResult:
        """Run a single evaluation case.

        Args:
            case: The eval case to run

        Returns:
            ToolParamsEvalResult with comparison of expected vs actual filters
        """
        timer = Timer()
        timer.start()

        # Capture tool calls
        tool_calls: list[ToolCallCapture] = []
        applied_filters: dict[str, Any] = {}

        def _create_eval_retrieval_tool():
            """Create a retrieval tool that captures filter parameters."""

            @function_tool
            def search_knowledge_base(
                query: str,
                speaker: Optional[str] = None,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                source: Optional[str] = None,
                doc_type: Optional[str] = None,
            ) -> str:
                """Search the knowledge base for relevant information.

                Use this tool to retrieve relevant passages from podcast transcripts.
                You can filter results by speaker name, date range, source, or document type.

                IMPORTANT: When the user asks about what a specific person said or thinks,
                you MUST use the speaker parameter to filter by that person's name.

                Args:
                    query: The search query to find relevant information.
                    speaker: Filter by speaker name (e.g., "Elon Musk", "Sam Altman").
                            Use this when the user asks about what someone specific said.
                    start_date: Only return results from after this date (ISO format: YYYY-MM-DD).
                    end_date: Only return results from before this date (ISO format: YYYY-MM-DD).
                    source: Filter by source (e.g., "youtube", "dwarkesh").
                    doc_type: Filter by document type (e.g., "transcript", "article").

                Returns:
                    Formatted string containing relevant passages from the knowledge base.
                """
                nonlocal tool_calls, applied_filters

                # Build filters dict from parameters
                filters: dict[str, Any] = {}
                if speaker:
                    filters["speaker"] = speaker
                if start_date:
                    filters["start_date"] = start_date
                if end_date:
                    filters["end_date"] = end_date
                if source:
                    filters["source"] = source
                if doc_type:
                    filters["doc_type"] = doc_type

                # Capture the tool call
                tool_calls.append(
                    ToolCallCapture(
                        tool_name="search_knowledge_base",
                        query=query,
                        filters=filters.copy(),
                        raw_args={
                            "query": query,
                            "speaker": speaker,
                            "start_date": start_date,
                            "end_date": end_date,
                            "source": source,
                            "doc_type": doc_type,
                        },
                    )
                )

                # Store applied filters for comparison
                applied_filters.update(filters)

                # Actually retrieve chunks with filters
                retrieval_params = {
                    **self.retrieval_params,
                    "filters": filters if filters else None,
                }

                try:
                    chunks = retrieve_chunks(query, retrieval_params)

                    if not chunks:
                        return "No relevant information found in the knowledge base."

                    context_parts: list[str] = []
                    for i, chunk in enumerate(chunks, 1):
                        title = chunk.metadata.get("title", "Unknown")
                        speaker_name = chunk.speaker
                        context_parts.append(
                            f"[Source {i}: {title} - {speaker_name}]\n{chunk.text}"
                        )

                    return "\n\n---\n\n".join(context_parts)
                except Exception as e:
                    return f"Error retrieving information: {str(e)}"

            return search_knowledge_base

        try:
            # Create the evaluation tool
            eval_tool = _create_eval_retrieval_tool()

            # Create agent with filter-aware instructions
            agent = Agent(  # pyrefly: ignore
                name="Eval RAG Assistant",
                model=settings.chat_model,
                instructions="""You are a helpful assistant that answers questions using a knowledge base of podcast transcripts.

CRITICAL INSTRUCTIONS FOR TOOL USE:

1. ALWAYS use the search_knowledge_base tool to find information before answering.

2. SPEAKER FILTERING: When the user asks about what a SPECIFIC PERSON said, thought, or discussed:
   - Extract the person's name from the query
   - Pass it to the 'speaker' parameter of search_knowledge_base
   - Examples:
     - "What has Elon Musk said about AI?" -> speaker="Elon Musk"
     - "According to Sam Altman..." -> speaker="Sam Altman"
     - "John Smith's views on X" -> speaker="John Smith"

3. DATE FILTERING: When the user mentions specific dates or years:
   - "in 2024" -> start_date="2024-01-01", end_date="2024-12-31"
   - "after March 2023" -> start_date="2023-03-01"
   - "before 2022" -> end_date="2021-12-31"

4. When NO specific person is mentioned, do NOT use the speaker filter.

5. Base your answer ONLY on the retrieved information. If no relevant info is found, say so.

6. Be concise and cite your sources.""",
                tools=[eval_tool],
            )

            # Run the agent
            result = await Runner.run(
                starting_agent=agent,
                input=case.query,
            )

            answer = str(result.final_output) if result.final_output else ""

            # Compare filters
            filter_matches, overall_match = _compare_filters(
                case.expected_filters, applied_filters
            )

            latency_ms = timer.stop()

            return ToolParamsEvalResult(
                case_id=case.id,
                query=case.query,
                expected_filters=case.expected_filters,
                actual_filters=applied_filters,
                tool_calls=tool_calls,
                filter_matches=filter_matches,
                overall_match=overall_match,
                answer=answer,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = timer.stop()
            return ToolParamsEvalResult(
                case_id=case.id,
                query=case.query,
                expected_filters=case.expected_filters,
                actual_filters=applied_filters,
                tool_calls=tool_calls,
                filter_matches={},
                overall_match=False,
                latency_ms=latency_ms,
                error=str(e),
            )

    async def run_all(self, cases: list[EvalCase]) -> list[ToolParamsEvalResult]:
        """Run all evaluation cases sequentially.

        Args:
            cases: List of eval cases to run

        Returns:
            List of ToolParamsEvalResult for each case
        """
        results = []
        for case in tqdm(cases, desc="Evaluating tool params", unit="case"):
            result = await self.run_case(case)
            results.append(result)
        return results


def serialize_result(result: Any) -> Any:
    """Serialize an EvalResult to a JSON-compatible dict."""
    if hasattr(result, "__dataclass_fields__"):
        d = {}
        for field_name in result.__dataclass_fields__:
            value = getattr(result, field_name)
            d[field_name] = serialize_result(value)
        return d
    elif isinstance(result, list):
        return [serialize_result(item) for item in result]
    elif isinstance(result, dict):
        return {k: serialize_result(v) for k, v in result.items()}
    else:
        return result


async def run_evaluation(
    category: Optional[str] = None,
    case_id: Optional[str] = None,
    num_samples: Optional[int] = None,
    verbose: bool = False,
    output_dir: str = "evals/results",
    retrieval_mode: str = "fts",
) -> int:
    """Run the evaluation and print results.

    Args:
        category: Optional category to filter cases
        case_id: Optional specific case ID to run
        num_samples: Optional number of samples to run
        verbose: Whether to print detailed results
        output_dir: Directory to write results
        retrieval_mode: Retrieval mode (fts, vector, hybrid)

    Returns:
        Exit code (0 for success, 1 for failures)
    """
    dataset = ToolParamsDataset()

    # Filter cases
    if case_id:
        case = dataset.get_by_id(case_id)
        if case is None:
            print(f"Error: Case '{case_id}' not found", file=sys.stderr)
            return 1
        cases = [case]
    elif category:
        cases = dataset.by_category(category)
        if not cases:
            print(f"Error: No cases found for category '{category}'", file=sys.stderr)
            print(f"Available categories: {', '.join(dataset.categories())}")
            return 1
    else:
        cases = list(dataset)

    # Sample if requested
    if num_samples and num_samples < len(cases):
        cases = cases[:num_samples]

    print(f"Running {len(cases)} evaluation case(s)...")
    print(f"Retrieval mode: {retrieval_mode}")
    print()

    # Create harness
    harness = ToolParamsHarness(
        retrieval_params={
            "mode": retrieval_mode,
            "operator": "or",
            "fts_candidates": 100,
            "max_returned": 5,
        }
    )

    # Run evaluations
    results = await harness.run_all(cases)

    # Compute metrics
    metrics = compute_tool_params_metrics(results)

    # Print report
    print()
    print(format_metrics_report(metrics))

    if verbose:
        print()
        print(format_detailed_results(results))

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"tool_params_{timestamp}.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "retrieval_mode": retrieval_mode,
            "category_filter": category,
            "case_id_filter": case_id,
            "num_samples": num_samples,
        },
        "metrics": {
            "total_cases": metrics.total_cases,
            "passed": metrics.passed,
            "failed": metrics.failed,
            "errors": metrics.errors,
            "overall_accuracy": metrics.overall_accuracy,
            "avg_latency_ms": metrics.avg_latency_ms,
        },
        "results": [serialize_result(r) for r in results],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Return exit code based on results
    return 0 if metrics.overall_accuracy >= 0.8 else 1


def main() -> None:
    """Main entry point for tool parameter evaluations."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    dataset = ToolParamsDataset()

    if args.list_categories:
        print("Available categories:")
        for cat in sorted(dataset.categories()):
            count = len(dataset.by_category(cat))
            print(f"  {cat}: {count} cases")
        sys.exit(0)

    if args.list_cases:
        print("Available cases:")
        for case in dataset:
            print(f"  [{case.category}] {case.id}: {case.description}")
        sys.exit(0)

    exit_code = asyncio.run(
        run_evaluation(
            category=args.category,
            case_id=args.case_id,
            num_samples=args.num_samples,
            verbose=args.verbose,
            output_dir=args.output_dir,
            retrieval_mode=args.mode,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
