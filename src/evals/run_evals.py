#!/usr/bin/env python3
"""CLI script to run agent tool parameter evaluations.

Usage:
    python -m src.evals.run_evals [OPTIONS]

Options:
    --category CATEGORY   Run only cases in specified category
    --case-id ID         Run only a specific case by ID
    --verbose            Print detailed results for each case
    --output FILE        Write results to JSON file
    --mode MODE          Retrieval mode: fts, vector, hybrid (default: fts)

Examples:
    # Run all evals
    python -m src.evals.run_evals

    # Run only speaker filter tests
    python -m src.evals.run_evals --category speaker_filter

    # Run specific case with verbose output
    python -m src.evals.run_evals --case-id speaker_001 --verbose

    # Save results to file
    python -m src.evals.run_evals --output results.json
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

from src.evals.dataset import EvalDataset
from src.evals.harness import EvalHarness
from src.evals.metrics import (
    compute_metrics,
    format_detailed_results,
    format_metrics_report,
)


def serialize_result(result: Any) -> dict:
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
    category: str | None = None,
    case_id: str | None = None,
    verbose: bool = False,
    output_file: str | None = None,
    retrieval_mode: str = "fts",
) -> int:
    """Run the evaluation and print results.

    Args:
        category: Optional category to filter cases
        case_id: Optional specific case ID to run
        verbose: Whether to print detailed results
        output_file: Optional file to write JSON results
        retrieval_mode: Retrieval mode (fts, vector, hybrid)

    Returns:
        Exit code (0 for success, 1 for failures)
    """
    dataset = EvalDataset()

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

    print(f"Running {len(cases)} evaluation case(s)...")
    print(f"Retrieval mode: {retrieval_mode}")
    print()

    # Create harness
    harness = EvalHarness(
        retrieval_params={
            "mode": retrieval_mode,
            "operator": "or",
            "fts_candidates": 100,
            "max_returned": 5,
        }
    )

    # Run evaluations
    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] Running {case.id}...", end=" ", flush=True)
        result = await harness.run_case(case)
        results.append(result)

        status = "PASS" if result.overall_match else "FAIL"
        if result.error:
            status = "ERROR"
        print(f"[{status}] ({result.latency_ms:.0f}ms)")

    print()

    # Compute metrics
    metrics = compute_metrics(results)

    # Print report
    print(format_metrics_report(metrics))

    if verbose:
        print()
        print(format_detailed_results(results))

    # Save to file if requested
    if output_file:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "retrieval_mode": retrieval_mode,
                "category_filter": category,
                "case_id_filter": case_id,
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


def main():
    """Main entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run agent tool parameter evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed results for each case",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Write results to JSON file",
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

    args = parser.parse_args()

    dataset = EvalDataset()

    if args.list_categories:
        print("Available categories:")
        for cat in sorted(dataset.categories()):
            count = len(dataset.by_category(cat))
            print(f"  {cat}: {count} cases")
        return 0

    if args.list_cases:
        print("Available cases:")
        for case in dataset:
            print(f"  [{case.category}] {case.id}: {case.description}")
        return 0

    exit_code = asyncio.run(
        run_evaluation(
            category=args.category,
            case_id=args.case_id,
            verbose=args.verbose,
            output_file=args.output,
            retrieval_mode=args.mode,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
