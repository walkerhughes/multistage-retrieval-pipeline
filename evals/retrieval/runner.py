#!/usr/bin/env python
"""Evaluation runner for RAG agents.

This script runs retrieval evaluation on the curated eval dataset,
computing retrieval quality metrics against ground truth source_chunk_ids.

Usage:
    python -m evals.retrieval.runner --agent vanilla --k 5 10 15
    python -m evals.retrieval.runner --agent multi-query --num-samples 10
    python -m evals.retrieval.runner --help

Example:
    # Run multi-query agent with hybrid retrieval, evaluate at k=5,10,15
    python -m evals.retrieval.runner \
        --agent multi-query \
        --k 5 10 15 \
        --fts-candidates 100 \
        --max-returned 15 \
        --output-dir evals/results/
"""

import argparse
import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from evals.loaders import load_eval_dataset
from evals.metrics.retrieval import RetrievalMetrics, compute_retrieval_metrics
from evals.results.schemas import (
    EvalResult,
    EvalRunResults,
    MetricsBreakdown,
    build_metrics_breakdown,
)
from evals.schemas.task import DifficultyLevel, EvalTask, QuestionType
from src.agents.factory import get_agent
from src.agents.helpers import flush_traces, get_trace_id, initialize_tracing
from src.agents.models import AgentResponse, AgentType
from src.database.connection import close_db_pool, init_db_pool

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation on RAG agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (multi-query, hybrid, k=5,10,15)
  python -m evals.retrieval.runner

  # Vanilla agent with 10 samples
  python -m evals.retrieval.runner --agent vanilla --num-samples 10

  # Custom retrieval settings
  python -m evals.retrieval.runner --fts-candidates 200 --max-returned 20
        """,
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="multi-query",
        choices=["vanilla", "multi-query"],
        help="Agent type to evaluate (default: multi-query)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to eval dataset JSON (default: evals/datasets/eval_questions.json)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of examples to evaluate (default: all)",
    )

    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5, 10, 15],
        help="K values for @k metrics (default: 5 10 15)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["fts", "vector", "hybrid"],
        help="Retrieval mode (default: hybrid)",
    )

    parser.add_argument(
        "--fts-candidates",
        type=int,
        default=100,
        help="Number of FTS candidates for hybrid mode (default: 100)",
    )

    parser.add_argument(
        "--max-returned",
        type=int,
        default=15,
        help="Number of chunks returned after reranking (default: 15)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="evals/results",
        help="Output directory for results (default: evals/results)",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout per example in seconds (default: None)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def build_retrieval_params(args: argparse.Namespace) -> dict[str, Any]:
    """Build retrieval params dict from CLI arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Dict with retrieval configuration
    """
    return {
        "mode": args.mode,
        "operator": "or",
        "fts_candidates": args.fts_candidates,
        "max_returned": args.max_returned,
    }


async def run_single_eval(
    agent: Any,
    task: EvalTask,
    retrieval_params: dict[str, Any],
    k_values: list[int],
    timeout: float | None = None,
) -> EvalResult:
    """Run evaluation on a single example.

    Args:
        agent: RAG agent instance
        task: Eval task to evaluate
        retrieval_params: Retrieval configuration
        k_values: List of k values for @k metrics
        timeout: Optional timeout in seconds

    Returns:
        EvalResult with metrics and metadata
    """
    # Get ground truth chunk IDs
    ground_truth = task.source_chunk_ids or []

    try:
        # Run agent with optional timeout
        if timeout:
            response: AgentResponse = await asyncio.wait_for(
                agent.generate(task.question, retrieval_params),
                timeout=timeout,
            )
        else:
            response = await agent.generate(task.question, retrieval_params)

        # Extract retrieved chunk IDs
        retrieved_ids = [chunk.chunk_id for chunk in response.retrieved_chunks]

        # Compute metrics for each k value
        metrics_by_k: dict[int, RetrievalMetrics] = {}
        for k in k_values:
            metrics_by_k[k] = compute_retrieval_metrics(
                retrieved=retrieved_ids,
                ground_truth=ground_truth,
                k=k,
            )

        # Get trace ID
        trace_id = get_trace_id()

        return EvalResult(
            eval_id=task.id,
            question=task.question,
            question_type=task.question_type,
            difficulty_level=task.difficulty_level,
            reference_answer=task.reference_answer,
            expected_chunk_ids=ground_truth,
            generated_answer=response.answer,
            retrieved_chunk_ids=retrieved_ids,
            metrics_by_k=metrics_by_k,
            latency_ms=response.latency_ms,
            model_used=response.model_used,
            tokens_used=response.tokens_used,
            trace_id=trace_id,
            sub_queries=response.sub_queries if response.sub_queries else None,
            deduplication_stats=response.deduplication_stats if response.deduplication_stats else None,
            success=True,
            error=None,
        )

    except asyncio.TimeoutError:
        return EvalResult(
            eval_id=task.id,
            question=task.question,
            question_type=task.question_type,
            difficulty_level=task.difficulty_level,
            reference_answer=task.reference_answer,
            expected_chunk_ids=ground_truth,
            generated_answer="",
            retrieved_chunk_ids=[],
            metrics_by_k={k: compute_retrieval_metrics([], ground_truth, k) for k in k_values},
            latency_ms=timeout * 1000 if timeout else 0,
            model_used="",
            tokens_used={},
            trace_id=None,
            success=False,
            error="timeout",
        )

    except Exception as e:
        logger.error(f"Error evaluating {task.id}: {e}")
        return EvalResult(
            eval_id=task.id,
            question=task.question,
            question_type=task.question_type,
            difficulty_level=task.difficulty_level,
            reference_answer=task.reference_answer,
            expected_chunk_ids=ground_truth,
            generated_answer="",
            retrieved_chunk_ids=[],
            metrics_by_k={k: compute_retrieval_metrics([], ground_truth, k) for k in k_values},
            latency_ms=0,
            model_used="",
            tokens_used={},
            trace_id=None,
            success=False,
            error=str(e),
        )


def generate_markdown_report(
    run_results: EvalRunResults,
    output_path: Path,
) -> None:
    """Generate a markdown summary report.

    Args:
        run_results: Complete evaluation results
        output_path: Path to write markdown file
    """
    lines = [
        "# Evaluation Results",
        "",
        "## Configuration",
        f"- **Run ID:** {run_results.run_id}",
        f"- **Agent:** {run_results.agent_type}",
        f"- **Dataset:** {run_results.dataset_path}",
        f"- **Dataset Version:** {run_results.dataset_version}",
        f"- **Retrieval Mode:** {run_results.retrieval_mode}",
        f"- **FTS Candidates:** {run_results.fts_candidates}",
        f"- **Max Returned:** {run_results.max_returned}",
        f"- **K Values:** {run_results.k_values}",
        f"- **Started:** {run_results.started_at.isoformat()}",
        f"- **Completed:** {run_results.completed_at.isoformat()}",
        f"- **Duration:** {run_results.total_duration_seconds:.1f}s",
        "",
        "## Summary",
        f"- **Total Examples:** {run_results.total_examples}",
        f"- **Successful:** {run_results.num_successful} ({run_results.success_rate:.1%})",
        f"- **Failed:** {run_results.num_failed}",
        "",
    ]

    # Overall metrics table
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append("| Metric | " + " | ".join(f"k={k}" for k in run_results.k_values) + " |")
    lines.append("|--------|" + "|".join("------" for _ in run_results.k_values) + "|")

    metric_names = ["Recall", "Precision", "Hit Rate", "MRR", "NDCG"]
    metric_keys = ["recall", "precision", "hit_rate", "mrr", "ndcg"]

    for name, key in zip(metric_names, metric_keys):
        row = f"| {name} |"
        for k in run_results.k_values:
            if k in run_results.overall_by_k:
                breakdown = run_results.overall_by_k[k]
                stats = getattr(breakdown, key)
                row += f" {stats.mean:.3f} ± {stats.std:.3f} |"
            else:
                row += " - |"
        lines.append(row)

    lines.append("")

    # Latency
    if run_results.k_values and run_results.k_values[0] in run_results.overall_by_k:
        latency = run_results.overall_by_k[run_results.k_values[0]].latency_ms
        lines.append("## Latency")
        lines.append(f"- **Mean:** {latency.mean:.0f}ms")
        lines.append(f"- **Median:** {latency.median:.0f}ms")
        lines.append(f"- **Min:** {latency.min:.0f}ms")
        lines.append(f"- **Max:** {latency.max:.0f}ms")
        lines.append("")

    # By difficulty (for first k value)
    if run_results.by_difficulty and run_results.k_values:
        k = run_results.k_values[0]
        lines.append(f"## By Difficulty (k={k})")
        lines.append("")
        lines.append("| Difficulty | Count | Recall | Precision | MRR |")
        lines.append("|------------|-------|--------|-----------|-----|")
        for level in ["easy", "medium", "hard"]:
            if level in run_results.by_difficulty and k in run_results.by_difficulty[level]:
                breakdown = run_results.by_difficulty[level][k]
                lines.append(
                    f"| {level} | {breakdown.count} | "
                    f"{breakdown.recall.mean:.3f} | {breakdown.precision.mean:.3f} | "
                    f"{breakdown.mrr.mean:.3f} |"
                )
        lines.append("")

    # By question type (for first k value)
    if run_results.by_question_type and run_results.k_values:
        k = run_results.k_values[0]
        lines.append(f"## By Question Type (k={k})")
        lines.append("")
        lines.append("| Type | Count | Recall | Precision | MRR |")
        lines.append("|------|-------|--------|-----------|-----|")
        for qtype in ["factual", "analytical", "opinion"]:
            if qtype in run_results.by_question_type and k in run_results.by_question_type[qtype]:
                breakdown = run_results.by_question_type[qtype][k]
                lines.append(
                    f"| {qtype} | {breakdown.count} | "
                    f"{breakdown.recall.mean:.3f} | {breakdown.precision.mean:.3f} | "
                    f"{breakdown.mrr.mean:.3f} |"
                )
        lines.append("")

    # Failed examples
    if run_results.errors:
        lines.append("## Failed Examples")
        lines.append("")
        for error in run_results.errors:
            lines.append(f"- **{error['eval_id']}**: {error['error']}")
        lines.append("")

    # Write file
    output_path.write_text("\n".join(lines))
    logger.info(f"Markdown report saved to {output_path}")


async def run_all_evals(
    agent: Any,
    examples: list[EvalTask],
    retrieval_params: dict[str, Any],
    k_values: list[int],
    timeout: float | None,
    agent_name: str,
) -> list[EvalResult]:
    """Run all evaluations in a single event loop.

    Args:
        agent: RAG agent instance
        examples: List of eval tasks to evaluate
        retrieval_params: Retrieval configuration
        k_values: List of k values for @k metrics
        timeout: Optional timeout in seconds
        agent_name: Agent name for progress display

    Returns:
        List of evaluation results
    """
    results: list[EvalResult] = []
    for task in tqdm(examples, desc=f"Evaluating {agent_name}", unit="example"):
        result = await run_single_eval(
            agent=agent,
            task=task,
            retrieval_params=retrieval_params,
            k_values=k_values,
            timeout=timeout,
        )
        results.append(result)
    return results


def main() -> None:
    """Main evaluation runner entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Generate run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

    # Load dataset
    logger.info("Loading eval dataset...")
    dataset = load_eval_dataset(args.dataset)
    logger.info(f"Loaded {dataset.count} examples (version {dataset.version})")

    # Sample if requested
    examples = dataset.examples
    if args.num_samples and args.num_samples < len(examples):
        examples = examples[: args.num_samples]
        logger.info(f"Sampling {len(examples)} examples")

    # Initialize database and tracing
    logger.info("Initializing database connection...")
    init_db_pool()
    initialize_tracing()

    # Verify database has data
    from src.database.connection import get_db_connection
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM chunks")
            result = cur.fetchone()
            chunk_count = result["count"] if result else 0  # type: ignore[index]
            logger.info(f"Database contains {chunk_count} chunks")
            if chunk_count == 0:
                logger.warning("WARNING: No chunks in database! Metrics will be 0.")

    try:
        # Create agent
        agent_type = AgentType.VANILLA if args.agent == "vanilla" else AgentType.MULTI_QUERY
        agent = get_agent(agent_type)
        logger.info(f"Using {args.agent} agent")

        # Build retrieval params
        retrieval_params = build_retrieval_params(args)
        logger.info(f"Retrieval config: mode={args.mode}, fts_candidates={args.fts_candidates}, max_returned={args.max_returned}")

        # Run evaluations in a single event loop
        started_at = datetime.now()
        logger.info(f"Running evaluations with k={args.k}...")

        results = asyncio.run(
            run_all_evals(
                agent=agent,
                examples=examples,
                retrieval_params=retrieval_params,
                k_values=args.k,
                timeout=args.timeout,
                agent_name=args.agent,
            )
        )

        completed_at = datetime.now()

        # Flush traces
        flush_traces()

        # Compute aggregates
        logger.info("Computing aggregate metrics...")

        # Overall metrics by k
        overall_by_k: dict[int, MetricsBreakdown] = {}
        for k in args.k:
            overall_by_k[k] = build_metrics_breakdown(results, k)

        # By difficulty
        by_difficulty: dict[str, dict[int, MetricsBreakdown]] = {}
        for level in DifficultyLevel:
            level_results = [r for r in results if r.difficulty_level == level]
            if level_results:
                by_difficulty[level.value] = {}
                for k in args.k:
                    by_difficulty[level.value][k] = build_metrics_breakdown(level_results, k)

        # By question type
        by_question_type: dict[str, dict[int, MetricsBreakdown]] = {}
        for qtype in QuestionType:
            type_results = [r for r in results if r.question_type == qtype]
            if type_results:
                by_question_type[qtype.value] = {}
                for k in args.k:
                    by_question_type[qtype.value][k] = build_metrics_breakdown(type_results, k)

        # Collect errors
        errors = [
            {"eval_id": r.eval_id, "error": r.error or "unknown"}
            for r in results
            if not r.success
        ]

        # Build full results
        run_results = EvalRunResults(
            run_id=run_id,
            agent_type=args.agent,
            dataset_path=args.dataset or "evals/datasets/eval_questions.json",
            dataset_version=dataset.version,
            retrieval_mode=args.mode,
            fts_candidates=args.fts_candidates,
            max_returned=args.max_returned,
            k_values=args.k,
            started_at=started_at,
            completed_at=completed_at,
            results=results,
            overall_by_k=overall_by_k,
            by_difficulty=by_difficulty,
            by_question_type=by_question_type,
            num_successful=sum(1 for r in results if r.success),
            num_failed=sum(1 for r in results if not r.success),
            errors=errors,
        )

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        json_path = output_dir / f"{run_id}_results.json"
        json_path.write_text(run_results.model_dump_json(indent=2))
        logger.info(f"JSON results saved to {json_path}")

        # Generate markdown report
        md_path = output_dir / f"{run_id}_summary.md"
        generate_markdown_report(run_results, md_path)

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Agent: {args.agent}")
        print(f"Examples: {run_results.total_examples} ({run_results.num_successful} successful)")
        print(f"Duration: {run_results.total_duration_seconds:.1f}s")
        print()

        for k in args.k:
            if k in overall_by_k:
                breakdown = overall_by_k[k]
                print(f"k={k}:")
                print(f"  Recall:    {breakdown.recall.mean:.3f} ± {breakdown.recall.std:.3f}")
                print(f"  Precision: {breakdown.precision.mean:.3f} ± {breakdown.precision.std:.3f}")
                print(f"  Hit Rate:  {breakdown.hit_rate.mean:.3f}")
                print(f"  MRR:       {breakdown.mrr.mean:.3f} ± {breakdown.mrr.std:.3f}")
                print(f"  NDCG:      {breakdown.ndcg.mean:.3f} ± {breakdown.ndcg.std:.3f}")
                print()

        print(f"Results: {json_path}")
        print(f"Summary: {md_path}")
        print("=" * 60)

    finally:
        # Cleanup - always runs, even on exception
        close_db_pool()


if __name__ == "__main__":
    main()
