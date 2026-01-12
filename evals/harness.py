#!/usr/bin/env python
"""Unified evaluation harness for running different eval batches.

This is the main entry point for running evaluations. It supports different
eval types through CLI arguments and can be easily extended for new eval batches.

Usage:
    # Run retrieval evals with default settings
    python -m evals.harness --eval-type retrieval

    # Run retrieval evals with custom settings
    python -m evals.harness --eval-type retrieval --agent vanilla --num-samples 10 --k 5 10

    # List available eval types
    python -m evals.harness --list

Examples:
    # CI quick check (3 samples)
    python -m evals.harness --eval-type retrieval --agent vanilla --num-samples 3 --k 5 10

    # Full retrieval eval suite
    python -m evals.harness --eval-type retrieval --agent multi-query --k 5 10 15
"""

import argparse
import sys
from typing import NoReturn


# Registry of available eval types
EVAL_TYPES = {
    "retrieval": "Run retrieval quality evaluation against ground truth chunks",
    "tool-params": "Run agent tool parameter extraction evaluation",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse harness-level arguments, passing through eval-specific args.

    Returns:
        Tuple of (harness args namespace, list of remaining args for eval runner)
    """
    parser = argparse.ArgumentParser(
        description="Unified evaluation harness for running different eval batches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available eval types:
  retrieval    Run retrieval quality evaluation against ground truth chunks
  tool-params  Run agent tool parameter extraction evaluation

Examples:
  # Run retrieval evals
  python -m evals.harness --eval-type retrieval --agent vanilla --num-samples 3

  # Run tool parameter evals
  python -m evals.harness --eval-type tool-params --category speaker_filter

  # List available eval types
  python -m evals.harness --list
        """,
    )

    parser.add_argument(
        "--eval-type",
        "-t",
        type=str,
        choices=list(EVAL_TYPES.keys()),
        help="Type of evaluation to run",
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available eval types and exit",
    )

    # Parse known args, leave the rest for the eval runner
    args, remaining = parser.parse_known_args()

    return args, remaining


def list_eval_types() -> NoReturn:
    """Print available eval types and exit."""
    print("Available evaluation types:")
    print("-" * 50)
    for name, description in EVAL_TYPES.items():
        print(f"  {name:15} {description}")
    print("-" * 50)
    print("\nUsage: python -m evals.harness --eval-type <type> [options]")
    print("Run with --help after --eval-type for type-specific options.")
    sys.exit(0)


def run_retrieval_evals(extra_args: list[str]) -> int:
    """Run retrieval evaluation with given arguments.

    Args:
        extra_args: Additional CLI arguments to pass to retrieval runner

    Returns:
        Exit code (0 for success)
    """
    # Patch sys.argv for the retrieval runner's argparse
    import sys
    original_argv = sys.argv
    sys.argv = ["evals.retrieval.runner"] + extra_args

    try:
        from evals.retrieval.runner import main
        main()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except Exception as e:
        print(f"Error running retrieval evals: {e}")
        return 1
    finally:
        sys.argv = original_argv


def run_tool_params_evals(extra_args: list[str]) -> int:
    """Run tool parameter extraction evaluation with given arguments.

    Args:
        extra_args: Additional CLI arguments to pass to tool-params runner

    Returns:
        Exit code (0 for success)
    """
    # Patch sys.argv for the tool-params runner's argparse
    import sys
    original_argv = sys.argv
    sys.argv = ["evals.tool_params.runner"] + extra_args

    try:
        from evals.tool_params.runner import main
        main()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except Exception as e:
        print(f"Error running tool-params evals: {e}")
        return 1
    finally:
        sys.argv = original_argv


def main() -> int:
    """Main harness entry point.

    Returns:
        Exit code (0 for success)
    """
    args, extra_args = parse_args()

    if args.list:
        list_eval_types()

    if not args.eval_type:
        print("Error: --eval-type is required")
        print("Run with --list to see available eval types")
        print("Run with --help for usage information")
        return 1

    print(f"Running {args.eval_type} evaluation...")
    print("=" * 60)

    if args.eval_type == "retrieval":
        return run_retrieval_evals(extra_args)
    elif args.eval_type == "tool-params":
        return run_tool_params_evals(extra_args)
    else:
        print(f"Unknown eval type: {args.eval_type}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
