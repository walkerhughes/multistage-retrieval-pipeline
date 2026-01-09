"""Evaluation harness for capturing and validating agent tool calls.

This module provides the infrastructure to run evaluation cases against
an agent and capture the tool calls it makes, comparing actual parameters
against expected values.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from agents import Agent, Runner, function_tool  # pyrefly: ignore

from src.agents.helpers import get_trace_id, initialize_tracing, retrieve_chunks
from src.config import settings
from src.evals.dataset import EvalCase, ExpectedFilters
from src.utils.timing import Timer


@dataclass
class ToolCallCapture:
    """Captured tool call with its parameters.

    Attributes:
        tool_name: Name of the tool that was called
        query: The search query passed to the tool
        filters: Dictionary of filter parameters applied
        raw_args: Raw arguments as passed to the tool
    """

    tool_name: str
    query: str
    filters: dict[str, Any] = field(default_factory=dict)
    raw_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of running a single evaluation case.

    Attributes:
        case_id: ID of the eval case
        query: Original query
        expected_filters: Expected filter values
        actual_filters: Filters actually applied by the agent
        tool_calls: List of all tool calls made
        filter_matches: Dict mapping filter names to match status
        overall_match: True if all expected filters were correctly applied
        answer: The agent's generated answer
        latency_ms: Time taken for the evaluation
        error: Error message if evaluation failed
    """

    case_id: str
    query: str
    expected_filters: ExpectedFilters
    actual_filters: dict[str, Any]
    tool_calls: list[ToolCallCapture]
    filter_matches: dict[str, bool]
    overall_match: bool
    answer: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None


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
        matches["speaker"] = expected_speaker in actual_speaker or actual_speaker in expected_speaker

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


class EvalHarness:
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

    async def run_case(self, case: EvalCase) -> EvalResult:
        """Run a single evaluation case.

        Args:
            case: The eval case to run

        Returns:
            EvalResult with comparison of expected vs actual filters
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
     - "What has Elon Musk said about AI?" → speaker="Elon Musk"
     - "According to Sam Altman..." → speaker="Sam Altman"
     - "John Smith's views on X" → speaker="John Smith"

3. DATE FILTERING: When the user mentions specific dates or years:
   - "in 2024" → start_date="2024-01-01", end_date="2024-12-31"
   - "after March 2023" → start_date="2023-03-01"
   - "before 2022" → end_date="2021-12-31"

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

            return EvalResult(
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
            return EvalResult(
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

    async def run_all(self, cases: list[EvalCase]) -> list[EvalResult]:
        """Run all evaluation cases sequentially.

        Args:
            cases: List of eval cases to run

        Returns:
            List of EvalResult for each case
        """
        results = []
        for case in cases:
            result = await self.run_case(case)
            results.append(result)
        return results
