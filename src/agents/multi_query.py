"""Multi-Query RAG Agent with query decomposition and parallel retrieval.

This agent decomposes user queries into 2-5 MECE (Mutually Exclusive, Collectively
Exhaustive) sub-queries, retrieves chunks for each in parallel, deduplicates results
with score boosting, and generates a comprehensive answer.

The agent uses the OpenAI Agents SDK with a retrieval tool that accepts multiple
queries and executes them in parallel using asyncio.gather.

LangSmith tracing is enabled via OpenAIAgentsTracingProcessor when configured.
"""

import concurrent.futures
from typing import Any

from agents import Agent, Runner, function_tool  # pyrefly: ignore

from src.agents.helpers import get_trace_id, initialize_tracing, retrieve_chunks
from src.agents.models import AgentResponse, RetrievedChunk
from src.config import settings
from src.utils.timing import Timer


def _deduplicate_chunks(
    results_by_query: dict[str, list[RetrievedChunk]],
    max_returned: int = 15,
    boost_factor: float = 0.2,
) -> tuple[list[RetrievedChunk], dict[str, Any]]:
    """Deduplicate chunks across sub-queries with score boosting.

    Deduplication Strategy:
    - Uses chunk_id as the deduplication key
    - For chunks appearing in multiple sub-query results, the score is boosted
      by boost_factor (default 20%) for each additional occurrence
    - This reflects the intuition that chunks relevant to multiple aspects of
      a question are more important overall

    Example:
        A chunk appearing in 3 sub-query results with max score 0.8:
        boosted_score = 0.8 * (1 + 0.2 * (3 - 1)) = 0.8 * 1.4 = 1.12

    Args:
        results_by_query: Dict mapping sub-query string to list of chunks
        max_returned: Maximum number of chunks to return after deduplication
        boost_factor: Score boost per additional occurrence (default: 0.2 = 20%)

    Returns:
        Tuple of (deduplicated_chunks, stats_dict) where stats includes:
        - total_before_dedup: Total chunks before deduplication
        - unique_chunks: Number of unique chunk_ids
        - duplicates_removed: Number of duplicate occurrences removed
        - chunks_boosted: Number of chunks that appeared in multiple results
        - max_occurrences: Maximum times any single chunk appeared
    """
    # Track chunk_id -> (best_chunk, occurrence_count, max_score)
    chunk_map: dict[int, tuple[RetrievedChunk, int, float]] = {}

    total_before = 0
    for query, chunks in results_by_query.items():
        for chunk in chunks:
            total_before += 1
            if chunk.chunk_id in chunk_map:
                existing_chunk, count, max_score = chunk_map[chunk.chunk_id]
                chunk_map[chunk.chunk_id] = (
                    existing_chunk,
                    count + 1,
                    max(max_score, chunk.score),
                )
            else:
                chunk_map[chunk.chunk_id] = (chunk, 1, chunk.score)

    # Calculate boosted scores and create final chunks
    boosted_chunks: list[RetrievedChunk] = []
    max_occurrences = 0
    chunks_boosted = 0

    for chunk_id, (chunk, count, max_score) in chunk_map.items():
        max_occurrences = max(max_occurrences, count)
        if count > 1:
            chunks_boosted += 1

        # Boost: +boost_factor for each additional occurrence
        boost_multiplier = 1.0 + (boost_factor * (count - 1))
        boosted_score = max_score * boost_multiplier

        # Create new chunk with boosted score
        boosted_chunks.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=boosted_score,
                metadata=chunk.metadata,
                ord=chunk.ord,
            )
        )

    # Sort by boosted score descending and take top N
    boosted_chunks.sort(key=lambda x: x.score, reverse=True)
    final_chunks = boosted_chunks[:max_returned]

    # Calculate stats
    stats = {
        "total_before_dedup": total_before,
        "unique_chunks": len(chunk_map),
        "duplicates_removed": total_before - len(chunk_map),
        "chunks_boosted": chunks_boosted,
        "max_occurrences": max_occurrences,
        "chunks_returned": len(final_chunks),
    }

    return final_chunks, stats


class MultiQueryRAGAgent:
    """Multi-query RAG agent with query decomposition and parallel retrieval.

    This agent implements a sophisticated retrieval strategy:
    1. Provides a retrieval tool that accepts multiple queries
    2. The LLM decomposes the user question into 2-5 MECE sub-queries
    3. Executes searches for all sub-queries in parallel
    4. Deduplicates results with score boosting for chunks appearing multiple times
    5. Generates a comprehensive answer synthesizing information from all sub-queries

    The agent gives the LLM control over query decomposition through the tool interface,
    allowing it to decide how to break down the question based on context.

    Implements AgentProtocol for compatibility with the agent factory.

    Attributes:
        DEFAULT_MAX_RETURNED: Default number of chunks to return (15)
        BOOST_FACTOR: Score boost per additional occurrence (0.2 = 20%)
    """

    DEFAULT_MAX_RETURNED = 15
    BOOST_FACTOR = 0.2

    def __init__(self) -> None:
        """Initialize the agent.

        Sets up LangSmith tracing if configured. The actual Agent instance
        is created per-request in generate() to capture request-specific parameters.
        """
        initialize_tracing()

    async def generate(
        self,
        question: str,
        retrieval_params: dict[str, Any],
    ) -> AgentResponse:
        """Generate an answer using multi-query RAG with parallel retrieval.

        The agent will:
        1. Analyze the question and decompose it into sub-queries
        2. Call the multi-retrieval tool with those sub-queries
        3. Receive deduplicated, score-boosted results
        4. Synthesize a comprehensive answer

        Args:
            question: User question to answer
            retrieval_params: Retrieval configuration with keys:
                - mode: "fts", "vector", or "hybrid" (default: "hybrid")
                - operator: "and" or "or" (default: "or")
                - fts_candidates: int for hybrid mode (default: 100)
                - max_returned: int, chunks per sub-query (default: 15)
                - filters: Optional dict with metadata filters

        Returns:
            AgentResponse with answer, retrieved chunks, and multi-query metadata:
            - sub_queries: List of sub-queries used
            - chunks_per_subquery: Dict mapping sub-query to chunk count
            - deduplication_stats: Stats about the deduplication process
        """
        timer = Timer()
        timer.start()

        # Set default max_returned for multi-query if not specified
        if "max_returned" not in retrieval_params:
            retrieval_params = {**retrieval_params, "max_returned": self.DEFAULT_MAX_RETURNED}

        # Storage for results captured in tool closure
        all_results_by_query: dict[str, list[RetrievedChunk]] = {}
        final_deduplicated_chunks: list[RetrievedChunk] = []
        dedup_stats: dict[str, Any] = {}
        captured_sub_queries: list[str] = []

        # Capture these for use in closure
        max_returned = retrieval_params.get("max_returned", self.DEFAULT_MAX_RETURNED)
        boost_factor = self.BOOST_FACTOR

        def _create_multi_retrieval_tool():
            """Create a retrieval tool that accepts multiple queries and runs them in parallel."""

            @function_tool
            def retrieve_for_queries(queries: list[str]) -> str:
                """Search the knowledge base with multiple queries in parallel.

                Use this tool to retrieve relevant information for multiple search queries
                simultaneously. This is useful when a complex question needs to be broken
                down into multiple aspects or sub-questions.

                The tool will:
                1. Execute all queries in parallel for efficiency
                2. Deduplicate results (same chunks found by multiple queries)
                3. Boost scores for chunks that appear in multiple query results
                4. Return the most relevant chunks across all queries

                Args:
                    queries: List of 2-5 search queries. Each query should target
                            a specific aspect of the information needed.

                Returns:
                    A formatted string containing the most relevant document chunks
                    from across all queries, deduplicated and ranked by relevance.
                """
                nonlocal all_results_by_query, final_deduplicated_chunks, dedup_stats, captured_sub_queries

                # Validate query count
                if len(queries) < 1:
                    return "Error: At least one query is required."
                if len(queries) > 5:
                    queries = queries[:5]  # Limit to 5 queries

                # Store sub-queries for response metadata
                captured_sub_queries[:] = queries  # Replace contents in-place

                # Execute retrievals in parallel with error handling
                def retrieve_all_sync() -> dict[str, list[RetrievedChunk]]:
                    """Run all retrievals using ThreadPoolExecutor for parallelism.

                    Handles individual query failures gracefully - failed queries
                    return empty results rather than crashing the entire retrieval.
                    """
                    results: dict[str, list[RetrievedChunk]] = {}

                    def retrieve_one(query: str) -> tuple[str, list[RetrievedChunk]]:
                        chunks = retrieve_chunks(query, retrieval_params)
                        return query, chunks

                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as executor:
                        futures = {executor.submit(retrieve_one, q): q for q in queries}
                        for future in concurrent.futures.as_completed(futures):
                            original_query = futures[future]
                            try:
                                query, chunks = future.result()
                                results[query] = chunks
                            except Exception:
                                # Log could be added here for debugging
                                # Failed queries return empty results rather than crashing
                                results[original_query] = []

                    return results

                # Run parallel retrieval
                results_by_query = retrieve_all_sync()

                # Update closure state (use slice assignment for in-place update)
                all_results_by_query.clear()
                all_results_by_query.update(results_by_query)

                # Deduplicate with score boosting
                deduplicated, stats = _deduplicate_chunks(
                    results_by_query,
                    max_returned=max_returned,
                    boost_factor=boost_factor,
                )

                # Update closure state
                final_deduplicated_chunks[:] = deduplicated
                dedup_stats.clear()
                dedup_stats.update(stats)

                # Format chunks as context for the agent
                if not deduplicated:
                    return "No relevant information found in the knowledge base for any of the queries."

                # Build summary header
                query_summary = "\n".join(
                    f"  - {q}: {len(results_by_query.get(q, []))} chunks"
                    for q in queries
                )
                header = (
                    f"Retrieved {stats['unique_chunks']} unique chunks from {len(queries)} queries:\n"
                    f"{query_summary}\n\n"
                    f"After deduplication and ranking, top {len(deduplicated)} chunks:\n"
                )

                context_parts: list[str] = [header]
                for i, chunk in enumerate(deduplicated, 1):
                    title = chunk.metadata.get("title", "Unknown")
                    score_info = f"[Score: {chunk.score:.3f}]"
                    context_parts.append(f"[Source {i}: {title}] {score_info}\n{chunk.text}")

                return "\n\n---\n\n".join(context_parts)

            return retrieve_for_queries

        # Create the multi-retrieval tool
        multi_retrieval_tool = _create_multi_retrieval_tool()

        # Create agent with instructions for multi-query retrieval
        agent = Agent(  # pyrefly: ignore
            name="Multi-Query RAG Assistant",
            model=settings.chat_model,
            instructions=f"""You are a helpful assistant that answers questions using a knowledge base of YouTube video transcripts.

You have access to a powerful multi-query retrieval tool. To answer questions effectively:

1. FIRST, analyze the user's question and break it down into 2-5 MECE (Mutually Exclusive, Collectively Exhaustive) sub-queries
2. Each sub-query should target a specific aspect of the question
3. Call retrieve_for_queries with your list of sub-queries
4. Synthesize the retrieved information into a comprehensive answer

Example decomposition for "What are the benefits and risks of AGI?":
- Sub-queries: ["benefits of AGI", "risks of AGI", "AGI safety concerns", "AGI potential applications"]

Guidelines:
- Use 2-3 sub-queries for simple questions, 4-5 for complex multi-part questions
- Make sub-queries specific and searchable
- Base your answer ONLY on the retrieved information
- If the information doesn't fully answer the question, say so clearly
- Reference sources when citing specific information
- Provide a comprehensive answer that addresses all aspects of the question

User question: {question}""",
            tools=[multi_retrieval_tool],
        )

        # Run the agent
        result = await Runner.run(
            starting_agent=agent,
            input=question,
        )

        # Extract the answer
        answer = str(result.final_output) if result.final_output else ""

        # Get trace ID if available
        trace_id = get_trace_id()

        # Calculate total token usage from all model responses
        total_input_tokens = 0
        total_output_tokens = 0
        for response in result.raw_responses:
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

        tokens_used = {
            "prompt_tokens": total_input_tokens,
            "completion_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }

        latency_ms = timer.stop()

        # Build chunks_per_subquery from captured results
        chunks_per_subquery = {
            query: len(chunks) for query, chunks in all_results_by_query.items()
        }

        return AgentResponse(
            answer=answer,
            trace_id=trace_id,
            latency_ms=latency_ms,
            retrieved_chunks=final_deduplicated_chunks,
            model_used=settings.chat_model,
            tokens_used=tokens_used,
            sub_queries=captured_sub_queries,
            chunks_per_subquery=chunks_per_subquery,
            deduplication_stats=dedup_stats,
        )
