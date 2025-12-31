"""Vanilla RAG Agent implementation using OpenAI Agents SDK.

This agent uses the OpenAI Agents SDK to perform retrieval-augmented generation.
The retrieval step is implemented as a tool that the agent calls to fetch
relevant chunks from the knowledge base.

LangSmith tracing is enabled via OpenAIAgentsTracingProcessor when configured.
"""

from typing import Any

from agents import Agent, Runner, function_tool, set_trace_processors  # pyrefly: ignore
from langsmith import get_current_run_tree
from langsmith.wrappers import OpenAIAgentsTracingProcessor

from src.agents.models import AgentResponse, RetrievedChunk
from src.config import settings
from src.retrieval import (
    FullTextSearchRetriever,
    HybridRetriever,
    VectorSimilarityRetriever,
)
from src.utils.timing import Timer

# Track whether tracing has been initialized (singleton pattern)
_tracing_initialized = False


def _retrieve_chunks(
    query: str,
    retrieval_params: dict[str, Any],
) -> list[RetrievedChunk]:
    """Retrieve relevant chunks using the retrieval system directly.

    Args:
        query: Search query
        retrieval_params: Retrieval configuration

    Returns:
        List of RetrievedChunk objects
    """
    mode = retrieval_params.get("mode", "hybrid")
    max_returned = retrieval_params.get("max_returned", 10)
    operator = retrieval_params.get("operator", "or")
    fts_candidates = retrieval_params.get("fts_candidates", 100)
    filters = retrieval_params.get("filters")

    # Select retriever based on mode (mode is always a string from API layer)
    if mode == "fts":
        retriever = FullTextSearchRetriever()
        result = retriever.retrieve(
            query=query,
            n=max_returned,
            filters=filters,
            operator=operator,
        )
    elif mode == "vector":
        retriever = VectorSimilarityRetriever()
        result = retriever.retrieve(
            query=query,
            n=max_returned,
            filters=filters,
        )
    elif mode == "hybrid":
        retriever = HybridRetriever()
        result = retriever.retrieve(
            query=query,
            n=max_returned,
            filters=filters,
            fts_candidates=fts_candidates,
            operator=operator,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'fts', 'vector', or 'hybrid'.")

    return [
        RetrievedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            score=chunk.score,
            metadata=chunk.metadata,
            ord=chunk.ord,
        )
        for chunk in result.chunks
    ]


def _initialize_tracing() -> None:
    """Initialize LangSmith tracing if configured.

    Sets up the OpenAIAgentsTracingProcessor to capture agent execution
    traces in LangSmith. Only initializes once, and only if LANGSMITH_API_KEY
    is set and tracing is enabled.
    """
    global _tracing_initialized

    if _tracing_initialized:
        return

    if settings.langsmith_api_key and settings.langsmith_tracing_enabled:
        processor = OpenAIAgentsTracingProcessor(
            project_name=settings.langsmith_project,
        )
        set_trace_processors([processor])

    _tracing_initialized = True


def _get_trace_id() -> str | None:
    """Get the current LangSmith trace ID if available.

    Returns:
        The trace ID string if tracing is active, None otherwise.
    """
    try:
        run_tree = get_current_run_tree()
        if run_tree:
            return str(run_tree.id)
    except Exception:
        pass
    return None


class VanillaRAGAgent:
    """Vanilla RAG agent using OpenAI Agents SDK with tool-based retrieval.

    This agent:
    1. Receives a user question
    2. Calls the retrieve_context tool to search the knowledge base
    3. Generates an answer based on the retrieved context

    The agent uses the OpenAI Agents SDK for execution and supports
    LangSmith tracing for observability.

    Implements AgentProtocol for compatibility with the agent factory.
    """

    def __init__(self) -> None:
        """Initialize the agent.

        Note: The actual Agent instance is created per-request in generate()
        because the retrieval tool needs to capture the retrieval_params.
        """
        # Initialize tracing once (idempotent)
        _initialize_tracing()

    async def generate(
        self,
        question: str,
        retrieval_params: dict[str, Any],
    ) -> AgentResponse:
        """Generate an answer using the OpenAI Agents SDK with tool-based retrieval.

        Args:
            question: User question to answer
            retrieval_params: Retrieval configuration with keys:
                - mode: "fts", "vector", or "hybrid"
                - operator: "and" or "or"
                - fts_candidates: int (for hybrid mode)
                - max_returned: int
                - filters: Optional dict with metadata filters

        Returns:
            AgentResponse with answer, trace_id, latency, chunks, and token usage
        """
        timer = Timer()
        timer.start()

        # Instance-level storage for retrieved chunks (thread-safe)
        retrieved_chunks: list[RetrievedChunk] = []

        def _create_retrieval_tool():
            """Create a retrieval tool that captures chunks in the closure."""

            @function_tool
            def retrieve_context(query: str) -> str:
                """Search the knowledge base to find relevant information.

                Use this tool to retrieve relevant document chunks from the
                knowledge base. The tool searches through YouTube video transcripts
                and returns the most relevant passages.

                Args:
                    query: The search query to find relevant information.

                Returns:
                    A formatted string containing relevant document chunks.
                """
                nonlocal retrieved_chunks

                # Retrieve chunks using the configured parameters
                chunks = _retrieve_chunks(query, retrieval_params)

                # Store chunks for later inclusion in AgentResponse
                retrieved_chunks.extend(chunks)

                # Format chunks as context for the agent
                if not chunks:
                    return "No relevant information found in the knowledge base."

                context_parts: list[str] = []
                for i, chunk in enumerate(chunks, 1):
                    title = chunk.metadata.get("title", "Unknown")
                    context_parts.append(f"[Source {i}: {title}]\n{chunk.text}")

                return "\n\n---\n\n".join(context_parts)

            return retrieve_context

        # Create retrieval tool with params baked in
        retrieval_tool = _create_retrieval_tool()

        # Create agent with the retrieval tool
        agent = Agent(  # pyrefly: ignore
            name="RAG Assistant",
            model=settings.chat_model,
            instructions="""You are a helpful assistant that answers questions using a knowledge base of YouTube video transcripts.

When answering questions:
1. ALWAYS use the retrieve_context tool first to search for relevant information
2. Base your answer ONLY on the information returned by the tool
3. If the retrieved information doesn't contain the answer, say so clearly
4. Be concise and accurate in your responses
5. When referencing information, mention which source it came from""",
            tools=[retrieval_tool],
        )

        # Run the agent
        result = await Runner.run(
            starting_agent=agent,
            input=question,
        )

        # Extract the answer
        answer = str(result.final_output) if result.final_output else ""

        # Get trace ID if available
        trace_id = _get_trace_id()

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

        return AgentResponse(
            answer=answer,
            trace_id=trace_id,
            latency_ms=latency_ms,
            retrieved_chunks=retrieved_chunks,
            model_used=settings.chat_model,
            tokens_used=tokens_used,
        )
