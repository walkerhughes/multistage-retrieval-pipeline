"""Multi-Query RAG Agent stub.

This is a placeholder for Issue #13 implementation. The multi-query agent
will decompose queries into sub-queries, retrieve for each, deduplicate,
and generate a comprehensive response.

See Issue #13 for full specification.
"""

from typing import Any

from src.agents.models import AgentResponse


class MultiQueryRAGAgent:
    """Multi-query RAG agent with query decomposition.

    NOT YET IMPLEMENTED - See Issue #13.

    This agent will:
    1. Decompose user query into MECE sub-queries
    2. Retrieve relevant chunks for each sub-query
    3. Deduplicate and rerank merged results
    4. Generate comprehensive answer from merged context

    Implements AgentProtocol for compatibility with the agent factory.
    """

    async def generate(
        self,
        question: str,
        retrieval_params: dict[str, Any],
    ) -> AgentResponse:
        """Generate an answer using multi-query RAG pipeline.

        Args:
            question: User question to answer
            retrieval_params: Retrieval configuration

        Raises:
            NotImplementedError: This agent is not yet implemented.
                See Issue #13 for implementation plan.
        """
        raise NotImplementedError(
            "MultiQueryRAGAgent is not yet implemented. "
            "See Issue #13 for the implementation plan. "
            "Use agent='vanilla' instead."
        )
