"""Agent factory for selecting and instantiating RAG agents."""

from src.agents.models import AgentType
from src.agents.vanilla import VanillaRAGAgent
from src.agents.multi_query import MultiQueryRAGAgent


def get_agent(agent_type: AgentType) -> VanillaRAGAgent | MultiQueryRAGAgent:
    """Factory function to get an agent instance by type.

    Args:
        agent_type: Type of agent to instantiate

    Returns:
        Agent instance implementing AgentProtocol

    Raises:
        ValueError: If agent_type is not recognized

    Example:
        >>> agent = get_agent(AgentType.VANILLA)
        >>> response = await agent.generate("What is AGI?", retrieval_params)
    """
    if agent_type == AgentType.VANILLA:
        return VanillaRAGAgent()
    elif agent_type == AgentType.MULTI_QUERY:
        return MultiQueryRAGAgent()
    else:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Valid types are: {[t.value for t in AgentType]}"
        )
