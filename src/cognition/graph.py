"""
LangGraph StateGraph Workflow

Defines the multi-agent workflow that orchestrates Supervisor, Analyst,
and Negotiator agents for supply chain decision-making.

Workflow:
    START → Supervisor → {Analyst, Negotiator, Human, END}
                Analyst → Supervisor
            Negotiator → Supervisor
"""

import logging
from typing import Any, Dict, Optional

from .state import SupplyChainState
from .supervisor import create_supervisor_agent
from .analyst import create_analyst_agent
from .negotiator import create_negotiator_agent

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import END, StateGraph
    from langgraph.checkpoint.memory import MemorySaver
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    logger.warning("langgraph not installed. Graph creation will use fallback mode.")


def _route_supervisor(state: SupplyChainState) -> str:
    """
    Routing function for supervisor's conditional edges.

    Returns the next node based on supervisor's decision.
    """
    next_agent = state.get("next_agent", "end")

    if next_agent == "analyst":
        return "analyst"
    elif next_agent == "negotiator":
        return "negotiator"
    elif next_agent == "human":
        return "human_review"
    else:
        return "end"


def _human_review_node(state: SupplyChainState) -> Dict[str, Any]:
    """
    Human-in-the-loop review node.

    Marks workflow as requiring human approval. In production,
    this would pause execution and wait for human input.
    """
    try:
        from langchain_core.messages import AIMessage
        message = AIMessage(
            content="[System] This decision requires human review. "
            f"Pending recommendations: {len(state.get('recommendations', []))}. "
            f"Please review and provide feedback."
        )
        return {
            "messages": [message],
            "next_agent": "end",
        }
    except ImportError:
        return {"next_agent": "end"}


def create_supply_chain_graph(
    llm=None,
    enable_checkpointing: bool = True,
) -> Any:
    """
    Create the LangGraph supply chain cognitive workflow.

    Args:
        llm: LangChain LLM instance (ChatOllama, ChatGroq, etc.)
             If None, agents use rule-based fallback logic.
        enable_checkpointing: Whether to enable state checkpointing
                              for pause/resume capability.

    Returns:
        Compiled LangGraph workflow ready for invocation

    Usage:
        graph = create_supply_chain_graph(llm=my_llm)
        initial = create_initial_state(alert=my_alert)
        result = graph.invoke(initial, config={"configurable": {"thread_id": "1"}})
    """
    if not HAS_LANGGRAPH:
        logger.info("LangGraph not available, returning fallback graph")
        return _create_fallback_graph(llm)

    # Create agent nodes
    supervisor = create_supervisor_agent(llm)
    analyst = create_analyst_agent(llm)
    negotiator = create_negotiator_agent(llm)

    # Build the graph
    graph = StateGraph(SupplyChainState)

    # Add nodes
    graph.add_node("supervisor", supervisor)
    graph.add_node("analyst", analyst)
    graph.add_node("negotiator", negotiator)
    graph.add_node("human_review", _human_review_node)

    # Set entry point
    graph.set_entry_point("supervisor")

    # Add conditional edges from supervisor
    graph.add_conditional_edges(
        "supervisor",
        _route_supervisor,
        {
            "analyst": "analyst",
            "negotiator": "negotiator",
            "human_review": "human_review",
            "end": END,
        },
    )

    # Analyst and Negotiator always report back to Supervisor
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("negotiator", "supervisor")

    # Human review goes to END
    graph.add_edge("human_review", END)

    # Compile with optional checkpointing
    compile_kwargs = {}
    if enable_checkpointing:
        compile_kwargs["checkpointer"] = MemorySaver()

    compiled = graph.compile(**compile_kwargs)
    logger.info("Supply chain cognitive graph compiled successfully")

    return compiled


class FallbackGraph:
    """
    Fallback sequential workflow when LangGraph is not installed.

    Runs agents in sequence: Supervisor → Analyst → Supervisor → END
    """

    def __init__(self, llm=None):
        self.supervisor = create_supervisor_agent(llm)
        self.analyst = create_analyst_agent(llm)
        self.negotiator = create_negotiator_agent(llm)

    def invoke(self, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run the workflow sequentially."""
        current_state = dict(state)

        for iteration in range(10):  # Max iterations
            # Supervisor decides
            try:
                update = self.supervisor(current_state)
            except Exception:
                # If langchain_core not available, use rule-based
                from .supervisor import _rule_based_decision
                decision = _rule_based_decision(current_state)
                update = {
                    "next_agent": {
                        "analyze": "analyst",
                        "negotiate": "negotiator",
                        "respond": "end",
                        "escalate": "end",
                    }.get(decision.get("action", "respond"), "end"),
                    "iteration_count": current_state.get("iteration_count", 0) + 1,
                }

            current_state.update(update)

            next_agent = current_state.get("next_agent", "end")

            if next_agent == "end" or next_agent == "human":
                break
            elif next_agent == "analyst":
                try:
                    analyst_update = self.analyst(current_state)
                except Exception:
                    from .analyst import _rule_based_analysis
                    analysis = _rule_based_analysis(current_state)
                    analyst_update = {
                        "analysis_results": analysis,
                        "next_agent": "supervisor",
                    }
                current_state.update(analyst_update)
            elif next_agent == "negotiator":
                try:
                    neg_update = self.negotiator(current_state)
                except Exception:
                    from .negotiator import _rule_based_negotiation
                    negotiation = _rule_based_negotiation(current_state)
                    neg_update = {
                        "negotiation_results": negotiation,
                        "next_agent": "supervisor",
                    }
                current_state.update(neg_update)

        return current_state


def _create_fallback_graph(llm=None) -> FallbackGraph:
    """Create a fallback graph when LangGraph is not installed."""
    return FallbackGraph(llm)
