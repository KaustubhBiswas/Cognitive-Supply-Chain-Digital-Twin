"""
Supervisor Agent Implementation

Central coordinator for the multi-agent supply chain system.
Receives alerts, delegates to Analyst/Negotiator, and makes final decisions.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .state import SupplyChainState

logger = logging.getLogger(__name__)

# Load system prompt
_PROMPT_PATH = Path(__file__).parent / "prompts" / "supervisor.txt"


def _load_prompt() -> str:
    """Load the supervisor system prompt."""
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Supervisor prompt file not found, using default")
        return "You are a supply chain supervisor agent. Respond with JSON: {\"action\": \"respond\", \"reasoning\": \"...\", \"message\": \"...\", \"priority\": \"medium\"}"


def _format_context(state: SupplyChainState) -> str:
    """Format the current context for the supervisor."""
    parts = []

    # Alert info
    alert = state.get("current_alert")
    if alert:
        parts.append(
            f"ACTIVE ALERT:\n"
            f"  Type: {alert.get('alert_type', 'unknown')}\n"
            f"  Severity: {alert.get('severity', 'unknown')}\n"
            f"  Affected Nodes: {alert.get('affected_nodes', [])}\n"
            f"  Details: {json.dumps(alert.get('details', {}), indent=2)}"
        )

    # Simulation state
    sim_state = state.get("simulation_state", {})
    if sim_state:
        parts.append(
            f"SIMULATION STATE:\n"
            f"  Step: {sim_state.get('current_step', 'N/A')}\n"
            f"  Total Inventory: {sim_state.get('total_inventory', 'N/A')}\n"
            f"  Bullwhip Ratio: {sim_state.get('bullwhip_ratio', 'N/A')}"
        )

    # Analysis results if available
    analysis = state.get("analysis_results")
    if analysis:
        parts.append(
            f"ANALYST REPORT:\n"
            f"  Findings: {analysis.get('findings', 'N/A')}\n"
            f"  Risk Level: {analysis.get('risk_level', 'N/A')}\n"
            f"  Recommendations: {len(analysis.get('recommendations', []))} items"
        )

    # Negotiation results if available
    negotiation = state.get("negotiation_results")
    if negotiation:
        parts.append(
            f"NEGOTIATOR REPORT:\n"
            f"  Plan: {negotiation.get('coordination_plan', 'N/A')}\n"
            f"  Proposals: {len(negotiation.get('proposals', []))} items"
        )

    # Human feedback
    feedback = state.get("human_feedback")
    if feedback:
        parts.append(f"HUMAN FEEDBACK: {feedback}")

    return "\n\n".join(parts) if parts else "No context available."


def _parse_response(response_text: str) -> Dict[str, Any]:
    """Parse the LLM response, extracting JSON."""
    # Try to find JSON in the response
    text = response_text.strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    for marker in ["```json", "```"]:
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

    # Try to find JSON object in text
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    # Fallback
    logger.warning("Could not parse supervisor response as JSON")
    return {
        "action": "respond",
        "reasoning": "Could not parse structured response",
        "message": text,
        "priority": "medium",
    }


def create_supervisor_agent(llm=None):
    """
    Create the supervisor agent node function.

    Args:
        llm: LangChain LLM instance (ChatOllama, ChatGroq, etc.)
             If None, uses a simple rule-based fallback.

    Returns:
        Supervisor node function for LangGraph
    """

    system_prompt = _load_prompt()

    def supervisor_node(state: SupplyChainState) -> Dict[str, Any]:
        """
        Supervisor agent node.

        Receives state, decides next action, routes to appropriate agent.
        """
        context = _format_context(state)
        iteration = state.get("iteration_count", 0)

        # Guard against infinite loops
        if iteration >= 10:
            logger.warning("Max iterations reached, ending workflow")
            return {
                "next_agent": "end",
                "iteration_count": iteration + 1,
                "error": "Maximum iteration count reached",
            }

        # Use LLM if available
        if llm is not None:
            try:
                response = _llm_decision(llm, system_prompt, context, state)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                response = _rule_based_decision(state)
        else:
            response = _rule_based_decision(state)

        # Map action to next agent
        action = response.get("action", "respond")
        next_agent = {
            "analyze": "analyst",
            "negotiate": "negotiator",
            "respond": "end",
            "escalate": "human",
        }.get(action, "end")

        # Build return state update
        from langchain_core.messages import AIMessage
        message_content = json.dumps(response, indent=2)

        return {
            "messages": [AIMessage(content=f"[Supervisor] {message_content}")],
            "next_agent": next_agent,
            "iteration_count": iteration + 1,
        }

    return supervisor_node


def _llm_decision(
    llm, system_prompt: str, context: str, state: SupplyChainState
) -> Dict[str, Any]:
    """Make a decision using the LLM."""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Current situation:\n\n{context}\n\n"
            f"Decide your next action and explain your reasoning."
        ),
    ]

    response = llm.invoke(messages)
    return _parse_response(response.content)


def _rule_based_decision(state: SupplyChainState) -> Dict[str, Any]:
    """
    Rule-based fallback when no LLM is available.

    Uses the alert type, severity, and state to make decisions.
    """
    alert = state.get("current_alert")
    analysis = state.get("analysis_results")
    negotiation = state.get("negotiation_results")

    # If we already have analysis and negotiation results, synthesize
    if analysis and negotiation:
        return {
            "action": "respond",
            "reasoning": "Analysis and negotiation complete, synthesizing results",
            "message": f"Analysis risk: {analysis.get('risk_level', 'unknown')}. "
            f"Proposals: {len(negotiation.get('proposals', []))} coordination actions.",
            "priority": analysis.get("risk_level", "medium"),
        }

    # If we have analysis but not negotiation, and recommendations exist
    if analysis and not negotiation:
        recs = analysis.get("recommendations", [])
        # Trigger negotiator for any recommendation that requires coordination
        negotiation_types = (
            "adjust_order_quantity",
            "adjust_safety_stock", 
            "adjust_reorder_point",
            "jit_buffer_increase",
            "emergency_buffer_increase",
            "coordinate_orders",
            "proactive_preparation",
            "synchronized_ordering",
        )
        if any(r.get("type") in negotiation_types for r in recs):
            return {
                "action": "negotiate",
                "reasoning": "Analyst recommends adjustments requiring supply chain coordination",
                "message": f"Please coordinate these adjustments: {json.dumps(recs)}",
                "priority": analysis.get("risk_level", "medium"),
            }
        return {
            "action": "respond",
            "reasoning": "Analysis complete, no negotiation needed",
            "message": f"Findings: {analysis.get('findings', 'N/A')}",
            "priority": analysis.get("risk_level", "medium"),
        }

    # No alert - nothing to do
    if not alert:
        return {
            "action": "respond",
            "reasoning": "No active alert to process",
            "message": "Supply chain operating normally",
            "priority": "low",
        }

    alert_type = alert.get("alert_type", "")
    severity = alert.get("severity", "medium")
    affected = alert.get("affected_nodes", [])

    # Critical severity or many nodes affected -> escalate
    if severity == "critical" or len(affected) > 3:
        return {
            "action": "escalate",
            "reasoning": f"Critical alert ({alert_type}) affecting {len(affected)} nodes",
            "message": f"URGENT: {alert_type} requires human review",
            "priority": "critical",
        }

    # Bullwhip or forecast issues -> analyze
    if alert_type in ("bullwhip_detected", "forecast_deviation", "demand_spike", "demand_drop"):
        return {
            "action": "analyze",
            "reasoning": f"Alert type {alert_type} requires data analysis",
            "message": f"Analyze {alert_type} for nodes {affected}",
            "priority": severity,
        }

    # Supply disruption -> negotiate
    if alert_type in ("supply_disruption", "lead_time_change", "capacity_constraint"):
        return {
            "action": "negotiate",
            "reasoning": f"Supply-side issue requires coordination: {alert_type}",
            "message": f"Coordinate response to {alert_type} for nodes {affected}",
            "priority": severity,
        }

    # Inventory issues -> analyze first
    if alert_type in ("inventory_low", "inventory_excess", "stockout"):
        return {
            "action": "analyze",
            "reasoning": f"Inventory issue needs analysis: {alert_type}",
            "message": f"Analyze inventory for nodes {affected}",
            "priority": severity,
        }

    # Default: analyze
    return {
        "action": "analyze",
        "reasoning": f"Unknown alert type {alert_type}, defaulting to analysis",
        "message": f"Please analyze: {alert_type}",
        "priority": "medium",
    }
