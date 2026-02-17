"""
Negotiator Agent Implementation

Specialized agent for optimizing order quantities, coordinating between
supply chain partners, and proposing collaborative arrangements.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .state import SupplyChainState
from .tools import (
    get_upstream_suppliers,
    get_downstream_customers,
    forecast_demand,
    propose_order_adjustment,
    propose_policy_change,
    get_node_inventory,
    get_supply_chain_metrics,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "negotiator.txt"


def _load_prompt() -> str:
    """Load the negotiator system prompt."""
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Negotiator prompt file not found, using default")
        return "You are a supply chain negotiator. Propose coordination plans in JSON format."


def _format_negotiation_context(state: SupplyChainState) -> str:
    """Format context for negotiation."""
    parts = []

    alert = state.get("current_alert")
    if alert:
        parts.append(
            f"ISSUE TO RESOLVE:\n"
            f"  Type: {alert.get('alert_type')}\n"
            f"  Severity: {alert.get('severity')}\n"
            f"  Nodes: {alert.get('affected_nodes')}\n"
            f"  Details: {json.dumps(alert.get('details', {}))}"
        )

    analysis = state.get("analysis_results")
    if analysis:
        parts.append(
            f"ANALYST FINDINGS:\n"
            f"  {analysis.get('findings', 'N/A')}\n"
            f"  Risk: {analysis.get('risk_level', 'N/A')}"
        )
        recs = analysis.get("recommendations", [])
        if recs:
            parts.append(f"ANALYST RECOMMENDATIONS: {json.dumps(recs, indent=2)}")

    sim_state = state.get("simulation_state", {})
    if sim_state:
        parts.append(
            f"SYSTEM STATE:\n"
            f"  Total Inventory: {sim_state.get('total_inventory', 'N/A')}\n"
            f"  Bullwhip Ratio: {sim_state.get('bullwhip_ratio', 'N/A')}"
        )

    return "\n\n".join(parts) if parts else "No context available."


def _parse_response(response_text: str) -> Dict[str, Any]:
    """Parse negotiator response JSON."""
    text = response_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for marker in ["```json", "```"]:
        if marker in text:
            try:
                start = text.index(marker) + len(marker)
                end = text.index("```", start)
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    return {
        "proposals": [],
        "coordination_plan": text,
        "expected_impact": {},
    }


def create_negotiator_agent(llm=None):
    """
    Create the negotiator agent node function.

    Args:
        llm: LangChain LLM instance. If None, uses rule-based negotiation.

    Returns:
        Negotiator node function for LangGraph
    """
    system_prompt = _load_prompt()

    def negotiator_node(state: SupplyChainState) -> Dict[str, Any]:
        """
        Negotiator agent node.

        Analyzes supply chain relationships and proposes coordination strategies.
        """
        if llm is not None:
            try:
                result = _llm_negotiation(llm, system_prompt, state)
            except Exception as e:
                logger.error(f"LLM negotiation failed: {e}")
                result = _rule_based_negotiation(state)
        else:
            result = _rule_based_negotiation(state)

        # Convert proposals to state recommendations
        recommendations = state.get("recommendations", []).copy()
        for proposal in result.get("proposals", []):
            target_nodes = []
            if "target_node" in proposal:
                target_nodes.append(proposal["target_node"])
            if "source_node" in proposal:
                target_nodes.append(proposal["source_node"])

            recommendations.append({
                "recommendation_type": proposal.get("type", "adjust_order_quantity"),
                "target_nodes": target_nodes,
                "parameters": proposal.get("parameters", {}),
                "reasoning": proposal.get("benefit", ""),
                "confidence": 0.7,
                "source_agent": "negotiator",
                "requires_approval": True,  # Negotiation proposals always need approval
            })

        from langchain_core.messages import AIMessage
        message = (
            f"[Negotiator] Plan: {result.get('coordination_plan', 'N/A')}. "
            f"Proposals: {len(result.get('proposals', []))}."
        )

        return {
            "messages": [AIMessage(content=message)],
            "negotiation_results": result,
            "recommendations": recommendations,
            "next_agent": "supervisor",  # Report back to supervisor
        }

    return negotiator_node


def _llm_negotiation(
    llm, system_prompt: str, state: SupplyChainState
) -> Dict[str, Any]:
    """Perform negotiation using the LLM."""
    from langchain_core.messages import HumanMessage, SystemMessage

    context = _format_negotiation_context(state)
    network_data = _gather_network_data(state)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Develop a coordination plan for this situation:\n\n"
            f"{context}\n\n"
            f"Supply network data:\n{json.dumps(network_data, indent=2)}\n\n"
            f"Propose collaborative solutions."
        ),
    ]

    response = llm.invoke(messages)
    return _parse_response(response.content)


def _gather_network_data(state: SupplyChainState) -> Dict[str, Any]:
    """Gather supply network data for negotiation."""
    data = {}
    alert = state.get("current_alert")

    if not alert:
        return data

    affected = alert.get("affected_nodes", [])

    for node_id in affected[:5]:
        node_info = {}

        try:
            upstream = get_upstream_suppliers.invoke({"node_id": node_id})
            if isinstance(upstream, dict) and upstream.get("success"):
                node_info["suppliers"] = upstream.get("upstream_suppliers", [])
        except Exception:
            pass

        try:
            downstream = get_downstream_customers.invoke({"node_id": node_id})
            if isinstance(downstream, dict) and downstream.get("success"):
                node_info["customers"] = downstream.get("downstream_customers", [])
        except Exception:
            pass

        try:
            inv = get_node_inventory.invoke({"node_id": node_id})
            if isinstance(inv, dict) and inv.get("success"):
                node_info["inventory"] = inv
        except Exception:
            pass

        if node_info:
            data[str(node_id)] = node_info

    return data


def _rule_based_negotiation(state: SupplyChainState) -> Dict[str, Any]:
    """
    Rule-based negotiation when no LLM is available.

    Generates coordination proposals based on alert type and analysis results.
    """
    alert = state.get("current_alert")
    analysis = state.get("analysis_results")
    proposals = []

    if not alert:
        return {
            "proposals": [],
            "coordination_plan": "No active issue to negotiate",
            "expected_impact": {},
        }

    alert_type = alert.get("alert_type", "")
    affected = alert.get("affected_nodes", [])
    details = alert.get("details", {})

    if alert_type in ("demand_spike", "inventory_low", "stockout"):
        # Strategy: Request expedited orders from upstream suppliers
        for node_id in affected:
            try:
                upstream = get_upstream_suppliers.invoke({"node_id": node_id})
                if isinstance(upstream, dict) and upstream.get("success"):
                    suppliers = upstream.get("upstream_suppliers", [])
                    for supplier in suppliers:
                        sup_id = supplier.get("node_id")
                        sup_inventory = supplier.get("inventory", 0)
                        if sup_inventory > 0:
                            proposals.append({
                                "type": "order_adjustment",
                                "source_node": sup_id,
                                "target_node": node_id,
                                "parameters": {
                                    "quantity": min(sup_inventory * 0.3, 50),
                                    "urgency": "high",
                                    "reason": f"Emergency replenishment due to {alert_type}",
                                },
                                "benefit": f"Reduces stockout risk at node {node_id}",
                                "trade_off": f"Reduces supplier {sup_id} buffer by ~30%",
                            })
            except Exception:
                pass

    elif alert_type == "supply_disruption":
        # Strategy: Find alternative suppliers or redistribute inventory
        for node_id in affected:
            try:
                # Check if downstream nodes have excess
                downstream = get_downstream_customers.invoke({"node_id": node_id})
                if isinstance(downstream, dict) and downstream.get("success"):
                    customers = downstream.get("downstream_customers", [])
                    for customer in customers:
                        proposals.append({
                            "type": "inventory_sharing",
                            "source_node": node_id,
                            "target_node": customer.get("node_id"),
                            "parameters": {
                                "action": "reduce_orders_temporarily",
                                "duration": 5,
                                "reason": "Upstream supply disruption",
                            },
                            "benefit": "Prevents cascading stockouts downstream",
                            "trade_off": "Temporary reduction in downstream fulfillment",
                        })
            except Exception:
                pass

    elif alert_type == "bullwhip_detected":
        # Strategy: Order smoothing across tiers
        ratio = details.get("ratio", 2.0)
        proposals.append({
            "type": "policy_change",
            "source_node": None,
            "target_node": None,
            "parameters": {
                "action": "order_smoothing",
                "smoothing_factor": 0.3,
                "apply_to": "all_nodes",
                "reason": f"Reduce bullwhip effect (ratio: {ratio:.2f})",
            },
            "benefit": f"Expected {min(ratio * 10, 30):.0f}% reduction in order variance",
            "trade_off": "Slightly slower response to genuine demand changes",
        })

    elif alert_type == "lead_time_change":
        for node_id in affected:
            proposals.append({
                "type": "policy_change",
                "source_node": node_id,
                "target_node": node_id,
                "parameters": {
                    "adjust_safety_stock": True,
                    "increase_pct": 20,
                    "reason": "Compensate for increased lead time",
                },
                "benefit": "Maintains service level despite longer lead times",
                "trade_off": "Higher holding costs from increased safety stock",
            })

    # Build coordination plan
    if proposals:
        plan = (
            f"Coordinating {len(proposals)} actions across "
            f"{len(affected)} affected nodes to address {alert_type}. "
            f"All proposals require human approval before execution."
        )
    else:
        plan = f"No coordination actions identified for {alert_type}"

    # Estimate impact
    impact = {
        "cost_change_pct": -2.0 * len(proposals),
        "service_level_change_pct": 5.0 * len(proposals),
        "bullwhip_reduction_pct": 10.0 if alert_type == "bullwhip_detected" else 0,
    }

    return {
        "proposals": proposals,
        "coordination_plan": plan,
        "expected_impact": impact,
    }
