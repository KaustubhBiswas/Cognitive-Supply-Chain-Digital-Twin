"""
Analyst Agent Implementation

Specialized agent for demand analysis, anomaly detection,
Bullwhip Effect assessment, and inventory policy recommendations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .state import RecommendationType, SupplyChainState
from .tools import (compute_bullwhip_ratio, forecast_demand,
                    get_all_inventories, get_historical_orders,
                    get_node_inventory, get_supply_chain_metrics)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "analyst.txt"


def _load_prompt() -> str:
    """Load the analyst system prompt."""
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Analyst prompt file not found, using default")
        return "You are a supply chain analyst. Analyze the data and provide recommendations in JSON format."


def _format_analysis_context(state: SupplyChainState) -> str:
    """Format the analysis context with available data."""
    parts = []

    alert = state.get("current_alert")
    if alert:
        parts.append(
            f"ALERT TO ANALYZE:\n"
            f"  Type: {alert.get('alert_type')}\n"
            f"  Severity: {alert.get('severity')}\n"
            f"  Nodes: {alert.get('affected_nodes')}\n"
            f"  Details: {json.dumps(alert.get('details', {}))}"
        )

    sim_state = state.get("simulation_state", {})
    if sim_state:
        parts.append(
            f"CURRENT STATE:\n"
            f"  Total Inventory: {sim_state.get('total_inventory', 'N/A')}\n"
            f"  Bullwhip Ratio: {sim_state.get('bullwhip_ratio', 'N/A')}"
        )

    forecasts = state.get("forecasts", {})
    if forecasts:
        parts.append(f"AVAILABLE FORECASTS: {len(forecasts)} nodes")

    return "\n\n".join(parts) if parts else "No data available for analysis."


def _parse_response(response_text: str) -> Dict[str, Any]:
    """Parse analyst response JSON."""
    text = response_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from code blocks
    for marker in ["```json", "```"]:
        if marker in text:
            try:
                start = text.index(marker) + len(marker)
                end = text.index("```", start)
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

    # Try to find JSON object
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    return {
        "findings": text,
        "risk_level": "medium",
        "recommendations": [],
        "metrics": {},
    }


def create_analyst_agent(llm=None):
    """
    Create the analyst agent node function.

    Args:
        llm: LangChain LLM instance. If None, uses rule-based analysis.

    Returns:
        Analyst node function for LangGraph
    """
    system_prompt = _load_prompt()

    def analyst_node(state: SupplyChainState) -> Dict[str, Any]:
        """
        Analyst agent node.

        Gathers data, performs analysis, returns findings and recommendations.
        """
        # Use LLM if available
        if llm is not None:
            try:
                analysis = _llm_analysis(llm, system_prompt, state)
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
                analysis = _rule_based_analysis(state)
        else:
            analysis = _rule_based_analysis(state)

        # Ensure assessed_severity is computed (for both LLM and rule-based paths)
        if "assessed_severity" not in analysis:
            alert = state.get("current_alert", {})
            metrics = analysis.get("metrics", {})
            risk_level = analysis.get("risk_level", "low")
            analysis["assessed_severity"] = _compute_severity(risk_level, alert, metrics)

        # Convert recommendations to state format
        recommendations = state.get("recommendations", []).copy()
        for rec in analysis.get("recommendations", []):
            recommendations.append({
                "recommendation_type": rec.get("type", "no_action"),
                "target_nodes": rec.get("target_nodes", []),
                "parameters": rec.get("parameters", {}),
                "reasoning": rec.get("reasoning", ""),
                "confidence": rec.get("confidence", 0.5),
                "source_agent": "analyst",
                "requires_approval": rec.get("confidence", 0) < 0.9,
            })

        # Build message
        from langchain_core.messages import AIMessage
        message = (
            f"[Analyst] Risk: {analysis.get('risk_level', 'unknown')}. "
            f"Findings: {analysis.get('findings', 'N/A')}. "
            f"Recommendations: {len(analysis.get('recommendations', []))}."
        )

        return {
            "messages": [AIMessage(content=message)],
            "analysis_results": analysis,
            "recommendations": recommendations,
            "next_agent": "supervisor",  # Report back to supervisor
        }

    return analyst_node


def _llm_analysis(
    llm, system_prompt: str, state: SupplyChainState
) -> Dict[str, Any]:
    """Perform analysis using the LLM."""
    from langchain_core.messages import HumanMessage, SystemMessage

    context = _format_analysis_context(state)

    # Gather data using tools
    tool_data = _gather_analysis_data(state)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Analyze the following supply chain situation:\n\n"
            f"{context}\n\n"
            f"Additional data:\n{json.dumps(tool_data, indent=2)}\n\n"
            f"Provide your analysis and recommendations."
        ),
    ]

    response = llm.invoke(messages)
    return _parse_response(response.content)


def _gather_analysis_data(state: SupplyChainState) -> Dict[str, Any]:
    """Gather data from tools for analysis."""
    data = {}

    try:
        # Get overall metrics
        metrics = get_supply_chain_metrics.invoke({})
        if isinstance(metrics, dict) and metrics.get("success"):
            data["metrics"] = metrics
    except Exception as e:
        logger.debug(f"Could not gather metrics: {e}")

    try:
        # Get bullwhip ratio
        bullwhip = compute_bullwhip_ratio.invoke({})
        if isinstance(bullwhip, dict) and bullwhip.get("success"):
            data["bullwhip"] = bullwhip
    except Exception as e:
        logger.debug(f"Could not compute bullwhip: {e}")

    # Get data for affected nodes
    alert = state.get("current_alert")
    if alert:
        affected = alert.get("affected_nodes", [])
        node_data = {}
        for node_id in affected[:5]:  # Limit to 5 nodes
            try:
                inv = get_node_inventory.invoke({"node_id": node_id})
                if isinstance(inv, dict) and inv.get("success"):
                    node_data[node_id] = inv
            except Exception:
                pass
        if node_data:
            data["affected_node_details"] = node_data

    return data


def _compute_severity(risk_level: str, alert: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """
    Compute assessed severity dynamically based on risk level, alert context, and metrics.
    
    The cognition brain determines severity by analyzing:
    - Base risk level from rule-based analysis
    - Alert type and specific details
    - Supply chain metrics (bullwhip ratio, stockouts, etc.)
    
    Returns:
        str: Assessed severity - "low", "medium", "high", or "critical"
    """
    if not alert:
        return "low"
    
    alert_type = alert.get("type", "")
    details = alert.get("details", {})
    affected_nodes = alert.get("affected_nodes", [])
    
    # Start with base severity from risk level
    severity_score = {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(risk_level, 1)
    
    # Escalate based on alert type severity weights
    type_weights = {
        "stockout": 1.5,           # Stockouts are critical - direct customer impact
        "demand_spike": 1.2,       # Demand spikes can cascade
        "bullwhip_detected": 1.3,  # Bullwhip amplifies problems
        "inventory_low": 1.1,      # Warning level
        "forecast_deviation": 1.0, # Informational
    }
    severity_score *= type_weights.get(alert_type, 1.0)
    
    # Escalate based on number of affected nodes
    num_affected = len(affected_nodes) if affected_nodes else 0
    if num_affected > 5:
        severity_score *= 1.3
    elif num_affected > 2:
        severity_score *= 1.1
    
    # Escalate based on specific metrics
    bullwhip_ratio = metrics.get("bullwhip_ratio", 1.0)
    if bullwhip_ratio > 3.0:
        severity_score *= 1.2
    
    num_stockouts = metrics.get("num_stockouts", 0)
    if num_stockouts > 3:
        severity_score *= 1.3
    elif num_stockouts > 0:
        severity_score *= 1.1
    
    # Escalate based on alert-specific details
    if alert_type == "demand_spike":
        current = details.get("current", 0)
        previous = details.get("previous", 1)
        if previous > 0 and current / previous > 2.5:
            severity_score *= 1.2
    elif alert_type == "bullwhip_detected":
        ratio = details.get("ratio", 1.0)
        if ratio > 3.0:
            severity_score *= 1.2
    elif alert_type == "forecast_deviation":
        deviation = details.get("deviation_pct", 0)
        if deviation > 40:
            severity_score *= 1.2
    
    # Map score to severity level
    if severity_score >= 4.5:
        return "critical"
    elif severity_score >= 3.0:
        return "high"
    elif severity_score >= 2.0:
        return "medium"
    else:
        return "low"


def _rule_based_analysis(state: SupplyChainState) -> Dict[str, Any]:
    """
    Rule-based analysis when no LLM is available.

    Uses thresholds and heuristics to generate findings and recommendations.
    """
    alert = state.get("current_alert")
    sim_state = state.get("simulation_state", {})

    findings = []
    recommendations = []
    risk_level = "low"
    metrics = {}

    # Get current metrics
    try:
        sc_metrics = get_supply_chain_metrics.invoke({})
        if isinstance(sc_metrics, dict) and sc_metrics.get("success"):
            metrics["health_score"] = sc_metrics.get("health_score", 100)
            metrics["bullwhip_ratio"] = sc_metrics.get("bullwhip_ratio", 1.0)
            metrics["num_stockouts"] = sc_metrics.get("num_stockouts", 0)

            if sc_metrics.get("bullwhip_ratio", 1.0) > 2.0:
                findings.append(
                    f"Severe bullwhip effect detected (ratio: {sc_metrics['bullwhip_ratio']:.2f})"
                )
                risk_level = "high"

            if sc_metrics.get("num_stockouts", 0) > 0:
                findings.append(
                    f"{sc_metrics['num_stockouts']} nodes experiencing stockouts"
                )
                risk_level = max(risk_level, "medium", key=lambda x: ["low", "medium", "high", "critical"].index(x))
    except Exception:
        pass

    # Analyze the specific alert
    if alert:
        alert_type = alert.get("alert_type", "")
        affected = alert.get("affected_nodes", [])
        details = alert.get("details", {})

        if alert_type == "demand_spike":
            increase = details.get("current", 0) / max(details.get("previous", 1), 1)
            findings.append(f"Demand spike of {increase:.1f}x at nodes {affected}")

            for node_id in affected:
                recommendations.append({
                    "type": "adjust_safety_stock",
                    "target_nodes": [node_id],
                    "parameters": {"increase_pct": min(increase * 20, 50)},
                    "reasoning": f"Demand spike of {increase:.1f}x requires safety stock increase",
                    "confidence": 0.75,
                })
            risk_level = "medium" if increase < 2.0 else "high"

        elif alert_type == "inventory_low":
            findings.append(f"Low inventory at nodes {affected}")
            for node_id in affected:
                recommendations.append({
                    "type": "adjust_reorder_point",
                    "target_nodes": [node_id],
                    "parameters": {"increase_pct": 25},
                    "reasoning": "Inventory below critical threshold, raise reorder point",
                    "confidence": 0.8,
                })
            risk_level = "medium"

        elif alert_type == "bullwhip_detected":
            ratio = details.get("ratio", 2.0)
            findings.append(f"Bullwhip ratio at {ratio:.2f}")
            recommendations.append({
                "type": "adjust_order_quantity",
                "target_nodes": affected if affected else [],
                "parameters": {"smoothing_factor": 0.3},
                "reasoning": f"Order smoothing to reduce bullwhip (ratio: {ratio:.2f})",
                "confidence": 0.7,
            })
            risk_level = "high" if ratio > 2.5 else "medium"

        elif alert_type == "stockout":
            findings.append(f"Stockout at nodes {affected}")
            for node_id in affected:
                recommendations.append({
                    "type": "adjust_reorder_point",
                    "target_nodes": [node_id],
                    "parameters": {"increase_pct": 50},
                    "reasoning": "Stockout recovery: significantly raise reorder point",
                    "confidence": 0.85,
                })
            risk_level = "high"

        elif alert_type == "forecast_deviation":
            deviation = details.get("deviation_pct", 0)
            findings.append(f"Forecast deviation of {deviation:.1f}% at nodes {affected}")
            risk_level = "medium" if deviation < 30 else "high"

        else:
            findings.append(f"Alert: {alert_type} at nodes {affected}")

    if not findings:
        findings.append("No significant issues detected")

    metrics["demand_trend"] = "stable"
    metrics["risk_score"] = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}.get(risk_level, 0.5)
    
    # Compute assessed severity based on risk_level and alert context
    assessed_severity = _compute_severity(risk_level, alert, metrics)

    return {
        "findings": ". ".join(findings),
        "risk_level": risk_level,
        "assessed_severity": assessed_severity,
        "recommendations": recommendations,
        "metrics": metrics,
    }
