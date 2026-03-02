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
from .tools import (analyze_disruption_propagation, compute_bullwhip_ratio,
                    estimate_time_to_impact, forecast_demand,
                    generate_cross_node_recommendations,
                    generate_proactive_alerts, get_all_inventories,
                    get_historical_orders, get_jit_recommendations,
                    get_node_inventory, get_supply_chain_metrics,
                    simulate_disruption_ripple)

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
        
        # JIT: Analyze disruption propagation for affected nodes
        alert_type = alert.get("alert_type", alert.get("type", "stockout"))
        if isinstance(alert_type, str):
            disruption_type = alert_type
        else:
            disruption_type = getattr(alert_type, "value", "stockout")
        
        try:
            jit_analysis = get_jit_recommendations.invoke({
                "disrupted_nodes": affected[:5],
                "disruption_type": disruption_type,
            })
            if isinstance(jit_analysis, dict) and jit_analysis.get("success"):
                data["jit_propagation"] = {
                    "total_affected_nodes": jit_analysis.get("total_affected_nodes", 0),
                    "priority_actions": jit_analysis.get("priority_actions", [])[:3],
                    "estimated_recovery_time": jit_analysis.get("estimated_recovery_time", 0),
                    "impact_summary": jit_analysis.get("impact_summary", {}),
                }
        except Exception as e:
            logger.debug(f"JIT propagation analysis failed: {e}")
        
        # JIT: Get time-to-impact calculations
        try:
            if affected:
                time_impact = estimate_time_to_impact.invoke({
                    "source_node_id": affected[0],
                })
                if isinstance(time_impact, dict) and time_impact.get("success"):
                    data["time_to_impact"] = {
                        "source_node": time_impact.get("source_node"),
                        "earliest_impact": time_impact.get("earliest_impact"),
                        "average_propagation_time": time_impact.get("average_propagation_time"),
                        "total_nodes_affected": time_impact.get("total_nodes_affected"),
                    }
        except Exception as e:
            logger.debug(f"Time-to-impact analysis failed: {e}")
        
        # JIT: Get cross-node coordinated recommendations
        try:
            cross_node = generate_cross_node_recommendations.invoke({
                "disrupted_nodes": affected[:5],
                "optimization_goal": "minimize_impact",
            })
            if isinstance(cross_node, dict) and cross_node.get("success"):
                data["cross_node_coordination"] = {
                    "coordination_groups": cross_node.get("coordination_groups", [])[:3],
                    "sequence": cross_node.get("sequence", [])[:5],
                    "network_impact_score": cross_node.get("network_impact_score", 0),
                }
        except Exception as e:
            logger.debug(f"Cross-node recommendation analysis failed: {e}")
    
    # JIT: Generate proactive alerts for entire network
    try:
        proactive = generate_proactive_alerts.invoke({})
        if isinstance(proactive, dict) and proactive.get("success"):
            proactive_alerts = proactive.get("proactive_alerts", [])
            risk_nodes = proactive.get("risk_nodes", [])
            if proactive_alerts or risk_nodes:
                data["proactive_alerts"] = {
                    "predictive_alerts": proactive_alerts[:5],
                    "risk_nodes": risk_nodes[:5],
                    "total_nodes_at_risk": proactive.get("total_nodes_at_risk", 0),
                }
    except Exception as e:
        logger.debug(f"Proactive alert generation failed: {e}")

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
        
        # JIT: Add propagation analysis findings
        try:
            propagation = analyze_disruption_propagation.invoke({
                "node_id": affected[0] if affected else 0,
                "disruption_type": str(alert_type) if isinstance(alert_type, str) else getattr(alert_type, "value", "stockout"),
                "disruption_severity": 1.0,
            })
            if isinstance(propagation, dict) and propagation.get("success"):
                total_affected = propagation.get("total_nodes_affected", 0)
                if total_affected > 0:
                    findings.append(f"JIT Analysis: Disruption will propagate to {total_affected} additional nodes")
                    most_at_risk = propagation.get("most_at_risk", [])
                    if most_at_risk:
                        at_risk_ids = [n["node_id"] for n in most_at_risk[:3]]
                        findings.append(f"Most at-risk nodes: {at_risk_ids}")
                    
                    # Add JIT-based recommendations
                    critical_path = propagation.get("critical_path", [])
                    if len(critical_path) > 1:
                        for node_info in critical_path[1:3]:  # First 2 downstream nodes
                            recommendations.append({
                                "type": "jit_buffer_increase",
                                "target_nodes": [node_info["node_id"]],
                                "parameters": {"urgency": "proactive", "severity": node_info.get("severity", 0.5)},
                                "reasoning": f"Proactive buffer increase before disruption arrives (severity: {node_info.get('severity', 0.5):.2f})",
                                "confidence": 0.75,
                            })
                    
                    metrics["jit_propagation"] = {
                        "total_affected": total_affected,
                        "critical_path_length": len(critical_path),
                        "most_at_risk": [n["node_id"] for n in most_at_risk[:3]],
                    }
        except Exception as e:
            logger.debug(f"JIT propagation failed in rule-based analysis: {e}")
        
        # JIT: Time-to-impact calculator for precise timing
        try:
            if affected:
                time_impact = estimate_time_to_impact.invoke({
                    "source_node_id": affected[0],
                })
                if isinstance(time_impact, dict) and time_impact.get("success"):
                    earliest = time_impact.get("earliest_impact", {})
                    if earliest.get("node_id"):
                        findings.append(
                            f"Earliest impact in {earliest.get('time', 0)} steps at node {earliest.get('node_id')}"
                        )
                        # Add urgent recommendation for earliest impact node
                        if earliest.get("time", 999) <= 2:
                            recommendations.append({
                                "type": "emergency_buffer_increase",
                                "target_nodes": [earliest.get("node_id")],
                                "parameters": {"urgency": "critical", "time_available": earliest.get("time")},
                                "reasoning": f"URGENT: Impact arrives in {earliest.get('time')} steps - immediate action required",
                                "confidence": 0.9,
                            })
                    metrics["time_to_impact"] = {
                        "earliest_node": earliest.get("node_id"),
                        "earliest_time": earliest.get("time", 0),
                        "avg_propagation": time_impact.get("average_propagation_time", 0),
                    }
        except Exception as e:
            logger.debug(f"Time-to-impact failed: {e}")
        
        # JIT: Cross-node coordinated recommendations
        try:
            cross_node = generate_cross_node_recommendations.invoke({
                "disrupted_nodes": affected[:5],
                "optimization_goal": "minimize_impact",
            })
            if isinstance(cross_node, dict) and cross_node.get("success"):
                coord_groups = cross_node.get("coordination_groups", [])
                if coord_groups:
                    findings.append(f"Identified {len(coord_groups)} node groups requiring coordinated response")
                    
                    # Add coordinated recommendations
                    for group in coord_groups[:2]:
                        group_nodes = group.get("nodes", [])
                        if group_nodes:
                            recommendations.append({
                                "type": "coordinate_orders",
                                "target_nodes": group_nodes,
                                "parameters": {
                                    "coordination_type": group.get("coordination_type", "synchronized_ordering"),
                                    "echelon": group.get("echelon", 0),
                                },
                                "reasoning": group.get("reason", "Coordinate orders across nodes to prevent bullwhip"),
                                "confidence": 0.7,
                            })
                
                metrics["cross_node"] = {
                    "coordination_groups": len(coord_groups),
                    "network_impact_score": cross_node.get("network_impact_score", 0),
                }
        except Exception as e:
            logger.debug(f"Cross-node recommendations failed: {e}")
        
        # JIT: Proactive alerts for predictive warnings
        try:
            proactive = generate_proactive_alerts.invoke({})
            if isinstance(proactive, dict) and proactive.get("success"):
                proactive_alerts = proactive.get("proactive_alerts", [])
                if proactive_alerts:
                    findings.append(f"Generated {len(proactive_alerts)} proactive alerts for predicted disruptions")
                    
                    # Add proactive recommendations for predicted impacts
                    for pa in proactive_alerts[:3]:
                        if pa.get("predicted_severity", 0) > 0.5:
                            recommendations.append({
                                "type": "proactive_preparation",
                                "target_nodes": [pa.get("node_id")],
                                "parameters": {
                                    "predicted_time": pa.get("estimated_time_to_impact"),
                                    "action": pa.get("recommended_action", "increase_buffer"),
                                },
                                "reasoning": f"Proactive: {pa.get('predicted_issue', 'Future disruption predicted')}",
                                "confidence": 0.7,
                            })
                
                risk_nodes = proactive.get("risk_nodes", [])
                if risk_nodes:
                    findings.append(f"{len(risk_nodes)} nodes identified as vulnerable even without current alerts")
                    
                metrics["proactive"] = {
                    "alerts_generated": len(proactive_alerts),
                    "risk_nodes": len(risk_nodes),
                }
        except Exception as e:
            logger.debug(f"Proactive alerts failed: {e}")
        except Exception as e:
            logger.debug(f"JIT propagation failed in rule-based analysis: {e}")

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
