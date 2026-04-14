"""
Dynamic Tool Policy

Provides a lightweight tool selection policy so agents can choose tools
based on objective context and step ownership instead of fixed call paths.
"""

from typing import Any, Dict, List, Optional

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "get_supply_chain_metrics": {
        "roles": ["planner", "supervisor", "analyst", "negotiator"],
        "tags": ["overview", "kpi", "state"],
        "cost": "low",
    },
    "compute_bullwhip_ratio": {
        "roles": ["analyst", "supervisor"],
        "tags": ["bullwhip", "variance", "risk"],
        "cost": "low",
    },
    "get_node_inventory": {
        "roles": ["analyst", "negotiator", "supervisor"],
        "tags": ["inventory", "node", "stock"],
        "cost": "low",
    },
    "get_all_inventories": {
        "roles": ["analyst", "supervisor"],
        "tags": ["inventory", "global", "state"],
        "cost": "low",
    },
    "get_historical_orders": {
        "roles": ["analyst"],
        "tags": ["orders", "history", "trend"],
        "cost": "low",
    },
    "forecast_demand": {
        "roles": ["analyst", "negotiator"],
        "tags": ["forecast", "demand", "prediction"],
        "cost": "medium",
    },
    "get_upstream_suppliers": {
        "roles": ["negotiator", "analyst"],
        "tags": ["network", "upstream", "suppliers"],
        "cost": "low",
    },
    "get_downstream_customers": {
        "roles": ["negotiator", "analyst"],
        "tags": ["network", "downstream", "customers"],
        "cost": "low",
    },
}


def get_tool_registry() -> Dict[str, Dict[str, Any]]:
    """Return the registered tool metadata used by the policy layer."""
    return TOOL_REGISTRY


def select_tools_for_goal(
    goal: str,
    owner: str,
    available_tools: Optional[List[str]] = None,
    max_tools: int = 5,
) -> List[str]:
    """
    Select tools dynamically from registry metadata + goal keywords.

    This is an intentionally lightweight policy for Sprint 1.
    """
    goal_text = (goal or "").lower()
    max_tools = max(1, int(max_tools))

    if available_tools is None:
        available = list(TOOL_REGISTRY.keys())
    else:
        available = [t for t in available_tools if t in TOOL_REGISTRY]

    scored: List[tuple[float, str]] = []
    for tool_name in available:
        meta = TOOL_REGISTRY[tool_name]
        score = 0.0

        if owner in meta.get("roles", []):
            score += 2.0

        for tag in meta.get("tags", []):
            if tag in goal_text:
                score += 1.0

        if "risk" in goal_text and any(k in tool_name for k in ["bullwhip", "inventory", "metrics"]):
            score += 0.8
        if "coord" in goal_text and any(k in tool_name for k in ["upstream", "downstream", "forecast"]):
            score += 0.8

        if meta.get("cost") == "low":
            score += 0.2

        scored.append((score, tool_name))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [name for score, name in scored if score > 0][:max_tools]

    if not selected:
        fallback = [
            t for t in ["get_supply_chain_metrics", "get_node_inventory", "compute_bullwhip_ratio"]
            if t in available
        ]
        return fallback[:max_tools]

    return selected
