"""
Cognition Module for LangGraph Multi-Agent System

This module provides the cognitive layer for the supply chain digital twin,
implementing a multi-agent system using LangGraph for intelligent decision-making.

Components:
- State: Shared state definition for agent coordination
- Tools: LangChain tools wrapping simulation and forecasting
- Agents: Supervisor, Analyst, and Negotiator agents
- Graph: LangGraph StateGraph workflow
"""

from .state import (
    # Enums
    AlertSeverity,
    AlertType,
    RecommendationType,
    AgentRoute,
    # Dataclasses
    Alert,
    Recommendation,
    SimulationSnapshot,
    ForecastData,
    # State
    SupplyChainState,
    # Helper functions
    create_initial_state,
    add_recommendation,
    add_forecast,
    # Constants
    ALERT_THRESHOLDS,
    DECISION_THRESHOLDS,
)

from .tools import (
    # Initialization
    initialize_tools,
    is_initialized,
    # Forecasting tools
    forecast_demand,
    # Inventory tools
    get_node_inventory,
    get_all_inventories,
    get_historical_orders,
    # Metrics tools
    get_supply_chain_metrics,
    compute_bullwhip_ratio,
    # Action tools
    propose_order_adjustment,
    propose_policy_change,
    # Network tools
    get_upstream_suppliers,
    get_downstream_customers,
    # Utilities
    get_all_tools,
    get_tool_descriptions,
)

from .supervisor import create_supervisor_agent
from .analyst import create_analyst_agent
from .negotiator import create_negotiator_agent
from .graph import create_supply_chain_graph, FallbackGraph

__all__ = [
    # State - Enums
    "AlertSeverity",
    "AlertType",
    "RecommendationType",
    "AgentRoute",
    # State - Dataclasses
    "Alert",
    "Recommendation",
    "SimulationSnapshot",
    "ForecastData",
    # State - TypedDict
    "SupplyChainState",
    # State - Helpers
    "create_initial_state",
    "add_recommendation",
    "add_forecast",
    # State - Constants
    "ALERT_THRESHOLDS",
    "DECISION_THRESHOLDS",
    # Tools - Init
    "initialize_tools",
    "is_initialized",
    # Tools - Forecasting
    "forecast_demand",
    # Tools - Inventory
    "get_node_inventory",
    "get_all_inventories",
    "get_historical_orders",
    # Tools - Metrics
    "get_supply_chain_metrics",
    "compute_bullwhip_ratio",
    # Tools - Actions
    "propose_order_adjustment",
    "propose_policy_change",
    # Tools - Network
    "get_upstream_suppliers",
    "get_downstream_customers",
    # Tools - Utilities
    "get_all_tools",
    "get_tool_descriptions",
    # Agents
    "create_supervisor_agent",
    "create_analyst_agent",
    "create_negotiator_agent",
    # Graph
    "create_supply_chain_graph",
    "FallbackGraph",
]
