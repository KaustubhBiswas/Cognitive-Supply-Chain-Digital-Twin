"""
Cognition Module for LangGraph Multi-Agent System

This module provides the cognitive layer for the supply chain digital twin,
implementing a multi-agent system using LangGraph for intelligent decision-making.

Components:
- State: Shared state definition for agent coordination
- Tools: LangChain tools wrapping simulation and forecasting
- Agents: Supervisor, Analyst, and Negotiator agents
- Graph: LangGraph StateGraph workflow
- LLM: Flexible LLM initialization (Groq cloud or Ollama local)

Usage:
    from src.cognition import (
        create_supply_chain_graph, create_initial_state, initialize_tools,
        Alert, AlertType, AlertSeverity, create_groq_llm,
    )
    
    # Initialize LLM (optional - uses rule-based fallback if None)
    llm = create_groq_llm(api_key="gsk_...")  # or create_ollama_llm()
    
    # Create workflow
    graph = create_supply_chain_graph(llm=llm)
    state = create_initial_state(alert=my_alert)
    result = graph.invoke(state, config={"configurable": {"thread_id": "1"}})
"""

from .analyst import create_analyst_agent
from .graph import FallbackGraph, create_supply_chain_graph
from .llm import (DEFAULT_GROQ_MODEL, DEFAULT_OLLAMA_MODEL, GROQ_MODELS,
                  OLLAMA_MODELS, LLMConfig, create_groq_llm, create_llm,
                  create_ollama_llm)
from .negotiator import create_negotiator_agent
from .state import (  # Enums; Dataclasses; State; Helper functions; Constants
    ALERT_THRESHOLDS, DECISION_THRESHOLDS, AgentRoute, Alert, AlertSeverity,
    AlertType, ForecastData, Recommendation, RecommendationType,
    SimulationSnapshot, SupplyChainState, add_forecast, add_recommendation,
    create_initial_state)
from .supervisor import create_supervisor_agent
from .tools import (  # Initialization; Forecasting tools; Inventory tools; Metrics tools; Action tools; Network tools; Utilities
    compute_bullwhip_ratio, forecast_demand, get_all_inventories,
    get_all_tools, get_downstream_customers, get_historical_orders,
    get_node_inventory, get_supply_chain_metrics, get_tool_descriptions,
    get_upstream_suppliers, initialize_tools, is_initialized,
    propose_order_adjustment, propose_policy_change)

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
    # LLM
    "LLMConfig",
    "create_llm",
    "create_groq_llm",
    "create_ollama_llm",
    "GROQ_MODELS",
    "OLLAMA_MODELS",
    "DEFAULT_GROQ_MODEL",
    "DEFAULT_OLLAMA_MODEL",
]
