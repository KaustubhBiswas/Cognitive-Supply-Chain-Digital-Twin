"""
LangGraph Tool Definitions

Callable tools that wrap simulation queries and GNN model inference.
These tools are made available to cognitive agents (Supervisor, Analyst,
Negotiator) to gather information and take actions.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    from langchain_core.tools import tool
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    # Fallback decorator that just returns the function
    def tool(func):
        func.is_tool = True
        return func

import numpy as np

if TYPE_CHECKING:
    from ..perception.predictor import SupplyChainPredictor
    from ..simulation.model import SupplyChainModel

logger = logging.getLogger(__name__)


# =============================================================================
# Global References (initialized at runtime)
# =============================================================================

_simulation: Optional["SupplyChainModel"] = None
_forecaster: Optional["SupplyChainPredictor"] = None
_node_features: Optional[np.ndarray] = None  # For GNN context


def initialize_tools(
    simulation: "SupplyChainModel",
    forecaster: Optional["SupplyChainPredictor"] = None,
    node_features: Optional[np.ndarray] = None,
) -> None:
    """
    Initialize tool references to simulation and forecaster.
    
    Must be called before using any tools.
    
    Args:
        simulation: The active SupplyChainModel instance
        forecaster: Optional GNN predictor for demand forecasting
        node_features: Optional node feature array for context
    """
    global _simulation, _forecaster, _node_features
    _simulation = simulation
    _forecaster = forecaster
    _node_features = node_features
    logger.info("Cognition tools initialized")


def is_initialized() -> bool:
    """Check if tools have been initialized."""
    return _simulation is not None


# =============================================================================
# Forecasting Tools
# =============================================================================

@tool
def forecast_demand(
    node_ids: List[int],
    horizon: int = 7,
) -> Dict[str, Any]:
    """
    Forecast demand for specified supply chain nodes using the GNN model.
    
    Use this tool to get AI-powered demand predictions for planning
    inventory levels and order quantities.
    
    Args:
        node_ids: List of node IDs to forecast demand for
        horizon: Number of future timesteps to predict (default: 7 days)
        
    Returns:
        Dictionary with predictions per node and metadata:
        - predictions: Dict mapping node_id to list of predicted values
        - confidence: Dict mapping node_id to confidence score (0-1)
        - model_type: Type of model used for prediction
    """
    global _forecaster, _simulation
    
    if _forecaster is None:
        # Fallback to simple trend-based forecast
        logger.warning("No forecaster available, using fallback")
        return _fallback_forecast(node_ids, horizon)
    
    try:
        predictions = {}
        confidences = {}
        
        for node_id in node_ids:
            # Get prediction from GNN model
            result = _forecaster.predict_node(node_id, horizon=horizon)
            predictions[node_id] = result.predictions.tolist()
            confidences[node_id] = float(result.confidence) if hasattr(result, 'confidence') else 0.8
        
        return {
            "predictions": predictions,
            "confidence": confidences,
            "model_type": "A3TGCN",
            "horizon": horizon,
            "success": True,
        }
        
    except Exception as e:
        logger.error(f"Forecasting failed: {e}")
        return _fallback_forecast(node_ids, horizon)


def _fallback_forecast(node_ids: List[int], horizon: int) -> Dict[str, Any]:
    """Simple moving average fallback when GNN is unavailable."""
    global _simulation
    
    predictions = {}
    
    for node_id in node_ids:
        agent = _get_agent(node_id)
        if agent and hasattr(agent, 'demand_history') and agent.demand_history:
            # Use last 7 days moving average
            recent = agent.demand_history[-7:]
            avg = sum(recent) / len(recent) if recent else 50.0
            predictions[node_id] = [avg] * horizon
        else:
            predictions[node_id] = [50.0] * horizon  # Default
    
    return {
        "predictions": predictions,
        "confidence": {nid: 0.5 for nid in node_ids},
        "model_type": "moving_average_fallback",
        "horizon": horizon,
        "success": True,
    }


# =============================================================================
# Inventory & State Query Tools
# =============================================================================

@tool
def get_node_inventory(node_id: int) -> Dict[str, Any]:
    """
    Get current inventory status for a supply chain node.
    
    Use this tool to check inventory levels, pending orders, and
    policy parameters for a specific node.
    
    Args:
        node_id: The unique ID of the node to query
        
    Returns:
        Dictionary with inventory information:
        - inventory: Current inventory level
        - reorder_point: Threshold for triggering orders
        - order_up_to: Target inventory level
        - pending_orders: Total quantity in pending orders
        - lead_time: Standard lead time for this node
        - node_type: Type of node (supplier, manufacturer, etc.)
    """
    agent = _get_agent(node_id)
    
    if agent is None:
        return {"error": f"Node {node_id} not found", "success": False}
    
    pending_qty = 0.0
    if hasattr(agent, 'pending_orders'):
        pending_qty = sum(
            o.get("quantity", 0) if isinstance(o, dict) else getattr(o, 'quantity', 0)
            for o in agent.pending_orders
        )
    
    return {
        "node_id": node_id,
        "inventory": float(agent.inventory),
        "reorder_point": float(getattr(agent, 'reorder_point', 20.0)),
        "order_up_to": float(getattr(agent, 'order_up_to', 100.0)),
        "pending_orders": float(pending_qty),
        "lead_time": int(getattr(agent, 'lead_time', 2)),
        "node_type": getattr(agent, 'node_type', 'unknown'),
        "success": True,
    }


@tool
def get_all_inventories() -> Dict[str, Any]:
    """
    Get inventory levels for all nodes in the supply chain.
    
    Use this tool to get a complete picture of inventory across
    the entire supply chain network.
    
    Returns:
        Dictionary with:
        - inventories: Dict mapping node_id to inventory level
        - total_inventory: Sum of all inventory
        - low_inventory_nodes: List of nodes below reorder point
        - excess_inventory_nodes: List of nodes with excess stock
    """
    global _simulation
    
    if _simulation is None:
        return {"error": "Simulation not initialized", "success": False}
    
    inventories = {}
    low_nodes = []
    excess_nodes = []
    
    for agent in _simulation.schedule.agents:
        node_id = agent.unique_id
        inventory = float(agent.inventory)
        inventories[node_id] = inventory
        
        reorder_point = getattr(agent, 'reorder_point', 20.0)
        order_up_to = getattr(agent, 'order_up_to', 100.0)
        
        if inventory < reorder_point * 0.5:
            low_nodes.append(node_id)
        elif inventory > order_up_to * 1.5:
            excess_nodes.append(node_id)
    
    return {
        "inventories": inventories,
        "total_inventory": sum(inventories.values()),
        "num_nodes": len(inventories),
        "low_inventory_nodes": low_nodes,
        "excess_inventory_nodes": excess_nodes,
        "success": True,
    }


@tool
def get_historical_orders(node_id: int, periods: int = 10) -> Dict[str, Any]:
    """
    Get historical order data for a specific node.
    
    Use this tool to analyze past ordering patterns and detect
    trends or anomalies in ordering behavior.
    
    Args:
        node_id: The node ID to get order history for
        periods: Number of past periods to retrieve (default: 10)
        
    Returns:
        Dictionary with:
        - orders_placed: List of order quantities placed
        - demands_received: List of demands received
        - average_order: Average order quantity
        - order_variance: Variance in order quantities
    """
    agent = _get_agent(node_id)
    
    if agent is None:
        return {"error": f"Node {node_id} not found", "success": False}
    
    orders = getattr(agent, 'orders_placed', [])[-periods:]
    demands = getattr(agent, 'demand_history', [])[-periods:]
    
    orders_list = [float(o) for o in orders]
    demands_list = [float(d) for d in demands]
    
    avg_order = np.mean(orders_list) if orders_list else 0.0
    order_var = np.var(orders_list) if orders_list else 0.0
    
    return {
        "node_id": node_id,
        "orders_placed": orders_list,
        "demands_received": demands_list,
        "average_order": float(avg_order),
        "order_variance": float(order_var),
        "periods": periods,
        "success": True,
    }


# =============================================================================
# Supply Chain Metrics Tools
# =============================================================================

@tool
def get_supply_chain_metrics() -> Dict[str, Any]:
    """
    Get overall supply chain health metrics.
    
    Use this tool to assess the current state of the entire supply
    chain and identify potential issues.
    
    Returns:
        Dictionary with:
        - total_inventory: Total inventory across all nodes
        - bullwhip_ratio: Order variance amplification ratio
        - average_fill_rate: Average order fulfillment rate
        - stockout_nodes: Nodes currently experiencing stockouts
        - health_score: Overall health score (0-100)
    """
    global _simulation
    
    if _simulation is None:
        return {"error": "Simulation not initialized", "success": False}
    
    # Calculate metrics
    total_inv = sum(agent.inventory for agent in _simulation.schedule.agents)
    bullwhip = _compute_bullwhip_ratio()
    stockouts = []
    fill_rates = []
    
    for agent in _simulation.schedule.agents:
        if agent.inventory <= 0:
            stockouts.append(agent.unique_id)
        
        # Calculate fill rate if available
        if hasattr(agent, 'fulfilled_demand') and hasattr(agent, 'total_demand'):
            if agent.total_demand > 0:
                fill_rates.append(agent.fulfilled_demand / agent.total_demand)
    
    avg_fill_rate = np.mean(fill_rates) if fill_rates else 1.0
    
    # Calculate health score (simple heuristic)
    health_score = 100.0
    health_score -= len(stockouts) * 10  # -10 per stockout
    health_score -= max(0, (bullwhip - 1.5) * 20)  # Penalty for high bullwhip
    health_score = max(0, min(100, health_score))
    
    return {
        "total_inventory": float(total_inv),
        "bullwhip_ratio": float(bullwhip),
        "average_fill_rate": float(avg_fill_rate),
        "stockout_nodes": stockouts,
        "num_stockouts": len(stockouts),
        "health_score": float(health_score),
        "current_step": _simulation.schedule.time,
        "success": True,
    }


@tool
def compute_bullwhip_ratio() -> Dict[str, Any]:
    """
    Compute the Bullwhip Effect ratio for the supply chain.
    
    The Bullwhip Effect is the amplification of order variance as
    we move upstream in the supply chain. A ratio > 1 indicates
    order amplification.
    
    Returns:
        Dictionary with:
        - overall_ratio: Average bullwhip ratio across tiers
        - tier_ratios: Bullwhip ratio for each tier transition
        - interpretation: Human-readable assessment
    """
    global _simulation
    
    if _simulation is None:
        return {"error": "Simulation not initialized", "success": False}
    
    ratio = _compute_bullwhip_ratio()
    
    # Interpretation
    if ratio < 1.0:
        interpretation = "Order dampening - supply chain is smoothing demand"
    elif ratio < 1.5:
        interpretation = "Mild bullwhip - acceptable order amplification"
    elif ratio < 2.0:
        interpretation = "Moderate bullwhip - consider policy adjustments"
    else:
        interpretation = "Severe bullwhip - urgent attention needed"
    
    return {
        "overall_ratio": float(ratio),
        "interpretation": interpretation,
        "threshold_mild": 1.5,
        "threshold_severe": 2.0,
        "success": True,
    }


def _compute_bullwhip_ratio() -> float:
    """Internal function to compute bullwhip ratio."""
    global _simulation
    
    if _simulation is None:
        return 1.0
    
    # Use simulation's built-in method if available
    if hasattr(_simulation, '_compute_bullwhip_ratio'):
        return _simulation._compute_bullwhip_ratio()
    
    # Fallback calculation
    order_variances = []
    demand_variances = []
    
    for agent in _simulation.schedule.agents:
        orders = getattr(agent, 'orders_placed', [])
        demands = getattr(agent, 'demand_history', [])
        
        if len(orders) > 2:
            order_variances.append(np.var(orders))
        if len(demands) > 2:
            demand_variances.append(np.var(demands))
    
    if not order_variances or not demand_variances:
        return 1.0
    
    avg_order_var = np.mean(order_variances)
    avg_demand_var = np.mean(demand_variances)
    
    if avg_demand_var < 1e-6:
        return 1.0
    
    return avg_order_var / avg_demand_var


# =============================================================================
# Action Tools
# =============================================================================

@tool
def propose_order_adjustment(
    node_id: int,
    new_order_quantity: float,
    reason: str,
) -> Dict[str, Any]:
    """
    Propose an adjustment to a node's order quantity.
    
    This creates a recommendation that may require human approval
    before being applied to the simulation.
    
    Args:
        node_id: The node to adjust orders for
        new_order_quantity: The proposed new order quantity
        reason: Explanation for the adjustment
        
    Returns:
        Dictionary with proposal details and approval status
    """
    agent = _get_agent(node_id)
    
    if agent is None:
        return {"error": f"Node {node_id} not found", "success": False}
    
    current_quantity = getattr(agent, 'order_up_to', 100.0) - agent.inventory
    change_pct = abs(new_order_quantity - current_quantity) / max(current_quantity, 1) * 100
    
    return {
        "node_id": node_id,
        "current_order_quantity": float(current_quantity),
        "proposed_order_quantity": float(new_order_quantity),
        "change_percentage": float(change_pct),
        "reason": reason,
        "requires_approval": change_pct > 20,  # Large changes need approval
        "proposal_id": f"order_adj_{node_id}_{_simulation.schedule.time if _simulation else 0}",
        "success": True,
    }


@tool
def propose_policy_change(
    node_id: int,
    parameter: str,
    new_value: float,
    reason: str,
) -> Dict[str, Any]:
    """
    Propose a change to a node's inventory policy parameters.
    
    Policy parameters include reorder_point, order_up_to, and safety_stock.
    Changes require human approval before being applied.
    
    Args:
        node_id: The node to change policy for
        parameter: The parameter to change (reorder_point, order_up_to, safety_stock)
        new_value: The proposed new value
        reason: Explanation for the change
        
    Returns:
        Dictionary with proposal details
    """
    agent = _get_agent(node_id)
    
    if agent is None:
        return {"error": f"Node {node_id} not found", "success": False}
    
    valid_params = ["reorder_point", "order_up_to", "safety_stock", "lead_time"]
    if parameter not in valid_params:
        return {"error": f"Invalid parameter. Valid: {valid_params}", "success": False}
    
    current_value = getattr(agent, parameter, None)
    if current_value is None:
        return {"error": f"Parameter {parameter} not found on agent", "success": False}
    
    change_pct = abs(new_value - current_value) / max(current_value, 1) * 100
    
    return {
        "node_id": node_id,
        "parameter": parameter,
        "current_value": float(current_value),
        "proposed_value": float(new_value),
        "change_percentage": float(change_pct),
        "reason": reason,
        "requires_approval": True,  # Policy changes always need approval
        "proposal_id": f"policy_{node_id}_{parameter}_{_simulation.schedule.time if _simulation else 0}",
        "success": True,
    }


@tool
def get_upstream_suppliers(node_id: int) -> Dict[str, Any]:
    """
    Get information about upstream suppliers for a node.
    
    Use this to understand the supply network and identify
    alternative suppliers or potential bottlenecks.
    
    Args:
        node_id: The node to get suppliers for
        
    Returns:
        Dictionary with supplier information
    """
    global _simulation
    
    if _simulation is None:
        return {"error": "Simulation not initialized", "success": False}
    
    if not hasattr(_simulation, 'grid'):
        return {"error": "Simulation has no grid", "success": False}
    
    upstream = _simulation.grid.get_upstream_neighbors(node_id)
    
    suppliers = []
    for sup_id in upstream:
        agent = _get_agent(sup_id)
        if agent:
            suppliers.append({
                "node_id": sup_id,
                "node_type": getattr(agent, 'node_type', 'unknown'),
                "inventory": float(agent.inventory),
                "available_capacity": float(getattr(agent, 'capacity', 0) - agent.inventory),
            })
    
    return {
        "node_id": node_id,
        "upstream_suppliers": suppliers,
        "num_suppliers": len(suppliers),
        "success": True,
    }


@tool
def get_downstream_customers(node_id: int) -> Dict[str, Any]:
    """
    Get information about downstream customers for a node.
    
    Use this to understand demand patterns and customer needs.
    
    Args:
        node_id: The node to get customers for
        
    Returns:
        Dictionary with customer information
    """
    global _simulation
    
    if _simulation is None:
        return {"error": "Simulation not initialized", "success": False}
    
    if not hasattr(_simulation, 'grid'):
        return {"error": "Simulation has no grid", "success": False}
    
    downstream = _simulation.grid.get_downstream_neighbors(node_id)
    
    customers = []
    for cust_id in downstream:
        agent = _get_agent(cust_id)
        if agent:
            recent_demand = getattr(agent, 'demand_history', [])[-1] if getattr(agent, 'demand_history', []) else 0
            customers.append({
                "node_id": cust_id,
                "node_type": getattr(agent, 'node_type', 'unknown'),
                "recent_demand": float(recent_demand),
            })
    
    return {
        "node_id": node_id,
        "downstream_customers": customers,
        "num_customers": len(customers),
        "success": True,
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _get_agent(node_id: int):
    """Get agent by ID from simulation."""
    global _simulation
    
    if _simulation is None:
        return None
    
    for agent in _simulation.schedule.agents:
        if agent.unique_id == node_id:
            return agent
    
    return None


def get_all_tools() -> List:
    """
    Get list of all available tools for the cognitive agents.
    
    Returns:
        List of tool functions decorated with @tool
    """
    return [
        forecast_demand,
        get_node_inventory,
        get_all_inventories,
        get_historical_orders,
        get_supply_chain_metrics,
        compute_bullwhip_ratio,
        propose_order_adjustment,
        propose_policy_change,
        get_upstream_suppliers,
        get_downstream_customers,
    ]


def get_tool_descriptions() -> Dict[str, str]:
    """
    Get descriptions of all available tools.
    
    Returns:
        Dictionary mapping tool name to description
    """
    tools = get_all_tools()
    result = {}
    for tool in tools:
        # Handle both plain functions and LangChain StructuredTool objects
        if hasattr(tool, 'name'):
            # LangChain StructuredTool
            name = tool.name
            desc = tool.description if hasattr(tool, 'description') else ""
        else:
            # Plain function
            name = tool.__name__
            desc = tool.__doc__.split("\n")[1].strip() if tool.__doc__ else ""
        result[name] = desc
    return result
