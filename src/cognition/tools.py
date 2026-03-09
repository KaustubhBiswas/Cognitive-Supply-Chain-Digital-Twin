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
    from .rag import SupplyChainRetriever

logger = logging.getLogger(__name__)


# =============================================================================
# Global References (initialized at runtime)
# =============================================================================

_simulation: Optional["SupplyChainModel"] = None
_forecaster: Optional["SupplyChainPredictor"] = None
_node_features: Optional[np.ndarray] = None  # For GNN context
_rag_retriever: Optional["SupplyChainRetriever"] = None  # For RAG knowledge retrieval


def initialize_tools(
    simulation: "SupplyChainModel",
    forecaster: Optional["SupplyChainPredictor"] = None,
    node_features: Optional[np.ndarray] = None,
    rag_retriever: Optional["SupplyChainRetriever"] = None,
) -> None:
    """
    Initialize tool references to simulation, forecaster, and RAG retriever.
    
    Must be called before using any tools.
    
    Args:
        simulation: The active SupplyChainModel instance
        forecaster: Optional GNN predictor for demand forecasting
        node_features: Optional node feature array for context
        rag_retriever: Optional RAG retriever for knowledge search
    """
    global _simulation, _forecaster, _node_features, _rag_retriever
    _simulation = simulation
    _forecaster = forecaster
    _node_features = node_features
    _rag_retriever = rag_retriever
    
    components = ["simulation"]
    if forecaster:
        components.append("forecaster")
    if rag_retriever:
        components.append("RAG retriever")
    logger.info(f"Cognition tools initialized with: {', '.join(components)}")


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
    
    grid = _get_grid()
    if grid is None:
        return {"error": "Simulation has no grid", "success": False}
    
    upstream = grid.get_upstream_neighbors(node_id)
    
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
    
    grid = _get_grid()
    if grid is None:
        return {"error": "Simulation has no grid", "success": False}
    
    downstream = grid.get_downstream_neighbors(node_id)
    
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
# JIT Disruption Propagation Tools
# =============================================================================

@tool
def analyze_disruption_propagation(
    node_id: int,
    disruption_type: str = "stockout",
    disruption_severity: float = 1.0,
) -> Dict[str, Any]:
    """
    Analyze how a disruption at one node will propagate through the supply chain.
    
    This is a Just-in-Time analysis tool that traces upstream and downstream
    effects, calculating time-to-impact for each affected node based on lead
    times and network topology.
    
    Args:
        node_id: The node where disruption originates
        disruption_type: Type of disruption (stockout, demand_spike, capacity_loss, delay)
        disruption_severity: Severity multiplier (0.0-1.0, where 1.0 is complete failure)
        
    Returns:
        Dictionary with propagation analysis:
        - source_node: The originating disruption node
        - affected_downstream: Nodes that will be impacted (customers)
        - affected_upstream: Nodes that may need to respond (suppliers)
        - impact_timeline: Dict of node_id -> estimated time steps until impact
        - severity_decay: Dict of node_id -> expected severity at that node
        - critical_path: The path with highest cumulative impact
        - total_nodes_affected: Count of all affected nodes
    """
    global _simulation
    
    if _simulation is None:
        return {"error": "Simulation not initialized", "success": False}
    
    grid = _get_grid()
    if grid is None:
        return {"error": "Simulation has no grid", "success": False}
    
    # Check source node exists
    source_agent = _get_agent(node_id)
    if source_agent is None:
        return {"error": f"Node {node_id} not found", "success": False}
    
    # Track affected nodes with their impact details
    downstream_impacts = {}
    upstream_impacts = {}
    impact_timeline = {}
    severity_decay = {}
    
    # === DOWNSTREAM PROPAGATION (toward customers) ===
    # Disruptions flow downstream: stockout at manufacturer affects distributors/retailers
    visited = set()
    queue = [(node_id, 0, disruption_severity)]  # (node, cumulative_time, severity)
    
    while queue:
        current_node, cumulative_time, current_severity = queue.pop(0)
        
        if current_node in visited:
            continue
        visited.add(current_node)
        
        # Get downstream neighbors (customers)
        downstream = grid.get_downstream_neighbors(current_node)
        
        for down_id in downstream:
            down_agent = _get_agent(down_id)
            if down_agent is None:
                continue
            
            # Time to impact = cumulative lead time to reach this node
            lead_time = getattr(down_agent, 'lead_time', 2)
            time_to_impact = cumulative_time + lead_time
            
            # Severity decays as it propagates (each hop reduces severity by 20%)
            decayed_severity = current_severity * 0.8
            
            # Check inventory buffer - nodes with high inventory absorb more impact
            inventory = down_agent.inventory
            reorder_point = getattr(down_agent, 'reorder_point', 20.0)
            buffer_ratio = min(inventory / max(reorder_point, 1), 2.0)  # Cap at 2x
            
            # Higher inventory = lower effective severity
            effective_severity = decayed_severity / max(buffer_ratio, 0.5)
            effective_severity = min(effective_severity, 1.0)  # Cap at 1.0
            
            downstream_impacts[down_id] = {
                "node_type": getattr(down_agent, 'node_type', 'unknown'),
                "time_to_impact": time_to_impact,
                "severity": round(effective_severity, 3),
                "inventory_buffer": round(buffer_ratio, 2),
                "echelon": grid.get_node_echelon(down_id),
            }
            
            impact_timeline[down_id] = time_to_impact
            severity_decay[down_id] = round(effective_severity, 3)
            
            # Continue propagation if severity is still significant
            if effective_severity > 0.1:
                queue.append((down_id, time_to_impact, effective_severity))
    
    # === UPSTREAM PROPAGATION (toward suppliers) ===
    # Information/demand changes flow upstream: demand spike at retailer affects manufacturer
    visited = {node_id}
    queue = [(node_id, 0, disruption_severity)]
    
    while queue:
        current_node, cumulative_time, current_severity = queue.pop(0)
        
        # Get upstream neighbors (suppliers)
        upstream = grid.get_upstream_neighbors(current_node)
        
        for up_id in upstream:
            if up_id in visited:
                continue
            visited.add(up_id)
            
            up_agent = _get_agent(up_id)
            if up_agent is None:
                continue
            
            # Upstream response time (reaction delay)
            lead_time = getattr(up_agent, 'lead_time', 2)
            response_time = cumulative_time + 1  # Info travels faster upstream
            
            # Severity for upstream = how much they need to adjust
            # Demand spikes amplify upstream (bullwhip), stockouts reduce demand
            if disruption_type in ["demand_spike", "bullwhip_detected"]:
                upstream_severity = current_severity * 1.2  # Amplification
            else:
                upstream_severity = current_severity * 0.7  # Dampening
            
            upstream_severity = min(upstream_severity, 1.0)
            
            upstream_impacts[up_id] = {
                "node_type": getattr(up_agent, 'node_type', 'unknown'),
                "response_needed_by": response_time,
                "adjustment_severity": round(upstream_severity, 3),
                "capacity": float(getattr(up_agent, 'capacity', 0)),
                "echelon": grid.get_node_echelon(up_id),
            }
            
            if up_id not in impact_timeline:
                impact_timeline[up_id] = response_time
            if up_id not in severity_decay:
                severity_decay[up_id] = round(upstream_severity, 3)
            
            # Continue upstream propagation
            if upstream_severity > 0.1:
                queue.append((up_id, response_time, upstream_severity))
    
    # === FIND CRITICAL PATH ===
    # Path with highest cumulative severity impact
    critical_path = _find_critical_path(grid, node_id, severity_decay)
    
    # === SUMMARY ===
    total_affected = len(downstream_impacts) + len(upstream_impacts)
    
    # Identify most at-risk nodes (highest severity, lowest buffer)
    at_risk_nodes = sorted(
        [(nid, info["severity"]) for nid, info in downstream_impacts.items()],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    # Convert all values to native Python types for JSON serialization
    return _to_python_type({
        "source_node": node_id,
        "disruption_type": disruption_type,
        "disruption_severity": disruption_severity,
        "affected_downstream": downstream_impacts,
        "affected_upstream": upstream_impacts,
        "impact_timeline": impact_timeline,
        "severity_decay": severity_decay,
        "critical_path": critical_path,
        "total_nodes_affected": total_affected,
        "most_at_risk": [{"node_id": nid, "severity": sev} for nid, sev in at_risk_nodes],
        "success": True,
    })


def _find_critical_path(grid, source_node: int, severity_decay: Dict[int, float]) -> List[Dict]:
    """Find the path with highest cumulative severity from source."""
    # Simple DFS to find path to most severely affected end node
    if not severity_decay:
        return [{"node_id": source_node, "severity": 1.0}]
    
    # Find the most severely affected node
    most_affected = max(severity_decay.items(), key=lambda x: x[1])
    target_node = most_affected[0]
    
    # Get path from source to target
    try:
        import networkx as nx
        path = nx.shortest_path(grid.graph, source_node, target_node)
        return [
            {"node_id": n, "severity": severity_decay.get(n, 1.0 if n == source_node else 0.0)}
            for n in path
        ]
    except:
        return [
            {"node_id": source_node, "severity": 1.0},
            {"node_id": target_node, "severity": most_affected[1]},
        ]


@tool
def estimate_time_to_impact(
    source_node_id: int,
    target_node_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Calculate precise time-to-impact for disruptions propagating to specific nodes.
    
    This is a Just-in-Time calculator that computes detailed propagation timing
    based on lead times, processing delays, and network topology. More detailed
    than the basic propagation analysis.
    
    Args:
        source_node_id: The node where disruption originates
        target_node_ids: Specific nodes to calculate impact time for (None = all reachable)
        
    Returns:
        Dictionary with detailed timing analysis:
        - source_node: The originating disruption node
        - timelines: Dict mapping node_id to detailed timing breakdown
        - earliest_impact: Node that will be affected first
        - latest_impact: Node that will be affected last
        - average_propagation_time: Average time across all affected nodes
    """
    global _simulation
    
    if _simulation is None:
        return {"error": "Simulation not initialized", "success": False}
    
    grid = _get_grid()
    if grid is None:
        return {"error": "Simulation has no grid", "success": False}
    
    source_agent = _get_agent(source_node_id)
    if source_agent is None:
        return {"error": f"Node {source_node_id} not found", "success": False}
    
    timelines = {}
    visited = set()
    queue = [(source_node_id, 0, [])]  # (node, cumulative_time, path)
    
    while queue:
        current_node, cumulative_time, path = queue.pop(0)
        
        if current_node in visited:
            continue
        visited.add(current_node)
        
        # Get downstream neighbors
        downstream = grid.get_downstream_neighbors(current_node)
        
        for down_id in downstream:
            down_agent = _get_agent(down_id)
            if down_agent is None:
                continue
            
            # Calculate detailed timing components
            lead_time = getattr(down_agent, 'lead_time', 2)
            order_processing_time = 1  # Time to process incoming order
            transit_time = max(1, lead_time - order_processing_time)
            
            time_to_impact = cumulative_time + lead_time
            new_path = path + [{"node": current_node, "time": cumulative_time}]
            
            # Only add if target_node_ids is None or this node is in target list
            if target_node_ids is None or down_id in target_node_ids:
                timelines[down_id] = {
                    "total_time_to_impact": time_to_impact,
                    "breakdown": {
                        "upstream_delay": cumulative_time,
                        "processing_time": order_processing_time,
                        "transit_time": transit_time,
                        "node_lead_time": lead_time,
                    },
                    "propagation_path": new_path + [{"node": down_id, "time": time_to_impact}],
                    "echelon": grid.get_node_echelon(down_id),
                    "node_type": getattr(down_agent, 'node_type', 'unknown'),
                }
            
            # Continue propagation
            queue.append((down_id, time_to_impact, new_path))
    
    # Summary statistics
    if timelines:
        times = [t["total_time_to_impact"] for t in timelines.values()]
        earliest_id = min(timelines.keys(), key=lambda nid: timelines[nid]["total_time_to_impact"])
        latest_id = max(timelines.keys(), key=lambda nid: timelines[nid]["total_time_to_impact"])
        avg_time = sum(times) / len(times)
    else:
        earliest_id = None
        latest_id = None
        avg_time = 0
    
    return _to_python_type({
        "source_node": source_node_id,
        "timelines": timelines,
        "earliest_impact": {
            "node_id": earliest_id,
            "time": timelines.get(earliest_id, {}).get("total_time_to_impact", 0) if earliest_id else 0,
        },
        "latest_impact": {
            "node_id": latest_id,
            "time": timelines.get(latest_id, {}).get("total_time_to_impact", 0) if latest_id else 0,
        },
        "average_propagation_time": round(avg_time, 2),
        "total_nodes_affected": len(timelines),
        "success": True,
    })


@tool
def generate_cross_node_recommendations(
    disrupted_nodes: List[int],
    optimization_goal: str = "minimize_impact",
) -> Dict[str, Any]:
    """
    Generate coordinated recommendations across multiple supply chain nodes.
    
    This tool optimizes recommendations network-wide to prevent issues like
    bullwhip effect and ensure coordinated responses to disruptions.
    
    Args:
        disrupted_nodes: List of node IDs experiencing disruptions
        optimization_goal: Goal for optimization (minimize_impact, balance_inventory, 
                          prevent_bullwhip, expedite_recovery)
        
    Returns:
        Dictionary with coordinated recommendations:
        - node_specific_actions: Actions for each affected node
        - coordination_groups: Groups of nodes that should coordinate
        - sequence: Recommended order of implementing actions
        - network_impact_score: Expected reduction in network impact (0-1)
    """
    global _simulation
    
    if _simulation is None:
        return {"error": "Simulation not initialized", "success": False}
    
    grid = _get_grid()
    if grid is None:
        return {"error": "Simulation has no grid", "success": False}
    
    # Analyze propagation for all disrupted nodes
    all_affected = {}
    upstream_responses = {}
    
    for node_id in disrupted_nodes:
        result = analyze_disruption_propagation.invoke({
            "node_id": node_id,
            "disruption_type": "stockout",
            "disruption_severity": 1.0,
        })
        
        if result.get("success"):
            for nid, impact in result.get("affected_downstream", {}).items():
                if nid not in all_affected or impact["severity"] > all_affected[nid].get("severity", 0):
                    all_affected[nid] = impact
                    all_affected[nid]["disruption_source"] = node_id
            
            for nid, impact in result.get("affected_upstream", {}).items():
                if nid not in upstream_responses:
                    upstream_responses[nid] = impact
    
    # Generate node-specific coordinated actions
    node_specific_actions = {}
    coordination_groups = []
    
    # Group by echelon for coordinated response
    echelon_groups = {}
    for nid, info in all_affected.items():
        echelon = info.get("echelon", 0)
        if echelon not in echelon_groups:
            echelon_groups[echelon] = []
        echelon_groups[echelon].append(nid)
    
    # Generate actions based on optimization goal
    for nid, info in all_affected.items():
        severity = info.get("severity", 0)
        echelon = info.get("echelon", 0)
        
        actions = []
        
        if optimization_goal in ["minimize_impact", "expedite_recovery"]:
            if severity > 0.7:
                actions.append({
                    "type": "emergency_order",
                    "priority": "critical",
                    "quantity_multiplier": 1.5,
                })
                actions.append({
                    "type": "activate_backup_supplier",
                    "priority": "high",
                })
            elif severity > 0.4:
                actions.append({
                    "type": "increase_safety_stock",
                    "priority": "high",
                    "increase_pct": 25,
                })
        
        if optimization_goal in ["balance_inventory", "prevent_bullwhip"]:
            # Coordinate order quantities across echelon
            actions.append({
                "type": "synchronized_ordering",
                "priority": "medium",
                "coordinate_with": echelon_groups.get(echelon, []),
            })
            actions.append({
                "type": "dampen_order_variance",
                "priority": "high",
                "smoothing_factor": 0.3,
            })
        
        if actions:
            node_specific_actions[nid] = {
                "actions": actions,
                "echelon": echelon,
                "severity": severity,
            }
    
    # Upstream response coordination
    for nid, info in upstream_responses.items():
        if nid not in node_specific_actions:
            node_specific_actions[nid] = {"actions": [], "echelon": info.get("echelon", 0)}
        
        node_specific_actions[nid]["actions"].append({
            "type": "increase_production_capacity",
            "priority": "medium",
            "response_deadline": info.get("response_needed_by", 5),
        })
    
    # Create coordination groups
    for echelon, nodes in echelon_groups.items():
        if len(nodes) > 1:
            coordination_groups.append({
                "echelon": echelon,
                "nodes": nodes,
                "coordination_type": "synchronized_ordering",
                "reason": f"Echelon {echelon} nodes should coordinate to prevent bullwhip",
            })
    
    # Recommended sequence (upstream first, then by severity)
    sequence = []
    
    # 1. Upstream capacity increases
    for nid in upstream_responses.keys():
        sequence.append({"node_id": nid, "phase": 1, "action_type": "increase_capacity"})
    
    # 2. Critical downstream nodes (severity > 0.7)
    critical_nodes = [(nid, info["severity"]) for nid, info in all_affected.items() if info.get("severity", 0) > 0.7]
    critical_nodes.sort(key=lambda x: x[1], reverse=True)
    for nid, sev in critical_nodes:
        sequence.append({"node_id": nid, "phase": 2, "action_type": "emergency_buffer"})
    
    # 3. Medium severity nodes
    medium_nodes = [(nid, info["severity"]) for nid, info in all_affected.items() if 0.4 < info.get("severity", 0) <= 0.7]
    for nid, sev in medium_nodes:
        sequence.append({"node_id": nid, "phase": 3, "action_type": "increase_safety_stock"})
    
    # Calculate network impact score
    total_severity = sum(info.get("severity", 0) for info in all_affected.values())
    mitigated_severity = sum(
        0.5 * info.get("severity", 0)  # Assume 50% mitigation with coordinated response
        for nid, info in all_affected.items() 
        if nid in node_specific_actions
    )
    network_impact_score = round(mitigated_severity / max(total_severity, 1), 2)
    
    return _to_python_type({
        "disrupted_nodes": disrupted_nodes,
        "optimization_goal": optimization_goal,
        "node_specific_actions": node_specific_actions,
        "coordination_groups": coordination_groups,
        "sequence": sequence,
        "network_impact_score": network_impact_score,
        "total_nodes_coordinated": len(node_specific_actions),
        "success": True,
    })


@tool
def generate_proactive_alerts(
    current_alerts: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Generate proactive/predictive alerts for nodes that will be affected by current disruptions.
    
    Analyzes current network state and active alerts to predict which nodes
    will experience issues in the future, generating early warning alerts.
    
    Args:
        current_alerts: List of current alert dictionaries (optional, will scan network if not provided)
        
    Returns:
        Dictionary with proactive alerts:
        - proactive_alerts: List of predicted future alerts
        - risk_nodes: Nodes at risk even without current alerts
        - early_warning_timeline: When to expect issues at each node
    """
    global _simulation
    
    if _simulation is None:
        return {"error": "Simulation not initialized", "success": False}
    
    grid = _get_grid()
    if grid is None:
        return {"error": "Simulation has no grid", "success": False}
    
    proactive_alerts = []
    risk_nodes = []
    early_warning_timeline = {}
    
    # Get current network state
    current_disruptions = []
    
    # Scan all agents for current issues
    agents = []
    if hasattr(_simulation, '_agents_by_id'):
        agents = list(_simulation._agents_by_id.values())
    elif hasattr(_simulation, 'agents'):
        agents = list(_simulation.agents)
    
    for agent in agents:
        node_id = agent.unique_id
        inventory = getattr(agent, 'inventory', 0)
        reorder_point = getattr(agent, 'reorder_point', 20)
        safety_stock = getattr(agent, 'safety_stock', 10)
        
        # Detect current disruptions
        if inventory <= 0:
            current_disruptions.append({
                "node_id": node_id,
                "type": "stockout",
                "severity": 1.0,
            })
        elif inventory < safety_stock:
            current_disruptions.append({
                "node_id": node_id,
                "type": "low_stock",
                "severity": 0.7,
            })
    
    # Also use current_alerts if provided
    if current_alerts:
        for alert in current_alerts:
            node_id = alert.get("node_id") or alert.get("agent_id")
            if node_id and not any(d["node_id"] == node_id for d in current_disruptions):
                current_disruptions.append({
                    "node_id": node_id,
                    "type": alert.get("alert_type", "unknown"),
                    "severity": 0.8,
                })
    
    # Analyze propagation from each current disruption
    for disruption in current_disruptions:
        result = analyze_disruption_propagation.invoke({
            "node_id": disruption["node_id"],
            "disruption_type": disruption["type"],
            "disruption_severity": disruption["severity"],
        })
        
        if not result.get("success"):
            continue
        
        # Generate proactive alerts for downstream nodes
        for nid, impact in result.get("affected_downstream", {}).items():
            time_to_impact = impact.get("time_to_impact", 5)
            severity = impact.get("severity", 0)
            
            if severity > 0.3:  # Only alert for significant impacts
                proactive_alerts.append({
                    "node_id": nid,
                    "alert_type": "proactive_warning",
                    "predicted_issue": f"Potential {disruption['type']} propagation from node {disruption['node_id']}",
                    "estimated_time_to_impact": time_to_impact,
                    "predicted_severity": round(severity, 2),
                    "source_disruption": disruption["node_id"],
                    "recommended_action": _get_proactive_recommendation(severity, time_to_impact),
                })
                
                early_warning_timeline[nid] = {
                    "issue_expected_in": time_to_impact,
                    "severity": round(severity, 2),
                }
    
    # Identify risk nodes (low inventory buffer even without current alerts)
    for agent in agents:
        node_id = agent.unique_id
        inventory = getattr(agent, 'inventory', 0)
        reorder_point = getattr(agent, 'reorder_point', 20)
        
        # Check if node is vulnerable
        if inventory < reorder_point * 1.5 and inventory > 0:
            # Not in stockout, but low buffer
            demand_rate = getattr(agent, 'demand_mean', 10)
            days_of_stock = inventory / max(demand_rate, 1)
            
            if days_of_stock < 5:
                risk_nodes.append({
                    "node_id": node_id,
                    "risk_type": "low_buffer",
                    "days_of_stock": round(days_of_stock, 1),
                    "recommendation": "Increase safety stock" if days_of_stock < 3 else "Monitor closely",
                })
    
    return _to_python_type({
        "current_disruptions_detected": len(current_disruptions),
        "proactive_alerts": proactive_alerts,
        "risk_nodes": risk_nodes,
        "early_warning_timeline": early_warning_timeline,
        "total_nodes_at_risk": len(set([a["node_id"] for a in proactive_alerts] + [r["node_id"] for r in risk_nodes])),
        "success": True,
    })


def _get_proactive_recommendation(severity: float, time_to_impact: int) -> str:
    """Get appropriate recommendation based on severity and time available."""
    if severity > 0.7 and time_to_impact <= 2:
        return "URGENT: Activate emergency stock or expedite orders immediately"
    elif severity > 0.7:
        return "Place expedited order with backup supplier"
    elif severity > 0.4 and time_to_impact <= 3:
        return "Increase order quantity by 50% on next order"
    elif severity > 0.4:
        return "Review and increase safety stock levels"
    else:
        return "Monitor situation, consider minor safety stock adjustment"


@tool
def simulate_disruption_ripple(
    scenario: Dict[str, Any],
    simulation_steps: int = 10,
) -> Dict[str, Any]:
    """
    Simulate the ripple effect of a hypothetical disruption scenario.
    
    This is a what-if analysis tool that models how a potential disruption
    would propagate through the network over time, without modifying the
    actual simulation state.
    
    Args:
        scenario: Description of disruption scenario:
            - node_id: Node where disruption occurs
            - disruption_type: Type (stockout, demand_spike, capacity_loss, delay)
            - severity: Severity 0.0-1.0
            - duration: How long the disruption lasts (in steps)
        simulation_steps: Number of steps to simulate forward
        
    Returns:
        Dictionary with simulation results:
        - time_series: Step-by-step impact across network
        - peak_impact: When and where maximum impact occurs
        - recovery_trajectory: How network recovers
        - total_cost_estimate: Estimated impact cost
    """
    global _simulation
    
    if _simulation is None:
        return {"error": "Simulation not initialized", "success": False}
    
    grid = _get_grid()
    if grid is None:
        return {"error": "Simulation has no grid", "success": False}
    
    # Extract scenario parameters
    node_id = scenario.get("node_id")
    disruption_type = scenario.get("disruption_type", "stockout")
    severity = scenario.get("severity", 1.0)
    duration = scenario.get("duration", 5)
    
    if node_id is None:
        return {"error": "scenario must include node_id", "success": False}
    
    source_agent = _get_agent(node_id)
    if source_agent is None:
        return {"error": f"Node {node_id} not found", "success": False}
    
    # Get all agents for impact tracking
    agents = []
    if hasattr(_simulation, '_agents_by_id'):
        agents = list(_simulation._agents_by_id.values())
    elif hasattr(_simulation, 'agents'):
        agents = list(_simulation.agents)
    
    # Initialize tracking
    time_series = []
    node_states = {}
    
    # Initialize node states with current inventory levels
    for agent in agents:
        nid = agent.unique_id
        node_states[nid] = {
            "inventory": getattr(agent, 'inventory', 100),
            "affected": False,
            "severity": 0,
            "demand_rate": getattr(agent, 'demand_mean', 10),
        }
    
    # Get propagation impact
    prop_result = analyze_disruption_propagation.invoke({
        "node_id": node_id,
        "disruption_type": disruption_type,
        "disruption_severity": severity,
    })
    
    impact_timeline = prop_result.get("impact_timeline", {}) if prop_result.get("success") else {}
    severity_decay = prop_result.get("severity_decay", {}) if prop_result.get("success") else {}
    
    # Simulate each time step
    peak_impact_step = 0
    peak_impact_value = 0
    
    for step in range(simulation_steps):
        step_state = {"step": step, "nodes_affected": 0, "total_severity": 0, "node_impacts": {}}
        
        # Apply disruption at source during duration
        if step < duration:
            node_states[node_id]["affected"] = True
            node_states[node_id]["severity"] = severity
        else:
            # Recovery begins
            node_states[node_id]["severity"] *= 0.7
        
        # Propagate effects
        for nid, impact_time in impact_timeline.items():
            if step >= impact_time:
                # Impact has reached this node
                base_severity = severity_decay.get(nid, 0)
                
                # Severity increases then decreases based on timing
                time_since_impact = step - impact_time
                if time_since_impact < duration:
                    current_sev = base_severity * min(1.0, (time_since_impact + 1) / 2)
                else:
                    # Recovery phase
                    current_sev = base_severity * max(0, 1 - (time_since_impact - duration) / 5)
                
                node_states[nid]["affected"] = current_sev > 0.1
                node_states[nid]["severity"] = current_sev
        
        # Calculate step metrics
        for nid, state in node_states.items():
            if state["affected"]:
                step_state["nodes_affected"] += 1
                step_state["total_severity"] += state["severity"]
                step_state["node_impacts"][nid] = round(state["severity"], 3)
        
        # Track peak impact
        if step_state["total_severity"] > peak_impact_value:
            peak_impact_value = step_state["total_severity"]
            peak_impact_step = step
        
        time_series.append(step_state)
    
    # Calculate recovery trajectory
    recovery_trajectory = []
    for i, ts in enumerate(time_series):
        if i > 0:
            prev_sev = time_series[i-1]["total_severity"]
            curr_sev = ts["total_severity"]
            if prev_sev > 0:
                recovery_rate = (prev_sev - curr_sev) / prev_sev
            else:
                recovery_rate = 0
            recovery_trajectory.append({
                "step": i,
                "nodes_affected": ts["nodes_affected"],
                "recovery_rate": round(recovery_rate, 3),
            })
    
    # Estimate total cost (simplified model)
    # Each unit of severity per step = 100 units of cost
    total_cost = sum(ts["total_severity"] * 100 for ts in time_series)
    
    # Find recovery step (when < 10% of peak)
    recovery_step = simulation_steps
    for ts in time_series[peak_impact_step:]:
        if ts["total_severity"] < peak_impact_value * 0.1:
            recovery_step = ts["step"]
            break
    
    return _to_python_type({
        "scenario": scenario,
        "simulation_steps": simulation_steps,
        "time_series": time_series,
        "peak_impact": {
            "step": peak_impact_step,
            "nodes_affected": time_series[peak_impact_step]["nodes_affected"] if time_series else 0,
            "total_severity": round(peak_impact_value, 2),
        },
        "recovery_trajectory": recovery_trajectory,
        "estimated_recovery_step": recovery_step,
        "total_cost_estimate": round(total_cost, 2),
        "success": True,
    })


@tool
def get_jit_recommendations(
    disrupted_nodes: List[int],
    disruption_type: str = "stockout",
) -> Dict[str, Any]:
    """
    Generate Just-in-Time recommendations for multiple disrupted nodes.
    
    Analyzes propagation for all disrupted nodes and generates prioritized
    recommendations to minimize network-wide impact.
    
    Args:
        disrupted_nodes: List of node IDs experiencing disruptions
        disruption_type: Type of disruption affecting these nodes
        
    Returns:
        Dictionary with JIT recommendations:
        - priority_actions: Ranked list of immediate actions needed
        - buffer_adjustments: Nodes that need inventory buffer changes
        - rerouting_options: Alternative supply paths identified
        - estimated_recovery_time: Time steps until network stabilizes
    """
    global _simulation
    
    if _simulation is None:
        return {"error": "Simulation not initialized", "success": False}
    
    grid = _get_grid()
    if grid is None:
        return {"error": "Simulation has no grid", "success": False}
    
    # Analyze propagation for each disrupted node
    all_affected = {}
    all_timelines = {}
    
    for node_id in disrupted_nodes:
        result = analyze_disruption_propagation.invoke({
            "node_id": node_id,
            "disruption_type": disruption_type,
            "disruption_severity": 1.0,
        })
        
        if result.get("success"):
            # Merge downstream impacts
            for nid, impact in result.get("affected_downstream", {}).items():
                if nid not in all_affected or impact["severity"] > all_affected[nid]["severity"]:
                    all_affected[nid] = impact
                    all_affected[nid]["disruption_source"] = node_id
            
            # Merge timelines (take earliest impact)
            for nid, time in result.get("impact_timeline", {}).items():
                if nid not in all_timelines or time < all_timelines[nid]:
                    all_timelines[nid] = time
    
    # Generate priority actions
    priority_actions = []
    
    # 1. Immediate buffer increases for nodes with earliest impact
    urgent_nodes = sorted(
        [(nid, info) for nid, info in all_affected.items() if info.get("time_to_impact", 999) <= 3],
        key=lambda x: x[1].get("time_to_impact", 999)
    )
    
    for nid, info in urgent_nodes[:5]:
        priority_actions.append({
            "action": "emergency_buffer_increase",
            "node_id": nid,
            "urgency": "critical" if info["time_to_impact"] <= 1 else "high",
            "reasoning": f"Impact arrives in {info['time_to_impact']} steps, severity {info['severity']:.2f}",
            "recommended_increase_pct": int(info["severity"] * 50),
        })
    
    # 2. Buffer adjustments for medium-term affected nodes
    buffer_adjustments = []
    medium_term = [(nid, info) for nid, info in all_affected.items() 
                   if 3 < info.get("time_to_impact", 999) <= 7]
    
    for nid, info in medium_term:
        buffer_adjustments.append({
            "node_id": nid,
            "current_buffer": info.get("inventory_buffer", 1.0),
            "recommended_buffer": max(1.5, info.get("inventory_buffer", 1.0) + 0.5),
            "time_available": info.get("time_to_impact", 5),
        })
    
    # 3. Estimate recovery time (max timeline + buffer)
    max_impact_time = max(all_timelines.values()) if all_timelines else 0
    estimated_recovery = max_impact_time + 5  # Add buffer for stabilization
    
    # Convert all values to native Python types for JSON serialization
    return _to_python_type({
        "disrupted_nodes": disrupted_nodes,
        "disruption_type": disruption_type,
        "total_affected_nodes": len(all_affected),
        "priority_actions": priority_actions,
        "buffer_adjustments": buffer_adjustments,
        "estimated_recovery_time": estimated_recovery,
        "impact_summary": {
            "critical": len([n for n, i in all_affected.items() if i.get("severity", 0) > 0.7]),
            "high": len([n for n, i in all_affected.items() if 0.4 < i.get("severity", 0) <= 0.7]),
            "medium": len([n for n, i in all_affected.items() if 0.2 < i.get("severity", 0) <= 0.4]),
            "low": len([n for n, i in all_affected.items() if i.get("severity", 0) <= 0.2]),
        },
        "success": True,
    })


# =============================================================================
# Helper Functions
# =============================================================================

def _get_grid():
    """Get the network grid from simulation, checking multiple possible attribute names."""
    global _simulation
    
    if _simulation is None:
        return None
    
    # Try different attribute names
    if hasattr(_simulation, 'grid'):
        return _simulation.grid
    elif hasattr(_simulation, 'network_grid'):
        return _simulation.network_grid
    
    return None


def _to_python_type(value):
    """Convert numpy types to native Python types for JSON serialization."""
    if hasattr(value, 'item'):  # numpy scalar
        return value.item()
    elif isinstance(value, dict):
        return {_to_python_type(k): _to_python_type(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_to_python_type(v) for v in value]
    return value


def _get_agent(node_id: int):
    """Get agent by ID from simulation."""
    global _simulation
    
    if _simulation is None:
        return None
    
    # Try quick lookup first (Mesa 3.x models often have this)
    if hasattr(_simulation, '_agents_by_id'):
        return _simulation._agents_by_id.get(node_id)
    
    # Try schedule.agents (Mesa 2.x style)
    if hasattr(_simulation, 'schedule') and hasattr(_simulation.schedule, 'agents'):
        for agent in _simulation.schedule.agents:
            if agent.unique_id == node_id:
                return agent
    
    # Try self.agents (Mesa 3.x style)
    if hasattr(_simulation, 'agents'):
        for agent in _simulation.agents:
            if agent.unique_id == node_id:
                return agent
    
    return None


# =============================================================================
# RAG (Retrieval-Augmented Generation) Tools
# =============================================================================

@tool
def search_supply_chain_knowledge(
    query: str,
    n_results: int = 5,
    include_best_practices: bool = True,
    include_case_studies: bool = True,
    include_recent_news: bool = True,
) -> Dict[str, Any]:
    """
    Search the RAG knowledge base for supply chain information.
    
    Use this tool to retrieve relevant context about:
    - Historical disruptions and their resolutions
    - Best practices for handling specific situations
    - Case studies from similar scenarios
    - Recent industry news and trends
    
    The retrieved context can inform analysis and recommendations.
    
    Args:
        query: Natural language query describing what information is needed
        n_results: Maximum number of results to return (default: 5)
        include_best_practices: Include best practice documents
        include_case_studies: Include case study documents
        include_recent_news: Include recent news articles
        
    Returns:
        Dictionary with search results:
        - results: List of relevant documents with content and metadata
        - context: Formatted context string ready for LLM consumption
        - query_analysis: Analysis of query intent and routing
        - success: Whether search was successful
    """
    global _rag_retriever
    
    if _rag_retriever is None:
        return {
            "success": False,
            "error": "RAG retriever not initialized",
            "results": [],
            "context": "",
        }
    
    try:
        # Determine which collections to search based on flags
        from .rag import CollectionType
        
        collections = []
        if include_best_practices:
            collections.append(CollectionType.BEST_PRACTICES)
        if include_case_studies:
            collections.append(CollectionType.CASE_STUDIES)
        if include_recent_news:
            collections.extend([CollectionType.NEWS, CollectionType.DISRUPTIONS])
        
        # Use query routing if no specific collections requested
        if not collections:
            collections = None  # Let router decide
        
        # Perform retrieval
        results = _rag_retriever.retrieve(
            query=query,
            n_results=n_results,
            collections=collections,
        )
        
        # Get formatted context for LLM
        context = _rag_retriever.get_context_for_llm(
            query=query,
            max_tokens=2000,
            n_results=n_results,
        )
        
        # Get query analysis
        query_analysis = _rag_retriever.query_router.analyze_query(query)
        
        # Format results
        formatted_results = []
        for r in results.results:
            formatted_results.append({
                "content": r.content[:500] + "..." if len(r.content) > 500 else r.content,
                "score": round(r.score, 4),
                "doc_id": r.doc_id,
                "collection": r.collection,
                "metadata": {
                    k: v for k, v in r.metadata.items()
                    if k in ["doc_type", "severity", "disruption_type", "region", "timestamp"]
                },
            })
        
        return {
            "success": True,
            "results": formatted_results,
            "total_found": len(results),
            "context": context,
            "query_analysis": {
                "intent": query_analysis.intent.value,
                "confidence": round(query_analysis.confidence, 2),
                "detected_disruption_type": (
                    query_analysis.detected_disruption_type.value
                    if query_analysis.detected_disruption_type else None
                ),
                "detected_region": (
                    query_analysis.detected_region.value
                    if query_analysis.detected_region else None
                ),
                "is_urgent": query_analysis.is_urgent,
            },
            "search_time_ms": round(results.search_time_ms, 2),
        }
        
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "context": "",
        }


@tool
def get_disruption_context(
    disruption_type: str,
    severity: str = "high",
    include_mitigation: bool = True,
) -> Dict[str, Any]:
    """
    Get contextual knowledge about a specific type of supply chain disruption.
    
    Use this when analyzing a disruption to get:
    - Historical information about similar disruptions
    - Best practices for mitigation
    - Case studies with lessons learned
    
    Args:
        disruption_type: Type of disruption (stockout, demand_spike, supplier_failure, 
                        logistics_delay, quality_issue, etc.)
        severity: Severity level (low, medium, high, critical)
        include_mitigation: Whether to include mitigation strategies
        
    Returns:
        Dictionary with contextual information:
        - disruption_info: Historical context about this disruption type
        - mitigation_strategies: Relevant best practices (if requested)
        - similar_cases: Case studies of similar situations
        - success: Whether retrieval was successful
    """
    global _rag_retriever
    
    if _rag_retriever is None:
        return {
            "success": False,
            "error": "RAG retriever not initialized",
            "disruption_info": "",
            "mitigation_strategies": [],
            "similar_cases": [],
        }
    
    try:
        from .rag import DisruptionType

        # Map string to enum
        dtype_map = {
            "stockout": DisruptionType.STOCKOUT,
            "demand_spike": DisruptionType.DEMAND_SPIKE,
            "supplier_failure": DisruptionType.SUPPLIER_FAILURE,
            "logistics_delay": DisruptionType.LOGISTICS_DELAY,
            "quality_issue": DisruptionType.QUALITY_ISSUE,
            "natural_disaster": DisruptionType.NATURAL_DISASTER,
            "geopolitical": DisruptionType.GEOPOLITICAL,
            "cyber_attack": DisruptionType.CYBER_ATTACK,
            "labor_shortage": DisruptionType.LABOR_SHORTAGE,
            "raw_material_shortage": DisruptionType.RAW_MATERIAL_SHORTAGE,
        }
        
        dtype = dtype_map.get(disruption_type.lower(), DisruptionType.GENERAL)
        
        # Search for disruption information
        results = _rag_retriever.retrieve_for_disruption(
            disruption_description=f"{severity} {disruption_type} disruption",
            disruption_type=dtype,
            include_best_practices=include_mitigation,
            include_case_studies=True,
            n_results=8,
        )
        
        # Categorize results
        disruption_info = []
        mitigation_strategies = []
        similar_cases = []
        
        for r in results.results:
            doc_type = r.metadata.get("doc_type", "")
            content_summary = r.content[:300] + "..." if len(r.content) > 300 else r.content
            
            if doc_type == "best_practice":
                mitigation_strategies.append({
                    "strategy": content_summary,
                    "relevance": round(r.score, 3),
                })
            elif doc_type == "case_study":
                similar_cases.append({
                    "case": content_summary,
                    "relevance": round(r.score, 3),
                })
            else:
                disruption_info.append({
                    "info": content_summary,
                    "source": doc_type,
                    "relevance": round(r.score, 3),
                })
        
        return {
            "success": True,
            "disruption_type": disruption_type,
            "severity": severity,
            "disruption_info": disruption_info[:3],
            "mitigation_strategies": mitigation_strategies[:3],
            "similar_cases": similar_cases[:2],
            "total_sources": len(results),
        }
        
    except Exception as e:
        logger.error(f"Disruption context retrieval failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "disruption_info": "",
            "mitigation_strategies": [],
            "similar_cases": [],
        }


@tool
def get_best_practices(
    topic: str,
    industry: Optional[str] = None,
    max_results: int = 3,
) -> Dict[str, Any]:
    """
    Retrieve best practices for a specific supply chain topic.
    
    Use this to get expert recommendations and proven strategies
    for handling various supply chain challenges.
    
    Args:
        topic: Topic to find best practices for (e.g., "inventory management",
               "supplier diversification", "demand forecasting")
        industry: Optional industry filter (e.g., "manufacturing", "retail")
        max_results: Maximum number of best practices to return
        
    Returns:
        Dictionary with best practices:
        - practices: List of relevant best practices
        - topic_analyzed: The topic that was searched
        - success: Whether retrieval was successful
    """
    global _rag_retriever
    
    if _rag_retriever is None:
        return {
            "success": False,
            "error": "RAG retriever not initialized",
            "practices": [],
        }
    
    try:
        results = _rag_retriever.retrieve_best_practices(
            topic=topic,
            industry=industry,
            n_results=max_results,
        )
        
        practices = []
        for r in results.results:
            practices.append({
                "practice": r.content,
                "relevance": round(r.score, 3),
                "industry": r.metadata.get("industry", "general"),
                "source_type": r.metadata.get("doc_type", "best_practice"),
            })
        
        return {
            "success": True,
            "topic_analyzed": topic,
            "industry_filter": industry,
            "practices": practices,
            "total_found": len(results),
        }
        
    except Exception as e:
        logger.error(f"Best practices retrieval failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "practices": [],
        }


def is_rag_available() -> bool:
    """Check if RAG retriever is initialized and available."""
    return _rag_retriever is not None


def get_rag_stats() -> Dict[str, Any]:
    """Get statistics about the RAG knowledge base."""
    global _rag_retriever
    
    if _rag_retriever is None:
        return {"available": False}
    
    try:
        stats = _rag_retriever.vector_store.get_collection_stats()
        total_chunks = sum(s.get("count", 0) for s in stats.values())
        
        return {
            "available": True,
            "total_chunks": total_chunks,
            "collections": {
                k: v.get("count", 0) for k, v in stats.items()
            },
        }
    except Exception as e:
        return {"available": True, "error": str(e)}


def get_all_tools() -> List:
    """
    Get list of all available tools for the cognitive agents.
    
    Returns:
        List of tool functions decorated with @tool
    """
    tools = [
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
        # JIT Tools
        analyze_disruption_propagation,
        get_jit_recommendations,
        estimate_time_to_impact,
        generate_cross_node_recommendations,
        generate_proactive_alerts,
        simulate_disruption_ripple,
    ]
    
    # Add RAG tools if retriever is available
    if is_rag_available():
        tools.extend([
            search_supply_chain_knowledge,
            get_disruption_context,
            get_best_practices,
        ])
    
    return tools


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
