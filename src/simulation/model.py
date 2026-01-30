"""
Supply Chain Simulation Model

Mesa model that simulates a multi-echelon supply chain network.
Designed to naturally exhibit the Bullwhip Effect under standard conditions.

Compatible with Mesa 3.x API.
"""

from mesa import Model, Agent, DataCollector
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass

from .agents import (
    SupplyChainAgent,
    SupplierAgent,
    ManufacturerAgent,
    DistributorAgent,
    RetailerAgent,
)
from .grid import SupplyNetworkGrid


class EventType(Enum):
    """Types of events that can occur in the simulation."""
    DEMAND_SHOCK = "demand_shock"
    SUPPLY_DISRUPTION = "supply_disruption"
    LEAD_TIME_INCREASE = "lead_time_increase"
    FACTORY_ISSUE = "factory_issue"


@dataclass
class SimulationEvent:
    """Represents an event injected into the simulation."""
    event_type: EventType
    affected_nodes: List[int]
    magnitude: float
    duration: int
    start_step: int


class SupplyChainModel(Model):
    """
    Agent-based supply chain simulation model.
    
    Features:
    - Multi-echelon network structure
    - Configurable agent parameters
    - Data collection for Bullwhip Effect analysis
    - Event injection for scenario testing
    - Integration hooks for GNN perception and LLM cognition
    
    Compatible with Mesa 3.x API.
    """
    
    def __init__(
        self,
        graph: nx.DiGraph,
        node_types: Dict[int, str],
        node_features: Optional[Dict[int, Dict[str, float]]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the supply chain model.
        
        Args:
            graph: Directed graph representing supply chain topology
            node_types: Mapping of node IDs to types (supplier, manufacturer, etc.)
            node_features: Optional initial features for each node
            random_seed: Random seed for reproducibility
        """
        super().__init__(seed=random_seed)
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.graph = graph
        self.node_types = node_types
        self.current_step = 0
        
        # Active events
        self.active_events: List[SimulationEvent] = []
        self.event_history: List[SimulationEvent] = []
        
        # Network grid (our custom implementation)
        self.network_grid = SupplyNetworkGrid(graph)
        
        # Store agents by ID for quick lookup
        self._agents_by_id: Dict[int, SupplyChainAgent] = {}
        
        # Create agents
        self._create_agents(node_features or {})
        
        # Set up connections between agents
        self._setup_connections()
        
        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "step": lambda m: m.current_step,
                "total_inventory": lambda m: m._total_inventory(),
                "total_backlog": lambda m: m._total_backlog(),
                "bullwhip_ratio": lambda m: m._compute_bullwhip_ratio(),
                "avg_order_variance": lambda m: m._avg_order_variance(),
                "active_events": lambda m: len(m.active_events),
            },
            agent_reporters={
                "inventory": "inventory",
                "backlog": "backlog",
                "orders_placed": lambda a: a.orders_placed[-1] if a.orders_placed else 0,
                "demand": lambda a: a.demand_history[-1] if a.demand_history else 0,
            }
        )
    
    def _create_agents(self, node_features: Dict[int, Dict[str, float]]):
        """Create agents based on node types."""
        agent_classes = {
            "supplier": SupplierAgent,
            "manufacturer": ManufacturerAgent,
            "distributor": DistributorAgent,
            "retailer": RetailerAgent,
        }
        
        for node_id in self.graph.nodes():
            node_type = self.node_types.get(node_id, "distributor")
            agent_class = agent_classes.get(node_type, SupplyChainAgent)
            
            # Get node-specific features
            features = node_features.get(node_id, {})
            
            # Filter out known kwargs
            known_kwargs = {"initial_inventory", "reorder_point", "order_up_to", "lead_time"}
            extra_kwargs = {k: v for k, v in features.items() if k not in known_kwargs}
            
            # Create agent with features
            agent = agent_class(
                unique_id=node_id,
                model=self,
                initial_inventory=features.get("initial_inventory", 100.0),
                reorder_point=features.get("reorder_point", 20.0),
                order_up_to=features.get("order_up_to", 100.0),
                lead_time=features.get("lead_time", 2),
                **extra_kwargs
            )
            
            self._agents_by_id[node_id] = agent
            self.network_grid.place_agent(agent, node_id)
    
    def _setup_connections(self):
        """Set up supplier-customer connections between agents."""
        for node_id in self.graph.nodes():
            agent = self._agents_by_id.get(node_id)
            if agent is None:
                continue
            
            # Set suppliers (upstream)
            for predecessor in self.graph.predecessors(node_id):
                supplier = self._agents_by_id.get(predecessor)
                if supplier:
                    agent.suppliers.append(supplier)
            
            # Set customers (downstream)
            for successor in self.graph.successors(node_id):
                customer = self._agents_by_id.get(successor)
                if customer:
                    agent.customers.append(customer)
    
    def step(self):
        """Advance the simulation by one step."""
        self.current_step += 1
        
        # Process active events
        self._process_events()
        
        # Execute agent steps (Mesa 3.x: iterate over self.agents)
        for agent in self.agents:
            agent.step()
        
        # Collect data
        self.datacollector.collect(self)
    
    def run(self, steps: int = 100) -> Dict[str, Any]:
        """
        Run the simulation for a specified number of steps.
        
        Args:
            steps: Number of simulation steps
            
        Returns:
            Dictionary containing simulation results
        """
        for _ in range(steps):
            self.step()
        
        return {
            "model_data": self.datacollector.get_model_vars_dataframe(),
            "agent_data": self.datacollector.get_agent_vars_dataframe(),
            "final_step": self.current_step,
            "events": self.event_history,
        }
    
    def inject_event(
        self,
        event_type: EventType,
        affected_nodes: Optional[List[int]] = None,
        magnitude: float = 2.0,
        duration: int = 10,
    ):
        """
        Inject an event into the simulation.
        
        Args:
            event_type: Type of event
            affected_nodes: Nodes affected (None = all of relevant type)
            magnitude: Intensity multiplier
            duration: How many steps the event lasts
        """
        if affected_nodes is None:
            # Default: affect all nodes of relevant type
            if event_type == EventType.DEMAND_SHOCK:
                affected_nodes = [
                    n for n, t in self.node_types.items() if t == "retailer"
                ]
            elif event_type == EventType.SUPPLY_DISRUPTION:
                affected_nodes = [
                    n for n, t in self.node_types.items() if t == "supplier"
                ]
            elif event_type == EventType.FACTORY_ISSUE:
                affected_nodes = [
                    n for n, t in self.node_types.items() if t == "manufacturer"
                ]
            else:
                affected_nodes = list(self.graph.nodes())
        
        event = SimulationEvent(
            event_type=event_type,
            affected_nodes=affected_nodes,
            magnitude=magnitude,
            duration=duration,
            start_step=self.current_step,
        )
        
        self.active_events.append(event)
        self.event_history.append(event)
        
        # Apply immediate effects
        self._apply_event(event)
    
    def _apply_event(self, event: SimulationEvent):
        """Apply the effects of an event."""
        for node_id in event.affected_nodes:
            agent = self._agents_by_id.get(node_id)
            if agent is None:
                continue
            
            if event.event_type == EventType.DEMAND_SHOCK:
                if isinstance(agent, RetailerAgent):
                    agent.base_demand *= event.magnitude
            
            elif event.event_type == EventType.SUPPLY_DISRUPTION:
                if isinstance(agent, SupplierAgent):
                    agent.production_capacity /= event.magnitude
            
            elif event.event_type == EventType.FACTORY_ISSUE:
                if isinstance(agent, ManufacturerAgent):
                    agent.has_factory_issue = True
            
            elif event.event_type == EventType.LEAD_TIME_INCREASE:
                agent.lead_time += int(event.magnitude)
    
    def _reverse_event(self, event: SimulationEvent):
        """Reverse the effects of an expired event."""
        for node_id in event.affected_nodes:
            agent = self._agents_by_id.get(node_id)
            if agent is None:
                continue
            
            if event.event_type == EventType.DEMAND_SHOCK:
                if isinstance(agent, RetailerAgent):
                    agent.base_demand /= event.magnitude
            
            elif event.event_type == EventType.SUPPLY_DISRUPTION:
                if isinstance(agent, SupplierAgent):
                    agent.production_capacity *= event.magnitude
            
            elif event.event_type == EventType.FACTORY_ISSUE:
                if isinstance(agent, ManufacturerAgent):
                    agent.has_factory_issue = False
            
            elif event.event_type == EventType.LEAD_TIME_INCREASE:
                agent.lead_time -= int(event.magnitude)
    
    def _process_events(self):
        """Process and expire active events."""
        still_active = []
        for event in self.active_events:
            if self.current_step >= event.start_step + event.duration:
                # Event expired, reverse effects
                self._reverse_event(event)
            else:
                still_active.append(event)
        self.active_events = still_active
    
    # ==================== Metrics ====================
    
    def _total_inventory(self) -> float:
        """Get total inventory across all agents."""
        return sum(agent.inventory for agent in self.agents)
    
    def _total_backlog(self) -> float:
        """Get total backlog across all agents."""
        return sum(agent.backlog for agent in self.agents)
    
    def _avg_order_variance(self) -> float:
        """Get average order variance across agents."""
        variances = []
        for agent in self.agents:
            if len(agent.orders_placed) > 1:
                variances.append(np.var(agent.orders_placed))
        return np.mean(variances) if variances else 0.0
    
    def _compute_bullwhip_ratio(self) -> float:
        """
        Compute Bullwhip Effect ratio.
        
        Ratio = Variance(Upstream Orders) / Variance(Downstream Demand)
        A ratio > 1 indicates order amplification (Bullwhip Effect).
        """
        num_echelons = self.network_grid.get_num_echelons()
        if num_echelons < 2:
            return 1.0
        
        # Get variance at each echelon
        echelon_variances = []
        for echelon in range(num_echelons):
            agents = self.network_grid.get_agents_at_echelon(echelon)
            orders = []
            for agent in agents:
                orders.extend(agent.orders_placed)
            if orders:
                echelon_variances.append(np.var(orders))
        
        if len(echelon_variances) < 2:
            return 1.0
        
        # Ratio of upstream (first echelon) to downstream (last echelon)
        upstream_var = echelon_variances[0]
        downstream_var = echelon_variances[-1]
        
        if downstream_var == 0:
            return float('inf') if upstream_var > 0 else 1.0
        
        return upstream_var / downstream_var
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of the current simulation state.
        
        Used for integration with cognitive agents.
        """
        return {
            "step": self.current_step,
            "total_inventory": self._total_inventory(),
            "total_backlog": self._total_backlog(),
            "bullwhip_ratio": self._compute_bullwhip_ratio(),
            "active_events": [
                {
                    "type": e.event_type.value,
                    "nodes": e.affected_nodes,
                    "magnitude": e.magnitude,
                    "remaining_steps": e.start_step + e.duration - self.current_step,
                }
                for e in self.active_events
            ],
            "agents": {
                agent.unique_id: agent.get_metrics()
                for agent in self.agents
            },
            "network": self.network_grid.to_dict(),
        }


def create_sample_supply_chain(
    num_suppliers: int = 2,
    num_manufacturers: int = 3,
    num_distributors: int = 4,
    num_retailers: int = 6,
    seed: Optional[int] = None,
) -> SupplyChainModel:
    """
    Create a sample multi-echelon supply chain for testing.
    
    Creates a layered network:
    Suppliers → Manufacturers → Distributors → Retailers
    """
    if seed:
        np.random.seed(seed)
    
    G = nx.DiGraph()
    node_types = {}
    current_id = 0
    
    # Create nodes by echelon
    supplier_ids = list(range(current_id, current_id + num_suppliers))
    current_id += num_suppliers
    
    manufacturer_ids = list(range(current_id, current_id + num_manufacturers))
    current_id += num_manufacturers
    
    distributor_ids = list(range(current_id, current_id + num_distributors))
    current_id += num_distributors
    
    retailer_ids = list(range(current_id, current_id + num_retailers))
    
    # Add nodes
    for nid in supplier_ids:
        G.add_node(nid)
        node_types[nid] = "supplier"
    
    for nid in manufacturer_ids:
        G.add_node(nid)
        node_types[nid] = "manufacturer"
    
    for nid in distributor_ids:
        G.add_node(nid)
        node_types[nid] = "distributor"
    
    for nid in retailer_ids:
        G.add_node(nid)
        node_types[nid] = "retailer"
    
    # Create edges (each node connects to 1-2 nodes in next echelon)
    for sup_id in supplier_ids:
        targets = np.random.choice(
            manufacturer_ids,
            size=min(2, len(manufacturer_ids)),
            replace=False
        )
        for target in targets:
            G.add_edge(sup_id, target)
    
    for man_id in manufacturer_ids:
        targets = np.random.choice(
            distributor_ids,
            size=min(2, len(distributor_ids)),
            replace=False
        )
        for target in targets:
            G.add_edge(man_id, target)
    
    for dist_id in distributor_ids:
        targets = np.random.choice(
            retailer_ids,
            size=min(2, len(retailer_ids)),
            replace=False
        )
        for target in targets:
            G.add_edge(dist_id, target)
    
    return SupplyChainModel(G, node_types, random_seed=seed)
