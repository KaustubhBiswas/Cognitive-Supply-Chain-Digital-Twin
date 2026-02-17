"""
Supply Chain Schedulers

Custom Mesa activation schedulers for supply chain simulation.
Provides staged activation that respects the supply chain flow direction.
"""

from typing import List, Dict, Optional, Callable, TYPE_CHECKING
from collections import defaultdict
import random

if TYPE_CHECKING:
    from mesa import Agent, Model


class SupplyChainScheduler:
    """
    Base scheduler for supply chain agents.
    
    Activates agents in order based on their echelon level,
    from downstream (retailers) to upstream (suppliers) for demand propagation,
    or upstream to downstream for fulfillment.
    """
    
    def __init__(self, model: "Model", shuffle_within_echelon: bool = True):
        """
        Initialize the scheduler.
        
        Args:
            model: The Mesa model instance
            shuffle_within_echelon: Whether to randomize order within each echelon level
        """
        self.model = model
        self.shuffle_within_echelon = shuffle_within_echelon
        self._agents: Dict[int, "Agent"] = {}
        self._echelon_agents: Dict[int, List[int]] = defaultdict(list)
        self._agent_echelons: Dict[int, int] = {}
        self.time = 0
    
    @property
    def agents(self) -> List["Agent"]:
        """Return list of all agents."""
        return list(self._agents.values())
    
    def add(self, agent: "Agent", echelon: int = 0):
        """
        Add an agent to the scheduler.
        
        Args:
            agent: The agent to add
            echelon: The echelon level (0 = suppliers, higher = downstream)
        """
        self._agents[agent.unique_id] = agent
        self._echelon_agents[echelon].append(agent.unique_id)
        self._agent_echelons[agent.unique_id] = echelon
    
    def remove(self, agent: "Agent"):
        """Remove an agent from the scheduler."""
        if agent.unique_id in self._agents:
            echelon = self._agent_echelons.get(agent.unique_id, 0)
            del self._agents[agent.unique_id]
            if agent.unique_id in self._echelon_agents[echelon]:
                self._echelon_agents[echelon].remove(agent.unique_id)
            if agent.unique_id in self._agent_echelons:
                del self._agent_echelons[agent.unique_id]
    
    def get_echelon_levels(self) -> List[int]:
        """Get sorted list of echelon levels."""
        return sorted(self._echelon_agents.keys())
    
    def get_agents_at_echelon(self, echelon: int) -> List["Agent"]:
        """Get all agents at a specific echelon level."""
        agent_ids = self._echelon_agents.get(echelon, [])
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]
    
    def step(self):
        """Execute one step of the scheduler (override in subclasses)."""
        raise NotImplementedError("Subclasses must implement step()")


class DownstreamFirstScheduler(SupplyChainScheduler):
    """
    Scheduler that activates agents from downstream to upstream.
    
    Order: Retailers → Distributors → Manufacturers → Suppliers
    
    This is useful for demand propagation where customer orders
    flow up the supply chain.
    """
    
    def step(self):
        """Execute one step, activating downstream agents first."""
        echelons = self.get_echelon_levels()
        
        # Reverse order: highest echelon (downstream) first
        for echelon in reversed(echelons):
            agents = self.get_agents_at_echelon(echelon)
            
            if self.shuffle_within_echelon:
                random.shuffle(agents)
            
            for agent in agents:
                agent.step()
        
        self.time += 1


class UpstreamFirstScheduler(SupplyChainScheduler):
    """
    Scheduler that activates agents from upstream to downstream.
    
    Order: Suppliers → Manufacturers → Distributors → Retailers
    
    This is useful for fulfillment/delivery propagation where
    goods flow down the supply chain.
    """
    
    def step(self):
        """Execute one step, activating upstream agents first."""
        echelons = self.get_echelon_levels()
        
        # Normal order: lowest echelon (upstream) first
        for echelon in echelons:
            agents = self.get_agents_at_echelon(echelon)
            
            if self.shuffle_within_echelon:
                random.shuffle(agents)
            
            for agent in agents:
                agent.step()
        
        self.time += 1


class StagedSupplyChainScheduler(SupplyChainScheduler):
    """
    Multi-stage scheduler for complex supply chain dynamics.
    
    Executes multiple stages per step, with each stage using
    a different activation order. This allows for realistic
    simulation of order placement (downstream-first) followed
    by fulfillment (upstream-first).
    
    Default stages:
    1. demand_generation: Retailers generate customer demand
    2. order_placement: Downstream-to-upstream order propagation  
    3. fulfillment: Upstream-to-downstream delivery
    4. metrics_update: All agents update their metrics
    """
    
    DEFAULT_STAGES = [
        ("demand_generation", "downstream_only"),  # Only retailers
        ("order_placement", "downstream_first"),   # Demand propagates up
        ("fulfillment", "upstream_first"),         # Deliveries flow down
        ("metrics_update", "all"),                 # All agents update
    ]
    
    def __init__(
        self, 
        model: "Model", 
        stages: Optional[List[tuple]] = None,
        shuffle_within_echelon: bool = True
    ):
        """
        Initialize staged scheduler.
        
        Args:
            model: The Mesa model instance
            stages: List of (stage_name, activation_order) tuples
                   activation_order can be: "downstream_first", "upstream_first", 
                   "all", "downstream_only", "upstream_only"
            shuffle_within_echelon: Whether to randomize within echelons
        """
        super().__init__(model, shuffle_within_echelon)
        self.stages = stages or self.DEFAULT_STAGES
    
    def step(self):
        """Execute all stages in sequence."""
        for stage_name, activation_order in self.stages:
            self._execute_stage(stage_name, activation_order)
        
        self.time += 1
    
    def _execute_stage(self, stage_name: str, activation_order: str):
        """
        Execute a single stage with the specified activation order.
        
        Args:
            stage_name: Name of the stage (used to call agent.{stage_name}())
            activation_order: Order to activate agents
        """
        echelons = self.get_echelon_levels()
        
        if activation_order == "downstream_first":
            echelon_order = reversed(echelons)
        elif activation_order == "upstream_first":
            echelon_order = echelons
        elif activation_order == "downstream_only":
            # Only the highest echelon (retailers)
            echelon_order = [max(echelons)] if echelons else []
        elif activation_order == "upstream_only":
            # Only the lowest echelon (suppliers)
            echelon_order = [min(echelons)] if echelons else []
        elif activation_order == "all":
            echelon_order = echelons
        else:
            echelon_order = echelons
        
        for echelon in echelon_order:
            agents = self.get_agents_at_echelon(echelon)
            
            if self.shuffle_within_echelon:
                random.shuffle(agents)
            
            for agent in agents:
                # Call the stage-specific method if it exists
                stage_method = getattr(agent, stage_name, None)
                if callable(stage_method):
                    stage_method()
                else:
                    # Fallback to generic step
                    agent.step()


class SimultaneousSupplyChainScheduler(SupplyChainScheduler):
    """
    Scheduler that collects all agent actions before applying them.
    
    All agents compute their actions simultaneously (based on the 
    current state), then all actions are applied at once. This prevents
    order-dependent effects within a single timestep.
    
    Useful for analyzing idealized supply chain dynamics without
    information asymmetry within a timestep.
    """
    
    def step(self):
        """Execute simultaneous activation."""
        # Phase 1: All agents compute actions
        actions = {}
        for agent in self.agents:
            if hasattr(agent, 'compute_action'):
                actions[agent.unique_id] = agent.compute_action()
        
        # Phase 2: All agents apply actions
        for agent in self.agents:
            if hasattr(agent, 'apply_action') and agent.unique_id in actions:
                agent.apply_action(actions[agent.unique_id])
            else:
                agent.step()
        
        self.time += 1


class PriorityScheduler(SupplyChainScheduler):
    """
    Scheduler that activates agents based on dynamic priority.
    
    Priority can be based on inventory levels, demand urgency,
    or custom priority functions. Higher priority agents are
    activated first.
    """
    
    def __init__(
        self, 
        model: "Model",
        priority_func: Optional[Callable[["Agent"], float]] = None,
        shuffle_within_echelon: bool = True
    ):
        """
        Initialize priority scheduler.
        
        Args:
            model: The Mesa model instance
            priority_func: Function that takes an agent and returns priority (higher = first)
                          Defaults to inventory-based priority (lower inventory = higher priority)
            shuffle_within_echelon: Whether to shuffle agents with equal priority
        """
        super().__init__(model, shuffle_within_echelon)
        self.priority_func = priority_func or self._default_priority
    
    def _default_priority(self, agent: "Agent") -> float:
        """
        Default priority based on inventory level.
        
        Lower inventory = higher priority (needs attention first).
        """
        if hasattr(agent, 'inventory') and hasattr(agent, 'reorder_point'):
            # Priority increases as inventory drops below reorder point
            return agent.reorder_point - agent.inventory
        return 0.0
    
    def step(self):
        """Execute step with priority-based ordering."""
        # Sort agents by priority (descending)
        sorted_agents = sorted(
            self.agents,
            key=self.priority_func,
            reverse=True
        )
        
        for agent in sorted_agents:
            agent.step()
        
        self.time += 1


# Helper function to create appropriate scheduler
def create_scheduler(
    model: "Model",
    scheduler_type: str = "staged",
    **kwargs
) -> SupplyChainScheduler:
    """
    Factory function to create schedulers.
    
    Args:
        model: The Mesa model instance
        scheduler_type: One of "staged", "downstream_first", "upstream_first",
                       "simultaneous", "priority"
        **kwargs: Additional arguments for the scheduler
    
    Returns:
        Configured scheduler instance
    """
    schedulers = {
        "staged": StagedSupplyChainScheduler,
        "downstream_first": DownstreamFirstScheduler,
        "upstream_first": UpstreamFirstScheduler,
        "simultaneous": SimultaneousSupplyChainScheduler,
        "priority": PriorityScheduler,
    }
    
    scheduler_class = schedulers.get(scheduler_type.lower())
    if scheduler_class is None:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. "
                        f"Available: {list(schedulers.keys())}")
    
    return scheduler_class(model, **kwargs)
