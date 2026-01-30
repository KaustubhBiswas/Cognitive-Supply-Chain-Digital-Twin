"""
Supply Chain Agents

Mesa agents representing entities in the supply chain network.
Each agent implements ordering policies that can produce the Bullwhip Effect.
"""

from mesa import Agent
from typing import Optional, List, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .model import SupplyChainModel


@dataclass
class Order:
    """Represents an order in the supply chain."""
    quantity: float
    source_id: int
    target_id: int
    placed_step: int
    delivery_step: int
    fulfilled: bool = False


class SupplyChainAgent(Agent):
    """
    Base agent for supply chain entities.
    
    Implements a standard (s, S) inventory policy where:
    - s = reorder point
    - S = order-up-to level
    
    This policy naturally produces order amplification (Bullwhip Effect)
    when combined with demand variability and lead times.
    """
    
    def __init__(
        self,
        unique_id: int,
        model: "SupplyChainModel",
        node_type: str,
        initial_inventory: float = 100.0,
        reorder_point: float = 20.0,
        order_up_to: float = 100.0,
        lead_time: int = 2,
    ):
        super().__init__(model)
        self.unique_id = unique_id
        self.node_type = node_type
        self.inventory = initial_inventory
        self.reorder_point = reorder_point
        self.order_up_to = order_up_to
        self.lead_time = lead_time
        
        # Order tracking
        self.pending_orders: List[Order] = []
        self.orders_placed: List[float] = []
        self.orders_received: List[float] = []
        self.demand_history: List[float] = []
        self.inventory_history: List[float] = []
        
        # Backlog for unfulfilled demand
        self.backlog: float = 0.0
        
        # Network connections (set during model initialization)
        self.suppliers: List["SupplyChainAgent"] = []
        self.customers: List["SupplyChainAgent"] = []
    
    def step(self):
        """Execute one simulation step."""
        # 1. Receive pending deliveries
        self._receive_deliveries()
        
        # 2. Record inventory before fulfillment
        self.inventory_history.append(self.inventory)
        
        # 3. Check inventory and place orders to suppliers
        self._check_and_order()
    
    def receive_demand(self, quantity: float, from_agent: "SupplyChainAgent") -> float:
        """
        Process incoming demand from a customer.
        
        Args:
            quantity: Amount requested
            from_agent: The requesting agent
            
        Returns:
            Amount actually fulfilled
        """
        self.demand_history.append(quantity)
        
        # Try to fulfill from inventory
        total_demand = quantity + self.backlog
        fulfilled = min(total_demand, self.inventory)
        self.inventory -= fulfilled
        
        # Track unfulfilled as backlog
        self.backlog = total_demand - fulfilled
        
        return fulfilled if quantity <= fulfilled else fulfilled - self.backlog
    
    def _receive_deliveries(self):
        """Process deliveries that have arrived (lead time elapsed)."""
        current_step = self.model.current_step
        received_qty = 0.0
        
        still_pending = []
        for order in self.pending_orders:
            if order.delivery_step <= current_step and not order.fulfilled:
                # Delivery arrived
                self.inventory += order.quantity
                order.fulfilled = True
                received_qty += order.quantity
            elif not order.fulfilled:
                still_pending.append(order)
        
        self.pending_orders = still_pending
        self.orders_received.append(received_qty)
    
    def _check_and_order(self):
        """Implement (s, S) ordering policy with demand forecasting."""
        # Calculate effective inventory position
        pending_qty = sum(o.quantity for o in self.pending_orders)
        inventory_position = self.inventory + pending_qty - self.backlog
        
        if inventory_position <= self.reorder_point:
            # Calculate order quantity using recent demand
            if len(self.demand_history) >= 3:
                # Use moving average of recent demand
                avg_demand = np.mean(self.demand_history[-3:])
                safety_factor = 1.5  # Conservative safety stock
                order_quantity = max(
                    self.order_up_to - inventory_position,
                    avg_demand * self.lead_time * safety_factor
                )
            else:
                order_quantity = self.order_up_to - inventory_position
            
            self._place_order(order_quantity)
        else:
            self.orders_placed.append(0.0)
    
    def _place_order(self, quantity: float):
        """Place order with upstream supplier."""
        if not self.suppliers:
            # This is a raw material supplier, infinite supply
            self._self_replenish(quantity)
            return
        
        # Distribute order among suppliers (simple even split)
        qty_per_supplier = quantity / len(self.suppliers)
        
        for supplier in self.suppliers:
            # Supplier fulfills what it can
            fulfilled = supplier.receive_demand(qty_per_supplier, self)
            
            # Create order record
            order = Order(
                quantity=fulfilled,
                source_id=supplier.unique_id,
                target_id=self.unique_id,
                placed_step=self.model.current_step,
                delivery_step=self.model.current_step + self.lead_time,
            )
            self.pending_orders.append(order)
        
        self.orders_placed.append(quantity)
    
    def _self_replenish(self, quantity: float):
        """For suppliers: self-replenish with lead time."""
        order = Order(
            quantity=quantity,
            source_id=self.unique_id,
            target_id=self.unique_id,
            placed_step=self.model.current_step,
            delivery_step=self.model.current_step + self.lead_time,
        )
        self.pending_orders.append(order)
        self.orders_placed.append(quantity)
    
    def get_metrics(self) -> dict:
        """Get current agent metrics."""
        return {
            "unique_id": self.unique_id,
            "node_type": self.node_type,
            "inventory": self.inventory,
            "backlog": self.backlog,
            "pending_orders": sum(o.quantity for o in self.pending_orders),
            "avg_demand": np.mean(self.demand_history) if self.demand_history else 0,
            "order_variance": np.var(self.orders_placed) if len(self.orders_placed) > 1 else 0,
        }


class SupplierAgent(SupplyChainAgent):
    """
    Raw material supplier with production capacity.
    
    Suppliers are the source of the supply chain and have
    theoretically infinite raw materials but limited production capacity.
    """
    
    def __init__(
        self,
        unique_id: int,
        model: "SupplyChainModel",
        production_capacity: float = 80.0,
        **kwargs
    ):
        super().__init__(unique_id, model, node_type="supplier", **kwargs)
        self.production_capacity = production_capacity
        self.production_history: List[float] = []
    
    def _self_replenish(self, quantity: float):
        """Produce up to capacity."""
        actual_production = min(quantity, self.production_capacity)
        self.production_history.append(actual_production)
        
        order = Order(
            quantity=actual_production,
            source_id=self.unique_id,
            target_id=self.unique_id,
            placed_step=self.model.current_step,
            delivery_step=self.model.current_step + self.lead_time,
        )
        self.pending_orders.append(order)
        self.orders_placed.append(actual_production)


class ManufacturerAgent(SupplyChainAgent):
    """
    Manufacturer that transforms raw materials into products.
    
    Has production capacity constraints and may experience
    factory issues that reduce capacity.
    """
    
    def __init__(
        self,
        unique_id: int,
        model: "SupplyChainModel",
        production_capacity: float = 60.0,
        **kwargs
    ):
        super().__init__(unique_id, model, node_type="manufacturer", **kwargs)
        self.production_capacity = production_capacity
        self.capacity_utilization: List[float] = []
        self.has_factory_issue = False
    
    def receive_demand(self, quantity: float, from_agent: "SupplyChainAgent") -> float:
        """Process demand with capacity constraints."""
        # Apply capacity constraint
        effective_capacity = self.production_capacity
        if self.has_factory_issue:
            effective_capacity *= 0.5  # 50% capacity during issues
        
        max_fulfillable = min(quantity, self.inventory, effective_capacity)
        self.demand_history.append(quantity)
        
        fulfilled = min(quantity + self.backlog, max_fulfillable)
        self.inventory -= fulfilled
        self.backlog = max(0, quantity + self.backlog - fulfilled)
        
        utilization = fulfilled / effective_capacity if effective_capacity > 0 else 0
        self.capacity_utilization.append(utilization)
        
        return fulfilled


class DistributorAgent(SupplyChainAgent):
    """
    Distributor handling logistics between manufacturers and retailers.
    
    Distributors aggregate demand and distribute products,
    potentially amplifying or dampening the Bullwhip Effect.
    """
    
    def __init__(
        self,
        unique_id: int,
        model: "SupplyChainModel",
        **kwargs
    ):
        super().__init__(unique_id, model, node_type="distributor", **kwargs)
        self.throughput_history: List[float] = []
    
    def step(self):
        """Track throughput in addition to base step."""
        super().step()
        
        # Calculate throughput (demand fulfilled)
        if self.demand_history:
            fulfilled = self.demand_history[-1] - self.backlog
            self.throughput_history.append(max(0, fulfilled))


class RetailerAgent(SupplyChainAgent):
    """
    Retailer facing end-customer demand.
    
    Retailers generate stochastic customer demand which propagates
    upstream through the supply chain.
    """
    
    def __init__(
        self,
        unique_id: int,
        model: "SupplyChainModel",
        base_demand: float = 10.0,
        demand_std: float = 3.0,
        **kwargs
    ):
        super().__init__(unique_id, model, node_type="retailer", **kwargs)
        self.base_demand = base_demand
        self.demand_std = demand_std
        self.customer_demand_history: List[float] = []
        self.lost_sales: List[float] = []
    
    def generate_customer_demand(self) -> float:
        """Generate stochastic customer demand."""
        demand = max(0, np.random.normal(self.base_demand, self.demand_std))
        self.customer_demand_history.append(demand)
        return demand
    
    def fulfill_customer_demand(self, demand: float) -> float:
        """
        Fulfill customer demand from inventory.
        
        Returns:
            Amount fulfilled (lost sales = demand - fulfilled)
        """
        fulfilled = min(demand, self.inventory)
        self.inventory -= fulfilled
        lost = demand - fulfilled
        self.lost_sales.append(lost)
        self.demand_history.append(demand)
        return fulfilled
    
    def step(self):
        """Retailer step includes customer demand generation."""
        # Receive deliveries first
        self._receive_deliveries()
        
        # Generate and fulfill customer demand
        customer_demand = self.generate_customer_demand()
        self.fulfill_customer_demand(customer_demand)
        
        # Record inventory
        self.inventory_history.append(self.inventory)
        
        # Check reorder
        self._check_and_order()
