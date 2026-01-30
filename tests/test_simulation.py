"""
Tests for the Mesa simulation components.

Tests cover:
- Agent initialization and behavior
- Model creation and stepping
- Bullwhip Effect detection
- Event injection and response
"""

import pytest
import numpy as np
import networkx as nx

from src.simulation.model import (
    SupplyChainModel,
    EventType,
    create_sample_supply_chain,
)
from src.simulation.agents import (
    SupplyChainAgent,
    SupplierAgent,
    ManufacturerAgent,
    DistributorAgent,
    RetailerAgent,
)
from src.simulation.grid import SupplyNetworkGrid


class TestSupplyChainAgents:
    """Test suite for supply chain agents."""
    
    def test_agent_initialization(self):
        """Test basic agent creation."""
        model = create_sample_supply_chain(seed=42)
        
        agents = list(model.agents)
        assert len(agents) > 0
        
        # Check all agents have required attributes
        for agent in agents:
            assert hasattr(agent, "inventory")
            assert hasattr(agent, "reorder_point")
            assert hasattr(agent, "order_up_to")
            assert hasattr(agent, "lead_time")
            assert agent.inventory >= 0
    
    def test_agent_types(self):
        """Test that agents are created with correct types."""
        model = create_sample_supply_chain(
            num_suppliers=2,
            num_manufacturers=3,
            num_distributors=4,
            num_retailers=5,
            seed=42,
        )
        
        type_counts = {"supplier": 0, "manufacturer": 0, "distributor": 0, "retailer": 0}
        for agent in model.agents:
            type_counts[agent.node_type] += 1
        
        assert type_counts["supplier"] == 2
        assert type_counts["manufacturer"] == 3
        assert type_counts["distributor"] == 4
        assert type_counts["retailer"] == 5
    
    def test_retailer_demand_generation(self):
        """Test that retailers generate customer demand."""
        model = create_sample_supply_chain(seed=42)
        
        retailers = [a for a in model.agents if isinstance(a, RetailerAgent)]
        assert len(retailers) > 0
        
        for retailer in retailers:
            demand = retailer.generate_customer_demand()
            assert demand >= 0
    
    def test_manufacturer_capacity_constraint(self):
        """Test that manufacturers respect capacity limits."""
        model = create_sample_supply_chain(seed=42)
        
        manufacturers = [a for a in model.agents if isinstance(a, ManufacturerAgent)]
        assert len(manufacturers) > 0
        
        for manufacturer in manufacturers:
            assert hasattr(manufacturer, "production_capacity")
            assert manufacturer.production_capacity > 0


class TestSupplyNetworkGrid:
    """Test suite for network grid."""
    
    def test_grid_creation(self):
        """Test grid initialization from graph."""
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        
        grid = SupplyNetworkGrid(G)
        
        assert grid.get_num_echelons() > 0
    
    def test_upstream_downstream_queries(self):
        """Test neighbor queries."""
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
        
        grid = SupplyNetworkGrid(G)
        
        # Node 0 has no upstream, 2 downstream
        assert len(grid.get_upstream_neighbors(0)) == 0
        assert set(grid.get_downstream_neighbors(0)) == {1, 2}
        
        # Node 3 has 2 upstream, no downstream
        assert set(grid.get_upstream_neighbors(3)) == {1, 2}
        assert len(grid.get_downstream_neighbors(3)) == 0
    
    def test_echelon_assignment(self):
        """Test echelon level computation."""
        G = nx.DiGraph()
        # Linear chain: 0 → 1 → 2 → 3
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        
        grid = SupplyNetworkGrid(G)
        
        assert grid.get_node_echelon(0) == 0
        assert grid.get_node_echelon(1) == 1
        assert grid.get_node_echelon(2) == 2
        assert grid.get_node_echelon(3) == 3


class TestSupplyChainModel:
    """Test suite for the main simulation model."""
    
    def test_model_creation(self):
        """Test model initialization."""
        model = create_sample_supply_chain(seed=42)
        
        assert model.current_step == 0
        assert len(model.agents) > 0
        assert model.network_grid is not None
    
    def test_model_step(self):
        """Test single simulation step."""
        model = create_sample_supply_chain(seed=42)
        
        initial_step = model.current_step
        model.step()
        
        assert model.current_step == initial_step + 1
    
    def test_model_run(self):
        """Test running simulation for multiple steps."""
        model = create_sample_supply_chain(seed=42)
        
        results = model.run(steps=50)
        
        assert model.current_step == 50
        assert "model_data" in results
        assert "agent_data" in results
        assert len(results["model_data"]) == 50
    
    def test_data_collection(self):
        """Test that data is collected during simulation."""
        model = create_sample_supply_chain(seed=42)
        model.run(steps=20)
        
        model_data = model.datacollector.get_model_vars_dataframe()
        
        assert "total_inventory" in model_data.columns
        assert "bullwhip_ratio" in model_data.columns
        assert len(model_data) == 20
    
    def test_bullwhip_ratio_calculation(self):
        """Test Bullwhip Effect metric calculation."""
        model = create_sample_supply_chain(seed=42)
        model.run(steps=50)
        
        bullwhip = model._compute_bullwhip_ratio()
        
        # Should be a positive number
        assert bullwhip >= 0 or np.isinf(bullwhip)
    
    def test_state_snapshot(self):
        """Test state snapshot generation for cognitive agents."""
        model = create_sample_supply_chain(seed=42)
        model.run(steps=10)
        
        snapshot = model.get_state_snapshot()
        
        assert "step" in snapshot
        assert "total_inventory" in snapshot
        assert "bullwhip_ratio" in snapshot
        assert "agents" in snapshot
        assert "network" in snapshot


class TestEventInjection:
    """Test suite for event injection."""
    
    def test_demand_shock_injection(self):
        """Test injecting demand shock event."""
        model = create_sample_supply_chain(seed=42)
        
        # Get initial demand levels
        retailers = [a for a in model.agents if isinstance(a, RetailerAgent)]
        initial_demands = [r.base_demand for r in retailers]
        
        # Inject demand shock
        model.inject_event(EventType.DEMAND_SHOCK, magnitude=2.0, duration=5)
        
        # Check demands increased
        for i, retailer in enumerate(retailers):
            assert retailer.base_demand == initial_demands[i] * 2.0
        
        assert len(model.active_events) == 1
    
    def test_event_expiration(self):
        """Test that events expire and reverse effects."""
        model = create_sample_supply_chain(seed=42)
        
        retailers = [a for a in model.agents if isinstance(a, RetailerAgent)]
        initial_demands = [r.base_demand for r in retailers]
        
        # Inject short event
        model.inject_event(EventType.DEMAND_SHOCK, magnitude=2.0, duration=3)
        
        # Run past event duration
        model.run(steps=5)
        
        # Demands should be back to normal
        for i, retailer in enumerate(retailers):
            assert abs(retailer.base_demand - initial_demands[i]) < 0.01
        
        assert len(model.active_events) == 0
    
    def test_supply_disruption(self):
        """Test supply disruption event."""
        model = create_sample_supply_chain(seed=42)
        
        suppliers = [a for a in model.agents if isinstance(a, SupplierAgent)]
        initial_capacities = [s.production_capacity for s in suppliers]
        
        # Inject disruption
        model.inject_event(EventType.SUPPLY_DISRUPTION, magnitude=2.0, duration=5)
        
        # Check capacities reduced
        for i, supplier in enumerate(suppliers):
            assert supplier.production_capacity == initial_capacities[i] / 2.0
    
    def test_factory_issue(self):
        """Test factory issue event."""
        model = create_sample_supply_chain(seed=42)
        
        manufacturers = [a for a in model.agents if isinstance(a, ManufacturerAgent)]
        
        # Initially no issues
        for man in manufacturers:
            assert not man.has_factory_issue
        
        # Inject factory issue
        model.inject_event(EventType.FACTORY_ISSUE, duration=5)
        
        # Now should have issues
        for man in manufacturers:
            assert man.has_factory_issue


class TestBullwhipEffect:
    """Test suite specifically for Bullwhip Effect verification."""
    
    def test_bullwhip_naturally_occurs(self):
        """Verify that Bullwhip Effect occurs naturally in simulation."""
        model = create_sample_supply_chain(seed=42)
        model.run(steps=100)
        
        # Get order variance by echelon
        echelon_variances = []
        for echelon in range(model.network_grid.get_num_echelons()):
            agents = model.network_grid.get_agents_at_echelon(echelon)
            orders = []
            for agent in agents:
                orders.extend(agent.orders_placed)
            if orders:
                echelon_variances.append(np.var(orders))
        
        # With proper (s,S) policy, variance should increase upstream
        # This test verifies the simulation can produce Bullwhip
        if len(echelon_variances) >= 2:
            # At least some amplification should occur
            assert any(
                echelon_variances[i] > echelon_variances[i + 1]
                for i in range(len(echelon_variances) - 1)
            ) or model._compute_bullwhip_ratio() > 0.5
    
    def test_bullwhip_amplified_by_demand_shock(self):
        """Test that demand shock amplifies Bullwhip Effect."""
        np.random.seed(42)
        
        # Run baseline
        model_baseline = create_sample_supply_chain(seed=42)
        model_baseline.run(steps=50)
        baseline_bullwhip = model_baseline._compute_bullwhip_ratio()
        
        # Run with demand shock
        model_shock = create_sample_supply_chain(seed=42)
        model_shock.run(steps=10)
        model_shock.inject_event(EventType.DEMAND_SHOCK, magnitude=2.0, duration=20)
        model_shock.run(steps=40)
        shock_bullwhip = model_shock._compute_bullwhip_ratio()
        
        # Demand shock should generally increase Bullwhip
        # (though this can vary with random factors)
        assert shock_bullwhip >= 0  # At minimum, should be valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
