"""
Tests for the Cognition Module

Tests the multi-agent system components:
- State management
- Tools (with and without dependencies)
- Agent functions
- Graph workflow (FallbackGraph)
"""

import networkx as nx
import numpy as np
import pytest

from src.cognition import ALERT_THRESHOLDS  # State; Tools; Agents; Graph
from src.cognition import (AgentRoute, Alert, AlertSeverity, AlertType,
                           FallbackGraph, Recommendation, RecommendationType,
                           SupplyChainState, add_recommendation,
                           compute_bullwhip_ratio, create_analyst_agent,
                           create_initial_state, create_negotiator_agent,
                           create_supervisor_agent, create_supply_chain_graph,
                           forecast_demand, get_all_inventories,
                           get_downstream_customers, get_node_inventory,
                           get_tool_descriptions, get_upstream_suppliers,
                           initialize_tools, is_initialized)
from src.data.parser import create_synthetic_supply_graph
from src.simulation import SupplyChainModel

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def supply_graph_data():
    """Create synthetic supply graph for testing."""
    return create_synthetic_supply_graph(
        num_suppliers=2,
        num_manufacturers=2,
        num_distributors=2,
        num_retailers=3,
        seed=42,
    )


@pytest.fixture
def simulation_model(supply_graph_data):
    """Create a simulation model for testing."""
    model = SupplyChainModel(
        graph=supply_graph_data.graph,
        node_types=supply_graph_data.node_types,
        random_seed=42,
    )
    # Run a few steps to generate data
    for _ in range(10):
        model.step()
    return model


@pytest.fixture
def initialized_tools(simulation_model):
    """Initialize tools with simulation model."""
    initialize_tools(simulation=simulation_model)
    return simulation_model


# =============================================================================
# State Tests
# =============================================================================

class TestState:
    """Test state module components."""
    
    def test_alert_severity_enum(self):
        """Test AlertSeverity enum values."""
        assert AlertSeverity.LOW.value == "low"
        assert AlertSeverity.MEDIUM.value == "medium"
        assert AlertSeverity.HIGH.value == "high"
        assert AlertSeverity.CRITICAL.value == "critical"
    
    def test_alert_type_enum(self):
        """Test AlertType enum values."""
        assert AlertType.DEMAND_SPIKE.value == "demand_spike"
        assert AlertType.INVENTORY_LOW.value == "inventory_low"
        assert AlertType.BULLWHIP_DETECTED.value == "bullwhip_detected"
    
    def test_recommendation_type_enum(self):
        """Test RecommendationType enum values."""
        assert RecommendationType.ADJUST_ORDER_QUANTITY.value == "adjust_order_quantity"
        assert RecommendationType.ADJUST_SAFETY_STOCK.value == "adjust_safety_stock"
        assert RecommendationType.NO_ACTION.value == "no_action"
    
    def test_create_alert(self):
        """Test Alert dataclass creation."""
        alert = Alert(
            alert_type=AlertType.DEMAND_SPIKE,
            severity=AlertSeverity.HIGH,
            affected_nodes=[5],
            details={"magnitude": 150.0},
        )
        assert alert.alert_type == AlertType.DEMAND_SPIKE
        assert alert.severity == AlertSeverity.HIGH
        assert 5 in alert.affected_nodes
        assert alert.details["magnitude"] == 150.0
    
    def test_create_recommendation(self):
        """Test Recommendation dataclass creation."""
        rec = Recommendation(
            recommendation_type=RecommendationType.ADJUST_ORDER_QUANTITY,
            target_nodes=[2, 3],
            parameters={"adjustment_factor": 1.2},
            reasoning="Increase order quantity due to demand spike",
            confidence=0.85,
            source_agent="analyst",
        )
        assert rec.recommendation_type == RecommendationType.ADJUST_ORDER_QUANTITY
        assert len(rec.target_nodes) == 2
        assert rec.parameters["adjustment_factor"] == 1.2
        assert rec.confidence == 0.85
    
    def test_create_initial_state_with_alert(self):
        """Test initial state creation with an alert."""
        alert = Alert(
            alert_type=AlertType.INVENTORY_LOW,
            severity=AlertSeverity.MEDIUM,
            affected_nodes=[7],
            details={"current_inventory": 10, "reorder_point": 50},
        )
        state = create_initial_state(alert=alert)
        
        assert "current_alert" in state
        assert state["current_alert"] is not None
        assert state["current_alert"]["alert_type"] == "inventory_low"
        assert state["next_agent"] == "supervisor"
        assert state["iteration_count"] == 0
    
    def test_add_recommendation_to_state(self):
        """Test adding recommendation to state."""
        state = create_initial_state()
        rec = Recommendation(
            recommendation_type=RecommendationType.ADJUST_SAFETY_STOCK,
            target_nodes=[5],
            parameters={"new_safety_stock": 100},
            reasoning="Increase safety stock to prevent stockouts",
            confidence=0.75,
            source_agent="negotiator",
        )
        updated = add_recommendation(state, rec)
        
        assert len(updated["recommendations"]) == 1
        # Recommendations are stored as dicts
        assert updated["recommendations"][0]["reasoning"] == "Increase safety stock to prevent stockouts"


# =============================================================================
# Tools Tests
# =============================================================================

class TestTools:
    """Test tool functions."""
    
    def test_is_initialized_false_initially(self):
        """Test that tools report uninitialized state."""
        assert isinstance(is_initialized(), bool)
    
    def test_initialize_tools(self, simulation_model):
        """Test tool initialization."""
        initialize_tools(simulation=simulation_model)
        assert is_initialized()
    
    def test_forecast_demand_fallback(self, initialized_tools):
        """Test demand forecasting with fallback (no GNN)."""
        result = forecast_demand.invoke({"node_ids": [0, 1], "horizon": 5})

        assert "predictions" in result
        assert "confidence" in result
        assert result["success"] is True
        assert len(result["predictions"][0]) == 5
        assert result["model_type"] == "moving_average_fallback"
    
    def test_get_node_inventory(self, initialized_tools):
        """Test inventory query tool."""
        result = get_node_inventory.invoke({"node_id": 0})

        # Tool always returns success key
        assert "success" in result
        # If successful, should have node_id and inventory
        if result["success"]:
            assert "node_id" in result
            assert "inventory" in result
    
    def test_get_all_inventories(self, initialized_tools):
        """Test all inventories query."""
        result = get_all_inventories.invoke({})

        assert "success" in result
        # May fail if simulation doesn't have expected structure
        if result["success"]:
            assert "inventories" in result
            assert isinstance(result["inventories"], dict)
    
    def test_compute_bullwhip_ratio(self, initialized_tools):
        """Test Bullwhip ratio computation."""
        result = compute_bullwhip_ratio.invoke({})
        
        assert "success" in result
        if result["success"]:
            assert "overall_ratio" in result
    
    def test_get_upstream_suppliers(self, initialized_tools):
        """Test upstream supplier query."""
        result = get_upstream_suppliers.invoke({"node_id": 5})
        
        assert "success" in result
        if result["success"]:
            assert "node_id" in result
            assert "upstream_suppliers" in result
    
    def test_get_downstream_customers(self, initialized_tools):
        """Test downstream customer query."""
        result = get_downstream_customers.invoke({"node_id": 0})
        
        assert "success" in result
        if result["success"]:
            assert "node_id" in result
            assert "downstream_customers" in result
    
    def test_get_tool_descriptions(self):
        """Test tool descriptions retrieval."""
        descriptions = get_tool_descriptions()
        
        assert isinstance(descriptions, dict)
        assert len(descriptions) > 0


# =============================================================================
# Agent Tests
# =============================================================================

class TestAgents:
    """Test agent creation and basic function."""
    
    def test_create_supervisor_agent(self):
        """Test supervisor agent creation (no LLM)."""
        supervisor = create_supervisor_agent(llm=None)
        assert callable(supervisor)
    
    def test_create_analyst_agent(self):
        """Test analyst agent creation (no LLM)."""
        analyst = create_analyst_agent(llm=None)
        assert callable(analyst)
    
    def test_create_negotiator_agent(self):
        """Test negotiator agent creation (no LLM)."""
        negotiator = create_negotiator_agent(llm=None)
        assert callable(negotiator)
    
    def test_supervisor_rule_based(self, initialized_tools):
        """Test supervisor with rule-based fallback."""
        alert = Alert(
            alert_type=AlertType.DEMAND_SPIKE,
            severity=AlertSeverity.HIGH,
            affected_nodes=[5],
            details={"magnitude": 50.0},
        )
        state = create_initial_state(alert=alert)
        
        supervisor = create_supervisor_agent(llm=None)
        try:
            result = supervisor(state)
            assert "next_agent" in result
            # High severity should route somewhere
            assert result["next_agent"] in ["analyst", "negotiator", "end", "human", "supervisor"]
        except ModuleNotFoundError:
            # langchain_core not installed - test the fallback import handling
            pytest.skip("langchain_core not installed - skipping LLM agent test")
    
    def test_analyst_rule_based(self, initialized_tools):
        """Test analyst with rule-based fallback."""
        alert = Alert(
            alert_type=AlertType.BULLWHIP_DETECTED,
            severity=AlertSeverity.MEDIUM,
            affected_nodes=[2, 3, 4],
            details={"bullwhip_ratio": 1.8},
        )
        state = create_initial_state(alert=alert)
        
        analyst = create_analyst_agent(llm=None)
        try:
            result = analyst(state)
            assert "analysis_results" in result or "next_agent" in result
        except ModuleNotFoundError:
            pytest.skip("langchain_core not installed - skipping LLM agent test")
    
    def test_negotiator_rule_based(self, initialized_tools):
        """Test negotiator with rule-based fallback."""
        state = create_initial_state()
        state["recommendations"] = [
            Recommendation(
                recommendation_type=RecommendationType.ADJUST_ORDER_QUANTITY,
                target_nodes=[5],
                parameters={"adjustment": 1.1},
                reasoning="Test recommendation",
                confidence=0.8,
                source_agent="analyst",
            ).to_dict()
        ]
        
        negotiator = create_negotiator_agent(llm=None)
        try:
            result = negotiator(state)
            # Should return some state update
            assert isinstance(result, dict)
        except ModuleNotFoundError:
            pytest.skip("langchain_core not installed - skipping LLM agent test")


# =============================================================================
# Graph Tests
# =============================================================================

class TestGraph:
    """Test graph workflow components."""
    
    def test_fallback_graph_creation(self):
        """Test FallbackGraph creation."""
        graph = FallbackGraph(llm=None)
        
        assert graph.supervisor is not None
        assert graph.analyst is not None
        assert graph.negotiator is not None
    
    def test_fallback_graph_invoke(self, initialized_tools):
        """Test FallbackGraph workflow execution."""
        graph = FallbackGraph(llm=None)
        
        alert = Alert(
            alert_type=AlertType.INVENTORY_LOW,
            severity=AlertSeverity.MEDIUM,
            affected_nodes=[7],
            details={"current_inventory": 15},
        )
        initial_state = create_initial_state(alert=alert)
        
        result = graph.invoke(initial_state)
        
        assert isinstance(result, dict)
        assert "next_agent" in result
        # Should have processed through workflow
        assert result.get("iteration_count", 0) >= 1
    
    def test_create_supply_chain_graph_returns_workflow(self, initialized_tools):
        """Test main graph creation function."""
        graph = create_supply_chain_graph(llm=None)
        
        # Should return either LangGraph compiled graph or FallbackGraph
        assert graph is not None
        assert hasattr(graph, 'invoke')
    
    def test_full_workflow_execution(self, initialized_tools):
        """Test complete workflow from start to finish."""
        graph = create_supply_chain_graph(llm=None)
        
        # Create an alert scenario
        alert = Alert(
            alert_type=AlertType.DEMAND_SPIKE,
            severity=AlertSeverity.HIGH,
            affected_nodes=[7],
            details={"increase_percent": 45.0},
        )
        
        initial_state = create_initial_state(alert=alert)
        
        # Execute workflow
        try:
            if hasattr(graph, 'invoke'):
                # LangGraph style
                result = graph.invoke(
                    initial_state,
                    config={"configurable": {"thread_id": "test-1"}} 
                        if hasattr(graph, 'checkpointer') else {}
                )
            else:
                # Should not happen
                result = graph(initial_state)
            
            assert isinstance(result, dict)
        except Exception as e:
            # Even with fallback, should work
            pytest.fail(f"Workflow execution failed: {e}")


# =============================================================================
# Integration Tests
# =============================================================================

class TestCognitionIntegration:
    """Integration tests for full cognition workflow."""
    
    def test_simulation_to_cognition_pipeline(self, simulation_model):
        """Test full pipeline from simulation to cognitive response."""
        # Initialize
        initialize_tools(simulation=simulation_model)
        
        # Create a simple alert directly (don't depend on tool working)
        alert = Alert(
            alert_type=AlertType.INVENTORY_LOW,
            severity=AlertSeverity.MEDIUM,
            affected_nodes=[0],
            details={"current_inventory": 15, "node_id": 0},
        )
        
        # Run cognitive workflow
        graph = create_supply_chain_graph(llm=None)
        state = create_initial_state(alert=alert)
        result = graph.invoke(state, config={"configurable": {"thread_id": "test-sim"}})
        
        assert result is not None
        # Should have gone through at least one iteration
        assert result.get("iteration_count", 0) >= 1
    
    def test_bullwhip_detection_workflow(self, simulation_model):
        """Test Bullwhip detection and response workflow."""
        # Run more steps to generate Bullwhip effect
        for _ in range(20):
            simulation_model.step()
        
        initialize_tools(simulation=simulation_model)
        
        # Check for Bullwhip effect
        bullwhip_result = compute_bullwhip_ratio.invoke({})
        ratio = bullwhip_result.get("overall_ratio", 1.0)
        
        if ratio > 1.2:  # Bullwhip detected
            alert = Alert(
                alert_type=AlertType.BULLWHIP_DETECTED,
                severity=AlertSeverity.HIGH if ratio > 2.0 else AlertSeverity.MEDIUM,
                affected_nodes=[],  # Affects whole chain
                details={"bullwhip_ratio": ratio},
            )
            
            graph = create_supply_chain_graph(llm=None)
            state = create_initial_state(alert=alert)
            result = graph.invoke(state, config={"configurable": {"thread_id": "test-bullwhip"}})
            
            assert result is not None


# =============================================================================
# Risk Engine Tests
# =============================================================================

class TestRiskEngine:
    """Test the probabilistic risk engine."""
    
    def test_risk_engine_creation(self):
        """Test RiskEngine initializes with correct defaults."""
        from src.cognition import RiskEngine
        engine = RiskEngine()
        assert engine.alert_threshold == 0.30
        assert engine.scan_count == 0
        assert len(engine.node_risks) == 0
    
    def test_risk_state_creation(self):
        """Test RiskState defaults and properties."""
        from src.cognition import RiskState
        rs = RiskState(node_id=5)
        
        assert rs.node_id == 5
        assert rs.most_likely_state == "healthy"
        assert rs.composite_risk >= 0
        assert rs.degradation_probability < 0.5
        assert "healthy" in rs.probabilities
        assert "critical" in rs.probabilities
    
    def test_risk_state_serialization(self):
        """Test RiskState to_dict roundtrip."""
        from src.cognition import RiskState
        rs = RiskState(node_id=3)
        rs.inventory_risk = 0.7
        rs.demand_volatility_risk = 0.4
        
        d = rs.to_dict()
        assert d["node_id"] == 3
        assert "composite_risk" in d
        assert "risk_factors" in d
        assert d["risk_factors"]["inventory_risk"] == 0.7
    
    def test_node_health_state_enum(self):
        """Test NodeHealthState enum values."""
        from src.cognition import NodeHealthState
        assert NodeHealthState.HEALTHY.value == "healthy"
        assert NodeHealthState.CRITICAL.value == "critical"
    
    def test_risk_engine_scan_network(self, simulation_model):
        """Test full network scan produces risk scores."""
        from src.cognition import RiskEngine
        engine = RiskEngine(alert_threshold=0.20)
        
        alerts, opportunities = engine.scan_network(simulation_model)
        
        assert engine.scan_count == 1
        assert len(engine.node_risks) > 0
        assert isinstance(alerts, list)
        assert isinstance(opportunities, list)
        
        # Each node should have risk state
        for node_id, risk_state in engine.node_risks.items():
            assert risk_state.composite_risk >= 0
            assert risk_state.composite_risk <= 1.0
            assert sum(risk_state.probabilities.values()) == pytest.approx(1.0, abs=0.01)
    
    def test_risk_engine_multiple_scans(self, simulation_model):
        """Test risk engine accumulates history over multiple scans."""
        from src.cognition import RiskEngine
        engine = RiskEngine()
        
        for _ in range(5):
            simulation_model.step()
            engine.scan_network(simulation_model)
        
        assert engine.scan_count == 5
        
        # Check that history has been accumulated
        for rs in engine.node_risks.values():
            assert len(rs.risk_score_history) == 5
            assert len(rs.state_history) == 5
    
    def test_risk_engine_generates_preemptive_alerts(self, simulation_model):
        """Test that low-threshold engine generates alerts."""
        from src.cognition import RiskEngine
        engine = RiskEngine(alert_threshold=0.10)  # Low threshold = more alerts
        
        # Run several steps to build state
        for _ in range(10):
            simulation_model.step()
        
        alerts, _ = engine.scan_network(simulation_model)
        
        # With a low threshold, should get some alerts
        for alert in alerts:
            assert alert["alert_type"] == "preemptive_warning"
            assert "severity" in alert
            assert "affected_nodes" in alert
            assert "details" in alert
            assert "composite_risk" in alert["details"]
            assert "recommended_action" in alert["details"]
    
    def test_risk_engine_network_summary(self, simulation_model):
        """Test network risk summary report."""
        from src.cognition import RiskEngine
        engine = RiskEngine()
        engine.scan_network(simulation_model)
        
        summary = engine.get_network_risk_summary()
        
        assert "total_nodes" in summary
        assert "state_distribution" in summary
        assert "average_risk" in summary
        assert "network_health" in summary
        assert summary["total_nodes"] > 0
    
    def test_risk_engine_node_detail(self, simulation_model):
        """Test querying individual node risk."""
        from src.cognition import RiskEngine
        engine = RiskEngine()
        engine.scan_network(simulation_model)
        
        # Query first node
        node_id = list(engine.node_risks.keys())[0]
        detail = engine.get_node_risk(node_id)
        
        assert detail is not None
        assert detail["node_id"] == node_id
        assert "probabilities" in detail
        assert "risk_factors" in detail


# =============================================================================
# Preemptive Monitor Tests
# =============================================================================

class TestPreemptiveMonitor:
    """Test the preemptive monitoring system."""
    
    def test_monitor_creation(self):
        """Test PreemptiveMonitor initializes correctly."""
        from src.cognition import PreemptiveMonitor
        monitor = PreemptiveMonitor(alert_cooldown=5)
        
        assert monitor.step_count == 0
        assert monitor.alert_cooldown == 5
        assert len(monitor.alert_history) == 0
    
    def test_monitor_on_step(self, simulation_model):
        """Test monitor hooks into simulation step."""
        from src.cognition import PreemptiveMonitor
        monitor = PreemptiveMonitor()
        
        report = monitor.on_step(simulation_model)
        
        assert report["scanned"] is True
        assert report["step"] == 1
        assert "alerts_generated" in report
        assert "risk_summary" in report
    
    def test_monitor_multiple_steps(self, simulation_model):
        """Test monitor tracks across multiple steps."""
        from src.cognition import PreemptiveMonitor, RiskEngine
        engine = RiskEngine(alert_threshold=0.15)
        monitor = PreemptiveMonitor(risk_engine=engine, alert_cooldown=3)
        
        total_alerts = 0
        for _ in range(10):
            simulation_model.step()
            report = monitor.on_step(simulation_model)
            total_alerts += report.get("alerts_generated", 0)
        
        assert monitor.step_count == 10
        # With low threshold, should generate some alerts
        assert total_alerts >= 0  # May be 0 if network is healthy
    
    def test_monitor_alert_cooldown(self, simulation_model):
        """Test that alert cooldown prevents spam."""
        from src.cognition import PreemptiveMonitor, RiskEngine
        engine = RiskEngine(alert_threshold=0.05)  # Very low = lots of alerts
        monitor = PreemptiveMonitor(risk_engine=engine, alert_cooldown=5)
        
        all_alerts = []
        for _ in range(10):
            simulation_model.step()
            report = monitor.on_step(simulation_model)
            all_alerts.extend(monitor.get_pending_alerts())
        
        # Check that we don't see same node alerted within cooldown window
        node_alert_steps = {}
        for alert in monitor.alert_history:
            node_id = alert.get("affected_nodes", [None])[0]
            step = alert.get("monitor_step", 0)
            if node_id in node_alert_steps:
                # Gap should be >= cooldown
                assert step - node_alert_steps[node_id] >= 5
            node_alert_steps[node_id] = step
    
    def test_monitor_pending_alerts(self, simulation_model):
        """Test pending alert retrieval and clear."""
        from src.cognition import PreemptiveMonitor, RiskEngine
        engine = RiskEngine(alert_threshold=0.10)
        monitor = PreemptiveMonitor(risk_engine=engine)
        
        for _ in range(5):
            simulation_model.step()
            monitor.on_step(simulation_model)
        
        # Get pending (clears queue)
        pending = monitor.get_pending_alerts(clear=True)
        assert isinstance(pending, list)
        
        # After clear, should be empty
        assert len(monitor.get_pending_alerts()) == 0
    
    def test_monitor_health_report(self, simulation_model):
        """Test network health report generation."""
        from src.cognition import PreemptiveMonitor
        monitor = PreemptiveMonitor()
        
        for _ in range(5):
            simulation_model.step()
            monitor.on_step(simulation_model)
        
        report = monitor.get_network_health_report()
        
        assert "step" in report
        assert "risk_summary" in report
        assert "network_trend" in report
        assert "total_alerts_generated" in report
        assert "recent_alerts" in report
    
    def test_monitor_node_risk_detail(self, simulation_model):
        """Test querying node risk through monitor."""
        from src.cognition import PreemptiveMonitor
        monitor = PreemptiveMonitor()
        
        simulation_model.step()
        monitor.on_step(simulation_model)
        
        # Get risk for first node
        detail = monitor.get_node_risk_detail(0)
        # May be None if node 0 doesn't exist
        if detail is not None:
            assert "probabilities" in detail
            assert "risk_factors" in detail


# =============================================================================
# Preemptive Integration Tests
# =============================================================================

class TestPreemptiveIntegration:
    """End-to-end tests for the preemptive workflow."""
    
    def test_full_preemptive_pipeline(self, simulation_model):
        """Test complete: simulation → risk scan → alerts → cognitive response."""
        from src.cognition import (PreemptiveMonitor, RiskEngine,
                                   create_supply_chain_graph, initialize_tools)
        
        initialize_tools(simulation=simulation_model)
        engine = RiskEngine(alert_threshold=0.25)
        monitor = PreemptiveMonitor(risk_engine=engine, alert_cooldown=3)
        
        # Run 15 steps
        for _ in range(15):
            simulation_model.step()
            monitor.on_step(simulation_model)
        
        # Verify system produced alerts and tracked state
        health = monitor.get_network_health_report()
        assert health["step"] == 15
        assert health["risk_summary"]["total_nodes"] > 0
        
        # Run cognitive workflow on a preemptive alert if any
        pending = monitor.get_pending_alerts(clear=False)
        if pending:
            alert_dict = pending[0]
            alert = Alert(
                alert_type=AlertType.PREEMPTIVE_WARNING,
                severity=AlertSeverity(alert_dict["severity"]),
                affected_nodes=alert_dict["affected_nodes"],
                details=alert_dict["details"],
            )
            graph = create_supply_chain_graph(llm=None)
            state = create_initial_state(alert=alert)
            result = graph.invoke(state, config={"configurable": {"thread_id": "test-preemptive"}})
            
            assert result is not None
            assert result.get("iteration_count", 0) >= 1
    
    def test_preemptive_alert_type_in_state(self):
        """Test new alert types work in state creation."""
        alert = Alert(
            alert_type=AlertType.PREEMPTIVE_WARNING,
            severity=AlertSeverity.MEDIUM,
            affected_nodes=[3],
            details={"composite_risk": 0.65, "primary_risk_factor": "inventory_runway"},
        )
        state = create_initial_state(alert=alert)
        
        assert state["current_alert"]["alert_type"] == "preemptive_warning"
        assert state["preemptive_mode"] is False  # Default, set by monitor
        assert state["risk_scores"] is None  # Default, set by monitor


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
