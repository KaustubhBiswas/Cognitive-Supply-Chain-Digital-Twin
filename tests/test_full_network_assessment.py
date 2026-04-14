"""Tests for full-network vulnerability assessment workflow."""

from src.cognition import Alert, AlertSeverity, AlertType
from src.integration.session import SessionManager


class _DummyGraph:
    def __init__(self):
        self.last_state = None

    def invoke(self, state, config=None):
        self.last_state = state
        return {
            "messages": [],
            "recommendations": [],
            "execution_log": [],
            "plan_status": "completed",
            "replan_count": 0,
        }


def test_run_full_network_assessment_builds_coverage_context(monkeypatch):
    mgr = SessionManager(seed=42)

    fake_alerts = [
        {
            "alert_type": "preemptive_warning",
            "severity": "high",
            "affected_nodes": [1],
            "details": {"composite_risk": 0.82},
            "alert_id": "a1",
        },
        {
            "alert_type": "preemptive_warning",
            "severity": "medium",
            "affected_nodes": [3],
            "details": {"composite_risk": 0.66},
            "alert_id": "a2",
        },
    ]

    monkeypatch.setattr(mgr.risk_engine, "scan_network", lambda model: (fake_alerts, []))

    captured = {}

    def _fake_run(alert):
        captured["alert"] = alert
        return {"messages": [], "recommendations": [], "execution_log": []}

    monkeypatch.setattr(mgr, "run_cognitive_workflow", _fake_run)

    result = mgr.run_full_network_assessment(
        alert_type=AlertType.PREEMPTIVE_WARNING,
        severity=AlertSeverity.HIGH,
    )

    assert "coverage_context" in result
    coverage = result["coverage_context"]
    assert coverage["scan_scope"] == "full_network"
    assert coverage["coverage_rate"] == 1.0
    assert coverage["vulnerability_count"] == 2
    assert set(coverage["vulnerable_node_ids"]) == {1, 3}

    emitted_alert = captured["alert"]
    assert emitted_alert.details["full_network_assessment"] is True
    assert emitted_alert.details["coverage_context"]["scan_scope"] == "full_network"


def test_run_cognitive_workflow_receives_coverage_context_state(monkeypatch):
    mgr = SessionManager(seed=123)
    dummy_graph = _DummyGraph()
    mgr.graph = dummy_graph
    all_nodes = sorted(int(agent.unique_id) for agent in mgr.model.agents)

    alert = Alert(
        alert_type=AlertType.PREEMPTIVE_WARNING,
        severity=AlertSeverity.HIGH,
        affected_nodes=[0, 1, 2],
        details={
            "objective": "Full-network vulnerability pass",
            "coverage_context": {
                "scan_scope": "full_network",
                "total_nodes_scanned": 3,
                "vulnerable_node_ids": [1, 2],
                "vulnerabilities_by_node": {
                    "1": [{"severity": "high"}],
                    "2": [{"severity": "medium"}],
                },
                "coverage_rate": 1.0,
                "vulnerability_count": 2,
            },
        },
    )

    _ = mgr.run_cognitive_workflow(alert)

    assert dummy_graph.last_state is not None
    assert dummy_graph.last_state["scan_scope"] == "full_network"
    assert dummy_graph.last_state["total_nodes_scanned"] == len(all_nodes)
    assert set(dummy_graph.last_state["vulnerable_node_ids"]) == {1, 2}
    assert dummy_graph.last_state["coverage_rate"] == 1.0
    assert dummy_graph.last_state["vulnerability_count"] == 2


def test_run_cognitive_workflow_custom_nodes_coverage_pass_through(monkeypatch):
    mgr = SessionManager(seed=7)
    dummy_graph = _DummyGraph()
    mgr.graph = dummy_graph

    alert = Alert(
        alert_type=AlertType.DEMAND_SPIKE,
        severity=AlertSeverity.MEDIUM,
        affected_nodes=[4, 6],
        details={
            "objective": "Custom-node assessment",
            "coverage_context": {
                "scan_scope": "custom_nodes",
                "total_nodes_scanned": 2,
                "vulnerable_node_ids": [4, 6],
                "vulnerabilities_by_node": {},
                "coverage_rate": 1.0,
                "vulnerability_count": 0,
            },
        },
    )

    result = mgr.run_cognitive_workflow(alert)

    assert dummy_graph.last_state is not None
    assert dummy_graph.last_state["scan_scope"] == "custom_nodes"
    assert dummy_graph.last_state["total_nodes_scanned"] == 2
    assert set(dummy_graph.last_state["vulnerable_node_ids"]) == {4, 6}
    assert "coverage_context" in result
    assert result["coverage_context"]["scan_scope"] == "custom_nodes"


def test_run_cognitive_workflow_normalizes_malformed_full_network_coverage(monkeypatch):
    mgr = SessionManager(seed=17)
    dummy_graph = _DummyGraph()
    mgr.graph = dummy_graph

    all_nodes = sorted(int(agent.unique_id) for agent in mgr.model.agents)
    assert len(all_nodes) >= 2
    n1, n2 = all_nodes[0], all_nodes[1]

    alert = Alert(
        alert_type=AlertType.PREEMPTIVE_WARNING,
        severity=AlertSeverity.HIGH,
        affected_nodes=[n1, n2],
        details={
            "objective": "Normalize malformed full-network payload",
            "coverage_context": {
                "scan_scope": "full_network",
                "total_nodes_scanned": 1,
                "vulnerable_node_ids": [n1, "bad", 999999],
                "vulnerabilities_by_node": {
                    str(n1): {"severity": "high"},
                    str(n2): [{"severity": "medium"}, "invalid_entry"],
                    "not-a-node": [{"severity": "low"}],
                },
                "coverage_rate": -0.4,
                "vulnerability_count": -3,
            },
        },
    )

    result = mgr.run_cognitive_workflow(alert)
    state = dummy_graph.last_state

    assert state is not None
    assert state["scan_scope"] == "full_network"
    assert state["total_nodes_scanned"] == len(all_nodes)
    assert state["coverage_rate"] == 1.0
    assert set(state["vulnerable_node_ids"]).issubset(set(all_nodes))
    assert 999999 not in state["vulnerable_node_ids"]
    assert state["vulnerability_count"] == sum(
        len(v) for v in state["vulnerabilities_by_node"].values()
    )

    assert "coverage_context" in result
    assert result["coverage_context"]["scan_scope"] == "full_network"
    assert result["coverage_context"]["coverage_rate"] == 1.0


def test_full_network_assessment_coverage_invariants(monkeypatch):
    mgr = SessionManager(seed=99)
    fake_alerts = [
        {
            "alert_type": "preemptive_warning",
            "severity": "critical",
            "affected_nodes": [2],
            "details": {"composite_risk": 0.91},
            "alert_id": "x1",
        }
    ]
    monkeypatch.setattr(mgr.risk_engine, "scan_network", lambda model: (fake_alerts, []))

    captured = {}

    def _fake_run(alert):
        captured["alert"] = alert
        return {"messages": [], "recommendations": [], "execution_log": []}

    monkeypatch.setattr(mgr, "run_cognitive_workflow", _fake_run)

    result = mgr.run_full_network_assessment()
    coverage = result["coverage_context"]

    assert coverage["scan_scope"] == "full_network"
    assert coverage["coverage_rate"] == 1.0
    assert coverage["total_nodes_scanned"] == len(captured["alert"].affected_nodes)
    assert set(coverage["vulnerable_node_ids"]).issubset(set(captured["alert"].affected_nodes))
    assert coverage["vulnerability_count"] >= len(coverage["vulnerable_node_ids"])
