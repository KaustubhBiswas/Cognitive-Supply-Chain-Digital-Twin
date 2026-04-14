"""Tests for governance policy decisions and Sprint 4 adaptation logic."""

from src.cognition.governance import (decide_rollout_execution,
                                      evaluate_recommendation_policy,
                                      tune_policy_thresholds)


def test_auto_approve_low_risk_high_confidence():
    rec = {
        "recommendation_type": "adjust_reorder_point",
        "target_nodes": [1, 2],
        "confidence": 0.82,
    }

    meta = evaluate_recommendation_policy(rec, alert_severity="medium")

    assert meta["decision"] == "auto_approve"
    assert meta["requires_human"] is False
    assert meta["risk_band"] == "low"


def test_high_risk_requires_human_review():
    rec = {
        "recommendation_type": "change_supplier",
        "target_nodes": [4],
        "confidence": 0.95,
    }

    meta = evaluate_recommendation_policy(rec, alert_severity="low")

    assert meta["decision"] == "human_review"
    assert meta["requires_human"] is True
    assert "High-risk" in meta["reason"]


def test_critical_low_confidence_requires_human_review():
    rec = {
        "recommendation_type": "expedite_order",
        "target_nodes": [3],
        "confidence": 0.62,
    }

    meta = evaluate_recommendation_policy(rec, alert_severity="critical")

    assert meta["decision"] == "human_review"
    assert meta["requires_human"] is True
    assert "Critical alert" in meta["reason"]


def test_env_thresholds_change_decision(monkeypatch):
    monkeypatch.setenv("GOV_BASELINE_MIN_CONFIDENCE", "0.90")

    rec = {
        "recommendation_type": "adjust_reorder_point",
        "target_nodes": [1],
        "confidence": 0.82,
    }

    meta = evaluate_recommendation_policy(rec, alert_severity="low")

    assert meta["decision"] == "human_review"
    assert "baseline automation threshold" in meta["reason"]


def test_tune_policy_thresholds_tightens_on_high_overrides():
    current = {
        "critical_min_confidence": 0.75,
        "medium_risk_min_confidence": 0.65,
        "baseline_min_confidence": 0.55,
    }
    history = [
        {
            "autonomous_completion_rate": 0.25,
            "human_override_rate": 0.45,
            "blocked_step_rate": 0.30,
            "plan_completion_rate": 0.48,
        }
        for _ in range(10)
    ]

    result = tune_policy_thresholds(current, history, min_samples=8)

    assert result["changed"] is True
    updated = result["updated_thresholds"]
    assert updated["baseline_min_confidence"] >= current["baseline_min_confidence"]
    assert updated["medium_risk_min_confidence"] >= current["medium_risk_min_confidence"]
    assert updated["critical_min_confidence"] >= current["critical_min_confidence"]


def test_tune_policy_thresholds_relaxes_on_stable_low_override():
    current = {
        "critical_min_confidence": 0.82,
        "medium_risk_min_confidence": 0.72,
        "baseline_min_confidence": 0.62,
    }
    history = [
        {
            "autonomous_completion_rate": 0.20,
            "human_override_rate": 0.05,
            "blocked_step_rate": 0.05,
            "plan_completion_rate": 0.80,
        }
        for _ in range(10)
    ]

    result = tune_policy_thresholds(current, history, min_samples=8)

    assert result["changed"] is True
    updated = result["updated_thresholds"]
    assert updated["baseline_min_confidence"] < current["baseline_min_confidence"]
    assert updated["medium_risk_min_confidence"] < current["medium_risk_min_confidence"]
    assert updated["baseline_min_confidence"] <= updated["medium_risk_min_confidence"] <= updated["critical_min_confidence"]


def test_rollout_shadow_never_executes_auto_approve():
    policy_meta = {
        "decision": "auto_approve",
        "risk_band": "low",
    }
    rollout = decide_rollout_execution(policy_meta, rollout_mode="shadow", autonomy_enabled=True)

    assert rollout["execute_now"] is False
    assert rollout["status"] == "pending"
    assert "Shadow mode" in rollout["reason"]


def test_rollout_constrained_blocks_non_low_risk():
    policy_meta = {
        "decision": "auto_approve",
        "risk_band": "medium",
    }
    rollout = decide_rollout_execution(policy_meta, rollout_mode="constrained_auto", autonomy_enabled=True)

    assert rollout["execute_now"] is False
    assert rollout["status"] == "pending"
    assert "low-risk" in rollout["reason"]


def test_rollout_full_executes_policy_auto_approve():
    policy_meta = {
        "decision": "auto_approve",
        "risk_band": "medium",
    }
    rollout = decide_rollout_execution(policy_meta, rollout_mode="full_auto", autonomy_enabled=True)

    assert rollout["execute_now"] is True
    assert rollout["status"] == "approved"


def test_rollout_disabled_autonomy_blocks_execution():
    policy_meta = {
        "decision": "auto_approve",
        "risk_band": "low",
    }
    rollout = decide_rollout_execution(policy_meta, rollout_mode="full_auto", autonomy_enabled=False)

    assert rollout["execute_now"] is False
    assert rollout["status"] == "pending"
    assert "disabled" in rollout["reason"]
