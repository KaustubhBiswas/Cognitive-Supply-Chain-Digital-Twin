"""Governance policies for recommendation approval and explainability."""

from __future__ import annotations

import os
from typing import Any, Dict, List

SEVERITY_SCORE = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}


RECOMMENDATION_RISK = {
    "adjust_reorder_point": "low",
    "adjust_order_quantity": "low",
    "adjust_safety_stock": "low",
    "increase_safety_stock": "low",
    "redistribute_inventory": "medium",
    "expedite_order": "medium",
    "increase_capacity": "medium",
    "change_supplier": "high",
    "no_action": "low",
}


RISK_SCORE = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_policy_thresholds() -> Dict[str, float]:
    """Load policy thresholds from environment with safe defaults."""
    return {
        "critical_min_confidence": _safe_float(os.getenv("GOV_CRITICAL_MIN_CONFIDENCE", 0.75), 0.75),
        "medium_risk_min_confidence": _safe_float(os.getenv("GOV_MEDIUM_RISK_MIN_CONFIDENCE", 0.65), 0.65),
        "baseline_min_confidence": _safe_float(os.getenv("GOV_BASELINE_MIN_CONFIDENCE", 0.55), 0.55),
    }


def get_default_policy_thresholds() -> Dict[str, float]:
    """Public accessor for active governance thresholds."""
    return _load_policy_thresholds().copy()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def tune_policy_thresholds(
    current_thresholds: Dict[str, float],
    kpi_history: List[Dict[str, Any]],
    min_samples: int = 8,
) -> Dict[str, Any]:
    """Tune policy thresholds from recent KPI outcomes.

    Returns a dict with fields: updated_thresholds, changed, reasons.
    """
    if len(kpi_history) < max(2, int(min_samples)):
        return {
            "updated_thresholds": current_thresholds.copy(),
            "changed": False,
            "reasons": ["Insufficient KPI history for tuning."],
        }

    window = kpi_history[-max(5, int(min_samples)) :]

    avg_auto = sum(float(item.get("autonomous_completion_rate", 0.0)) for item in window) / len(window)
    avg_override = sum(float(item.get("human_override_rate", 0.0)) for item in window) / len(window)
    avg_blocked = sum(float(item.get("blocked_step_rate", 0.0)) for item in window) / len(window)
    avg_completion = sum(float(item.get("plan_completion_rate", 0.0)) for item in window) / len(window)

    updated = {
        "critical_min_confidence": _safe_float(current_thresholds.get("critical_min_confidence", 0.75), 0.75),
        "medium_risk_min_confidence": _safe_float(current_thresholds.get("medium_risk_min_confidence", 0.65), 0.65),
        "baseline_min_confidence": _safe_float(current_thresholds.get("baseline_min_confidence", 0.55), 0.55),
    }
    reasons: List[str] = []

    # Strictness up when human overrides or blocked executions are high.
    if avg_override > 0.35 or avg_blocked > 0.25:
        updated["baseline_min_confidence"] += 0.02
        updated["medium_risk_min_confidence"] += 0.03
        updated["critical_min_confidence"] += 0.02
        reasons.append("Increased strictness due to high override/blocked rates.")

    # Strictness down when automation is low but outcomes remain stable.
    if avg_auto < 0.30 and avg_override < 0.15 and avg_blocked < 0.12 and avg_completion > 0.65:
        updated["baseline_min_confidence"] -= 0.02
        updated["medium_risk_min_confidence"] -= 0.02
        reasons.append("Relaxed thresholds to improve safe autonomy throughput.")

    # Recovery guard: low plan completion should tighten medium/high thresholds.
    if avg_completion < 0.50:
        updated["medium_risk_min_confidence"] += 0.02
        updated["critical_min_confidence"] += 0.02
        reasons.append("Tightened medium/critical thresholds due to low completion.")

    # Clamp and enforce ordering: baseline <= medium <= critical.
    updated["baseline_min_confidence"] = _clamp(updated["baseline_min_confidence"], 0.40, 0.90)
    updated["medium_risk_min_confidence"] = _clamp(updated["medium_risk_min_confidence"], 0.50, 0.95)
    updated["critical_min_confidence"] = _clamp(updated["critical_min_confidence"], 0.60, 0.98)

    updated["medium_risk_min_confidence"] = max(
        updated["medium_risk_min_confidence"],
        updated["baseline_min_confidence"],
    )
    updated["critical_min_confidence"] = max(
        updated["critical_min_confidence"],
        updated["medium_risk_min_confidence"],
    )

    changed = any(abs(updated[k] - _safe_float(current_thresholds.get(k, updated[k]), updated[k])) > 1e-9 for k in updated)

    return {
        "updated_thresholds": updated,
        "changed": changed,
        "reasons": reasons or ["No threshold adaptation required."],
    }


def decide_rollout_execution(
    policy_meta: Dict[str, Any],
    rollout_mode: str,
    autonomy_enabled: bool = True,
) -> Dict[str, Any]:
    """Decide whether to execute now under the configured rollout mode.

    Modes:
    - shadow: never execute automatically (observe-only)
    - constrained_auto: execute only low-risk policy-approved actions
    - full_auto: execute all policy-approved actions
    """
    mode = str(rollout_mode or "constrained_auto").strip().lower()
    decision = str(policy_meta.get("decision", "human_review"))
    risk_band = str(policy_meta.get("risk_band", "medium")).lower()

    if not autonomy_enabled:
        return {
            "execute_now": False,
            "status": "pending",
            "reason": "Autonomy disabled by operator rollback switch.",
            "mode": mode,
        }

    if decision != "auto_approve":
        return {
            "execute_now": False,
            "status": "pending",
            "reason": "Policy requires human review.",
            "mode": mode,
        }

    if mode == "shadow":
        return {
            "execute_now": False,
            "status": "pending",
            "reason": "Shadow mode: action logged but not auto-executed.",
            "mode": mode,
        }

    if mode == "constrained_auto":
        if risk_band != "low":
            return {
                "execute_now": False,
                "status": "pending",
                "reason": "Constrained mode permits auto-execution for low-risk only.",
                "mode": mode,
            }
        return {
            "execute_now": True,
            "status": "approved",
            "reason": "Constrained mode approved low-risk auto action.",
            "mode": mode,
        }

    # full_auto (and unknown fallback) executes policy-approved actions.
    return {
        "execute_now": True,
        "status": "approved",
        "reason": "Full-auto mode executing policy-approved action.",
        "mode": "full_auto" if mode == "full_auto" else mode,
    }


def _estimate_impact(rec_type: str, target_nodes: List[Any]) -> Dict[str, float]:
    """Return lightweight heuristic impact estimates for explainability cards."""
    node_count = max(1, len(target_nodes or []))

    if rec_type in {"adjust_reorder_point", "adjust_order_quantity", "adjust_safety_stock", "increase_safety_stock"}:
        return {
            "inventory_delta_pct": round(2.0 + 1.2 * node_count, 2),
            "backlog_delta_pct": round(-1.5 - 0.8 * node_count, 2),
        }
    if rec_type == "expedite_order":
        return {
            "inventory_delta_pct": round(1.0 + 0.6 * node_count, 2),
            "backlog_delta_pct": round(-3.5 - 0.7 * node_count, 2),
        }
    if rec_type == "redistribute_inventory":
        return {
            "inventory_delta_pct": round(0.5 + 0.5 * node_count, 2),
            "backlog_delta_pct": round(-2.0 - 0.6 * node_count, 2),
        }
    if rec_type == "increase_capacity":
        return {
            "inventory_delta_pct": round(3.0 + 1.0 * node_count, 2),
            "backlog_delta_pct": round(-2.5 - 0.8 * node_count, 2),
        }

    return {
        "inventory_delta_pct": 0.0,
        "backlog_delta_pct": 0.0,
    }


def evaluate_recommendation_policy(
    recommendation: Dict[str, Any],
    alert_severity: str,
    policy_thresholds: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Evaluate recommendation governance policy and produce explainability metadata."""
    rec_type = str(recommendation.get("recommendation_type", "unknown"))
    target_nodes = recommendation.get("target_nodes", []) or []
    confidence = _safe_float(recommendation.get("confidence", 0.0), default=0.0)

    severity_score = SEVERITY_SCORE.get(str(alert_severity).lower(), 1)
    risk_band = RECOMMENDATION_RISK.get(rec_type, "medium")
    risk_score = RISK_SCORE.get(risk_band, 1)
    thresholds = policy_thresholds or _load_policy_thresholds()
    critical_min_confidence = _safe_float(thresholds.get("critical_min_confidence", 0.75), 0.75)
    medium_risk_min_confidence = _safe_float(thresholds.get("medium_risk_min_confidence", 0.65), 0.65)
    baseline_min_confidence = _safe_float(thresholds.get("baseline_min_confidence", 0.55), 0.55)

    # Auto-approval guardrails:
    # 1) No high-risk action auto-approved.
    # 2) Critical alerts require confidence >= 0.75.
    # 3) Medium-risk actions require confidence >= 0.65.
    decision = "human_review"
    reason = "Escalated for human validation due to policy guardrails."

    if risk_score == 2:
        decision = "human_review"
        reason = "High-risk action requires explicit human approval."
    elif severity_score >= 3 and confidence < critical_min_confidence:
        decision = "human_review"
        reason = (
            "Critical alert requires confidence >= "
            f"{critical_min_confidence:.2f} for automation."
        )
    elif risk_score == 1 and confidence < medium_risk_min_confidence:
        decision = "human_review"
        reason = (
            "Medium-risk action confidence below auto-approval threshold "
            f"({medium_risk_min_confidence:.2f})."
        )
    elif confidence < baseline_min_confidence:
        decision = "human_review"
        reason = (
            "Confidence below baseline automation threshold "
            f"({baseline_min_confidence:.2f})."
        )
    else:
        decision = "auto_approve"
        reason = "Policy thresholds satisfied; safe for autonomous execution."

    return {
        "decision": decision,
        "reason": reason,
        "risk_band": risk_band,
        "confidence": confidence,
        "policy_thresholds": {
            "critical_min_confidence": critical_min_confidence,
            "medium_risk_min_confidence": medium_risk_min_confidence,
            "baseline_min_confidence": baseline_min_confidence,
        },
        "estimated_impact": _estimate_impact(rec_type, target_nodes),
        "requires_human": decision != "auto_approve",
    }
