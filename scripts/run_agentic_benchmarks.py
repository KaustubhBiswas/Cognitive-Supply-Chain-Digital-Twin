"""Deterministic agentic benchmark harness.

Runs a fixed scenario suite and emits per-scenario and aggregate metrics.

Usage:
    python scripts/run_agentic_benchmarks.py --trials 4 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cognition import Alert, AlertSeverity, AlertType, create_initial_state
from src.cognition.graph import FallbackGraph

try:
    from src.integration import SessionManager
except ModuleNotFoundError as e:  # pragma: no cover - environment dependent
    missing = getattr(e, "name", "unknown_dependency")
    raise SystemExit(
        "Cannot run benchmark harness because a runtime dependency is missing: "
        f"{missing}. Install project dependencies first, e.g. `pip install -e .` "
        "(and optional extras as needed)."
    )


def _node_sample(mgr: SessionManager, rng: random.Random, k: int = 2) -> List[int]:
    node_ids = list(mgr.supply_data.node_types.keys())
    if not node_ids:
        return [0]
    return rng.sample(node_ids, min(k, len(node_ids)))


def _invoke_graph_with_serialization_guard(mgr: SessionManager, state: Dict[str, Any]) -> Dict[str, Any]:
    config = {"configurable": {"thread_id": f"bench-{datetime.now().timestamp()}"}}
    if isinstance(mgr.graph, FallbackGraph):
        return mgr.graph.invoke(state)
    try:
        return mgr.graph.invoke(state, config=config)
    except TypeError as e:
        if "msgpack serializable" not in str(e):
            raise
        return FallbackGraph(llm=mgr.llm).invoke(state)


def _run_tool_failure_scenario(mgr: SessionManager, rng: random.Random) -> Dict[str, Any]:
    alert = Alert(
        alert_type=AlertType.SUPPLY_DISRUPTION,
        severity=AlertSeverity.HIGH,
        affected_nodes=_node_sample(mgr, rng, k=2),
        details={"objective": "Recover service levels while minimizing downstream stockouts."},
    )

    state = create_initial_state(alert=alert, objective="Recover from injected tool failure and continue safely.")
    state["plan_steps"] = [
        {
            "step_id": "F1",
            "title": "Injected invalid owner step",
            "description": "Intentional fault injection for recovery testing.",
            "owner": "invalid_owner",
            "status": "pending",
            "required_tools": [],
            "success_criteria": "System should detect and recover by replanning.",
        }
    ]
    state["plan_status"] = "in_progress"
    state["current_plan_step"] = 0
    payload = mgr._to_serializable(state)
    return _invoke_graph_with_serialization_guard(mgr, payload)


def _run_standard_scenario(mgr: SessionManager, scenario: str, rng: random.Random) -> Dict[str, Any]:
    nodes = _node_sample(mgr, rng, k=3)

    if scenario == "normal_operations":
        alert = Alert(
            alert_type=AlertType.DEMAND_SPIKE,
            severity=AlertSeverity.MEDIUM,
            affected_nodes=nodes,
            details={"objective": "Stabilize network with minimal coordination overhead."},
        )
    elif scenario == "disruption":
        alert = Alert(
            alert_type=AlertType.SUPPLY_DISRUPTION,
            severity=AlertSeverity.CRITICAL,
            affected_nodes=nodes,
            details={"objective": "Contain disruption while preserving service levels."},
        )
    elif scenario == "stale_context":
        alert = Alert(
            alert_type=AlertType.LEAD_TIME_RISK,
            severity=AlertSeverity.HIGH,
            affected_nodes=nodes,
            details={
                "objective": (
                    "Prior context says lead times are 2 days, but treat this as stale and "
                    "re-validate before committing.")
            },
        )
    elif scenario == "conflicting_goals":
        alert = Alert(
            alert_type=AlertType.INVENTORY_IMBALANCE,
            severity=AlertSeverity.HIGH,
            affected_nodes=nodes,
            details={
                "objective": (
                    "Reduce backlog aggressively while simultaneously minimizing all inventory increases."
                )
            },
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return mgr.run_cognitive_workflow(alert)


def _extract_run_metrics(result: Dict[str, Any], action_slice: List[Any]) -> Dict[str, Any]:
    log = result.get("execution_log", []) or []
    blocked = any(str(evt.get("event", "")).lower() == "step_blocked" for evt in log)
    replans = int(result.get("replan_count", 0) or 0)
    plan_status = str(result.get("plan_status", "unknown")).lower()

    auto_approved = sum(1 for a in action_slice if getattr(a, "status", "") == "approved")
    pending = sum(1 for a in action_slice if getattr(a, "status", "") == "pending")

    return {
        "plan_completed": plan_status == "completed",
        "plan_status": plan_status,
        "blocked": blocked,
        "replans": replans,
        "auto_approved": auto_approved,
        "pending": pending,
    }


def run_benchmarks(trials: int, seed: int, mode: str, autonomy_enabled: bool) -> Dict[str, Any]:
    rng = random.Random(seed)
    mgr = SessionManager(seed=seed)
    mgr.set_rollout_mode(mode)
    mgr.set_autonomy_enabled(autonomy_enabled)

    scenarios = [
        "normal_operations",
        "disruption",
        "stale_context",
        "conflicting_goals",
        "tool_failure_injection",
    ]

    records: List[Dict[str, Any]] = []
    for scenario in scenarios:
        for trial in range(max(1, trials)):
            before_idx = len(mgr.action_queue)
            if scenario == "tool_failure_injection":
                result = _run_tool_failure_scenario(mgr, rng)
            else:
                result = _run_standard_scenario(mgr, scenario, rng)
            metrics = _extract_run_metrics(result, mgr.action_queue[before_idx:])
            records.append(
                {
                    "scenario": scenario,
                    "trial": trial + 1,
                    **metrics,
                }
            )
            mgr.step()

    total = len(records)
    completed = sum(1 for r in records if r["plan_completed"])
    blocked = sum(1 for r in records if r["blocked"])
    auto_approved = sum(int(r["auto_approved"]) for r in records)
    pending = sum(int(r["pending"]) for r in records)

    failure_records = [r for r in records if r["scenario"] == "tool_failure_injection"]
    recovered = sum(1 for r in failure_records if (not r["blocked"]) or r["replans"] > 0)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "trials_per_scenario": max(1, trials),
        "rollout_mode": mode,
        "autonomy_enabled": autonomy_enabled,
        "scenario_count": len(scenarios),
        "run_count": total,
        "plan_success_rate": (completed / total) if total else 0.0,
        "blocked_step_rate": (blocked / total) if total else 0.0,
        "autonomous_completion_rate": (auto_approved / max(1, auto_approved + pending)),
        "failure_recovery_rate": (recovered / max(1, len(failure_records))),
        "records": records,
        "kpis": mgr.get_agentic_kpis(),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic agentic benchmark scenarios")
    parser.add_argument("--trials", type=int, default=4, help="Trials per scenario")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mode",
        type=str,
        default="constrained_auto",
        choices=["shadow", "constrained_auto", "full_auto"],
        help="Rollout mode",
    )
    parser.add_argument("--autonomy-enabled", action="store_true", default=False)
    parser.add_argument(
        "--output",
        type=str,
        default="data/benchmarks/agentic_benchmark_latest.json",
        help="Output JSON summary path",
    )
    args = parser.parse_args()

    summary = run_benchmarks(
        trials=args.trials,
        seed=args.seed,
        mode=args.mode,
        autonomy_enabled=args.autonomy_enabled,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Agentic Benchmark Summary")
    print("-" * 32)
    print(f"Output:                  {out_path}")
    print(f"Runs:                    {summary['run_count']}")
    print(f"Plan success rate:       {100.0 * summary['plan_success_rate']:.2f}%")
    print(f"Blocked-step rate:       {100.0 * summary['blocked_step_rate']:.2f}%")
    print(f"Autonomous completion:   {100.0 * summary['autonomous_completion_rate']:.2f}%")
    print(f"Failure recovery rate:   {100.0 * summary['failure_recovery_rate']:.2f}%")


if __name__ == "__main__":
    main()
