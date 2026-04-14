"""Replay governance decisions over generated scenarios and print summary metrics.

Usage:
    python scripts/evaluate_governance.py --episodes 40
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cognition import Alert, AlertSeverity, AlertType
from src.integration import SessionManager


def _build_random_alert(mgr: SessionManager, rng: random.Random) -> Alert:
    node_ids = list(mgr.supply_data.node_types.keys())
    sample_size = 1 if len(node_ids) <= 2 else rng.randint(1, min(3, len(node_ids)))
    chosen_nodes = rng.sample(node_ids, sample_size)

    alert_type = rng.choice(list(AlertType))
    severity = rng.choice(list(AlertSeverity))
    return Alert(
        alert_type=alert_type,
        severity=severity,
        affected_nodes=chosen_nodes,
        details={"source": "governance_replay"},
    )


def run_replay(episodes: int, seed: int, mode: str, autonomy_enabled: bool) -> dict:
    rng = random.Random(seed)
    mgr = SessionManager(seed=seed)
    mgr.set_rollout_mode(mode)
    mgr.set_autonomy_enabled(autonomy_enabled)

    for _ in range(max(1, episodes)):
        alert = _build_random_alert(mgr, rng)
        mgr.run_cognitive_workflow(alert)
        if rng.random() < 0.7:
            mgr.step()

    kpis = mgr.get_agentic_kpis()
    history = mgr.get_agentic_kpi_history(limit=1000)

    return {
        "episodes": episodes,
        "seed": seed,
        "rollout_mode": mgr.get_rollout_config().get("rollout_mode", mode),
        "autonomy_enabled": bool(mgr.get_rollout_config().get("autonomy_enabled", autonomy_enabled)),
        "workflow_runs": int(kpis["workflow_runs"]),
        "total_recommendations": int(kpis["total_recommendations"]),
        "autonomous_completion_rate": float(kpis["autonomous_completion_rate"]),
        "human_override_rate": float(kpis["human_override_rate"]),
        "mean_replans_per_run": float(kpis["mean_replans_per_run"]),
        "plan_completion_rate": float(kpis["plan_completion_rate"]),
        "blocked_step_rate": float(kpis["blocked_step_rate"]),
        "kpi_snapshots": len(history),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay governance policy over randomized alerts")
    parser.add_argument("--episodes", type=int, default=30, help="Number of replayed alert episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mode",
        type=str,
        default="constrained_auto",
        choices=["shadow", "constrained_auto", "full_auto"],
        help="Rollout mode for replay",
    )
    parser.add_argument(
        "--autonomy-enabled",
        action="store_true",
        default=False,
        help="Enable autonomous execution during replay",
    )
    args = parser.parse_args()

    result = run_replay(
        episodes=args.episodes,
        seed=args.seed,
        mode=args.mode,
        autonomy_enabled=args.autonomy_enabled,
    )

    print("Governance Replay Summary")
    print("-" * 32)
    print(f"Episodes:                {result['episodes']}")
    print(f"Mode:                    {result['rollout_mode']}")
    print(f"Autonomy enabled:        {result['autonomy_enabled']}")
    print(f"Workflow runs:           {result['workflow_runs']}")
    print(f"Recommendations:         {result['total_recommendations']}")
    print(f"Autonomous completion:   {100.0 * result['autonomous_completion_rate']:.2f}%")
    print(f"Human override rate:     {100.0 * result['human_override_rate']:.2f}%")
    print(f"Mean replans per run:    {result['mean_replans_per_run']:.3f}")
    print(f"Plan completion rate:    {100.0 * result['plan_completion_rate']:.2f}%")
    print(f"Blocked-step rate:       {100.0 * result['blocked_step_rate']:.2f}%")
    print(f"KPI snapshots captured:  {result['kpi_snapshots']}")


if __name__ == "__main__":
    main()
