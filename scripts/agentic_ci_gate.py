"""CI gate for agentic benchmark regression thresholds.

Usage:
    python scripts/agentic_ci_gate.py --benchmark data/benchmarks/agentic_benchmark_latest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _load_thresholds() -> Dict[str, float]:
    return {
        "min_plan_success_rate": _env_float("AGENTIC_GATE_MIN_PLAN_SUCCESS_RATE", 0.55),
        "max_blocked_step_rate": _env_float("AGENTIC_GATE_MAX_BLOCKED_STEP_RATE", 0.45),
        "min_autonomous_completion_rate": _env_float("AGENTIC_GATE_MIN_AUTONOMOUS_COMPLETION_RATE", 0.10),
        "min_failure_recovery_rate": _env_float("AGENTIC_GATE_MIN_FAILURE_RECOVERY_RATE", 0.50),
    }


def _evaluate(summary: Dict[str, float], thresholds: Dict[str, float]) -> List[Tuple[str, bool, float, float]]:
    checks: List[Tuple[str, bool, float, float]] = []

    plan_success = float(summary.get("plan_success_rate", 0.0))
    blocked = float(summary.get("blocked_step_rate", 1.0))
    autonomous = float(summary.get("autonomous_completion_rate", 0.0))
    recovery = float(summary.get("failure_recovery_rate", 0.0))

    checks.append(("plan_success_rate", plan_success >= thresholds["min_plan_success_rate"], plan_success, thresholds["min_plan_success_rate"]))
    checks.append(("blocked_step_rate", blocked <= thresholds["max_blocked_step_rate"], blocked, thresholds["max_blocked_step_rate"]))
    checks.append(("autonomous_completion_rate", autonomous >= thresholds["min_autonomous_completion_rate"], autonomous, thresholds["min_autonomous_completion_rate"]))
    checks.append(("failure_recovery_rate", recovery >= thresholds["min_failure_recovery_rate"], recovery, thresholds["min_failure_recovery_rate"]))

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic KPI CI regression gate")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="data/benchmarks/agentic_benchmark_latest.json",
        help="Path to benchmark JSON summary",
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        print(f"FAIL: benchmark file not found: {benchmark_path}")
        raise SystemExit(2)

    summary = json.loads(benchmark_path.read_text(encoding="utf-8"))
    thresholds = _load_thresholds()
    checks = _evaluate(summary, thresholds)

    print("Agentic CI Gate")
    print("-" * 20)
    failed = False
    for name, ok, value, threshold in checks:
        if name == "blocked_step_rate":
            line = f"{name}: {'PASS' if ok else 'FAIL'} (value={value:.4f}, max={threshold:.4f})"
        else:
            line = f"{name}: {'PASS' if ok else 'FAIL'} (value={value:.4f}, min={threshold:.4f})"
        print(line)
        if not ok:
            failed = True

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
