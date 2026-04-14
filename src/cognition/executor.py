"""
Execution Controller Node

Dispatches the active plan step to the correct specialist agent and records
structured execution logs for Sprint 1.
"""

from datetime import datetime
from typing import Any, Dict

from .state import SupplyChainState


def create_executor_node():
    """Create a plan-step execution controller node."""

    def executor_node(state: SupplyChainState) -> Dict[str, Any]:
        plan_steps = state.get("plan_steps", [])
        current = state.get("current_plan_step", 0)
        execution_log = state.get("execution_log", []).copy()
        reflection_notes = state.get("reflection_notes", []).copy()

        if not plan_steps or current >= len(plan_steps):
            execution_log.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "event": "executor_noop",
                    "details": "No active plan step to execute",
                }
            )
            return {
                "execution_log": execution_log,
                "next_agent": "end",
            }

        active = dict(plan_steps[current])
        owner = str(active.get("owner", "supervisor")).lower()

        if owner not in {"supervisor", "analyst", "negotiator", "human", "tools"}:
            active["status"] = "blocked"
            active["error"] = f"Unknown plan step owner: {owner}"
            plan_steps = plan_steps.copy()
            plan_steps[current] = active

            reflection_notes.append(
                f"Step {active.get('step_id', '?')} blocked due to unsupported owner '{owner}'"
            )
            execution_log.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "event": "step_blocked",
                    "step_id": active.get("step_id"),
                    "owner": owner,
                    "reason": active.get("error"),
                }
            )
            return {
                "plan_steps": plan_steps,
                "reflection_notes": reflection_notes,
                "execution_log": execution_log,
                "plan_status": "blocked",
                "next_agent": "supervisor",
            }

        if str(active.get("status", "pending")).lower() == "pending":
            active["status"] = "in_progress"
            plan_steps = plan_steps.copy()
            plan_steps[current] = active

        execution_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event": "step_started",
                "step_id": active.get("step_id"),
                "title": active.get("title"),
                "owner": owner,
                "required_tools": active.get("required_tools", []),
            }
        )

        next_agent = owner if owner != "tools" else "analyst"
        if owner == "human":
            next_agent = "human"

        return {
            "plan_steps": plan_steps,
            "execution_log": execution_log,
            "plan_status": "in_progress",
            "next_agent": next_agent,
        }

    return executor_node
