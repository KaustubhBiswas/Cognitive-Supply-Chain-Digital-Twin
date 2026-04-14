"""
Planner Agent Implementation

Creates an execution plan from a high-level objective and alert context.
This is the Sprint 1 entry point for goal decomposition.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .memory_store import get_default_memory_store
from .state import SupplyChainState
from .tool_policy import get_tool_registry, select_tools_for_goal

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "planner.txt"


def _load_prompt() -> str:
    """Load planner system prompt."""
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            "You are a planning agent. Break the objective into JSON steps with fields: "
            "title, owner, success_criteria."
        )


def _objective_from_state(state: SupplyChainState) -> str:
    objective = state.get("objective")
    if objective:
        return str(objective)

    alert = state.get("current_alert") or {}
    alert_type = alert.get("alert_type", "supply_chain_issue")
    affected = alert.get("affected_nodes", [])
    return f"Stabilize {alert_type} impact for nodes {affected} while minimizing bullwhip and backlog."


def _parse_plan_response(text: str) -> List[Dict[str, Any]]:
    payload = text.strip()
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            steps = parsed.get("steps", [])
        else:
            steps = parsed
        if isinstance(steps, list):
            return [s for s in steps if isinstance(s, dict)]
    except json.JSONDecodeError:
        pass
    return []


def _rule_based_plan(state: SupplyChainState) -> List[Dict[str, Any]]:
    objective = _objective_from_state(state)

    scan_scope = str(state.get("scan_scope", "custom_nodes"))
    vulnerable_nodes = state.get("vulnerable_node_ids", []) or []
    vulnerability_count = int(state.get("vulnerability_count", 0) or 0)

    if scan_scope == "full_network":
        step_specs = [
            {
                "title": "Triaging network-wide vulnerabilities",
                "owner": "analyst",
                "success_criteria": "High/critical risks ranked across vulnerable nodes",
            },
            {
                "title": "Design staged mitigation batches",
                "owner": "negotiator",
                "success_criteria": "Coordinated mitigation batches proposed for multi-node vulnerabilities",
            },
            {
                "title": "Validate mitigation coverage and residual risk",
                "owner": "analyst",
                "success_criteria": "Coverage and residual-risk report generated for all vulnerable nodes",
            },
            {
                "title": "Finalize governance-ready execution package",
                "owner": "supervisor",
                "success_criteria": "Actions classified by risk with escalation and rollback notes",
            },
        ]
        objective = (
            f"{objective} Prioritize {vulnerability_count} vulnerabilities across "
            f"{len(vulnerable_nodes)} vulnerable nodes."
        )
    else:
        step_specs = [
            {
                "title": "Assess network risk and KPIs",
                "owner": "analyst",
                "success_criteria": "Risk level and demand/inventory signals quantified",
            },
            {
                "title": "Design coordination actions",
                "owner": "negotiator",
                "success_criteria": "Actionable cross-node coordination proposals generated",
            },
            {
                "title": "Synthesize and escalate if needed",
                "owner": "supervisor",
                "success_criteria": "Final recommendation package produced with escalation decision",
            },
        ]

    steps: List[Dict[str, Any]] = []
    for i, spec in enumerate(step_specs, start=1):
        selected_tools = select_tools_for_goal(
            goal=f"{objective}. {spec['title']}",
            owner=spec["owner"],
            max_tools=4,
        )
        steps.append(
            {
                "step_id": f"P{i}",
                "title": spec["title"],
                "description": spec["title"],
                "owner": spec["owner"],
                "status": "pending",
                "required_tools": selected_tools,
                "success_criteria": spec["success_criteria"],
            }
        )
    return steps


def create_planner_agent(llm=None):
    """Create planner node for LangGraph."""
    system_prompt = _load_prompt()

    def planner_node(state: SupplyChainState) -> Dict[str, Any]:
        plan_status = state.get("plan_status", "not_started")
        existing_steps = state.get("plan_steps", [])
        if existing_steps and plan_status != "replanned":
            return {"next_agent": "supervisor"}

        objective = _objective_from_state(state)
        alert = state.get("current_alert") or {}
        alert_type = str(alert.get("alert_type", ""))
        scan_scope = str(state.get("scan_scope", "custom_nodes"))
        vulnerable_nodes = state.get("vulnerable_node_ids", []) or []
        vulnerability_count = int(state.get("vulnerability_count", 0) or 0)
        vulnerabilities_by_node = state.get("vulnerabilities_by_node", {}) or {}

        vulnerability_brief_lines: List[str] = []
        if scan_scope == "full_network":
            vulnerability_brief_lines.append(
                f"Scope: full_network; vulnerabilities={vulnerability_count}; vulnerable_nodes={len(vulnerable_nodes)}"
            )
            for node_id in vulnerable_nodes[:8]:
                entries = vulnerabilities_by_node.get(str(node_id), []) or []
                severities = [str(e.get("severity", "unknown")) for e in entries[:3]]
                severity_label = ",".join(severities) if severities else "unknown"
                vulnerability_brief_lines.append(
                    f"node {node_id}: count={len(entries)} severities={severity_label}"
                )
        vulnerability_brief = "\n".join(vulnerability_brief_lines) if vulnerability_brief_lines else "None"

        plan_steps: List[Dict[str, Any]] = []
        execution_log = state.get("execution_log", []).copy()

        memory_store = get_default_memory_store()
        retrieved_memories = memory_store.retrieve_relevant(
            objective=objective,
            alert_type=alert_type,
            limit=3,
        )
        memory_context = memory_store.build_prompt_memory_context(retrieved_memories)

        if llm is not None:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage

                registry = get_tool_registry()
                response = llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(
                            content=(
                                f"Objective: {objective}\n"
                                f"Vulnerability summary:\n{vulnerability_brief}\n"
                                f"Available tool registry: {json.dumps(registry, indent=2)}\n"
                                f"Relevant past episodes:\n{memory_context or 'None'}\n"
                                "Return compact JSON with a list of steps."
                            )
                        ),
                    ]
                )
                parsed_steps = _parse_plan_response(getattr(response, "content", ""))
                for i, step in enumerate(parsed_steps, start=1):
                    owner = str(step.get("owner", "analyst")).lower()
                    plan_steps.append(
                        {
                            "step_id": f"P{i}",
                            "title": step.get("title", f"Step {i}"),
                            "description": step.get("description", step.get("title", "")),
                            "owner": owner,
                            "status": "pending",
                            "required_tools": select_tools_for_goal(
                                goal=f"{objective}. {step.get('title', '')}",
                                owner=owner,
                                max_tools=4,
                            ),
                            "success_criteria": step.get("success_criteria", "Step completed"),
                        }
                    )
            except Exception as e:
                logger.warning("LLM planner failed, falling back to rule-based plan: %s", e)

        if not plan_steps:
            # Memory-aware fallback: reuse structure from best past completed episode if available.
            reused = None
            for mem in retrieved_memories:
                if str(mem.get("plan_status", "")).lower() == "completed" and mem.get("plan_steps"):
                    reused = mem
                    break

            if reused:
                template_steps = reused.get("plan_steps", [])
                for i, step in enumerate(template_steps[:5], start=1):
                    owner = str(step.get("owner", "analyst")).lower()
                    title = step.get("title", f"Step {i}")
                    plan_steps.append(
                        {
                            "step_id": f"M{i}",
                            "title": title,
                            "description": step.get("description", title),
                            "owner": owner,
                            "status": "pending",
                            "required_tools": select_tools_for_goal(
                                goal=f"{objective}. {title}",
                                owner=owner,
                                max_tools=4,
                            ),
                            "success_criteria": step.get("success_criteria", "Step completed"),
                        }
                    )
            else:
                plan_steps = _rule_based_plan(state)

        start_idx = 0
        if existing_steps and plan_status == "replanned":
            completed_prefix = [s for s in existing_steps if str(s.get("status", "")).lower() == "completed"]
            start_idx = len(completed_prefix)
            # Keep completed prefix and replace the unresolved tail with newly generated steps.
            for i, step in enumerate(plan_steps, start=1):
                step["step_id"] = f"RP{i}"
            plan_steps = completed_prefix + plan_steps
            execution_log.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "event": "plan_rebuilt",
                    "completed_prefix": len(completed_prefix),
                    "new_steps": len(plan_steps) - len(completed_prefix),
                }
            )
        else:
            execution_log.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "event": "plan_created",
                    "steps": len(plan_steps),
                    "objective": objective,
                    "scan_scope": scan_scope,
                    "vulnerability_count": vulnerability_count,
                    "vulnerable_nodes": len(vulnerable_nodes),
                }
            )

        try:
            from langchain_core.messages import AIMessage

            plan_message = AIMessage(
                content=(
                    f"[Planner] Objective: {objective}. "
                    f"Generated {len(plan_steps)} plan steps. "
                    f"Scope={scan_scope}, vulnerabilities={vulnerability_count}."
                )
            )
            messages = [plan_message]
        except Exception:
            messages = []

        return {
            "messages": messages,
            "objective": objective,
            "plan_steps": plan_steps,
            "current_plan_step": start_idx,
            "plan_status": "in_progress",
            "execution_log": execution_log,
            "retrieved_memories": retrieved_memories,
            "next_agent": "supervisor",
        }

    return planner_node
