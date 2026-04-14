"""
Session State Orchestrator

Wraps simulation, risk engine, preemptive monitor, and cognitive graph
into a single manager that Streamlit pages share via st.session_state.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np

from src.cognition import (Alert, AlertSeverity, AlertType, PreemptiveMonitor,
                           RiskEngine, create_initial_state,
                           create_supply_chain_graph, initialize_tools)
from src.cognition.governance import (decide_rollout_execution,
                                      evaluate_recommendation_policy,
                                      get_default_policy_thresholds,
                                      tune_policy_thresholds)
from src.cognition.graph import FallbackGraph
from src.cognition.llm import create_llm
from src.cognition.memory_store import EpisodeMemoryStore
from src.data.parser import create_synthetic_supply_graph
from src.simulation import SupplyChainModel

logger = logging.getLogger(__name__)


@dataclass
class ActionRecord:
    """A record of an agent recommendation and its human decision."""
    timestamp: str
    alert_type: str
    severity: str
    recommendation: Dict[str, Any]
    status: str = "pending"  # pending | approved | rejected
    feedback: str = ""


@dataclass
class ChatMessage:
    """A single message in the agent chat log."""
    agent: str  # supervisor | analyst | negotiator | system | human
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class SessionManager:
    """
    Central orchestrator that holds all backend state.

    Usage in Streamlit:
        if "session" not in st.session_state:
            st.session_state.session = SessionManager()
        mgr = st.session_state.session
    """

    def __init__(
        self,
        num_suppliers: int = 3,
        num_manufacturers: int = 4,
        num_distributors: int = 5,
        num_retailers: int = 6,
        seed: int = 42,
        risk_threshold: float = 0.30,
        alert_cooldown: int = 5,
    ):
        # ── Simulation ─────────────────────────────────────────
        self._seed = seed
        self._network_config = {
            "num_suppliers": num_suppliers,
            "num_manufacturers": num_manufacturers,
            "num_distributors": num_distributors,
            "num_retailers": num_retailers,
        }
        self.supply_data = create_synthetic_supply_graph(
            **self._network_config, seed=seed
        )
        self.model = SupplyChainModel(
            graph=self.supply_data.graph,
            node_types=self.supply_data.node_types,
            random_seed=seed,
        )

        # ── Risk & Monitor ─────────────────────────────────────
        self.risk_engine = RiskEngine(alert_threshold=risk_threshold)
        self.monitor = PreemptiveMonitor(
            risk_engine=self.risk_engine, alert_cooldown=alert_cooldown
        )

        # ── History ────────────────────────────────────────────
        self.chat_history: List[ChatMessage] = []
        self.action_queue: List[ActionRecord] = []
        self.event_log: List[Dict[str, Any]] = []
        self.monitor_reports: List[Dict[str, Any]] = []
        self.last_cognitive_result: Optional[Dict[str, Any]] = None
        self.memory_store = EpisodeMemoryStore()
        self.agentic_metrics: Dict[str, float] = {
            "workflow_runs": 0,
            "total_recommendations": 0,
            "auto_approved": 0,
            "human_approvals": 0,
            "human_rejections": 0,
            "replans_total": 0,
            "blocked_steps": 0,
            "completed_plans": 0,
        }
        self.agentic_kpi_history: List[Dict[str, Any]] = []
        self.policy_thresholds: Dict[str, float] = get_default_policy_thresholds()
        self.policy_adaptation_log: List[Dict[str, Any]] = []
        self.rollout_mode: str = str(os.getenv("AGENT_ROLLOUT_MODE", "constrained_auto")).strip().lower()
        if self.rollout_mode not in {"shadow", "constrained_auto", "full_auto"}:
            self.rollout_mode = "constrained_auto"
        self.autonomy_enabled: bool = os.getenv("AGENT_AUTONOMY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}

        # ── Cognition ──────────────────────────────────────────
        self.llm = None
        self.graph = None
        self.rag_retriever = None
        self._init_cognition()

        # ── State flags ────────────────────────────────────────
        self.is_running = False
        self.auto_step = False

    # ─── Initialization ────────────────────────────────────────

    def _init_cognition(self):
        """Initialize LLM and cognitive graph."""
        self.rag_retriever = self._init_rag()
        initialize_tools(simulation=self.model, rag_retriever=self.rag_retriever)

        # Try Groq first
        groq_key = os.getenv("GROQ_API_KEY", "")
        llm_model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

        if groq_key:
            self.llm = create_llm(
                provider="groq",
                api_key=groq_key,
                model=llm_model,
                test_connection=False,
            )
            if self.llm:
                logger.info(f"LLM connected: Groq/{llm_model}")

        self.graph = create_supply_chain_graph(llm=self.llm)
        self._add_system_message(
            f"System initialized • {sum(1 for _ in self.model.agents)} nodes • "
            f"LLM: {'Groq/' + llm_model if self.llm else 'Rule-based fallback'} • "
            f"RAG: {'enabled' if self.rag_retriever else 'disabled'}"
        )

    def _init_rag(self):
        """Initialize RAG retriever from persisted vector store if available."""
        rag_enabled = os.getenv("RAG_ENABLED", "true").lower() in {"1", "true", "yes"}
        if not rag_enabled:
            logger.info("RAG disabled via RAG_ENABLED")
            return None

        persist_dir = os.getenv("RAG_PERSIST_DIR", str(Path("data") / "vectorstore"))
        persist_path = Path(persist_dir)

        # Skip initialization when no persisted vector DB exists.
        if not persist_path.exists() or not any(persist_path.iterdir()):
            logger.info("RAG persist directory not found or empty: %s", persist_path)
            return None

        try:
            from src.cognition.rag import (ChromaVectorStore,
                                           SupplyChainEmbeddings,
                                           SupplyChainRetriever)

            embeddings = SupplyChainEmbeddings(use_cache=True)
            vector_store = ChromaVectorStore(
                persist_directory=str(persist_path),
                embedding_model=embeddings,
            )
            retriever = SupplyChainRetriever(
                vector_store=vector_store,
                embedding_model=embeddings,
            )

            logger.info("RAG retriever initialized from %s", persist_path)
            return retriever
        except Exception as e:
            logger.warning("RAG initialization failed: %s", e)
            return None

    def reset(self):
        """Reset simulation and all state."""
        self.supply_data = create_synthetic_supply_graph(
            **self._network_config, seed=self._seed
        )
        self.model = SupplyChainModel(
            graph=self.supply_data.graph,
            node_types=self.supply_data.node_types,
            random_seed=self._seed,
        )
        self.risk_engine = RiskEngine(alert_threshold=self.risk_engine.alert_threshold)
        self.monitor = PreemptiveMonitor(
            risk_engine=self.risk_engine,
            alert_cooldown=self.monitor.alert_cooldown,
        )
        self.chat_history.clear()
        self.action_queue.clear()
        self.event_log.clear()
        self.monitor_reports.clear()
        self.last_cognitive_result = None
        self.agentic_metrics = {
            "workflow_runs": 0,
            "total_recommendations": 0,
            "auto_approved": 0,
            "human_approvals": 0,
            "human_rejections": 0,
            "replans_total": 0,
            "blocked_steps": 0,
            "completed_plans": 0,
        }
        self.agentic_kpi_history = []
        self.policy_thresholds = get_default_policy_thresholds()
        self.policy_adaptation_log = []
        self.rollout_mode = str(os.getenv("AGENT_ROLLOUT_MODE", "constrained_auto")).strip().lower()
        if self.rollout_mode not in {"shadow", "constrained_auto", "full_auto"}:
            self.rollout_mode = "constrained_auto"
        self.autonomy_enabled = os.getenv("AGENT_AUTONOMY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        self._init_cognition()
        self.is_running = False
        self.auto_step = False

    # ─── Simulation Control ────────────────────────────────────

    def step(self) -> Dict[str, Any]:
        """Advance simulation by one step and run risk scan."""
        self.model.step()
        report = self.monitor.on_step(self.model)
        self.monitor_reports.append(report)

        # Auto-generate chat messages for new alerts
        pending = self.monitor.get_pending_alerts(clear=True)
        for alert_dict in pending:
            node_id = alert_dict.get("affected_nodes", [None])[0]
            risk = alert_dict.get("details", {}).get("composite_risk", 0)
            self._add_system_message(
                f"⚠️ Preemptive alert: Node {node_id} — "
                f"risk {risk:.0%} — {alert_dict.get('details', {}).get('recommended_action', '')}"
            )

        return report

    def run_steps(self, n: int) -> List[Dict[str, Any]]:
        """Run multiple steps."""
        reports = []
        for _ in range(n):
            reports.append(self.step())
        return reports

    def inject_event(self, event_type: str, magnitude: float = 2.0, duration: int = 10):
        """Inject a disruption event."""
        from src.simulation.model import EventType
        type_map = {
            "demand_shock": EventType.DEMAND_SHOCK,
            "supply_disruption": EventType.SUPPLY_DISRUPTION,
            "factory_issue": EventType.FACTORY_ISSUE,
            "lead_time_increase": EventType.LEAD_TIME_INCREASE,
        }
        et = type_map.get(event_type)
        if et is None:
            return
        self.model.inject_event(event_type=et, magnitude=magnitude, duration=duration)
        self.event_log.append({
            "step": self.model.current_step,
            "type": event_type,
            "magnitude": magnitude,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
        })
        self._add_system_message(
            f"🔴 Event injected: {event_type} (×{magnitude}, {duration} steps)"
        )

    # ─── Cognition ─────────────────────────────────────────────

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        """Recursively convert numpy/container types to plain Python types."""
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [SessionManager._to_serializable(v) for v in value.tolist()]
        if isinstance(value, dict):
            return {
                SessionManager._to_serializable(k): SessionManager._to_serializable(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [SessionManager._to_serializable(v) for v in value]
        return value

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        """Best-effort integer coercion for ids/counters from UI or alert payloads."""
        if isinstance(value, bool):
            return None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if np.isfinite(value) and float(value).is_integer():
                return int(value)
            return None
        if isinstance(value, str):
            text = value.strip()
            if text.lstrip("-").isdigit():
                return int(text)
        return None

    @staticmethod
    def _coerce_rate(value: Any) -> Optional[float]:
        """Best-effort rate coercion clamped to [0, 1]."""
        try:
            rate = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(rate):
            return None
        return max(0.0, min(1.0, rate))

    def _normalize_coverage_context(
        self,
        coverage_context: Optional[Dict[str, Any]],
        affected_nodes: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Normalize coverage payload so planner/executor always see consistent fields."""
        raw = coverage_context or {}

        all_nodes = sorted(int(agent.unique_id) for agent in self.model.agents)
        all_node_set = set(all_nodes)

        normalized_affected: List[int] = []
        for node in affected_nodes or []:
            node_id = self._coerce_int(node)
            if node_id is None:
                continue
            if not all_node_set or node_id in all_node_set:
                normalized_affected.append(node_id)
        normalized_affected = sorted(set(normalized_affected))

        scan_scope = str(raw.get("scan_scope", "custom_nodes")).strip().lower()
        if scan_scope not in {"custom_nodes", "full_network"}:
            scan_scope = "custom_nodes"
        if scan_scope == "full_network" and not all_nodes:
            scan_scope = "custom_nodes"

        scope_node_set = (
            all_node_set
            if scan_scope == "full_network"
            else (set(normalized_affected) if normalized_affected else all_node_set)
        )

        vulnerable_node_ids_set = set()
        raw_vulnerable = raw.get("vulnerable_node_ids", [])
        if isinstance(raw_vulnerable, (list, tuple, set)):
            for node in raw_vulnerable:
                node_id = self._coerce_int(node)
                if node_id is None:
                    continue
                if not scope_node_set or node_id in scope_node_set:
                    vulnerable_node_ids_set.add(node_id)

        normalized_vulnerabilities: Dict[str, List[Dict[str, Any]]] = {}
        raw_vulnerabilities = raw.get("vulnerabilities_by_node", {})
        if isinstance(raw_vulnerabilities, dict):
            for raw_node_id, entries in raw_vulnerabilities.items():
                node_id = self._coerce_int(raw_node_id)
                if node_id is None:
                    continue
                if scope_node_set and node_id not in scope_node_set:
                    continue

                normalized_entries: List[Dict[str, Any]] = []
                if isinstance(entries, dict):
                    normalized_entries = [entries]
                elif isinstance(entries, list):
                    for item in entries:
                        if isinstance(item, dict):
                            normalized_entries.append(item)
                        else:
                            normalized_entries.append({"detail": str(item)})
                elif entries is not None:
                    normalized_entries = [{"detail": str(entries)}]

                normalized_vulnerabilities[str(node_id)] = normalized_entries
                if normalized_entries:
                    vulnerable_node_ids_set.add(node_id)

        vulnerable_node_ids = sorted(vulnerable_node_ids_set)
        for node_id in vulnerable_node_ids:
            normalized_vulnerabilities.setdefault(str(node_id), [])

        computed_findings = sum(len(v) for v in normalized_vulnerabilities.values())
        findings_hint = self._coerce_int(raw.get("vulnerability_count"))
        if computed_findings > 0:
            vulnerability_count = computed_findings
        elif findings_hint is not None and findings_hint >= 0:
            vulnerability_count = max(len(vulnerable_node_ids), findings_hint)
        else:
            vulnerability_count = len(vulnerable_node_ids)

        if scan_scope == "full_network":
            total_nodes_scanned = len(all_nodes)
            coverage_rate = 1.0 if total_nodes_scanned > 0 else 0.0
        else:
            total_hint = self._coerce_int(raw.get("total_nodes_scanned"))
            if normalized_affected:
                total_nodes_scanned = len(normalized_affected)
            elif total_hint is not None and total_hint >= 0:
                max_nodes = len(all_nodes) if all_nodes else total_hint
                total_nodes_scanned = min(total_hint, max_nodes)
            else:
                total_nodes_scanned = len(vulnerable_node_ids)

            total_nodes_scanned = max(total_nodes_scanned, len(vulnerable_node_ids))

            rate_hint = self._coerce_rate(raw.get("coverage_rate"))
            if total_nodes_scanned == 0:
                coverage_rate = 0.0
            elif rate_hint is None:
                coverage_rate = 1.0
            else:
                coverage_rate = rate_hint

        return {
            "scan_scope": scan_scope,
            "total_nodes_scanned": int(total_nodes_scanned),
            "vulnerable_node_ids": vulnerable_node_ids,
            "vulnerabilities_by_node": normalized_vulnerabilities,
            "coverage_rate": float(coverage_rate),
            "vulnerability_count": int(vulnerability_count),
        }

    def run_cognitive_workflow(self, alert: Alert) -> Dict[str, Any]:
        """Run the full cognitive graph and capture messages."""
        coverage_context: Dict[str, Any] = {}
        if isinstance(alert.details, dict):
            raw_coverage_context = alert.details.get("coverage_context", {}) or {}
            if raw_coverage_context or bool(alert.details.get("full_network_assessment", False)):
                coverage_context = self._normalize_coverage_context(
                    coverage_context=raw_coverage_context,
                    affected_nodes=alert.affected_nodes,
                )

        objective = (
            alert.details.get("objective")
            if isinstance(alert.details, dict)
            else None
        )
        if not objective:
            objective = (
                f"Stabilize {alert.alert_type.value} for nodes {alert.affected_nodes} "
                "while minimizing backlog and bullwhip."
            )

        state = self._to_serializable(create_initial_state(alert=alert, objective=objective))
        if coverage_context:
            state["scan_scope"] = str(coverage_context.get("scan_scope", "custom_nodes"))
            state["total_nodes_scanned"] = int(coverage_context.get("total_nodes_scanned", 0) or 0)
            state["vulnerable_node_ids"] = [int(v) for v in (coverage_context.get("vulnerable_node_ids", []) or [])]
            state["vulnerabilities_by_node"] = coverage_context.get("vulnerabilities_by_node", {}) or {}
            state["coverage_rate"] = float(coverage_context.get("coverage_rate", 0.0) or 0.0)
            state["vulnerability_count"] = int(coverage_context.get("vulnerability_count", 0) or 0)

        config = {"configurable": {"thread_id": f"ui-{datetime.now().timestamp()}"}}

        if isinstance(self.graph, FallbackGraph):
            result = self.graph.invoke(state)
        else:
            try:
                result = self.graph.invoke(state, config=config)
            except TypeError as e:
                if "msgpack serializable" not in str(e):
                    raise
                logger.warning(
                    "LangGraph state serialization failed (%s); retrying with fallback graph",
                    e,
                )
                result = FallbackGraph(llm=self.llm).invoke(state)

        # Capture agent messages
        for msg in result.get("messages", []):
            content = msg.content if hasattr(msg, "content") else str(msg)
            # Determine agent from content
            agent = "supervisor"
            if "analyst" in content.lower() or "analysis" in content.lower():
                agent = "analyst"
            elif "negotiat" in content.lower() or "order" in content.lower():
                agent = "negotiator"
            self.chat_history.append(ChatMessage(
                agent=agent,
                content=content,
                timestamp=datetime.now().strftime("%H:%M:%S"),
            ))

        # Capture recommendations
        for rec in result.get("recommendations", []):
            policy_meta = evaluate_recommendation_policy(
                recommendation=rec,
                alert_severity=alert.severity.value,
                policy_thresholds=self.policy_thresholds,
            )
            rollout_meta = decide_rollout_execution(
                policy_meta=policy_meta,
                rollout_mode=self.rollout_mode,
                autonomy_enabled=self.autonomy_enabled,
            )
            policy_meta["rollout"] = rollout_meta
            rec["governance"] = policy_meta

            status = str(rollout_meta.get("status", "pending"))
            feedback = str(rollout_meta.get("reason", ""))

            self.action_queue.append(ActionRecord(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                alert_type=alert.alert_type.value,
                severity=alert.severity.value,
                recommendation=rec,
                status=status,
                feedback=feedback,
            ))

            if bool(rollout_meta.get("execute_now", False)):
                action = self.action_queue[-1]
                self._process_approved_action(
                    action=action,
                    actor_label=f"governance policy ({self.rollout_mode})",
                )
                self.agentic_metrics["auto_approved"] += 1
            elif status == "pending" and policy_meta.get("decision") == "auto_approve":
                self._add_system_message(
                    "🛰️ Auto-approved by policy but held by rollout control: "
                    f"{feedback}"
                )

        self.last_cognitive_result = result
        if coverage_context:
            self.last_cognitive_result["coverage_context"] = coverage_context
        self._persist_episode_memory(alert=alert, objective=objective, result=result)
        self._update_agentic_metrics(result)

        return result

    def run_full_network_assessment(
        self,
        alert_type: AlertType = AlertType.PREEMPTIVE_WARNING,
        severity: AlertSeverity = AlertSeverity.HIGH,
        objective: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run cognition with full-network vulnerability coverage context."""
        all_nodes = sorted(int(agent.unique_id) for agent in self.model.agents)
        total_nodes = len(all_nodes)

        raw_alerts, _ = self.risk_engine.scan_network(self.model)
        vulnerabilities_by_node: Dict[str, List[Dict[str, Any]]] = {}
        for item in raw_alerts:
            nodes = item.get("affected_nodes", []) or []
            if not nodes:
                continue
            try:
                node_id = int(nodes[0])
            except (TypeError, ValueError):
                continue

            vulnerabilities_by_node.setdefault(str(node_id), []).append(
                {
                    "alert_type": str(item.get("alert_type", "preemptive_warning")),
                    "severity": str(item.get("severity", "low")),
                    "details": item.get("details", {}) or {},
                    "alert_id": item.get("alert_id"),
                }
            )

        vulnerable_node_ids = sorted(int(n) for n in vulnerabilities_by_node.keys())
        coverage_context = {
            "scan_scope": "full_network",
            "total_nodes_scanned": total_nodes,
            "vulnerable_node_ids": vulnerable_node_ids,
            "vulnerabilities_by_node": vulnerabilities_by_node,
            "coverage_rate": 1.0 if total_nodes > 0 else 0.0,
            "vulnerability_count": sum(len(v) for v in vulnerabilities_by_node.values()),
        }
        coverage_context = self._normalize_coverage_context(
            coverage_context=coverage_context,
            affected_nodes=all_nodes,
        )

        run_objective = objective or (
            "Assess and mitigate vulnerabilities across the entire supply chain network "
            f"({coverage_context['total_nodes_scanned']} nodes scanned, "
            f"{len(coverage_context['vulnerable_node_ids'])} vulnerable nodes)."
        )

        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            affected_nodes=all_nodes,
            details={
                "triggered_manually": True,
                "step": self.model.current_step,
                "full_network_assessment": True,
                "objective": run_objective,
                "coverage_context": coverage_context,
            },
        )

        self.add_human_message(
            "Triggered full-network vulnerability assessment "
            f"({coverage_context['total_nodes_scanned']} nodes; "
            f"{len(coverage_context['vulnerable_node_ids'])} vulnerable)."
        )
        self._add_system_message(
            "🔎 Full-chain scan completed: "
            f"{coverage_context['vulnerability_count']} vulnerabilities found across "
            f"{len(coverage_context['vulnerable_node_ids'])} nodes."
        )

        result = self.run_cognitive_workflow(alert)
        result["coverage_context"] = coverage_context
        return result

    def _update_agentic_metrics(self, result: Dict[str, Any]) -> None:
        """Update aggregate Sprint 3 KPIs after each cognition run."""
        self.agentic_metrics["workflow_runs"] += 1
        recommendations = result.get("recommendations", []) or []
        self.agentic_metrics["total_recommendations"] += len(recommendations)
        self.agentic_metrics["replans_total"] += int(result.get("replan_count", 0) or 0)

        if str(result.get("plan_status", "")).lower() == "completed":
            self.agentic_metrics["completed_plans"] += 1

        execution_log = result.get("execution_log", []) or []
        blocked_events = sum(
            1 for evt in execution_log
            if str(evt.get("event", "")).lower() == "step_blocked"
        )
        self.agentic_metrics["blocked_steps"] += blocked_events
        self._snapshot_agentic_kpis()

    def _snapshot_agentic_kpis(self) -> None:
        """Append a time-stamped snapshot of current aggregate KPIs."""
        kpis = self.get_agentic_kpis()
        self.agentic_kpi_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "step": int(self.model.current_step),
                **kpis,
            }
        )
        if len(self.agentic_kpi_history) > 500:
            self.agentic_kpi_history = self.agentic_kpi_history[-500:]

        # Sprint 4: periodically self-tune governance thresholds.
        if len(self.agentic_kpi_history) % 5 == 0:
            self._auto_tune_policy_thresholds()

    def _auto_tune_policy_thresholds(self) -> None:
        """Adapt governance thresholds using recent KPI outcomes."""
        tuning = tune_policy_thresholds(
            current_thresholds=self.policy_thresholds,
            kpi_history=self.agentic_kpi_history,
            min_samples=8,
        )
        if not tuning.get("changed", False):
            return

        previous = self.policy_thresholds.copy()
        updated = tuning.get("updated_thresholds", {}) or {}
        self.policy_thresholds = {
            "critical_min_confidence": float(updated.get("critical_min_confidence", previous.get("critical_min_confidence", 0.75))),
            "medium_risk_min_confidence": float(updated.get("medium_risk_min_confidence", previous.get("medium_risk_min_confidence", 0.65))),
            "baseline_min_confidence": float(updated.get("baseline_min_confidence", previous.get("baseline_min_confidence", 0.55))),
        }

        reasons = tuning.get("reasons", []) or []
        self.policy_adaptation_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "before": previous,
                "after": self.policy_thresholds.copy(),
                "reasons": reasons,
                "history_points": len(self.agentic_kpi_history),
            }
        )
        if len(self.policy_adaptation_log) > 100:
            self.policy_adaptation_log = self.policy_adaptation_log[-100:]

        self._add_system_message(
            "🧪 Governance thresholds adapted: "
            f"baseline={self.policy_thresholds['baseline_min_confidence']:.2f}, "
            f"medium={self.policy_thresholds['medium_risk_min_confidence']:.2f}, "
            f"critical={self.policy_thresholds['critical_min_confidence']:.2f}."
        )

    def _persist_episode_memory(self, alert: Alert, objective: str, result: Dict[str, Any]) -> None:
        """Persist a compact cognitive episode for future memory-aware planning."""
        try:
            episode = {
                "timestamp": datetime.now().isoformat(),
                "objective": objective,
                "alert_type": alert.alert_type.value,
                "severity": alert.severity.value,
                "affected_nodes": list(alert.affected_nodes),
                "plan_status": result.get("plan_status", "unknown"),
                "replan_count": int(result.get("replan_count", 0) or 0),
                "plan_steps": result.get("plan_steps", []) or [],
                "execution_log": (result.get("execution_log", []) or [])[-20:],
                "reflection_notes": (result.get("reflection_notes", []) or [])[-10:],
                "recommendations_count": len(result.get("recommendations", []) or []),
            }
            self.memory_store.append_episode(episode)
        except Exception as e:
            logger.warning("Could not persist episode memory: %s", e)

    def review_action(self, action_index: int, decision: str, feedback: str = "") -> bool:
        """Approve or reject a queued recommendation and record human feedback."""
        if action_index < 0 or action_index >= len(self.action_queue):
            return False

        decision = decision.strip().lower()
        if decision not in {"approved", "rejected"}:
            return False

        action = self.action_queue[action_index]
        if action.status != "pending":
            return False

        action.status = decision
        action.feedback = feedback.strip()

        rec = action.recommendation
        rec_type = rec.get("recommendation_type", "unknown")
        rec_label = rec_type.replace("_", " ")

        if decision == "approved":
            self.add_human_message(
                f"Approved recommendation: {rec_label}. "
                f"Feedback: {action.feedback or 'none'}"
            )
            self._process_approved_action(action=action, actor_label="human reviewer")
            self.agentic_metrics["human_approvals"] += 1
            self._snapshot_agentic_kpis()
        else:
            self.add_human_message(
                f"Rejected recommendation: {rec_label}. "
                f"Feedback: {action.feedback or 'none'}"
            )
            self._add_system_message(
                "🛑 Recommendation rejected by human reviewer."
            )
            self.agentic_metrics["human_rejections"] += 1
            self._snapshot_agentic_kpis()

        return True

    def _process_approved_action(self, action: ActionRecord, actor_label: str) -> None:
        """Execute recommendation effects and add explainable approval logs."""
        rec = action.recommendation
        rec_type = rec.get("recommendation_type", "unknown")

        before = self.model.get_state_snapshot()
        execution_summary = self._execute_recommendation(rec)
        after = self.model.get_state_snapshot()
        self._refresh_telemetry_checkpoint(
            reason=f"approval:{rec_type}",
            before_snapshot=before,
            after_snapshot=after,
        )

        delta_inventory = after.get("total_inventory", 0.0) - before.get("total_inventory", 0.0)
        delta_backlog = after.get("total_backlog", 0.0) - before.get("total_backlog", 0.0)
        governance = rec.get("governance", {}) or {}
        reason = governance.get("reason", "")
        decision = governance.get("decision", "human_review")

        reason_suffix = f" Policy reason: {reason}" if reason else ""
        self._add_system_message(
            "✅ Recommendation approved and processed "
            f"({actor_label}; decision={decision}): "
            f"{execution_summary}. "
            f"Telemetry Δinventory={delta_inventory:+.2f}, "
            f"Δbacklog={delta_backlog:+.2f}."
            f"{reason_suffix}"
        )

    def _execute_recommendation(self, recommendation: Dict[str, Any]) -> str:
        """
        Apply approved recommendations to the simulation where possible.

        Returns a short execution summary for the chat log.
        """
        rec_type = recommendation.get("recommendation_type", "unknown")
        params = recommendation.get("parameters", {}) or {}
        target_nodes = recommendation.get("target_nodes", []) or []

        node_to_agent = {}
        for agent in self.model.agents:
            node_to_agent[int(agent.unique_id)] = agent

        applied = 0
        for node_id in target_nodes:
            try:
                agent = node_to_agent.get(int(node_id))
            except (TypeError, ValueError):
                agent = None

            if agent is None:
                continue

            if rec_type == "adjust_reorder_point":
                current = getattr(agent, "reorder_point", None)
                if current is None:
                    continue
                new_value = params.get("reorder_point", float(current) * 1.10)
                agent.reorder_point = float(new_value)
                applied += 1

            elif rec_type in {"adjust_order_quantity", "adjust_safety_stock", "increase_safety_stock"}:
                current = getattr(agent, "order_up_to", None)
                if current is None:
                    continue
                new_value = params.get("order_up_to", float(current) * 1.15)
                agent.order_up_to = float(new_value)

                # Reflect approved policy changes in telemetry by adding a small
                # tactical replenishment toward the new target level.
                current_inventory = float(getattr(agent, "inventory", 0.0))
                gap = max(0.0, float(agent.order_up_to) - current_inventory)
                buffer_add = min(gap * 0.25, float(agent.order_up_to) * 0.20)
                if buffer_add > 0:
                    agent.inventory = current_inventory + buffer_add
                applied += 1

            elif rec_type == "increase_capacity":
                current = getattr(agent, "production_capacity", None)
                if current is None:
                    continue
                multiplier = float(params.get("capacity_multiplier", 1.20))
                agent.production_capacity = float(current) * multiplier
                applied += 1

            elif rec_type == "expedite_order":
                current = getattr(agent, "lead_time", None)
                if current is None:
                    continue
                new_value = max(1, int(current) - 1)
                agent.lead_time = new_value

                # Pull incoming orders closer to delivery to make approval effects visible.
                for order in getattr(agent, "pending_orders", []):
                    order.delivery_step = max(self.model.current_step, order.delivery_step - 1)
                if hasattr(agent, "_receive_deliveries"):
                    agent._receive_deliveries()
                applied += 1

        if rec_type == "redistribute_inventory":
            applied = self._apply_redistribution(target_nodes)
            if applied > 0:
                return f"redistribute inventory applied across {applied} node(s)"
            return "redistribute inventory marked approved (manual coordination required)"

        if rec_type == "change_supplier":
            return "change supplier marked approved (manual sourcing workflow required)"
        if rec_type == "no_action":
            return "no operational change applied"

        if applied == 0:
            return f"{rec_type.replace('_', ' ')} approved (no direct simulation parameter updated)"

        return (
            f"{rec_type.replace('_', ' ')} applied to {applied} "
            f"node{'s' if applied != 1 else ''}"
        )

    def _apply_redistribution(self, target_nodes: List[Any]) -> int:
        """Redistribute inventory from high-stock nodes to target nodes."""
        if not target_nodes:
            return 0

        try:
            target_ids = [int(n) for n in target_nodes]
        except (TypeError, ValueError):
            return 0

        node_to_agent = {int(agent.unique_id): agent for agent in self.model.agents}
        targets = [node_to_agent[n] for n in target_ids if n in node_to_agent]
        if not targets:
            return 0

        donors = sorted(
            [a for a in self.model.agents if int(a.unique_id) not in target_ids and getattr(a, "inventory", 0) > 0],
            key=lambda a: float(getattr(a, "inventory", 0.0)),
            reverse=True,
        )
        if not donors:
            return 0

        changed_nodes = set()
        for target in targets:
            target_inventory = float(getattr(target, "inventory", 0.0))
            target_level = float(getattr(target, "order_up_to", max(50.0, target_inventory)))
            needed = max(0.0, target_level - target_inventory)
            if needed <= 0:
                continue

            for donor in donors:
                donor_inventory = float(getattr(donor, "inventory", 0.0))
                transferable = max(0.0, donor_inventory * 0.10)
                if transferable <= 0:
                    continue

                moved = min(transferable, needed)
                donor.inventory = donor_inventory - moved
                target.inventory = float(getattr(target, "inventory", 0.0)) + moved
                changed_nodes.add(int(donor.unique_id))
                changed_nodes.add(int(target.unique_id))
                needed -= moved

                if needed <= 0:
                    break

        return len(changed_nodes)

    def _refresh_telemetry_checkpoint(
        self,
        reason: str,
        before_snapshot: Optional[Dict[str, Any]] = None,
        after_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Collect telemetry immediately after an approved intervention."""
        try:
            self.model.datacollector.collect(self.model)
        except Exception as e:
            logger.warning("Telemetry collection after approval failed: %s", e)

        try:
            self.risk_engine.scan_network(self.model)
        except Exception as e:
            logger.warning("Risk refresh after approval failed: %s", e)

        self.monitor_reports.append(
            {
                "scanned": True,
                "step": self.model.current_step,
                "source": "human_approval",
                "reason": reason,
                "risk_summary": self.risk_engine.get_network_risk_summary(),
                "before": before_snapshot or {},
                "after": after_snapshot or self.model.get_state_snapshot(),
                "timestamp": datetime.now().isoformat(),
            }
        )

    # ─── Data Access ───────────────────────────────────────────

    def get_snapshot(self) -> Dict[str, Any]:
        """Current simulation state snapshot."""
        return self.model.get_state_snapshot()

    def get_model_dataframe(self):
        """Get time-series model data."""
        return self.model.datacollector.get_model_vars_dataframe()

    def get_agent_dataframe(self):
        """Get per-agent time-series data."""
        return self.model.datacollector.get_agent_vars_dataframe()

    def get_risk_summary(self) -> Dict[str, Any]:
        """Current risk engine summary."""
        return self.risk_engine.get_network_risk_summary()

    def get_health_report(self) -> Dict[str, Any]:
        """Current monitor health report."""
        return self.monitor.get_network_health_report()

    def get_last_intervention_impact(self) -> Optional[Dict[str, Any]]:
        """Return a compact summary of the latest human-approved intervention impact."""
        for report in reversed(self.monitor_reports):
            if report.get("source") != "human_approval":
                continue

            before = report.get("before", {}) or {}
            after = report.get("after", {}) or {}

            before_inventory = float(before.get("total_inventory", 0.0) or 0.0)
            after_inventory = float(after.get("total_inventory", 0.0) or 0.0)
            before_backlog = float(before.get("total_backlog", 0.0) or 0.0)
            after_backlog = float(after.get("total_backlog", 0.0) or 0.0)

            risk_summary = report.get("risk_summary", {}) or {}

            return {
                "timestamp": report.get("timestamp", ""),
                "reason": report.get("reason", ""),
                "delta_inventory": after_inventory - before_inventory,
                "delta_backlog": after_backlog - before_backlog,
                "network_health": float(risk_summary.get("network_health", 0.0) or 0.0),
            }

        return None

    def get_human_intervention_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent human-approved intervention checkpoints for visualization overlays."""
        history: List[Dict[str, Any]] = []
        for report in reversed(self.monitor_reports):
            if report.get("source") != "human_approval":
                continue
            history.append({
                "step": int(report.get("step", self.model.current_step)),
                "reason": str(report.get("reason", "human_approval")),
                "timestamp": str(report.get("timestamp", "")),
            })
            if len(history) >= max(1, limit):
                break

        history.reverse()
        return history

    def get_agentic_kpis(self) -> Dict[str, float]:
        """Return aggregate Sprint 3 agentic KPIs for dashboard monitoring."""
        runs = max(1.0, float(self.agentic_metrics["workflow_runs"]))
        total_recs = max(1.0, float(self.agentic_metrics["total_recommendations"]))
        human_reviews = float(self.agentic_metrics["human_approvals"] + self.agentic_metrics["human_rejections"])

        return {
            "workflow_runs": float(self.agentic_metrics["workflow_runs"]),
            "total_recommendations": float(self.agentic_metrics["total_recommendations"]),
            "autonomous_completion_rate": float(self.agentic_metrics["auto_approved"]) / total_recs,
            "human_override_rate": float(self.agentic_metrics["human_rejections"]) / max(1.0, human_reviews),
            "mean_replans_per_run": float(self.agentic_metrics["replans_total"]) / runs,
            "plan_completion_rate": float(self.agentic_metrics["completed_plans"]) / runs,
            "blocked_step_rate": float(self.agentic_metrics["blocked_steps"]) / total_recs,
        }

    def get_agentic_kpi_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return the latest KPI trend snapshots."""
        size = max(1, int(limit))
        return self.agentic_kpi_history[-size:]

    def get_policy_thresholds(self) -> Dict[str, float]:
        """Return active governance policy thresholds."""
        return self.policy_thresholds.copy()

    def get_policy_adaptation_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent policy adaptation events."""
        size = max(1, int(limit))
        return self.policy_adaptation_log[-size:]

    def set_rollout_mode(self, mode: str) -> str:
        """Set rollout mode: shadow | constrained_auto | full_auto."""
        mode_value = str(mode or "").strip().lower()
        if mode_value not in {"shadow", "constrained_auto", "full_auto"}:
            return self.rollout_mode
        self.rollout_mode = mode_value
        self._add_system_message(f"Rollout mode set to {mode_value}.")
        return self.rollout_mode

    def set_autonomy_enabled(self, enabled: bool) -> bool:
        """Enable/disable autonomous execution (rollback switch)."""
        self.autonomy_enabled = bool(enabled)
        state = "enabled" if self.autonomy_enabled else "disabled"
        self._add_system_message(f"Autonomy {state} by operator control.")
        return self.autonomy_enabled

    def get_rollout_config(self) -> Dict[str, Any]:
        """Return active rollout strategy configuration."""
        return {
            "rollout_mode": self.rollout_mode,
            "autonomy_enabled": self.autonomy_enabled,
        }

    # ─── Chat ──────────────────────────────────────────────────

    def _add_system_message(self, content: str):
        self.chat_history.append(ChatMessage(
            agent="system",
            content=content,
            timestamp=datetime.now().strftime("%H:%M:%S"),
        ))

    def add_human_message(self, content: str):
        self.chat_history.append(ChatMessage(
            agent="human",
            content=content,
            timestamp=datetime.now().strftime("%H:%M:%S"),
        ))
