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
from src.cognition.graph import FallbackGraph
from src.cognition.llm import create_llm
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

    def run_cognitive_workflow(self, alert: Alert) -> Dict[str, Any]:
        """Run the full cognitive graph and capture messages."""
        state = self._to_serializable(create_initial_state(alert=alert))
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
            self.action_queue.append(ActionRecord(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                alert_type=alert.alert_type.value,
                severity=alert.severity.value,
                recommendation=rec,
            ))

        return result

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
