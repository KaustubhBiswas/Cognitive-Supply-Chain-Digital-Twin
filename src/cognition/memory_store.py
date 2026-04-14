"""
Persistent Episode Memory Store

Stores cognitive workflow episodes on disk and retrieves relevant prior episodes
for memory-aware planning.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EpisodeMemoryStore:
    """Simple JSON-backed episodic memory store."""

    def __init__(self, file_path: Optional[str] = None, max_episodes: int = 500):
        path = file_path or os.getenv("AGENT_MEMORY_PATH", str(Path("data") / "agent_memory" / "episodes.json"))
        self.file_path = Path(path)
        self.max_episodes = max(10, int(max_episodes))
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file()

    def _ensure_file(self) -> None:
        if self.file_path.exists():
            return
        self._write_all([])

    def _read_all(self) -> List[Dict[str, Any]]:
        try:
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
            episodes = payload.get("episodes", [])
            if isinstance(episodes, list):
                return [e for e in episodes if isinstance(e, dict)]
        except Exception as e:
            logger.warning("Memory store read failed (%s). Resetting memory file.", e)
        self._write_all([])
        return []

    def _write_all(self, episodes: List[Dict[str, Any]]) -> None:
        payload = {"episodes": episodes[-self.max_episodes :]}
        self.file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def append_episode(self, episode: Dict[str, Any]) -> None:
        episodes = self._read_all()
        episodes.append(episode)
        self._write_all(episodes)

    def retrieve_relevant(
        self,
        objective: str,
        alert_type: Optional[str] = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        objective_tokens = set(str(objective).lower().split())
        episodes = self._read_all()

        scored: List[tuple[float, Dict[str, Any]]] = []
        for ep in episodes:
            score = 0.0
            ep_objective = str(ep.get("objective", "")).lower()
            ep_tokens = set(ep_objective.split())

            overlap = len(objective_tokens & ep_tokens)
            score += overlap * 0.3

            ep_alert = str(ep.get("alert_type", "")).lower()
            if alert_type and ep_alert == str(alert_type).lower():
                score += 2.0

            if str(ep.get("plan_status", "")).lower() == "completed":
                score += 1.2
            elif str(ep.get("plan_status", "")).lower() == "in_progress":
                score += 0.4

            if ep.get("replan_count", 0) == 0:
                score += 0.2

            if score > 0:
                scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[: max(1, int(limit))]]

    @staticmethod
    def build_prompt_memory_context(memories: List[Dict[str, Any]], max_chars: int = 1200) -> str:
        if not memories:
            return ""

        lines: List[str] = []
        for i, mem in enumerate(memories, start=1):
            lines.append(
                f"Memory {i}: objective={mem.get('objective', '')}; "
                f"alert={mem.get('alert_type', '')}; "
                f"status={mem.get('plan_status', '')}; "
                f"replans={mem.get('replan_count', 0)}"
            )
            steps = mem.get("plan_steps", []) or []
            compact_steps = []
            for step in steps[:4]:
                compact_steps.append(
                    f"{step.get('owner', '?')}:{step.get('title', '')}"
                )
            if compact_steps:
                lines.append("  Steps: " + " | ".join(compact_steps))

        text = "\n".join(lines)
        if len(text) > max_chars:
            return text[: max_chars - 3] + "..."
        return text


_DEFAULT_MEMORY_STORE: Optional[EpisodeMemoryStore] = None


def get_default_memory_store() -> EpisodeMemoryStore:
    """Return a singleton memory store instance for cognition modules."""
    global _DEFAULT_MEMORY_STORE
    if _DEFAULT_MEMORY_STORE is None:
        _DEFAULT_MEMORY_STORE = EpisodeMemoryStore()
    return _DEFAULT_MEMORY_STORE
