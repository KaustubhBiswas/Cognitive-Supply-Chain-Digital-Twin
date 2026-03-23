"""
Preemptive Monitor

Continuous monitoring layer that bridges the simulation and cognitive system.
Hooks into each simulation step to run probabilistic risk scanning, auto-generate
alerts, and feed them into the cognitive workflow — all before issues occur.

Usage:
    monitor = PreemptiveMonitor()
    # In simulation loop:
    for step in range(100):
        model.step()
        monitor.on_step(model, cognitive_graph)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .risk_engine import RiskEngine
from .state import Alert, AlertSeverity, AlertType, create_initial_state

logger = logging.getLogger(__name__)


class PreemptiveMonitor:
    """
    Continuous preemptive scanner that bridges simulation → cognition.
    
    Called after each simulation step to:
    1. Run risk engine scan
    2. Auto-generate Alert objects from risk predictions
    3. Optionally feed alerts into the cognitive graph
    4. Track alert history and network health trends
    5. Collect efficiency optimization opportunities
    
    This is the key component that transforms the system from reactive
    (wait for alert → respond) to preemptive (predict → prevent).
    """
    
    def __init__(
        self,
        risk_engine: Optional[RiskEngine] = None,
        auto_run_cognition: bool = False,
        alert_cooldown: int = 5,
        min_scan_interval: int = 1,
    ):
        """
        Args:
            risk_engine: RiskEngine instance (created if not provided)
            auto_run_cognition: Whether to automatically invoke cognitive graph
            alert_cooldown: Minimum steps between alerts for the same node
            min_scan_interval: Minimum steps between risk scans (default every step)
        """
        self.risk_engine = risk_engine or RiskEngine()
        self.auto_run_cognition = auto_run_cognition
        self.alert_cooldown = alert_cooldown
        self.min_scan_interval = min_scan_interval
        
        # Tracking
        self.step_count: int = 0
        self.alert_history: List[Dict[str, Any]] = []
        self.opportunity_history: List[Dict[str, Any]] = []
        self.cognitive_results: List[Dict[str, Any]] = []
        
        # Cooldown tracking: node_id → last alert step
        self._alert_cooldowns: Dict[int, int] = {}
        
        # Pending items for external consumption
        self._pending_alerts: List[Dict] = []
        self._pending_opportunities: List[Dict] = []
    
    def on_step(
        self,
        model,
        cognitive_graph=None,
    ) -> Dict[str, Any]:
        """
        Called after each simulation step. Runs risk scan and generates
        preemptive alerts.
        
        Args:
            model: SupplyChainModel instance
            cognitive_graph: Optional compiled LangGraph or FallbackGraph
                           (if provided and auto_run_cognition=True, alerts
                            are fed directly into the cognitive workflow)
        
        Returns:
            Step report with alerts, opportunities, and risk summary
        """
        self.step_count += 1
        
        # Check if we should scan this step
        if self.step_count % self.min_scan_interval != 0:
            return {"scanned": False, "step": self.step_count}
        
        # Run risk engine scan
        raw_alerts, raw_opportunities = self.risk_engine.scan_network(model)
        
        # Apply alert cooldowns (don't spam the same node)
        filtered_alerts = self._apply_cooldowns(raw_alerts)
        
        # Store pending items
        self._pending_alerts.extend(filtered_alerts)
        self._pending_opportunities.extend(raw_opportunities)
        
        # Record history
        for alert in filtered_alerts:
            self.alert_history.append({
                **alert,
                "monitor_step": self.step_count,
            })
        for opp in raw_opportunities:
            self.opportunity_history.append({
                **opp,
                "monitor_step": self.step_count,
            })
        
        # Auto-run cognitive graph if enabled
        cognitive_result = None
        if self.auto_run_cognition and cognitive_graph and filtered_alerts:
            cognitive_result = self._run_cognitive_workflow(
                filtered_alerts, cognitive_graph
            )
        
        # Build step report
        report = {
            "scanned": True,
            "step": self.step_count,
            "alerts_generated": len(filtered_alerts),
            "alerts_suppressed": len(raw_alerts) - len(filtered_alerts),
            "opportunities_found": len(raw_opportunities),
            "risk_summary": self.risk_engine.get_network_risk_summary(),
            "cognitive_result": cognitive_result,
        }
        
        if filtered_alerts:
            logger.info(
                f"Step {self.step_count}: Generated {len(filtered_alerts)} "
                f"preemptive alerts, {len(raw_opportunities)} opportunities"
            )
        
        return report
    
    def get_pending_alerts(self, clear: bool = True) -> List[Dict]:
        """
        Get alerts generated since last retrieval.
        
        Args:
            clear: If True, clears the pending queue after retrieval
            
        Returns:
            List of preemptive alert dicts
        """
        alerts = list(self._pending_alerts)
        if clear:
            self._pending_alerts.clear()
        return alerts
    
    def get_pending_opportunities(self, clear: bool = True) -> List[Dict]:
        """
        Get optimization opportunities generated since last retrieval.
        
        Args:
            clear: If True, clears the pending queue after retrieval
            
        Returns:
            List of optimization opportunity dicts
        """
        opps = list(self._pending_opportunities)
        if clear:
            self._pending_opportunities.clear()
        return opps
    
    def get_network_health_report(self) -> Dict[str, Any]:
        """
        Dashboard-ready network health summary.
        
        Returns:
            Comprehensive health report with risk summary, trends,
            recent alerts, and optimization opportunities
        """
        risk_summary = self.risk_engine.get_network_risk_summary()
        
        # Recent alerts (last 10)
        recent_alerts = self.alert_history[-10:] if self.alert_history else []
        
        # Recent opportunities (last 5)
        recent_opps = self.opportunity_history[-5:] if self.opportunity_history else []
        
        # Trend
        health_history = self.risk_engine.network_health_history
        if len(health_history) >= 2:
            trend = health_history[-1] - health_history[-5] if len(health_history) >= 5 else health_history[-1] - health_history[0]
            if trend > 0.02:
                trend_label = "improving"
            elif trend < -0.02:
                trend_label = "degrading"
            else:
                trend_label = "stable"
        else:
            trend_label = "insufficient_data"
            trend = 0.0
        
        return {
            "step": self.step_count,
            "risk_summary": risk_summary,
            "network_trend": trend_label,
            "trend_value": round(trend, 4),
            "total_alerts_generated": len(self.alert_history),
            "total_opportunities_found": len(self.opportunity_history),
            "recent_alerts": recent_alerts,
            "recent_opportunities": recent_opps,
            "cognitive_interventions": len(self.cognitive_results),
        }
    
    def get_node_risk_detail(self, node_id: int) -> Optional[Dict]:
        """Get detailed risk information for a specific node."""
        return self.risk_engine.get_node_risk(node_id)
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _apply_cooldowns(self, alerts: List[Dict]) -> List[Dict]:
        """Filter alerts based on per-node cooldown periods."""
        filtered = []
        for alert in alerts:
            nodes = alert.get("affected_nodes", [])
            if not nodes:
                filtered.append(alert)
                continue
            
            node_id = nodes[0]
            last_alert_step = self._alert_cooldowns.get(node_id, -999)
            
            if self.step_count - last_alert_step >= self.alert_cooldown:
                filtered.append(alert)
                self._alert_cooldowns[node_id] = self.step_count
        
        return filtered
    
    def _run_cognitive_workflow(
        self, alerts: List[Dict], cognitive_graph
    ) -> Optional[Dict]:
        """
        Run the cognitive workflow for the most critical preemptive alert.
        """
        if not alerts:
            return None
        
        # Pick the most critical alert
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        alerts_sorted = sorted(
            alerts,
            key=lambda a: severity_order.get(a.get("severity", "low"), 0),
            reverse=True,
        )
        top_alert = alerts_sorted[0]
        
        try:
            # Convert to Alert object
            alert_obj = Alert(
                alert_type=AlertType.PREEMPTIVE_WARNING,
                severity=AlertSeverity(top_alert.get("severity", "medium")),
                affected_nodes=top_alert.get("affected_nodes", []),
                details=top_alert.get("details", {}),
                alert_id=top_alert.get("alert_id"),
            )
            
            # Create initial state and run
            state = create_initial_state(alert=alert_obj)
            state["preemptive_mode"] = True
            state["risk_scores"] = self.risk_engine.get_all_risk_scores()
            
            result = cognitive_graph.invoke(state)
            
            self.cognitive_results.append({
                "step": self.step_count,
                "alert": top_alert,
                "recommendations": result.get("recommendations", []),
                "analysis": result.get("analysis_results"),
            })
            
            return {
                "processed": True,
                "alert_id": top_alert.get("alert_id"),
                "recommendations_count": len(result.get("recommendations", [])),
            }
            
        except Exception as e:
            logger.error(f"Cognitive workflow failed for preemptive alert: {e}")
            return {"processed": False, "error": str(e)}
