"""
Probabilistic Risk Engine

Bayesian risk scoring engine that continuously evaluates supply chain health,
predicts state transitions, and generates preemptive alerts BEFORE issues occur.

This is the core of the preemptive approach — instead of waiting for thresholds
to be crossed, it maintains probability distributions over node health states
and triggers cognitive workflows when degradation probability rises.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Risk State Definitions
# =============================================================================

class NodeHealthState(str, Enum):
    """Discrete health states for each supply chain node."""
    HEALTHY = "healthy"           # Normal operations, adequate buffers
    AT_RISK = "at_risk"           # Early warning signs, trending toward degradation
    DEGRADED = "degraded"         # Performance below targets, action needed
    CRITICAL = "critical"         # Imminent or active failure


@dataclass
class RiskState:
    """
    Probabilistic risk state for a single node.
    
    Maintains a probability distribution over health states rather than
    a single binary healthy/unhealthy classification.
    """
    node_id: int
    probabilities: Dict[str, float] = field(default_factory=lambda: {
        "healthy": 0.7,
        "at_risk": 0.2,
        "degraded": 0.08,
        "critical": 0.02,
    })
    # Contributing risk factors (0-1 each)
    inventory_risk: float = 0.0       # Low inventory runway
    demand_volatility_risk: float = 0.0  # High demand coefficient of variation
    bullwhip_exposure: float = 0.0    # Upstream order amplification
    lead_time_risk: float = 0.0       # Insufficient lead time buffer
    trend_risk: float = 0.0           # Worsening trajectory
    
    # History for trend detection
    risk_score_history: List[float] = field(default_factory=list)
    state_history: List[str] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    
    @property
    def composite_risk(self) -> float:
        """Weighted composite risk score (0-1)."""
        weights = {
            "inventory_risk": 0.35,
            "demand_volatility_risk": 0.20,
            "bullwhip_exposure": 0.15,
            "lead_time_risk": 0.15,
            "trend_risk": 0.15,
        }
        score = (
            self.inventory_risk * weights["inventory_risk"]
            + self.demand_volatility_risk * weights["demand_volatility_risk"]
            + self.bullwhip_exposure * weights["bullwhip_exposure"]
            + self.lead_time_risk * weights["lead_time_risk"]
            + self.trend_risk * weights["trend_risk"]
        )
        return min(max(score, 0.0), 1.0)
    
    @property
    def most_likely_state(self) -> str:
        """Return the most probable health state."""
        return max(self.probabilities, key=self.probabilities.get)
    
    @property
    def degradation_probability(self) -> float:
        """Probability of being in degraded or critical state."""
        return self.probabilities.get("degraded", 0) + self.probabilities.get("critical", 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for state transport."""
        return {
            "node_id": self.node_id,
            "probabilities": self.probabilities.copy(),
            "composite_risk": round(self.composite_risk, 4),
            "most_likely_state": self.most_likely_state,
            "degradation_probability": round(self.degradation_probability, 4),
            "risk_factors": {
                "inventory_risk": round(self.inventory_risk, 4),
                "demand_volatility_risk": round(self.demand_volatility_risk, 4),
                "bullwhip_exposure": round(self.bullwhip_exposure, 4),
                "lead_time_risk": round(self.lead_time_risk, 4),
                "trend_risk": round(self.trend_risk, 4),
            },
        }


@dataclass
class OptimizationOpportunity:
    """
    An identified efficiency improvement — not a problem to fix,
    but a way to make the already-working system perform better.
    """
    opportunity_type: str  # "redistribute_inventory", "smooth_orders", "reduce_safety_stock"
    target_nodes: List[int]
    estimated_benefit: float  # 0-1, estimated improvement magnitude
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "opportunity_type": self.opportunity_type,
            "target_nodes": self.target_nodes,
            "estimated_benefit": round(self.estimated_benefit, 4),
            "parameters": self.parameters,
            "reasoning": self.reasoning,
        }


# =============================================================================
# Transition Matrix
# =============================================================================

# Base Markov transition probabilities P(next_state | current_state)
# These are modulated by risk factors at runtime
BASE_TRANSITION_MATRIX = {
    "healthy":  {"healthy": 0.85, "at_risk": 0.12, "degraded": 0.025, "critical": 0.005},
    "at_risk":  {"healthy": 0.20, "at_risk": 0.55, "degraded": 0.20, "critical": 0.05},
    "degraded": {"healthy": 0.05, "at_risk": 0.15, "degraded": 0.55, "critical": 0.25},
    "critical": {"healthy": 0.01, "at_risk": 0.04, "degraded": 0.25, "critical": 0.70},
}


# =============================================================================
# Risk Engine
# =============================================================================

class RiskEngine:
    """
    Probabilistic risk scoring and prediction engine.
    
    Runs each simulation step to:
    1. Score every node's risk across multiple factors
    2. Update probability distributions over health states
    3. Predict future state transitions (Markov model)
    4. Generate preemptive alerts when P(degraded) crosses thresholds
    5. Identify efficiency optimization opportunities
    
    Usage:
        engine = RiskEngine()
        # Each simulation step:
        alerts, opportunities = engine.scan_network(simulation_model)
    """
    
    def __init__(
        self,
        alert_threshold: float = 0.30,
        opportunity_threshold: float = 0.15,
        history_window: int = 20,
        trend_window: int = 5,
    ):
        """
        Args:
            alert_threshold: P(degraded|critical) threshold to generate preemptive alert
            opportunity_threshold: Excess capacity threshold to flag optimization opportunity
            history_window: Number of steps to keep in risk history
            trend_window: Number of recent steps for trend calculation
        """
        self.alert_threshold = alert_threshold
        self.opportunity_threshold = opportunity_threshold
        self.history_window = history_window
        self.trend_window = trend_window
        
        # Per-node risk states
        self.node_risks: Dict[int, RiskState] = {}
        
        # Network-level metrics history
        self.network_health_history: List[float] = []
        self.scan_count: int = 0
    
    def scan_network(self, model) -> Tuple[List[Dict], List[Dict]]:
        """
        Scan the entire supply chain network and generate preemptive alerts
        and optimization opportunities.
        
        Args:
            model: SupplyChainModel instance
            
        Returns:
            Tuple of (preemptive_alerts, optimization_opportunities)
        """
        self.scan_count += 1
        now = datetime.now()
        
        agents = self._get_agents(model)
        if not agents:
            return [], []
        
        preemptive_alerts = []
        opportunities = []
        network_risk_sum = 0.0
        
        for agent in agents:
            node_id = agent.unique_id
            
            # Initialize risk state if new node
            if node_id not in self.node_risks:
                self.node_risks[node_id] = RiskState(node_id=node_id)
            
            risk_state = self.node_risks[node_id]
            
            # ---- Step 1: Score individual risk factors ----
            self._score_inventory_risk(risk_state, agent)
            self._score_demand_volatility(risk_state, agent)
            self._score_bullwhip_exposure(risk_state, agent)
            self._score_lead_time_risk(risk_state, agent)
            self._score_trend_risk(risk_state)
            
            # ---- Step 2: Update probability distribution ----
            self._update_probabilities(risk_state)
            
            # ---- Step 3: Predict state transitions ----
            predicted_probs = self._predict_transitions(risk_state)
            
            # ---- Step 4: Record history ----
            risk_state.risk_score_history.append(risk_state.composite_risk)
            if len(risk_state.risk_score_history) > self.history_window:
                risk_state.risk_score_history.pop(0)
            
            risk_state.state_history.append(risk_state.most_likely_state)
            if len(risk_state.state_history) > self.history_window:
                risk_state.state_history.pop(0)
            
            risk_state.last_updated = now
            network_risk_sum += risk_state.composite_risk
            
            # ---- Step 5: Check alert thresholds ----
            degradation_prob = risk_state.degradation_probability
            predicted_degradation = predicted_probs.get("degraded", 0) + predicted_probs.get("critical", 0)
            
            # Alert if current OR predicted degradation exceeds threshold
            if degradation_prob >= self.alert_threshold or predicted_degradation >= self.alert_threshold * 1.2:
                alert = self._create_preemptive_alert(risk_state, predicted_probs, agent)
                if alert:
                    preemptive_alerts.append(alert)
            
            # ---- Step 6: Check optimization opportunities ----
            opp = self._check_optimization_opportunity(risk_state, agent)
            if opp:
                opportunities.append(opp)
        
        # Record network-level health
        avg_risk = network_risk_sum / max(len(agents), 1)
        self.network_health_history.append(1.0 - avg_risk)
        if len(self.network_health_history) > self.history_window:
            self.network_health_history.pop(0)
        
        logger.debug(
            f"Risk scan #{self.scan_count}: {len(agents)} nodes, "
            f"avg_risk={avg_risk:.3f}, "
            f"alerts={len(preemptive_alerts)}, "
            f"opportunities={len(opportunities)}"
        )
        
        return preemptive_alerts, opportunities
    
    # =========================================================================
    # Risk Factor Scoring
    # =========================================================================
    
    def _score_inventory_risk(self, risk_state: RiskState, agent) -> None:
        """
        Score inventory risk based on runway (days of stock remaining).
        
        Uses a sigmoid function centered on the reorder point to produce
        a smooth risk score rather than a binary threshold.
        """
        inventory = getattr(agent, 'inventory', 0)
        reorder_point = getattr(agent, 'reorder_point', 20.0)
        demand_rate = getattr(agent, 'demand_mean', 10.0)
        
        if demand_rate <= 0:
            risk_state.inventory_risk = 0.0
            return
        
        days_of_stock = inventory / demand_rate
        # Sigmoid: 0 when days_of_stock >> reorder_days, 1 when days_of_stock << reorder_days
        reorder_days = reorder_point / demand_rate if demand_rate > 0 else 5.0
        # Center sigmoid at reorder_days, steepness controlled by k
        k = 2.0 / max(reorder_days, 1)
        x = k * (days_of_stock - reorder_days)

        # Numerically stable form of 1 / (1 + exp(x)) to avoid overflow.
        if x >= 0:
            z = math.exp(-x)
            risk_state.inventory_risk = z / (1.0 + z)
        else:
            z = math.exp(x)
            risk_state.inventory_risk = 1.0 / (1.0 + z)
    
    def _score_demand_volatility(self, risk_state: RiskState, agent) -> None:
        """
        Score demand volatility using coefficient of variation (CV).
        
        High CV means demand is unpredictable → harder to maintain optimal inventory.
        """
        demand_history = getattr(agent, 'demand_history', [])
        
        if len(demand_history) < 3:
            risk_state.demand_volatility_risk = 0.1  # Insufficient data
            return
        
        recent = demand_history[-self.history_window:]
        mean_demand = np.mean(recent)
        std_demand = np.std(recent)
        
        if mean_demand <= 0:
            risk_state.demand_volatility_risk = 0.0
            return
        
        cv = std_demand / mean_demand  # Coefficient of variation
        # Map CV to risk: CV < 0.2 = low risk, CV > 0.8 = high risk
        risk_state.demand_volatility_risk = min(cv / 0.8, 1.0)
    
    def _score_bullwhip_exposure(self, risk_state: RiskState, agent) -> None:
        """
        Score how exposed this node is to bullwhip amplification.
        
        Compares order variance to demand variance — high ratio means
        this node is amplifying demand signals upstream.
        """
        orders = getattr(agent, 'orders_placed', [])
        demands = getattr(agent, 'demand_history', [])
        
        if len(orders) < 5 or len(demands) < 5:
            risk_state.bullwhip_exposure = 0.1
            return
        
        recent_orders = orders[-self.history_window:]
        recent_demands = demands[-self.history_window:]
        
        order_var = np.var(recent_orders) if recent_orders else 0
        demand_var = np.var(recent_demands) if recent_demands else 0
        
        if demand_var < 1e-6:
            risk_state.bullwhip_exposure = 0.0
            return
        
        bullwhip_ratio = order_var / demand_var
        # Map ratio to risk: < 1.0 = dampening (good), > 2.0 = severe
        if bullwhip_ratio <= 1.0:
            risk_state.bullwhip_exposure = 0.0
        else:
            risk_state.bullwhip_exposure = min((bullwhip_ratio - 1.0) / 2.0, 1.0)
    
    def _score_lead_time_risk(self, risk_state: RiskState, agent) -> None:
        """
        Score whether the current inventory can cover the lead time.
        
        If inventory < demand_rate * lead_time, there's a stockout risk
        during the replenishment window.
        """
        inventory = getattr(agent, 'inventory', 0)
        lead_time = getattr(agent, 'lead_time', 2)
        demand_rate = getattr(agent, 'demand_mean', 10.0)
        pending = sum(
            o.get("quantity", 0) if isinstance(o, dict) else getattr(o, 'quantity', 0)
            for o in getattr(agent, 'pending_orders', [])
        )
        
        # Effective inventory = current + incoming orders
        effective_inventory = inventory + pending
        lead_time_demand = demand_rate * lead_time
        
        if lead_time_demand <= 0:
            risk_state.lead_time_risk = 0.0
            return
        
        coverage_ratio = effective_inventory / lead_time_demand
        # Risk increases as coverage drops below 1.5x (lean buffer)
        if coverage_ratio >= 2.0:
            risk_state.lead_time_risk = 0.0
        elif coverage_ratio >= 1.0:
            risk_state.lead_time_risk = 1.0 - (coverage_ratio - 1.0)
        else:
            risk_state.lead_time_risk = min(1.0, 1.5 - coverage_ratio * 0.5)
    
    def _score_trend_risk(self, risk_state: RiskState) -> None:
        """
        Score whether risk is trending upward (worsening trajectory).
        
        Uses the slope of recent risk scores — positive slope = worsening.
        """
        history = risk_state.risk_score_history
        
        if len(history) < self.trend_window:
            risk_state.trend_risk = 0.0
            return
        
        recent = history[-self.trend_window:]
        # Simple linear regression slope
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        
        # Positive slope = worsening, negative = improving
        if slope > 0:
            risk_state.trend_risk = min(slope * 5.0, 1.0)  # Scale factor
        else:
            risk_state.trend_risk = 0.0
    
    # =========================================================================
    # Probability Updates & Prediction
    # =========================================================================
    
    def _update_probabilities(self, risk_state: RiskState) -> None:
        """
        Update health state probabilities based on current risk factors.
        
        Uses a Bayesian-inspired update: prior probabilities are the
        transition matrix predictions, likelihood is derived from
        the composite risk score.
        """
        composite = risk_state.composite_risk
        
        # Generate likelihood of each state given the composite risk
        # These are soft assignments based on risk score ranges
        likelihoods = {
            "healthy": max(0, 1.0 - composite * 2.5),
            "at_risk": self._gaussian_likelihood(composite, mu=0.35, sigma=0.15),
            "degraded": self._gaussian_likelihood(composite, mu=0.65, sigma=0.15),
            "critical": min(1.0, max(0, composite * 2.5 - 1.5)),
        }
        
        # Bayesian update: posterior ∝ likelihood × prior
        prior = risk_state.probabilities
        unnormalized = {
            state: likelihoods[state] * prior.get(state, 0.25)
            for state in NodeHealthState
        }
        
        # Normalize
        total = sum(unnormalized.values())
        if total > 0:
            risk_state.probabilities = {
                state.value: round(prob / total, 6)
                for state, prob in unnormalized.items()
            }
        else:
            # Fallback uniform
            risk_state.probabilities = {s.value: 0.25 for s in NodeHealthState}
    
    def _predict_transitions(self, risk_state: RiskState) -> Dict[str, float]:
        """
        Predict next-step probabilities using the Markov transition matrix,
        modulated by current risk factors.
        """
        current_probs = risk_state.probabilities
        composite = risk_state.composite_risk
        
        # Modulate transition matrix based on risk
        # Higher risk → higher probability of transitioning to worse states
        predicted = {s.value: 0.0 for s in NodeHealthState}
        
        for current_state, prob in current_probs.items():
            if prob < 0.001:
                continue
            
            base_transitions = BASE_TRANSITION_MATRIX.get(current_state, {})
            
            for next_state, trans_prob in base_transitions.items():
                # Modulate: increase probability of worsening when risk is high
                if self._state_order(next_state) > self._state_order(current_state):
                    # Worsening transition — amplify with risk
                    modulated = trans_prob * (1 + composite * 1.5)
                elif self._state_order(next_state) < self._state_order(current_state):
                    # Improving transition — dampen with risk
                    modulated = trans_prob * (1 - composite * 0.5)
                else:
                    modulated = trans_prob
                
                predicted[next_state] += prob * max(modulated, 0)
        
        # Normalize
        total = sum(predicted.values())
        if total > 0:
            predicted = {k: round(v / total, 6) for k, v in predicted.items()}
        
        return predicted
    
    # =========================================================================
    # Alert & Opportunity Generation
    # =========================================================================
    
    def _create_preemptive_alert(
        self, risk_state: RiskState, predicted_probs: Dict, agent
    ) -> Optional[Dict]:
        """Create a preemptive alert for a node at risk."""
        
        # Determine the primary risk factor driving the alert
        risk_factors = {
            "inventory_runway": risk_state.inventory_risk,
            "demand_volatility": risk_state.demand_volatility_risk,
            "bullwhip_exposure": risk_state.bullwhip_exposure,
            "lead_time_coverage": risk_state.lead_time_risk,
            "worsening_trend": risk_state.trend_risk,
        }
        primary_factor = max(risk_factors, key=risk_factors.get)
        
        # Determine severity from probability distribution
        if risk_state.probabilities.get("critical", 0) > 0.3:
            severity = "critical"
        elif risk_state.degradation_probability > 0.5:
            severity = "high"
        elif risk_state.degradation_probability > 0.3:
            severity = "medium"
        else:
            severity = "low"
        
        return {
            "alert_type": "preemptive_warning",
            "severity": severity,
            "affected_nodes": [risk_state.node_id],
            "details": {
                "composite_risk": round(risk_state.composite_risk, 4),
                "degradation_probability": round(risk_state.degradation_probability, 4),
                "predicted_degradation": round(
                    predicted_probs.get("degraded", 0) + predicted_probs.get("critical", 0), 4
                ),
                "primary_risk_factor": primary_factor,
                "risk_factors": risk_factors,
                "probabilities": risk_state.probabilities.copy(),
                "most_likely_state": risk_state.most_likely_state,
                "node_type": getattr(agent, 'node_type', 'unknown'),
                "inventory": float(getattr(agent, 'inventory', 0)),
                "recommended_action": self._recommend_preemptive_action(
                    primary_factor, risk_state, agent
                ),
            },
            "timestamp": datetime.now().isoformat(),
            "alert_id": f"preemptive_{risk_state.node_id}_{self.scan_count}",
        }
    
    def _check_optimization_opportunity(
        self, risk_state: RiskState, agent
    ) -> Optional[Dict]:
        """Check if this node has an optimization opportunity."""
        inventory = getattr(agent, 'inventory', 0)
        order_up_to = getattr(agent, 'order_up_to', 100.0)
        demand_rate = getattr(agent, 'demand_mean', 10.0)
        
        # Only flag opportunities for healthy nodes
        if risk_state.most_likely_state != "healthy":
            return None
        
        # Check for excess inventory (could be redistributed or reduced)
        if demand_rate > 0:
            days_of_stock = inventory / demand_rate
            
            if days_of_stock > 15 and inventory > order_up_to * 1.3:
                return OptimizationOpportunity(
                    opportunity_type="redistribute_inventory",
                    target_nodes=[risk_state.node_id],
                    estimated_benefit=min((days_of_stock - 10) / 20, 0.5),
                    parameters={
                        "excess_inventory": round(inventory - order_up_to, 2),
                        "days_of_stock": round(days_of_stock, 1),
                    },
                    reasoning=(
                        f"Node {risk_state.node_id} has {days_of_stock:.0f} days of stock "
                        f"(excess: {inventory - order_up_to:.0f} units). "
                        f"Redistributing to at-risk nodes improves network resilience."
                    ),
                ).to_dict()
        
        # Check for order smoothing opportunity (low bullwhip = can order less frequently)
        if risk_state.bullwhip_exposure < 0.1 and risk_state.composite_risk < 0.15:
            orders = getattr(agent, 'orders_placed', [])
            if len(orders) > 5:
                order_cv = np.std(orders[-10:]) / max(np.mean(orders[-10:]), 1)
                if order_cv > 0.3:
                    return OptimizationOpportunity(
                        opportunity_type="smooth_orders",
                        target_nodes=[risk_state.node_id],
                        estimated_benefit=min(order_cv * 0.5, 0.3),
                        parameters={"current_order_cv": round(order_cv, 3)},
                        reasoning=(
                            f"Node {risk_state.node_id} is healthy but has variable ordering "
                            f"(CV={order_cv:.2f}). Smoothing reduces upstream amplification."
                        ),
                    ).to_dict()
        
        return None
    
    def _recommend_preemptive_action(
        self, primary_factor: str, risk_state: RiskState, agent
    ) -> str:
        """Generate a specific preemptive action recommendation."""
        recommendations = {
            "inventory_runway": (
                f"Increase reorder point by {int(risk_state.inventory_risk * 30)}% "
                f"or place an early replenishment order"
            ),
            "demand_volatility": (
                f"Increase safety stock to absorb demand variance "
                f"(CV={risk_state.demand_volatility_risk:.2f})"
            ),
            "bullwhip_exposure": (
                f"Apply order smoothing (α=0.3) to dampen upstream amplification "
                f"(bullwhip score={risk_state.bullwhip_exposure:.2f})"
            ),
            "lead_time_coverage": (
                f"Expedite pending orders or source from backup supplier — "
                f"current stock may not cover lead time"
            ),
            "worsening_trend": (
                f"Risk has been rising for {self.trend_window} periods — "
                f"review and adjust inventory policy proactively"
            ),
        }
        return recommendations.get(primary_factor, "Review node status and adjust policy")
    
    # =========================================================================
    # Reports & Queries
    # =========================================================================
    
    def get_network_risk_summary(self) -> Dict[str, Any]:
        """Get a summary of risk across the entire network."""
        if not self.node_risks:
            return {"error": "No risk data available", "scan_count": 0}
        
        states_count = {"healthy": 0, "at_risk": 0, "degraded": 0, "critical": 0}
        risk_scores = []
        
        for rs in self.node_risks.values():
            states_count[rs.most_likely_state] += 1
            risk_scores.append(rs.composite_risk)
        
        avg_risk = np.mean(risk_scores) if risk_scores else 0
        max_risk_node = max(self.node_risks.values(), key=lambda rs: rs.composite_risk)
        
        return {
            "scan_count": self.scan_count,
            "total_nodes": len(self.node_risks),
            "state_distribution": states_count,
            "average_risk": round(avg_risk, 4),
            "max_risk_node": max_risk_node.node_id,
            "max_risk_score": round(max_risk_node.composite_risk, 4),
            "network_health": round(1.0 - avg_risk, 4),
            "nodes_at_risk": [
                rs.node_id for rs in self.node_risks.values()
                if rs.degradation_probability >= self.alert_threshold
            ],
        }
    
    def get_node_risk(self, node_id: int) -> Optional[Dict]:
        """Get detailed risk state for a specific node."""
        rs = self.node_risks.get(node_id)
        return rs.to_dict() if rs else None
    
    def get_all_risk_scores(self) -> Dict[int, Dict]:
        """Get risk scores for all nodes."""
        return {nid: rs.to_dict() for nid, rs in self.node_risks.items()}
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    @staticmethod
    def _gaussian_likelihood(x: float, mu: float, sigma: float) -> float:
        """Gaussian PDF for likelihood computation."""
        return math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    @staticmethod
    def _state_order(state: str) -> int:
        """Numeric ordering of states for comparison."""
        return {"healthy": 0, "at_risk": 1, "degraded": 2, "critical": 3}.get(state, 0)
    
    @staticmethod
    def _get_agents(model) -> list:
        """Get agents from model, handling different Mesa versions."""
        if hasattr(model, '_agents_by_id'):
            return list(model._agents_by_id.values())
        elif hasattr(model, 'agents'):
            return list(model.agents)
        elif hasattr(model, 'schedule') and hasattr(model.schedule, 'agents'):
            return list(model.schedule.agents)
        return []
