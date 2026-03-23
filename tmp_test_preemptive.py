"""End-to-end test for preemptive cognition enhancements."""
from src.cognition import RiskEngine, PreemptiveMonitor, initialize_tools, create_supply_chain_graph
from src.data.parser import create_synthetic_supply_graph
from src.simulation import SupplyChainModel

# Setup simulation
data = create_synthetic_supply_graph(num_suppliers=2, num_manufacturers=2, num_distributors=2, num_retailers=3, seed=42)
model = SupplyChainModel(graph=data.graph, node_types=data.node_types, random_seed=42)

# Initialize
initialize_tools(simulation=model)
engine = RiskEngine(alert_threshold=0.25)
monitor = PreemptiveMonitor(risk_engine=engine, alert_cooldown=3)

# Run 20 steps with monitoring
total_alerts = 0
total_opps = 0
for i in range(20):
    model.step()
    report = monitor.on_step(model)
    total_alerts += report.get("alerts_generated", 0)
    total_opps += report.get("opportunities_found", 0)

# Results
health = monitor.get_network_health_report()
print(f"Steps: {health['step']}")
print(f"Total preemptive alerts: {total_alerts}")
print(f"Total optimization opportunities: {total_opps}")
print(f"Network health: {health['risk_summary']['network_health']}")
print(f"Nodes at risk: {health['risk_summary']['nodes_at_risk']}")
print(f"State distribution: {health['risk_summary']['state_distribution']}")
print(f"Trend: {health['network_trend']}")
print(f"Max risk node: {health['risk_summary']['max_risk_node']} (score: {health['risk_summary']['max_risk_score']})")

# Verify node risk detail
node_risk = monitor.get_node_risk_detail(health["risk_summary"]["max_risk_node"])
if node_risk:
    print(f"Max risk node probabilities: {node_risk['probabilities']}")
    print(f"Risk factors: {node_risk['risk_factors']}")

# Verify pending alerts are retrievable
pending = monitor.get_pending_alerts()
print(f"Pending alerts: {len(pending)}")

print("\nEND-TO-END TEST PASSED")
