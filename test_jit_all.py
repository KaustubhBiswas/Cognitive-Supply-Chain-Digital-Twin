"""Test all JIT implementations."""

from src.cognition.tools import (analyze_disruption_propagation,
                                 estimate_time_to_impact,
                                 generate_cross_node_recommendations,
                                 generate_proactive_alerts,
                                 get_jit_recommendations, initialize_tools,
                                 simulate_disruption_ripple)
from src.data.parser import create_synthetic_supply_graph
from src.simulation.model import SupplyChainModel


def main():
    print("=" * 70)
    print("TESTING ALL JIT IMPLEMENTATIONS")
    print("=" * 70)
    
    # Create supply chain graph
    data = create_synthetic_supply_graph(
        num_suppliers=3,
        num_manufacturers=3,
        num_distributors=4,
        num_retailers=5,
        seed=42,
    )
    
    # Create simulation
    model = SupplyChainModel(
        graph=data.graph,
        node_types=data.node_types,
        random_seed=42,
    )
    
    # Run a few steps to generate some state
    for _ in range(10):
        model.step()
    
    # Initialize tools
    initialize_tools(model)
    
    # Find a node to test with
    test_node = 5
    
    # =========================================================================
    # TEST 1: Time-to-Impact Calculator
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: TIME-TO-IMPACT CALCULATOR")
    print("=" * 70)
    
    result = estimate_time_to_impact.invoke({
        "source_node_id": test_node,
    })
    
    if result.get("success"):
        print(f"Source node: {result['source_node']}")
        print(f"Total nodes affected: {result['total_nodes_affected']}")
        print(f"Average propagation time: {result['average_propagation_time']} steps")
        print(f"\nEarliest impact: Node {result['earliest_impact']['node_id']} in {result['earliest_impact']['time']} steps")
        print(f"Latest impact: Node {result['latest_impact']['node_id']} in {result['latest_impact']['time']} steps")
        
        print("\nDetailed timelines:")
        for nid, timeline in list(result['timelines'].items())[:3]:
            print(f"  Node {nid}: {timeline['total_time_to_impact']} steps")
            print(f"    - Upstream delay: {timeline['breakdown']['upstream_delay']}")
            print(f"    - Transit time: {timeline['breakdown']['transit_time']}")
    else:
        print(f"Error: {result.get('error')}")
    
    # =========================================================================
    # TEST 2: Cross-Node Recommendations
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: CROSS-NODE RECOMMENDATIONS")
    print("=" * 70)
    
    result = generate_cross_node_recommendations.invoke({
        "disrupted_nodes": [test_node],
        "optimization_goal": "minimize_impact",
    })
    
    if result.get("success"):
        print(f"Disrupted nodes: {result['disrupted_nodes']}")
        print(f"Optimization goal: {result['optimization_goal']}")
        print(f"Total nodes coordinated: {result['total_nodes_coordinated']}")
        print(f"Network impact score: {result['network_impact_score']}")
        
        print("\nCoordination groups:")
        for group in result.get('coordination_groups', [])[:3]:
            print(f"  Echelon {group.get('echelon')}: Nodes {group.get('nodes')}")
            print(f"    Type: {group.get('coordination_type')}")
            print(f"    Reason: {group.get('reason')}")
        
        print("\nImplementation sequence (first 5):")
        for step in result.get('sequence', [])[:5]:
            print(f"  Phase {step['phase']}: Node {step['node_id']} - {step['action_type']}")
        
        print("\nNode-specific actions (first 3):")
        for nid, info in list(result.get('node_specific_actions', {}).items())[:3]:
            print(f"  Node {nid} (severity: {info.get('severity', 'N/A')}):")
            for action in info.get('actions', [])[:2]:
                print(f"    - {action.get('type')} (priority: {action.get('priority')})")
    else:
        print(f"Error: {result.get('error')}")
    
    # =========================================================================
    # TEST 3: Proactive Alert Generation
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: PROACTIVE ALERT GENERATION")
    print("=" * 70)
    
    result = generate_proactive_alerts.invoke({})
    
    if result.get("success"):
        print(f"Current disruptions detected: {result['current_disruptions_detected']}")
        print(f"Total nodes at risk: {result['total_nodes_at_risk']}")
        
        print("\nProactive alerts:")
        for alert in result.get('proactive_alerts', [])[:5]:
            print(f"  Node {alert.get('node_id')}:")
            print(f"    Issue: {alert.get('predicted_issue')}")
            print(f"    Severity: {alert.get('predicted_severity')}")
            print(f"    Time to impact: {alert.get('estimated_time_to_impact')} steps")
            print(f"    Recommendation: {alert.get('recommended_action')}")
        
        print("\nRisk nodes (vulnerable inventory):")
        for risk in result.get('risk_nodes', [])[:5]:
            print(f"  Node {risk.get('node_id')}: {risk.get('days_of_stock', 'N/A')} days of stock")
            print(f"    Recommendation: {risk.get('recommendation')}")
        
        print("\nEarly warning timeline:")
        for nid, timeline in list(result.get('early_warning_timeline', {}).items())[:5]:
            print(f"  Node {nid}: Issue expected in {timeline['issue_expected_in']} steps (severity: {timeline['severity']})")
    else:
        print(f"Error: {result.get('error')}")
    
    # =========================================================================
    # TEST 4: Ripple Effect Simulation
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: RIPPLE EFFECT SIMULATION")
    print("=" * 70)
    
    scenario = {
        "node_id": test_node,
        "disruption_type": "stockout",
        "severity": 1.0,
        "duration": 3,
    }
    
    result = simulate_disruption_ripple.invoke({
        "scenario": scenario,
        "simulation_steps": 10,
    })
    
    if result.get("success"):
        print(f"Scenario: {result['scenario']}")
        print(f"Simulation steps: {result['simulation_steps']}")
        
        print(f"\nPeak impact:")
        peak = result.get('peak_impact', {})
        print(f"  Step: {peak.get('step')}")
        print(f"  Nodes affected: {peak.get('nodes_affected')}")
        print(f"  Total severity: {peak.get('total_severity')}")
        
        print(f"\nEstimated recovery step: {result.get('estimated_recovery_step')}")
        print(f"Total cost estimate: {result.get('total_cost_estimate')}")
        
        print("\nTime series (key steps):")
        time_series = result.get('time_series', [])
        for ts in [time_series[0], time_series[len(time_series)//2], time_series[-1]] if len(time_series) >= 3 else time_series:
            print(f"  Step {ts['step']}: {ts['nodes_affected']} nodes, severity {ts['total_severity']:.2f}")
        
        print("\nRecovery trajectory:")
        for rt in result.get('recovery_trajectory', [])[:5]:
            print(f"  Step {rt['step']}: {rt['nodes_affected']} nodes, recovery rate {rt['recovery_rate']:.2%}")
    else:
        print(f"Error: {result.get('error')}")
    
    # =========================================================================
    # INTEGRATION TEST: Full JIT Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("INTEGRATION TEST: FULL JIT ANALYSIS FLOW")
    print("=" * 70)
    
    # Simulate a stockout at the test node
    agent = model._agents_by_id.get(test_node)
    if agent:
        print(f"Setting node {test_node} inventory to 0 to simulate stockout...")
        agent.inventory = 0
    
    # Run propagation analysis
    prop_result = analyze_disruption_propagation.invoke({
        "node_id": test_node,
        "disruption_type": "stockout",
        "disruption_severity": 1.0,
    })
    
    # Get JIT recommendations
    jit_result = get_jit_recommendations.invoke({
        "disrupted_nodes": [test_node],
        "disruption_type": "stockout",
    })
    
    # Get time-to-impact
    time_result = estimate_time_to_impact.invoke({
        "source_node_id": test_node,
    })
    
    # Get cross-node recommendations
    cross_result = generate_cross_node_recommendations.invoke({
        "disrupted_nodes": [test_node],
        "optimization_goal": "expedite_recovery",
    })
    
    # Get proactive alerts
    proactive_result = generate_proactive_alerts.invoke({})
    
    # Simulate ripple
    ripple_result = simulate_disruption_ripple.invoke({
        "scenario": {"node_id": test_node, "disruption_type": "stockout", "severity": 1.0, "duration": 5},
        "simulation_steps": 15,
    })
    
    print("\n=== FULL JIT ANALYSIS SUMMARY ===")
    print(f"Disruption source: Node {test_node}")
    print(f"Total nodes affected: {prop_result.get('total_nodes_affected', 0)}")
    print(f"Earliest impact: Node {time_result.get('earliest_impact', {}).get('node_id')} in {time_result.get('earliest_impact', {}).get('time')} steps")
    print(f"Estimated recovery: {jit_result.get('estimated_recovery_time', 0)} steps")
    print(f"Priority actions: {len(jit_result.get('priority_actions', []))}")
    print(f"Coordination groups: {len(cross_result.get('coordination_groups', []))}")
    print(f"Proactive alerts generated: {len(proactive_result.get('proactive_alerts', []))}")
    print(f"Peak impact at step {ripple_result.get('peak_impact', {}).get('step', 0)} with severity {ripple_result.get('peak_impact', {}).get('total_severity', 0):.2f}")
    print(f"Estimated total cost: {ripple_result.get('total_cost_estimate', 0):.2f}")
    
    print("\n" + "=" * 70)
    print("ALL JIT TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
