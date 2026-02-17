#!/usr/bin/env python3
"""
Cognition Layer Demo - Shows the multi-agent workflow in action
"""

import logging

logging.basicConfig(level=logging.WARNING)

from src.cognition import (Alert, AlertSeverity, AlertType, FallbackGraph,
                           create_initial_state, create_supply_chain_graph,
                           initialize_tools)
from src.data.parser import create_synthetic_supply_graph
from src.simulation import SupplyChainModel


def main():
    print("=" * 60)
    print("COGNITION LAYER DEMO")
    print("=" * 60)

    # 1. Setup simulation
    print("\n[1] Creating supply chain simulation...")
    data = create_synthetic_supply_graph()
    model = SupplyChainModel(data.graph, data.node_types, random_seed=42)
    for _ in range(20):
        model.step()
    print(f"    Simulation running: {model.current_step} steps, {len(list(model.agents))} agents")

    # 2. Initialize cognitive tools
    print("\n[2] Initializing cognitive tools...")
    initialize_tools(simulation=model)
    print("    Tools ready")

    # 3. Create an alert scenario
    # Try different scenarios by changing alert_type and details:
    # - DEMAND_SPIKE: details={"current": 150, "previous": 50} → 3x spike → HIGH risk
    # - STOCKOUT: details={} → HIGH risk
    # - BULLWHIP_DETECTED: details={"ratio": 3.5} → HIGH risk (>2.5)
    # - INVENTORY_LOW: details={} → MEDIUM risk
    # - FORECAST_DEVIATION: details={"deviation_pct": 45} → HIGH risk (>30)
    print("\n[3] Creating alert scenario...")
    alert = Alert(
        alert_type=AlertType.LEAD_TIME_CHANGE,
        severity=AlertSeverity.HIGH,
        affected_nodes=[5, 6, 7],
        details={
            "current": 200,
            "previous": 210,
        }
    )
    print(f"    Alert Type: {alert.alert_type.value}")
    print(f"    Severity: {alert.severity.value}")
    print(f"    Affected Nodes: {alert.affected_nodes}")
    print(f"    Details: {alert.details}")

    # 4. Create initial state for cognitive workflow
    print("\n[4] Creating cognitive state...")
    state = create_initial_state(alert=alert)
    print(f"    State keys: {list(state.keys())}")
    print(f"    Iteration count: {state.get('iteration_count', 0)}")

    # 5. Run the cognitive workflow
    print("\n[5] Running cognitive workflow...")
    print("-" * 60)
    
    try:
        graph = create_supply_chain_graph(llm=None)
        graph_type = "LangGraph" if not isinstance(graph, FallbackGraph) else "FallbackGraph"
        print(f"    Workflow type: {graph_type}")
        
        if isinstance(graph, FallbackGraph):
            result = graph.invoke(state)
        else:
            result = graph.invoke(state, config={"configurable": {"thread_id": "demo-1"}})
        
        print("\n[6] WORKFLOW RESULTS:")
        print("-" * 60)
        print(f"    Final iteration: {result.get('iteration', 'N/A')}")
        print(f"    Status: Complete")
        
        # Show analysis results
        analysis = result.get("analysis_results", {})
        if analysis:
            print("\n    ANALYSIS RESULTS:")
            for key, value in analysis.items():
                if isinstance(value, dict):
                    print(f"      {key}:")
                    for k, v in value.items():
                        print(f"        {k}: {v}")
                else:
                    print(f"      {key}: {value}")
        
        # Show recommendations
        recommendations = result.get("recommendations", [])
        if recommendations:
            print(f"\n    RECOMMENDATIONS ({len(recommendations)} total):")
            for i, rec in enumerate(recommendations, 1):
                if hasattr(rec, "recommendation_type"):
                    print(f"      {i}. [{rec.recommendation_type.value}] {rec.description}")
                    print(f"         Target: Node {rec.target_node_id}, Priority: {rec.priority}")
                else:
                    print(f"      {i}. {rec}")
        else:
            print("\n    No recommendations generated")
            
        # Show messages if available
        messages = result.get("messages", [])
        if messages:
            print(f"\n    AGENT MESSAGES ({len(messages)} total):")
            for msg in messages[-5:]:  # Show last 5 messages
                if hasattr(msg, "content"):
                    content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    print(f"      - {content}")
                else:
                    content = str(msg)[:100] + "..." if len(str(msg)) > 100 else str(msg)
                    print(f"      - {content}")
                
    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
