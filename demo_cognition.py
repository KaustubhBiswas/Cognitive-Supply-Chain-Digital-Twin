#!/usr/bin/env python3
"""
Cognition Layer Demo - Shows the multi-agent workflow in action

Requires:
    - Groq API key in .env file or GROQ_API_KEY environment variable
    - Get your free key at: https://console.groq.com
"""

import logging
import os

# Load .env file
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING)

from src.cognition import (GROQ_MODELS, Alert, AlertSeverity,  # LLM utilities
                           AlertType, FallbackGraph, create_groq_llm,
                           create_initial_state, create_ollama_llm,
                           create_supply_chain_graph, initialize_tools)
from src.data.parser import create_synthetic_supply_graph
from src.simulation import SupplyChainModel

# LLM Configuration (loaded from .env)
USE_LLM = True  # Set to False to use rule-based fallback
LLM_PROVIDER = "groq"  # "groq" (cloud) or "ollama" (local)
GROQ_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL = "llama3:8b"  # For local deployment

# API key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


def get_llm():
    """Initialize the LLM using the cognition module. Returns None if not available."""
    if not USE_LLM:
        print("    LLM disabled (USE_LLM=False), using rule-based fallback")
        return None
    
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            print("    GROQ_API_KEY not set, using rule-based fallback")
            print("    Get your free API key at: https://console.groq.com")
            return None
        
        llm = create_groq_llm(api_key=GROQ_API_KEY, model=GROQ_MODEL)
        if llm:
            print(f"    LLM connected: Groq/{GROQ_MODEL}")
        else:
            print("    Failed to connect to Groq, using rule-based fallback")
        return llm
    
    elif LLM_PROVIDER == "ollama":
        llm = create_ollama_llm(model=OLLAMA_MODEL)
        if llm:
            print(f"    LLM connected: Ollama/{OLLAMA_MODEL}")
        else:
            print("    Failed to connect to Ollama, using rule-based fallback")
            print("    Make sure Ollama is running: ollama serve")
        return llm
    
    else:
        print(f"    Unknown LLM provider: {LLM_PROVIDER}")
        return None


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

    # 2b. Initialize LLM
    print("\n[2b] Initializing LLM...")
    llm = get_llm()

    # 3. Create an alert scenario
    # Try different scenarios by changing alert_type and details:
    # - DEMAND_SPIKE: details={"current": 150, "previous": 50} → 3x spike → HIGH risk
    # - STOCKOUT: details={} → HIGH risk
    # - BULLWHIP_DETECTED: details={"ratio": 3.5} → HIGH risk (>2.5)
    # - INVENTORY_LOW: details={} → MEDIUM risk
    # - FORECAST_DEVIATION: details={"deviation_pct": 45} → HIGH risk (>30)
    print("\n[3] Creating alert scenario...")
    alert = Alert(
        alert_type=AlertType.DEMAND_DROP,
        severity=AlertSeverity.UNASSESSED,  # Severity will be determined by cognition brain
        affected_nodes=[5, 6, 7],
        details={
            "current": 90,
            "previous": 80,  # 200/80 = 2.5x spike
        }
    )
    print(f"    Alert Type: {alert.alert_type.value}")
    print(f"    Initial Severity: {alert.severity.value} (to be assessed by cognition)")
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
        graph = create_supply_chain_graph(llm=llm)  # Pass the LLM (or None for rule-based)
        graph_type = "LangGraph" if not isinstance(graph, FallbackGraph) else "FallbackGraph"
        llm_status = "with LLM" if llm else "rule-based"
        print(f"    Workflow type: {graph_type} ({llm_status})")
        
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
            # Highlight the assessed severity
            assessed = analysis.get("assessed_severity", "unknown")
            print(f"      >>> ASSESSED SEVERITY: {assessed.upper()} <<<")
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
