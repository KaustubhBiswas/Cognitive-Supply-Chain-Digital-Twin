# Supply Chain Cognitive Digital Twin

An AI-powered supply chain digital twin that combines **Mesa agent-based simulation**, **Graph Neural Networks (A3TGCN)** for demand forecasting, and **LangGraph multi-agent cognition** for intelligent decision-making.

## 🚀 Features

- **Agent-Based Simulation**: Multi-echelon supply chain modeled with Mesa
- **GNN Forecasting**: A3TGCN for spatio-temporal demand prediction
- **Multi-Agent Cognition**: Supervisor, Analyst, and Negotiator agents via LangGraph
- **Real-Time Dashboard**: Streamlit UI with Human-in-the-Loop approval
- **Scenario Testing**: Demand shocks, supply disruptions, lead time changes

## 📋 Prerequisites

- Python 3.10+
- Groq API Key (free at https://console.groq.com)

## 🛠️ Installation

```bash
# Clone the repository
cd "Supply Chain Cognitive Digital Twin"

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"

# Optional features
# RAG pipeline dependencies
pip install -e ".[rag]"
# Dataset loader dependencies (Kaggle API)
pip install -e ".[data]"

# Set up environment variables
copy .env.example .env
# Edit .env and add your GROQ_API_KEY
```

## 📊 Dataset Setup

This project uses the **SupplyGraph** dataset from CIOL-SUST.

```bash
# Download the dataset
python scripts/download_dataset.py
```

## 🏃 Quick Start

```bash
# Run the simulation dashboard
streamlit run app.py

# Train the GNN model
python -m src.perception.trainer

# Run tests
pytest tests/ -v

# Run deterministic agentic benchmark harness
python scripts/run_agentic_benchmarks.py --trials 4 --seed 42 --mode constrained_auto --autonomy-enabled

# Run agentic KPI CI gate on benchmark output
python scripts/agentic_ci_gate.py --benchmark data/benchmarks/agentic_benchmark_latest.json
```

## 🧭 Rollout Modes (Sprint 4)

The system supports staged autonomy rollout with rollback controls:

- `shadow`: policy decisions are computed but never auto-executed.
- `constrained_auto`: only low-risk policy-approved actions auto-execute.
- `full_auto`: all policy-approved actions auto-execute.

Key environment variables (see `.env.example`):

- `AGENT_ROLLOUT_MODE`
- `AGENT_AUTONOMY_ENABLED`
- `GOV_CRITICAL_MIN_CONFIDENCE`
- `GOV_MEDIUM_RISK_MIN_CONFIDENCE`
- `GOV_BASELINE_MIN_CONFIDENCE`

The dashboard sidebar also exposes runtime controls for mode switching and emergency rollback.

## 🧪 Agentic Benchmark and Gate

`scripts/run_agentic_benchmarks.py` runs a deterministic scenario suite:

- normal operations
- disruption
- stale context
- conflicting goals
- injected tool-failure path

It writes summary JSON to `data/benchmarks/agentic_benchmark_latest.json` by default.

`scripts/agentic_ci_gate.py` enforces regression thresholds on benchmark output.
Thresholds are configurable via environment variables:

- `AGENTIC_GATE_MIN_PLAN_SUCCESS_RATE`
- `AGENTIC_GATE_MAX_BLOCKED_STEP_RATE`
- `AGENTIC_GATE_MIN_AUTONOMOUS_COMPLETION_RATE`
- `AGENTIC_GATE_MIN_FAILURE_RECOVERY_RATE`

## 📁 Project Structure

```
├── src/
│   ├── simulation/     # Mesa simulation components
│   ├── data/           # Dataset parsing and preprocessing
│   ├── perception/     # A3TGCN model and training
│   ├── cognition/      # LangGraph multi-agent system
│   └── integration/    # Event loop and Streamlit dashboard
├── data/               # SupplyGraph dataset
├── models/             # Saved model checkpoints
├── tests/              # Unit and integration tests
└── notebooks/          # Jupyter notebooks for exploration
```

## 🤖 Agent System

| Agent | Role |
|-------|------|
| **Supervisor** | Orchestrates workflow, delegates tasks, escalates to humans |
| **Analyst** | Analyzes forecasts, detects Bullwhip Effect, recommends policies |
| **Negotiator** | Optimizes inter-node coordination, proposes order adjustments |

## 📈 Key Metrics

- **Bullwhip Ratio**: Measures order amplification through the supply chain
- **Forecast Accuracy**: MAPE, MAE, MSE for demand predictions
- **Response Time**: Agent decision latency

## 📄 License

MIT License
