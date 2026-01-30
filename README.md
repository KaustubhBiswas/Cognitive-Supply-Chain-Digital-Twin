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
streamlit run src/integration/dashboard.py

# Train the GNN model
python -m src.perception.trainer

# Run tests
pytest tests/ -v
```

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
