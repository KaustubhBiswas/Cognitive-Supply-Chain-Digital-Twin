"""
Supply Chain Cognitive Digital Twin — Human Control Plane
=========================================================
Streamlit application entrypoint.
Run:  streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── KPI metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1A1D23 0%, #23262F 100%);
    border: 1px solid #2D3139;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
[data-testid="stMetricLabel"] {
    font-size: 0.85rem !important;
    color: #9CA3AF !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] > div {
    font-size: 0.85rem !important;
}

/* ── Sidebar styling ── */
section[data-testid="stSidebar"] {
    background: #12151A;
    border-right: 1px solid #2D3139;
}
section[data-testid="stSidebar"] .stMarkdown h1 {
    font-size: 1.2rem;
    color: #6C63FF;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(108,99,255,0.3);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
}

/* ── Expander ── */
details {
    border: 1px solid #2D3139 !important;
    border-radius: 8px !important;
}

/* ── Plotly chart backgrounds ── */
.js-plotly-plot .plotly .main-svg {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ─── Session Initialization ─────────────────────────────────
def init_session():
    """Initialize session state on first load."""
    if "mgr" not in st.session_state:
        from src.integration import SessionManager
        st.session_state.mgr = SessionManager()
        st.session_state.auto_running = False


init_session()


def _format_compact_number(value: float) -> str:
    """Format large numbers into compact human-readable units."""
    abs_value = abs(float(value))
    sign = "-" if value < 0 else ""

    if abs_value < 1_000:
        return f"{value:.0f}"
    if abs_value < 1_000_000:
        return f"{sign}{abs_value / 1_000:.2f}K"
    if abs_value < 1_000_000_000:
        return f"{sign}{abs_value / 1_000_000:.2f}M"
    if abs_value < 1_000_000_000_000:
        return f"{sign}{abs_value / 1_000_000_000:.2f}B"
    return f"{sign}{abs_value / 1_000_000_000_000:.2f}T"


# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏭 Digital Twin")
    st.caption("Supply Chain Cognitive Control Plane")
    st.divider()

    mgr = st.session_state.mgr

    # Simulation info
    step = mgr.model.current_step
    n_agents = sum(1 for _ in mgr.model.agents)
    llm_status = "🟢 Groq" if mgr.llm else "🟡 Rule-based"

    st.markdown(f"**Step:** `{step}`")
    st.markdown(f"**Nodes:** `{n_agents}`")
    st.markdown(f"**LLM:** {llm_status}")

    st.divider()

    # ── Simulation Controls ──
    st.markdown("### ⚙️ Simulation")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Step", use_container_width=True):
            mgr.step()
            st.rerun()
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            mgr.reset()
            st.rerun()

    n_steps = st.slider("Batch steps", 1, 50, 10)
    if st.button(f"▶▶ Run {n_steps} Steps", use_container_width=True):
        with st.spinner(f"Running {n_steps} steps..."):
            mgr.run_steps(n_steps)
        st.rerun()

    st.divider()

    # ── Event Injection ──
    st.markdown("### 💥 Inject Event")
    event_type = st.selectbox(
        "Event Type",
        ["demand_shock", "supply_disruption", "factory_issue", "lead_time_increase"],
        format_func=lambda x: x.replace("_", " ").title(),
    )
    magnitude = st.slider("Magnitude", 1.0, 5.0, 2.0, 0.5)
    duration = st.slider("Duration (steps)", 1, 30, 10)
    if st.button("⚡ Inject", use_container_width=True):
        mgr.inject_event(event_type, magnitude, duration)
        st.rerun()


# ─── Main Area ───────────────────────────────────────────────
st.markdown("## 📊 Real-Time Dashboard")
st.caption(
    f"Step {mgr.model.current_step} • "
    f"{n_agents} nodes • "
    f"{len(mgr.event_log)} events injected"
)

# ── KPI Row ──
k1, k2, k3, k4 = st.columns(4)
snapshot = mgr.get_snapshot()

with k1:
    st.metric("Total Inventory", _format_compact_number(snapshot["total_inventory"]))
with k2:
    st.metric("Total Backlog", _format_compact_number(snapshot["total_backlog"]))
with k3:
    br = snapshot["bullwhip_ratio"]
    br_str = f"{br:.2f}" if br < 100 else "∞"
    st.metric("Bullwhip Ratio", br_str)
with k4:
    st.metric("Active Events", _format_compact_number(len(snapshot["active_events"])))


# ── Charts ──
if mgr.model.current_step > 0:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    model_df = mgr.get_model_dataframe()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Inventory & Backlog",
        "🌊 Bullwhip Effect",
        "🔴 Risk Overview",
        "🗺️ Network Topology",
    ])

    # ── Tab 1: Inventory ──
    with tab1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if "total_inventory" in model_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(model_df))),
                    y=model_df["total_inventory"],
                    name="Inventory",
                    line=dict(color="#6C63FF", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(108,99,255,0.1)",
                ),
                secondary_y=False,
            )
        if "total_backlog" in model_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(model_df))),
                    y=model_df["total_backlog"],
                    name="Backlog",
                    line=dict(color="#FF6B6B", width=2, dash="dash"),
                ),
                secondary_y=True,
            )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_xaxes(title_text="Step", gridcolor="#2D3139")
        fig.update_yaxes(title_text="Inventory", secondary_y=False, gridcolor="#2D3139")
        fig.update_yaxes(title_text="Backlog", secondary_y=True, gridcolor="#2D3139")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Bullwhip ──
    with tab2:
        if "bullwhip_ratio" in model_df.columns:
            bullwhip = model_df["bullwhip_ratio"].replace([np.inf, -np.inf], np.nan)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=list(range(len(bullwhip))),
                y=bullwhip,
                name="Bullwhip Ratio",
                line=dict(color="#A78BFA", width=2),
                fill="tozeroy",
                fillcolor="rgba(167,139,250,0.1)",
            ))
            fig2.add_hline(
                y=1.0, line_dash="dash", line_color="#4B5563",
                annotation_text="No amplification",
                annotation_font_color="#9CA3AF",
            )
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            fig2.update_xaxes(title_text="Step", gridcolor="#2D3139")
            fig2.update_yaxes(title_text="Ratio", gridcolor="#2D3139")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Run simulation to see Bullwhip data.")

    # ── Tab 3: Risk Overview ──
    with tab3:
        if mgr.model.current_step > 0 and len(mgr.risk_engine.node_risks) > 0:
            risk_summary = mgr.get_risk_summary()
            state_dist = risk_summary.get("state_distribution", {})

            rc1, rc2 = st.columns([1, 2])
            with rc1:
                st.metric("Network Health", f"{risk_summary.get('network_health', 0):.0%}")
                st.metric("Avg Risk", f"{risk_summary.get('average_risk', 0):.2f}")
                health_report = mgr.get_health_report()
                trend = health_report.get("network_trend", "unknown")
                trend_icon = {"improving": "📈", "stable": "➡️", "degrading": "📉"}.get(trend, "❓")
                st.metric("Trend", f"{trend_icon} {trend.title()}")

            with rc2:
                # State distribution donut
                labels = list(state_dist.keys())
                values = list(state_dist.values())
                colors = {
                    "healthy": "#10B981",
                    "at_risk": "#F59E0B",
                    "degraded": "#F97316",
                    "critical": "#EF4444",
                }
                fig3 = go.Figure(data=[go.Pie(
                    labels=[l.replace("_", " ").title() for l in labels],
                    values=values,
                    hole=0.55,
                    marker=dict(colors=[colors.get(l, "#6B7280") for l in labels]),
                    textinfo="label+value",
                    textfont=dict(size=13),
                )])
                fig3.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                )
                st.plotly_chart(fig3, use_container_width=True)

            # Per-node risk table
            with st.expander("📋 Node Risk Details"):
                rows = []
                for nid, rs in mgr.risk_engine.node_risks.items():
                    d = rs.to_dict()
                    rows.append({
                        "Node": nid,
                        "Type": mgr.supply_data.node_types.get(nid, "?"),
                        "State": d["most_likely_state"],
                        "Risk Score": f"{d['composite_risk']:.3f}",
                        "P(Degraded)": f"{d['probabilities'].get('degraded', 0):.2f}",
                        "P(Critical)": f"{d['probabilities'].get('critical', 0):.2f}",
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Run simulation to generate risk data.")

    # ── Tab 4: Network Topology ──
    with tab4:
        from src.data.visualization import visualize_topology
        fig_net = visualize_topology(
            mgr.supply_data.graph,
            mgr.supply_data.node_types,
            title="",
        )
        fig_net.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_net, use_container_width=True)

else:
    st.info("👆 Use the sidebar to step the simulation and see live data.")


# ── Event Log ──
if mgr.event_log:
    with st.expander(f"📜 Event Log ({len(mgr.event_log)} events)"):
        for ev in reversed(mgr.event_log[-10:]):
            st.markdown(
                f"**Step {ev['step']}** — "
                f"`{ev['type']}` ×{ev['magnitude']} for {ev['duration']} steps"
            )
