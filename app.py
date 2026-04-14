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

    rollout = mgr.get_rollout_config()
    st.markdown(f"**Rollout:** `{rollout.get('rollout_mode', 'constrained_auto')}`")
    st.markdown(f"**Autonomy:** {'🟢 ON' if rollout.get('autonomy_enabled', True) else '🔴 OFF'}")

    st.divider()

    # ── Rollout Controls (Sprint 4) ──
    st.markdown("### 🧭 Rollout")
    mode_options = ["shadow", "constrained_auto", "full_auto"]
    current_mode = str(rollout.get("rollout_mode", "constrained_auto"))
    mode_index = mode_options.index(current_mode) if current_mode in mode_options else 1
    selected_mode = st.selectbox(
        "Mode",
        mode_options,
        index=mode_index,
        format_func=lambda x: x.replace("_", " ").title(),
    )
    if selected_mode != current_mode:
        mgr.set_rollout_mode(selected_mode)
        st.rerun()

    autonomy_enabled = bool(rollout.get("autonomy_enabled", True))
    new_autonomy = st.toggle("Autonomy Enabled", value=autonomy_enabled)
    if new_autonomy != autonomy_enabled:
        mgr.set_autonomy_enabled(new_autonomy)
        st.rerun()

    if st.button("⏪ Emergency Rollback (Shadow)", use_container_width=True):
        mgr.set_rollout_mode("shadow")
        mgr.set_autonomy_enabled(False)
        st.rerun()

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

rollout = mgr.get_rollout_config()
st.caption(
    "Agent runtime • "
    f"mode {str(rollout.get('rollout_mode', 'constrained_auto')).replace('_', ' ')} • "
    f"autonomy {'on' if rollout.get('autonomy_enabled', True) else 'off'}"
)

last_impact = mgr.get_last_intervention_impact()
if last_impact:
    impact_time = str(last_impact.get("timestamp", "")).replace("T", " ")[:19]
    d_inv = float(last_impact.get("delta_inventory", 0.0))
    d_backlog = float(last_impact.get("delta_backlog", 0.0))
    net_health = float(last_impact.get("network_health", 0.0))
    reason = str(last_impact.get("reason", "human_approval")).replace("approval:", "")

    st.caption(
        "Reference • "
        f"last approved intervention ({reason}) at {impact_time}: "
        f"inventory {d_inv:+.2f}, backlog {d_backlog:+.2f}, "
        f"network health {net_health:.0%}"
    )

last_trace = getattr(mgr, "last_cognitive_result", None) or {}
if last_trace:
    coverage = last_trace.get("coverage_context", {}) or {}
    if coverage:
        total = int(coverage.get("total_nodes_scanned", 0) or 0)
        vulnerable = len(coverage.get("vulnerable_node_ids", []) or [])
        findings = int(coverage.get("vulnerability_count", 0) or 0)
        rate = 100.0 * float(coverage.get("coverage_rate", 0.0) or 0.0)
        scope = str(coverage.get("scan_scope", "custom_nodes")).replace("_", " ")
        st.caption(
            "Coverage • "
            f"scope {scope} • scanned {total} nodes • vulnerable {vulnerable} • "
            f"findings {findings} • coverage {rate:.1f}%"
        )

    trace_status_raw = str(last_trace.get("plan_status", "not_started")).lower()
    trace_status = trace_status_raw.replace("_", " ").title()
    trace_step = int(last_trace.get("current_plan_step", 0) or 0)
    trace_events = len(last_trace.get("execution_log", []) or [])
    status_color = {
        "completed": "#10B981",
        "in_progress": "#F59E0B",
        "replanned": "#3B82F6",
        "blocked": "#EF4444",
        "not_started": "#6B7280",
    }.get(trace_status_raw, "#6B7280")
    st.markdown(
        "<div style='font-size:0.82rem; color:#9CA3AF; margin-top:2px;'>"
        "Reference • latest plan trace: "
        f"<span style='color:{status_color}; font-weight:600;'>● {trace_status}</span>"
        f" • step {trace_step} • events {trace_events}"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("🧭 Last Plan Trace", expanded=False):
        plan_steps = last_trace.get("plan_steps", []) or []
        if plan_steps:
            rows = []
            for i, step in enumerate(plan_steps):
                rows.append(
                    {
                        "ID": step.get("step_id", f"P{i+1}"),
                        "Owner": str(step.get("owner", "?")).title(),
                        "Status": str(step.get("status", "pending")).replace("_", " ").title(),
                        "Title": step.get("title", ""),
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)

        exec_log = last_trace.get("execution_log", []) or []
        if exec_log:
            st.markdown("**Recent Events**")
            for evt in exec_log[-6:]:
                ts = str(evt.get("timestamp", "")).replace("T", " ")[:19]
                event = str(evt.get("event", "unknown")).replace("_", " ")
                sid = evt.get("step_id", "-")
                st.caption(f"{ts} — {event} • step {sid}")

        notes = last_trace.get("reflection_notes", []) or []
        if notes:
            st.markdown("**Reflection Notes**")
            for note in notes[-4:]:
                st.caption(f"• {note}")

# ── KPI Row ──
k1, k2, k3, k4 = st.columns(4)
snapshot = mgr.get_snapshot()
agentic_kpis = mgr.get_agentic_kpis()

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

with st.expander("🧪 Agentic KPI Monitor", expanded=False):
    policy = mgr.get_policy_thresholds()
    st.caption(
        "Active policy thresholds • "
        f"baseline {float(policy.get('baseline_min_confidence', 0.55)):.2f} • "
        f"medium {float(policy.get('medium_risk_min_confidence', 0.65)):.2f} • "
        f"critical {float(policy.get('critical_min_confidence', 0.75)):.2f}"
    )

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.metric(
            "Autonomous Completion",
            f"{100 * float(agentic_kpis.get('autonomous_completion_rate', 0.0)):.1f}%",
            help="Share of recommendations auto-approved by governance policy.",
        )
    with a2:
        st.metric(
            "Human Override Rate",
            f"{100 * float(agentic_kpis.get('human_override_rate', 0.0)):.1f}%",
            help="Rejected recommendations divided by all human-reviewed recommendations.",
        )
    with a3:
        st.metric(
            "Mean Replans / Run",
            f"{float(agentic_kpis.get('mean_replans_per_run', 0.0)):.2f}",
            help="Average number of replans triggered per workflow run.",
        )
    with a4:
        st.metric(
            "Plan Completion",
            f"{100 * float(agentic_kpis.get('plan_completion_rate', 0.0)):.1f}%",
            help="Share of workflow runs that ended with completed plan status.",
        )

    st.caption(
        f"Workflow runs: {int(agentic_kpis.get('workflow_runs', 0))} • "
        f"Recommendations: {int(agentic_kpis.get('total_recommendations', 0))} • "
        f"Blocked-step rate: {100 * float(agentic_kpis.get('blocked_step_rate', 0.0)):.1f}%"
    )

    kpi_history = mgr.get_agentic_kpi_history(limit=200)
    if len(kpi_history) >= 2:
        import plotly.graph_objects as go

        x_vals = list(range(1, len(kpi_history) + 1))
        auto_vals = [100.0 * float(item.get("autonomous_completion_rate", 0.0)) for item in kpi_history]
        override_vals = [100.0 * float(item.get("human_override_rate", 0.0)) for item in kpi_history]
        completion_vals = [100.0 * float(item.get("plan_completion_rate", 0.0)) for item in kpi_history]

        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(x=x_vals, y=auto_vals, mode="lines+markers", name="Autonomous %", line=dict(color="#10B981", width=2)))
        trend_fig.add_trace(go.Scatter(x=x_vals, y=override_vals, mode="lines+markers", name="Override %", line=dict(color="#EF4444", width=2)))
        trend_fig.add_trace(go.Scatter(x=x_vals, y=completion_vals, mode="lines+markers", name="Plan Completion %", line=dict(color="#3B82F6", width=2)))
        trend_fig.update_layout(
            template="plotly_dark",
            height=280,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        trend_fig.update_xaxes(title_text="Snapshot")
        trend_fig.update_yaxes(title_text="Rate (%)", range=[0, 100])
        st.plotly_chart(trend_fig, use_container_width=True)

    adaptation_log = mgr.get_policy_adaptation_log(limit=5)
    if adaptation_log:
        st.markdown("**Recent Policy Adaptations**")
        for evt in adaptation_log[::-1]:
            ts = str(evt.get("timestamp", "")).replace("T", " ")[:19]
            before = evt.get("before", {}) or {}
            after = evt.get("after", {}) or {}
            reasons = "; ".join(evt.get("reasons", [])[:2])
            st.caption(
                f"{ts} • baseline {float(before.get('baseline_min_confidence', 0.0)):.2f}→"
                f"{float(after.get('baseline_min_confidence', 0.0)):.2f}, medium "
                f"{float(before.get('medium_risk_min_confidence', 0.0)):.2f}→"
                f"{float(after.get('medium_risk_min_confidence', 0.0)):.2f}, critical "
                f"{float(before.get('critical_min_confidence', 0.0)):.2f}→"
                f"{float(after.get('critical_min_confidence', 0.0)):.2f}"
                + (f" • {reasons}" if reasons else "")
            )


# ── Charts ──
model_df = mgr.get_model_dataframe()
if len(model_df) > 0:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    interventions = mgr.get_human_intervention_history(limit=50)
    intervention_points = []
    if interventions and "step" in model_df.columns:
        # Map simulation step -> latest row index in the telemetry dataframe.
        step_to_idx = {}
        for idx, step_value in enumerate(model_df["step"].tolist()):
            try:
                step_to_idx[int(step_value)] = idx
            except (TypeError, ValueError):
                continue

        for item in interventions:
            step = item.get("step")
            if step not in step_to_idx:
                continue
            intervention_points.append({
                "x": step_to_idx[step],
                "step": step,
                "reason": str(item.get("reason", "human_approval")).replace("approval:", ""),
                "timestamp": str(item.get("timestamp", "")),
            })

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

        if intervention_points and "total_inventory" in model_df.columns:
            marker_x = [p["x"] for p in intervention_points]
            marker_y = [float(model_df["total_inventory"].iloc[p["x"]]) for p in intervention_points]
            marker_text = [
                f"Human intervention<br>step {p['step']}<br>{p['reason']}<br>{p['timestamp']}"
                for p in intervention_points
            ]
            fig.add_trace(
                go.Scatter(
                    x=marker_x,
                    y=marker_y,
                    mode="markers",
                    name="Intervention",
                    marker=dict(symbol="diamond-open", size=9, color="#9CA3AF", line=dict(width=1)),
                    text=marker_text,
                    hovertemplate="%{text}<extra></extra>",
                ),
                secondary_y=False,
            )
            for p in intervention_points:
                fig.add_vline(x=p["x"], line_width=1, line_dash="dot", line_color="rgba(156,163,175,0.45)")
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

            if intervention_points:
                marker_x = [p["x"] for p in intervention_points]
                marker_y = []
                marker_text = []
                for p in intervention_points:
                    y_val = bullwhip.iloc[p["x"]]
                    if pd.isna(y_val):
                        continue
                    marker_y.append(float(y_val))
                    marker_text.append(
                        f"Human intervention<br>step {p['step']}<br>{p['reason']}<br>{p['timestamp']}"
                    )
                if marker_y:
                    valid_x = [p["x"] for p in intervention_points if not pd.isna(bullwhip.iloc[p["x"]])]
                    fig2.add_trace(go.Scatter(
                        x=valid_x,
                        y=marker_y,
                        mode="markers",
                        name="Intervention",
                        marker=dict(symbol="diamond-open", size=9, color="#9CA3AF", line=dict(width=1)),
                        text=marker_text,
                        hovertemplate="%{text}<extra></extra>",
                    ))
                    for x in valid_x:
                        fig2.add_vline(x=x, line_width=1, line_dash="dot", line_color="rgba(156,163,175,0.45)")

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
