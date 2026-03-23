"""
Agent Chat Log — Page 2

Live view of the multi-agent cognitive workflow.
Supervisor 🟦, Analyst 🟩, Negotiator 🟧, System ⚙️, Human 👤
"""

import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Agent Chat", page_icon="🧠", layout="wide")

# ─── Ensure session ──────────────────────────────────────────
if "mgr" not in st.session_state:
    from src.integration import SessionManager
    st.session_state.mgr = SessionManager()

mgr = st.session_state.mgr

# ─── Agent styling ───────────────────────────────────────────
AGENT_STYLE = {
    "supervisor": {"icon": "🟦", "color": "#6C63FF", "label": "Supervisor"},
    "analyst":    {"icon": "🟩", "color": "#10B981", "label": "Analyst"},
    "negotiator": {"icon": "🟧", "color": "#F59E0B", "label": "Negotiator"},
    "system":     {"icon": "⚙️", "color": "#6B7280", "label": "System"},
    "human":      {"icon": "👤", "color": "#3B82F6", "label": "You"},
}

# ─── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
.chat-bubble {
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
    max-width: 85%;
    font-size: 0.92rem;
    line-height: 1.5;
}
.chat-bubble-left {
    background: #1E2128;
    border: 1px solid #2D3139;
    border-bottom-left-radius: 4px;
}
.chat-bubble-right {
    background: #2A2D55;
    border: 1px solid #3D4170;
    border-bottom-right-radius: 4px;
    margin-left: auto;
}
.chat-agent {
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.chat-time {
    font-size: 0.7rem;
    color: #6B7280;
    text-align: right;
    margin-top: 4px;
}
.rec-card {
    background: linear-gradient(135deg, #1A1D23, #23262F);
    border: 1px solid #2D3139;
    border-left: 4px solid;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────
st.markdown("## 🧠 Agent Chat Log")
st.caption(f"Step {mgr.model.current_step} • {len(mgr.chat_history)} messages")

# ─── Layout ──────────────────────────────────────────────────
chat_col, control_col = st.columns([3, 1])

# ─── Control Panel (right) ───────────────────────────────────
with control_col:
    st.markdown("### 🚨 Trigger Alert")

    from src.cognition import Alert, AlertSeverity, AlertType

    alert_type = st.selectbox(
        "Alert Type",
        [e.value for e in AlertType],
        format_func=lambda x: x.replace("_", " ").title(),
    )
    severity = st.selectbox(
        "Severity",
        [s.value for s in AlertSeverity],
        format_func=lambda x: x.upper(),
    )

    # Node selection
    all_nodes = list(mgr.supply_data.node_types.keys())
    node_options = {f"Node {n} ({mgr.supply_data.node_types[n]})": n for n in all_nodes}
    selected_labels = st.multiselect("Affected Nodes", list(node_options.keys()), default=[list(node_options.keys())[0]] if node_options else [])
    affected_nodes = [node_options[l] for l in selected_labels]

    if st.button("🧠 Run Cognitive Workflow", use_container_width=True, type="primary"):
        if not affected_nodes:
            st.warning("Select at least one node.")
        else:
            alert = Alert(
                alert_type=AlertType(alert_type),
                severity=AlertSeverity(severity),
                affected_nodes=affected_nodes,
                details={"triggered_manually": True, "step": mgr.model.current_step},
            )

            mgr.add_human_message(
                f"Triggered {alert_type} alert (severity: {severity}) "
                f"on nodes {affected_nodes}"
            )

            with st.spinner("Running cognitive workflow..."):
                result = mgr.run_cognitive_workflow(alert)

            st.rerun()

    st.divider()

    # Recommendations panel
    st.markdown("### 📋 Recommendations")
    if mgr.action_queue:
        for i, action in enumerate(reversed(mgr.action_queue[-5:])):
            rec = action.recommendation
            rec_type = rec.get("recommendation_type", "unknown")
            border_color = {
                "increase_safety_stock": "#6C63FF",
                "adjust_reorder_point": "#10B981",
                "redistribute_inventory": "#F59E0B",
                "expedite_order": "#EF4444",
            }.get(rec_type, "#6B7280")

            st.markdown(
                f'<div class="rec-card" style="border-left-color: {border_color}">'
                f'<div style="font-weight:600; color:{border_color}">{rec_type.replace("_", " ").title()}</div>'
                f'<div style="font-size:0.85rem; color:#9CA3AF; margin-top:4px">'
                f'Nodes: {rec.get("target_nodes", "—")} • {action.timestamp}</div>'
                f'<div style="font-size:0.85rem; margin-top:6px">{rec.get("reasoning", "")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No recommendations yet. Trigger an alert to start.")

# ─── Chat Feed (left) ────────────────────────────────────────
with chat_col:
    st.markdown("### 💬 Message Feed")

    if not mgr.chat_history:
        st.info("No messages yet. Use the panel on the right to trigger a cognitive workflow, or step the simulation from the Dashboard.")
    else:
        # Chat container with messages
        for msg in mgr.chat_history:
            style = AGENT_STYLE.get(msg.agent, AGENT_STYLE["system"])
            is_human = msg.agent == "human"
            bubble_class = "chat-bubble-right" if is_human else "chat-bubble-left"
            align = "flex-end" if is_human else "flex-start"

            st.markdown(
                f'<div style="display:flex; justify-content:{align}">'
                f'<div class="chat-bubble {bubble_class}">'
                f'<div class="chat-agent" style="color:{style["color"]}">'
                f'{style["icon"]} {style["label"]}</div>'
                f'{msg.content}'
                f'<div class="chat-time">{msg.timestamp}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # Quick actions at bottom
    st.divider()
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        if st.button("🔄 Step & Scan", use_container_width=True):
            mgr.step()
            st.rerun()
    with qc2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            mgr.chat_history.clear()
            st.rerun()
    with qc3:
        if st.button("📊 Go to Dashboard", use_container_width=True):
            st.switch_page("app.py")
