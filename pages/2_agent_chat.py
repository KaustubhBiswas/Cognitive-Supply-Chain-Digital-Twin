"""
Agent Chat Log — Page 2

Live view of the multi-agent cognitive workflow.
Supervisor 🟦, Analyst 🟩, Negotiator 🟧, System ⚙️, Human 👤
"""

from datetime import datetime

import streamlit as st

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

    objective = st.text_input(
        "Objective (optional)",
        value="Assess and mitigate vulnerabilities across the entire supply chain.",
    )

    coverage_mode = st.radio(
        "Coverage Mode",
        ["Entire Supply Chain", "Custom Nodes (Advanced)"],
        index=0,
    )

    # Node selection
    all_nodes = list(mgr.supply_data.node_types.keys())
    affected_nodes = all_nodes
    if coverage_mode == "Custom Nodes (Advanced)":
        node_options = {f"Node {n} ({mgr.supply_data.node_types[n]})": n for n in all_nodes}
        selected_labels = st.multiselect(
            "Affected Nodes",
            list(node_options.keys()),
            default=[list(node_options.keys())[0]] if node_options else [],
        )
        affected_nodes = [node_options[l] for l in selected_labels]

    button_label = "🧠 Run Full Vulnerability Assessment" if coverage_mode == "Entire Supply Chain" else "🧠 Run Cognitive Workflow"
    if st.button(button_label, use_container_width=True, type="primary"):
        if coverage_mode == "Entire Supply Chain":
            with st.spinner("Running full-network vulnerability assessment..."):
                mgr.run_full_network_assessment(
                    alert_type=AlertType(alert_type),
                    severity=AlertSeverity(severity),
                    objective=objective,
                )
            st.rerun()
        elif not affected_nodes:
            st.warning("Select at least one node.")
        else:
            alert = Alert(
                alert_type=AlertType(alert_type),
                severity=AlertSeverity(severity),
                affected_nodes=affected_nodes,
                details={
                    "triggered_manually": True,
                    "step": mgr.model.current_step,
                    "objective": objective,
                    "coverage_context": {
                        "scan_scope": "custom_nodes",
                        "total_nodes_scanned": len(affected_nodes),
                        "vulnerable_node_ids": affected_nodes,
                        "vulnerabilities_by_node": {},
                        "coverage_rate": 1.0,
                        "vulnerability_count": 0,
                    },
                },
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
        pending_count = sum(1 for a in mgr.action_queue if a.status == "pending")
        auto_count = sum(
            1 for a in mgr.action_queue
            if str(a.recommendation.get("governance", {}).get("decision", "")) == "auto_approve"
        )
        st.caption(f"{pending_count} pending review • {auto_count} auto-approved")

        start_index = max(0, len(mgr.action_queue) - 5)
        recent_indices = list(range(start_index, len(mgr.action_queue)))[::-1]
        for action_index in recent_indices:
            action = mgr.action_queue[action_index]
            rec = action.recommendation
            rec_type = rec.get("recommendation_type", "unknown")
            border_color = {
                "increase_safety_stock": "#6C63FF",
                "adjust_reorder_point": "#10B981",
                "redistribute_inventory": "#F59E0B",
                "expedite_order": "#EF4444",
            }.get(rec_type, "#6B7280")

            status_color = {
                "pending": "#F59E0B",
                "approved": "#10B981",
                "rejected": "#EF4444",
            }.get(action.status, "#6B7280")

            st.markdown(
                f'<div class="rec-card" style="border-left-color: {border_color}">'
                f'<div style="font-weight:600; color:{border_color}">{rec_type.replace("_", " ").title()}</div>'
                f'<div style="font-size:0.85rem; color:#9CA3AF; margin-top:4px">'
                f'Nodes: {rec.get("target_nodes", "—")} • {action.timestamp}</div>'
                f'<div style="font-size:0.8rem; color:{status_color}; margin-top:4px; font-weight:600">'
                f'Status: {action.status.upper()}</div>'
                f'<div style="font-size:0.85rem; margin-top:6px">{rec.get("reasoning", "")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            governance = rec.get("governance", {}) or {}
            if governance:
                impact = governance.get("estimated_impact", {}) or {}
                inv_impact = float(impact.get("inventory_delta_pct", 0.0) or 0.0)
                back_impact = float(impact.get("backlog_delta_pct", 0.0) or 0.0)
                st.caption(
                    "Policy: "
                    f"{str(governance.get('decision', 'human_review')).replace('_', ' ')} • "
                    f"Risk: {governance.get('risk_band', 'medium')} • "
                    f"Conf: {float(governance.get('confidence', 0.0) or 0.0):.2f}"
                )
                st.caption(
                    f"Estimated impact: inventory {inv_impact:+.2f}% • backlog {back_impact:+.2f}%"
                )
                st.caption(f"Rationale: {governance.get('reason', '')}")

            if action.feedback:
                st.caption(f"Feedback: {action.feedback}")

            if action.status == "pending":
                feedback_text = st.text_input(
                    f"Feedback for recommendation #{action_index + 1}",
                    key=f"rec_feedback_{action_index}",
                    placeholder="Optional rationale for approve/reject",
                )
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ Approve", key=f"approve_{action_index}", use_container_width=True):
                        mgr.review_action(action_index, "approved", feedback_text)
                        st.rerun()
                with c2:
                    if st.button("❌ Reject", key=f"reject_{action_index}", use_container_width=True):
                        mgr.review_action(action_index, "rejected", feedback_text)
                        st.rerun()
    else:
        st.caption("No recommendations yet. Trigger an alert to start.")

    st.divider()

    last_result = getattr(mgr, "last_cognitive_result", None) or {}
    coverage = last_result.get("coverage_context", {}) or {}
    if coverage:
        st.markdown("### 🌐 Coverage Summary")
        scope = str(coverage.get("scan_scope", "custom_nodes")).replace("_", " ").title()
        total = int(coverage.get("total_nodes_scanned", 0) or 0)
        vulnerable = len(coverage.get("vulnerable_node_ids", []) or [])
        findings = int(coverage.get("vulnerability_count", 0) or 0)
        rate = 100.0 * float(coverage.get("coverage_rate", 0.0) or 0.0)
        st.caption(
            f"Scope: {scope} • Nodes scanned: {total} • Vulnerable nodes: {vulnerable} • "
            f"Vulnerabilities: {findings} • Coverage: {rate:.1f}%"
        )
        with st.expander("View Vulnerable Nodes", expanded=False):
            vul_nodes = coverage.get("vulnerable_node_ids", []) or []
            if vul_nodes:
                st.caption(", ".join(str(n) for n in vul_nodes[:50]))
            else:
                st.caption("No vulnerable nodes detected in the latest full scan.")

    st.divider()

    # Plan execution trace (Sprint 1 observability)
    with st.expander("🧭 Plan Execution Trace", expanded=False):
        result = getattr(mgr, "last_cognitive_result", None) or {}
        plan_steps = result.get("plan_steps", [])
        execution_log = result.get("execution_log", [])
        reflection_notes = result.get("reflection_notes", [])
        plan_status = result.get("plan_status", "not_started")
        current_step_idx = int(result.get("current_plan_step", 0) or 0)

        st.caption(
            f"Status: {str(plan_status).replace('_', ' ').title()} • "
            f"Current step: {current_step_idx}"
        )

        if plan_steps:
            rows = []
            for i, step in enumerate(plan_steps):
                rows.append(
                    {
                        "ID": step.get("step_id", f"P{i+1}"),
                        "Owner": str(step.get("owner", "?")).title(),
                        "Title": step.get("title", ""),
                        "Status": str(step.get("status", "pending")).replace("_", " ").title(),
                        "Tools": ", ".join(step.get("required_tools", [])[:3]),
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No plan steps available yet.")

        if execution_log:
            st.markdown("**Recent Execution Events**")
            for evt in execution_log[-8:]:
                ts = str(evt.get("timestamp", "")).replace("T", " ")[:19]
                event = str(evt.get("event", "unknown")).replace("_", " ")
                step_id = evt.get("step_id", "-")
                reason = evt.get("reason")
                detail = f"{event} • step {step_id}"
                if reason:
                    detail += f" • {reason}"
                st.caption(f"{ts} — {detail}")

        if reflection_notes:
            st.markdown("**Reflection Notes**")
            for note in reflection_notes[-5:]:
                st.caption(f"• {note}")

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
