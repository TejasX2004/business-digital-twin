import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys

# Add parent directory to sys.path to allow importing from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import date

from backend.digital_twin.digital_twin import build_digital_twin, get_connection
from backend.digital_twin.simulation import Scenario, simulate
from agents.planner_agent import PlannerAgent
from agents.analysis_agent import AnalysisAgent
from agents.red_team_agent import RedTeamAgent
from agents.decision_agent import DecisionAgent
from backend.orchestration import run_digital_twin_pipeline, PipelineResult
from events import EventProcessor, EventStore, EventType
from mcp import MCPServer
from agents.operations_agent import OperationsAgent



# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Business Digital Twin",
    page_icon="🏢",
    layout="wide"
)

if "saved_scenarios" not in st.session_state:
    st.session_state["saved_scenarios"] = {}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.metric-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 10px;
}

.metric-title {
    font-size: 14px;
    color: #6b7280;
}

.metric-value {
    font-size: 25px;
    font-weight: 700;
}

.metric-change {
    font-size: 13px;
    margin-top: 5px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def money(value):
    """Format INR values."""
    if value is None:
        return "₹0"

    value = float(value)
    sign = "-" if value < 0 else ""
    v = abs(value)

    if v >= 10_000_000:
        return f"{sign}₹{v / 10_000_000:.2f} Cr"

    if v >= 100_000:
        return f"{sign}₹{v / 100_000:.2f} L"

    return f"{sign}₹{v:,.0f}"

def money_delta(value):
    """Format INR delta string so Streamlit correctly parses the leading negative sign."""
    if value is None or float(value) == 0:
        return "₹0"
    
    value = float(value)
    v = abs(value)
    
    if v >= 10_000_000:
        s = f"₹{v / 10_000_000:.2f} Cr"
    elif v >= 100_000:
        s = f"₹{v / 100_000:.2f} L"
    else:
        s = f"₹{v:,.0f}"
        
    return f"{s}" if value > 0 else f"-{s}"


def percentage(value):
    return f"{float(value):.1f}%"


def risk_color(risk):
    risk = str(risk).lower()

    if risk == "low":
        return "🟢"

    if risk == "medium":
        return "🟡"

    if risk == "high":
        return "🟠"

    if risk == "critical":
        return "🔴"

    return "⚪"


def get_value(dictionary, *keys, default=0):
    """
    Safely retrieve nested dictionary values.

    Example:
        get_value(data, "financials", "revenue")
    """

    current = dictionary

    for key in keys:

        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current


# ============================================================
# LOAD DIGITAL TWIN
# ============================================================

@st.cache_data(ttl=60)
def load_digital_twin():

    return build_digital_twin()

@st.cache_resource
def load_planner():
    return PlannerAgent(model="qwen3:4b-instruct-2507-q4_K_M")

@st.cache_resource
def load_analyst():
    return AnalysisAgent(model="qwen3:4b-instruct-2507-q4_K_M")

@st.cache_resource
def load_red_team():
    return RedTeamAgent(model="qwen3:4b-instruct-2507-q4_K_M")

@st.cache_resource
def load_decision_agent():
    return DecisionAgent(model="qwen3:4b-instruct-2507-q4_K_M")

@st.cache_resource
def load_operations_agent():
    return OperationsAgent(model="qwen3:4b-instruct-2507-q4_K_M")

@st.cache_resource
def load_mcp_server():
    return MCPServer()

# ============================================================
# EVENT HELPER DATA FETCHERS
# ============================================================

@st.cache_data(ttl=15)
def fetch_customers():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT customer_id, customer_name, city FROM customers ORDER BY customer_id;")
            return cur.fetchall()
    finally:
        conn.close()

@st.cache_data(ttl=15)
def fetch_products():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT product_id, product_name, selling_price, cost_price FROM products ORDER BY product_id;")
            return cur.fetchall()
    finally:
        conn.close()

@st.cache_data(ttl=15)
def fetch_pending_sales():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT s.sale_id, s.customer_id, c.customer_name, s.product_id, s.quantity, s.unit_price, (s.quantity * s.unit_price) AS total_amount
                FROM sales s
                JOIN customers c ON s.customer_id = c.customer_id
                WHERE s.payment_status = 'Pending'
                ORDER BY s.sale_id;
            """)
            return cur.fetchall()
    finally:
        conn.close()

@st.cache_data(ttl=15)
def fetch_inventory_status():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT i.product_id, p.product_name, i.stock_quantity, i.warehouse, i.reorder_level
                FROM inventory i
                JOIN products p ON i.product_id = p.product_id
                ORDER BY i.product_id;
            """)
            return cur.fetchall()
    finally:
        conn.close()

@st.cache_data(ttl=15)
def fetch_supplier_expenses():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT expense_id, expense_date, amount, description
                FROM expenses
                WHERE category = 'Supplier Payment'
                ORDER BY expense_date DESC;
            """)
            return cur.fetchall()
    finally:
        conn.close()


try:

    twin = load_digital_twin()
    planner = load_planner()
    analyst = load_analyst()
    red_team = load_red_team()
    decision_agent = load_decision_agent()

except Exception as e:

    st.error("Unable to load the Digital Twin or Planner Agent.")

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏢 Digital Twin")

    st.caption("Business Intelligence & Simulation")

    st.divider()

    st.subheader("Navigation")

    page = st.radio(
        "Go to",
        [
            "📊 Dashboard",
            "🤖 Autonomous Control Tower",
            "⚡ Business Events",
            "🔬 What-If Lab",
            "⚖️ Scenario Comparison"
        ]
    )

    st.divider()

    st.caption(
        "Deterministic simulation engine + AI reasoning layer"
    )


# ============================================================
# HEADER
# ============================================================

st.title("🏢 Business Digital Twin")

st.caption(
    "A live simulated representation of the business state"
)

st.divider()


# ============================================================
# BUSINESS OVERVIEW / DASHBOARD
# ============================================================

if page in ["📊 Dashboard", "Business Overview"]:

    st.header("Current Business State")

    financials = twin.get("financials", {})
    cash_flow = twin.get("cash_flow", {})
    working_capital = twin.get("working_capital", {})
    operations = twin.get("operations", {})
    risk = twin.get("risk", {})

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Revenue",
            money(financials.get("revenue", 0))
        )

    with col2:

        st.metric(
            "Gross Profit",
            money(financials.get("gross_profit", 0))
        )

    with col3:

        st.metric(
            "Net Profit",
            money(financials.get("net_profit", 0))
        )

    with col4:

        risk_level = risk.get("risk_level", "Unknown")

        st.metric(
            "Risk",
            f"{risk_color(risk_level)} {risk_level}"
        )

    # --------------------------------------------------------
    # SECOND ROW
    # --------------------------------------------------------

    st.markdown("### Working Capital")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Current Cash",
            money(
                cash_flow.get(
                    "current_cash",
                    cash_flow.get("ending_cash", 0)
                )
            )
        )

    with col2:

        st.metric(
            "Accounts Receivable",
            money(
                working_capital.get(
                    "accounts_receivable",
                    working_capital.get("ar", 0)
                )
            )
        )

    with col3:

        st.metric(
            "Inventory",
            money(
                working_capital.get(
                    "inventory_value",
                    working_capital.get("inventory", 0)
                )
            )
        )

    with col4:

        st.metric(
            "Inventory Days",
            f"{operations.get('inventory_days', 0):.1f}"
        )

    # --------------------------------------------------------
    # FINANCIAL DETAILS
    # --------------------------------------------------------

    st.markdown("### Financial Performance")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "COGS",
            money(financials.get("cogs", 0))
        )

        st.metric(
            "Gross Margin",
            percentage(financials.get("gross_margin", 0) * 100)
        )

    with col2:

        st.metric(
            "Operating Expenses",
            money(financials.get("operating_expenses", 0))
        )

        st.metric(
            "Net Margin",
            percentage(financials.get("net_margin", 0) * 100)
        )

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    st.markdown("### Risk Assessment")

    risk_score = risk.get("risk_score", 0)
    risk_level = risk.get("risk_level", "Unknown")

    st.progress(
        min(max(float(risk_score) / 100, 0), 1)
    )

    st.write(
        f"**Overall Risk:** "
        f"{risk_color(risk_level)} {risk_level}"
    )

    # --------------------------------------------------------
    # BUSINESS STATISTICS
    # --------------------------------------------------------

    st.markdown("### Operational Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Sales Transactions",
            f"{operations.get('sales_transactions', 0):,}"
        )

    with col2:

        st.metric(
            "Units Sold",
            f"{operations.get('units_sold', 0):,}"
        )

    with col3:

        st.metric(
            "Inventory Turnover",
            f"{operations.get('inventory_turnover', 0):.1f}x"
        )

    with col4:

        st.metric(
            "Avg Payment",
            f"{twin.get('payment_behavior', {}).get('average_payment_days', 0):.1f} days"
        )


# ============================================================
# AUTONOMOUS CONTROL TOWER (AGENTIC OPERATIONS MODE)
# ============================================================

elif page == "🤖 Autonomous Control Tower":

    st.header("🤖 Autonomous Control Tower")
    st.caption(
        "Agentic Operations Mode: Powered by Model Context Protocol (MCP). "
        "The autonomous operations agent inspects current business state, identifies material risks, "
        "runs what-if simulations, consults company policies through RAG, and produces recommendations with explicit human approval."
    )

    # --------------------------------------------------------
    # LIVE SNAPSHOT METRICS BANNER
    # --------------------------------------------------------
    cf = twin.get("cash_flow", {})
    wc = twin.get("working_capital", {})
    fin = twin.get("financials", {})
    risk = twin.get("risk", {})

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Current Cash", money(cf.get("current_cash", 0)))
    with kpi2:
        st.metric("Accounts Receivable", money(wc.get("accounts_receivable", 0)), delta=f"{wc.get('receivable_percentage', 0):.1f}% of Rev", delta_color="inverse")
    with kpi3:
        st.metric("Inventory Value", money(wc.get("inventory_value", 0)), delta=f"{wc.get('inventory_units', 0)} units")
    with kpi4:
        st.metric("Material Risk Level", f"{risk_color(risk.get('level', 'Low'))} {risk.get('level', 'Low')}")

    st.divider()

    # --------------------------------------------------------
    # OBJECTIVE INPUT & PROMPT SUGGESTIONS
    # --------------------------------------------------------
    st.subheader("🎯 Business Objective")
    st.caption("Enter a broad operational objective or question. The agent autonomously decides which tools it needs to investigate.")

    suggested_prompts = [
        "Find the biggest business risk right now.",
        "How is the business doing?",
        "Why is cash under pressure?",
        "Should we increase inventory if sales increase 30%?",
        "What should management focus on?"
    ]

    st.markdown("**Quick Inquiries:**")
    prompt_cols = st.columns(len(suggested_prompts))
    for i, p_text in enumerate(suggested_prompts):
        with prompt_cols[i]:
            if st.button(p_text, key=f"quick_prompt_{i}", use_container_width=True):
                st.session_state["control_tower_query"] = p_text

    default_query = st.session_state.get("control_tower_query", "Find the biggest business risk right now.")
    user_objective = st.text_input("Operational Objective:", value=default_query, key="objective_text_input")

    col_run, col_clear = st.columns([2, 6])
    with col_run:
        run_clicked = st.button("🚀 Run Investigation", type="primary", use_container_width=True, key="btn_run_tower")
    with col_clear:
        if st.button("🔄 Reset Investigation", use_container_width=False, key="btn_reset_tower"):
            st.session_state.pop("control_tower_result", None)
            st.session_state.pop("action_execution_result", None)
            st.rerun()

    if run_clicked and user_objective:
        agent = load_operations_agent()
        with st.spinner("🤖 Autonomous Control Tower investigating business state via MCP..."):
            result = agent.investigate(user_objective)
            st.session_state["control_tower_result"] = result
            st.session_state["control_tower_objective"] = user_objective
            st.session_state.pop("action_execution_result", None)

    # --------------------------------------------------------
    # DISPLAY INVESTIGATION RESULTS
    # --------------------------------------------------------
    if "control_tower_result" in st.session_state:
        res = st.session_state["control_tower_result"]
        st.divider()

        # 1. INVESTIGATION TIMELINE
        st.subheader("⏱️ Investigation Timeline")
        st.caption("Agentic conceptual loop: OBSERVE → IDENTIFY ISSUE → INVESTIGATE → SIMULATE → CHECK POLICY → COMPARE OPTIONS → RECOMMEND")

        timeline = res.get("timeline", [])
        if timeline:
            for idx, item in enumerate(timeline):
                stage = item.get("stage", "INVESTIGATE")
                icon = {
                    "OBSERVE": "🤖",
                    "IDENTIFY": "⚠️",
                    "INVESTIGATE": "🔍",
                    "SIMULATE": "📊",
                    "CHECK POLICY": "📚",
                    "COMPARE OPTIONS": "⚖️",
                    "RECOMMEND": "💡",
                }.get(stage, "🔹")

                with st.container():
                    st.markdown(
                        f"""
                        <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 6px; margin-bottom: 8px;">
                            <div style="font-weight: 600; font-size: 15px; color: #1e293b;">{icon} Step {item.get('step', idx+1)}: {item.get('description', '')}</div>
                            <div style="font-size: 13px; color: #64748b; margin-top: 4px;">
                                <code>Tool: {item.get('tool', 'None')}</code> &nbsp;|&nbsp; <strong>Stage:</strong> {stage}
                            </div>
                            <div style="font-size: 13px; color: #334155; margin-top: 4px; background: #ffffff; padding: 6px 10px; border-radius: 4px; border: 1px solid #e2e8f0;">
                                {item.get('summary', '')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                if idx < len(timeline) - 1:
                    st.markdown("<div style='text-align: center; color: #94a3b8; font-weight: bold; margin: -4px 0;'>↓</div>", unsafe_allow_html=True)
        else:
            st.info("No timeline events recorded.")

        st.divider()

        # 2. KEY FINDINGS & OBSERVED FACTS
        st.subheader("🔍 Key Findings & Observed Facts")
        col_facts, col_issues = st.columns(2)

        with col_facts:
            st.markdown("##### 📌 Authoritative Facts Observed (MCP)")
            facts = res.get("observed_facts", [])
            if facts:
                for f in facts:
                    st.markdown(f"- {f}")
            else:
                st.write("No specific numeric facts extracted.")

        with col_issues:
            st.markdown("##### ⚠️ Material Issues Identified")
            issues = res.get("identified_issues", [])
            if issues:
                for issue in issues:
                    st.warning(f"⚠️ {issue}")
            else:
                st.success("✅ No critical operational deviations detected.")

        st.divider()

        # 3. SIMULATIONS RUN & POLICY KNOWLEDGE CONSULTED
        st.subheader("📊 Simulations & Company Policies")
        col_sims, col_policies = st.columns(2)

        with col_sims:
            st.markdown("##### 📈 What-If Simulations Executed")
            sims = res.get("simulations_run", [])
            if sims:
                for s in sims:
                    if isinstance(s, dict):
                        st.markdown(f"**Scenario:** `{s.get('args', {})}`")
                        res_data = s.get("result", {})
                        if isinstance(res_data, dict):
                            fi = res_data.get("financial_impact", {})
                            li = res_data.get("liquidity_impact", {})
                            ra = res_data.get("risk_assessment", {})
                            st.write(f"- Projected Revenue: ₹{fi.get('projected_revenue', 0):,.0f}")
                            st.write(f"- Projected Net Profit: ₹{fi.get('projected_net_profit', 0):,.0f}")
                            st.write(f"- Projected Ending Cash: ₹{li.get('projected_ending_cash', 0):,.0f}")
                            st.write(f"- Inventory Risk: {ra.get('inventory_risk', 'Normal')} | Overall Risk: {ra.get('overall_risk', 'Normal')}")
                    else:
                        st.write(f"- {s}")
            else:
                st.write("No what-if scenario simulations required for this objective.")

        with col_policies:
            st.markdown("##### 📚 Authoritative Company Policies Consulted (RAG)")
            policies = res.get("business_knowledge_used", [])
            if policies:
                for p in policies:
                    if isinstance(p, dict):
                        st.info(f"**Policy Source:** `{p.get('source', 'Company Knowledge')}`\n\n{p.get('content', '')}")
                    else:
                        st.info(f"{p}")
            else:
                st.write("No business policy excerpts retrieved.")

        st.divider()

        # 4. OPTIONS CONSIDERED
        st.subheader("⚖️ Options Considered")
        options = res.get("options_considered", [])
        if options and isinstance(options, list):
            opt_data = []
            for opt in options:
                if isinstance(opt, dict):
                    opt_data.append({
                        "Option": opt.get("option", "Alternative"),
                        "Pros": opt.get("pros", ""),
                        "Cons": opt.get("cons", ""),
                        "Simulated Outcome": opt.get("outcome", ""),
                    })
            if opt_data:
                st.dataframe(pd.DataFrame(opt_data), use_container_width=True, hide_index=True)
        else:
            st.write("Standard operational parameters evaluated.")

        st.divider()

        # 5. RECOMMENDATION & ACTION GATE
        st.subheader("💡 Strategic Recommendation & Proposed Action")

        st.success(f"**Recommendation:** {res.get('recommendation', 'No recommendation generated.')}")
        st.info(f"**Concrete Next Step:** {res.get('recommended_action', 'Continue monitoring business operations.')}")

        c_conf, c_gate = st.columns([1, 3])
        with c_conf:
            st.metric("Agent Confidence", f"{res.get('confidence', 0.95) * 100:.0f}%")
        with c_gate:
            st.warning("🔒 **Human-in-the-Loop Safeguard:** The agent cannot alter business state without explicit approval.")

        # 6. ACTION APPROVAL & EXECUTION INTERFACE
        st.divider()
        st.subheader("🚀 Operational Action Gate")
        st.caption("Review proposed action parameters, deterministic validation findings, and policy sources before committing to PostgreSQL.")

        action_payload = res.get("action_payload") or {}
        action_val = res.get("action_validation") or {}
        val_status = res.get("validation_status") or action_val.get("status", "PASS")
        findings = action_val.get("findings", [])
        policy_sources = action_val.get("policy_sources", [])
        can_approve = res.get("can_approve", val_status == "PASS")

        ev_type = action_payload.get("event_type", "RESTRICT_CREDIT")
        ev_data = action_payload.get("payload", {})
        action_reason = action_payload.get("reason", res.get("recommended_action", ""))

        # Proposed Action Display (Requirement 10)
        st.markdown("##### 📌 Proposed Action")
        if ev_type in ["RESTRICT_CREDIT", "INITIATE_CREDIT_REVIEW"]:
            target_cid = ev_data.get("customer_id", "C004")
            st.info(f"**Action:** Restrict additional credit exposure for {target_cid}")
            st.markdown(f"**Description:** {action_reason}")
            st.markdown("**Specific new credit limit:** `NOT SPECIFIED` *(Do NOT invent one)*")
        else:
            st.info(f"**Action:** `{ev_type}` — {action_reason}")
            if "credit_limit" in str(ev_data) or "new_credit_limit" in str(ev_data):
                st.markdown("**Specific new credit limit:** `NOT SPECIFIED` *(Do NOT invent one)*")

        with st.expander("🔍 View Action Payload JSON", expanded=False):
            st.json(action_payload)

        # Validation Status & Findings (Requirement 10)
        st.markdown("##### 🛡️ Validation Status & Findings")
        col_v1, col_v2 = st.columns([1, 2])
        with col_v1:
            if val_status == "PASS":
                st.success(f"**Validation Status:** `{val_status}`")
            elif val_status == "REVIEW_REQUIRED":
                st.warning(f"**Validation Status:** `{val_status}`")
            else:
                st.error(f"**Validation Status:** `{val_status}`")

        with col_v2:
            if policy_sources:
                st.markdown(f"**Policy Sources:** {', '.join([f'`{ps}`' for ps in policy_sources])}")
            else:
                st.markdown("**Policy Sources:** `payment_policy.md`, `business_rules.md`")

        if findings:
            for f in findings:
                if "violation" in f.lower() or "unsupported" in f.lower() or "not exist" in f.lower():
                    st.warning(f"⚠️ {f}")
                else:
                    st.info(f"ℹ️ {f}")
        else:
            checks = action_val.get("checks", [])
            if checks:
                for chk in checks:
                    c_name = chk.get("check")
                    c_detail = chk.get("detail")
                    if chk.get("passed"):
                        st.markdown(f"- ✅ **{c_name}**: {c_detail}")
                    else:
                        st.markdown(f"- ❌ **{c_name}**: {c_detail}")
            else:
                st.markdown("- ✅ All domain constraints, entity references, and numeric bounds verified.")

        # Human Approval (Requirement 9 & 10)
        st.markdown("##### 👤 Human Approval")
        if can_approve and val_status == "PASS":
            col_act1, col_act2 = st.columns([2, 5])
            with col_act1:
                approve_btn = st.button("✅ Approve and Execute Action", type="primary", use_container_width=True, key="btn_approve_action")
        else:
            approve_btn = False
            st.error(
                "🔒 **Approval Blocked:** Deterministic validation or Red-Team checks flagged issues. "
                f"Action status is `{val_status}`. The approval button is withheld until all constraints pass."
            )

        if approve_btn:
            mcp_srv = load_mcp_server()
            # Capture state before mutation
            before_twin = build_digital_twin()
            before_state = {
                "Cash": before_twin.get("cash_flow", {}).get("current_cash", 0),
                "Accounts Receivable": before_twin.get("working_capital", {}).get("accounts_receivable", 0),
                "Inventory Value": before_twin.get("working_capital", {}).get("inventory_value", 0),
                "Revenue": before_twin.get("financials", {}).get("revenue", 0),
            }

            with st.spinner("Submitting approved event through Event System to PostgreSQL..."):
                exec_res = mcp_srv.call_tool(
                    "create_business_event",
                    {
                        "event_type": ev_type,
                        "payload": ev_data,
                        "approved": True,
                    }
                )

                st.cache_data.clear()
                after_twin = build_digital_twin()
                after_state = {
                    "Cash": after_twin.get("cash_flow", {}).get("current_cash", 0),
                    "Accounts Receivable": after_twin.get("working_capital", {}).get("accounts_receivable", 0),
                    "Inventory Value": after_twin.get("working_capital", {}).get("inventory_value", 0),
                    "Revenue": after_twin.get("financials", {}).get("revenue", 0),
                }

                st.session_state["action_execution_result"] = {
                    "exec_res": exec_res,
                    "before": before_state,
                    "after": after_state,
                }

        # Render execution verification if completed
        if "action_execution_result" in st.session_state:
            exec_info = st.session_state["action_execution_result"]
            exec_res = exec_info["exec_res"]
            s_data = exec_res.get("structured_data", {})

            if s_data.get("status") == "EXECUTED":
                st.success(f"🎉 **Action Executed Successfully!** Event ID: `{s_data.get('event_id')}` committed to PostgreSQL.")

                st.markdown("""
                **Execution Verification Pipeline:**
                `ACTION` → `Event System` → `PostgreSQL` → `Digital Twin Refresh` → `Verification ✅`
                """)

                # Display Before vs After Comparison
                st.markdown("##### 🔄 Before vs. After Business State")
                b_metrics = exec_info["before"]
                a_metrics = exec_info["after"]

                comp_rows = []
                for k in ["Cash", "Accounts Receivable", "Inventory Value", "Revenue"]:
                    b_val = b_metrics.get(k, 0)
                    a_val = a_metrics.get(k, 0)
                    diff = a_val - b_val
                    comp_rows.append({
                        "Metric": k,
                        "Before Action": money(b_val),
                        "After Action": money(a_val),
                        "Change": money_delta(diff),
                    })
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
            else:
                st.error(f"❌ Event execution failed: {s_data.get('error') or exec_res.get('content', [{}])[0].get('text')}")


# ============================================================
# BUSINESS EVENTS (EVENT-DRIVEN DIGITAL TWIN)
# ============================================================

elif page == "⚡ Business Events":

    st.header("⚡ Operational Business Events")
    st.caption(
        "Event-Driven Architecture: Ingest business transactions into PostgreSQL with ACID transactions. "
        "Processed events immediately update the database, log in the persistent audit table, and refresh the Digital Twin state."
    )

    # --------------------------------------------------------
    # LIVE DIGITAL TWIN METRICS BANNER
    # --------------------------------------------------------
    cf = twin.get("cash_flow", {})
    wc = twin.get("working_capital", {})
    fin = twin.get("financials", {})

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Current Cash", money(cf.get("current_cash", 0)))
    with kpi2:
        st.metric("Accounts Receivable", money(wc.get("accounts_receivable", 0)))
    with kpi3:
        st.metric("Total Revenue", money(fin.get("revenue", 0)))
    with kpi4:
        st.metric("Inventory Value", money(wc.get("inventory_value", 0)))

    st.divider()

    # --------------------------------------------------------
    # EVENT FORMS
    # --------------------------------------------------------
    st.subheader("Submit Operational Event")

    tab_sale, tab_pm, tab_inv, tab_exp, tab_supp = st.tabs([
        "🛒 SALE_CREATED",
        "💵 PAYMENT_RECEIVED",
        "📦 INVENTORY_UPDATED",
        "🧾 EXPENSE_RECORDED",
        "🏭 SUPPLIER_PAYMENT_UPDATED"
    ])

    processor = EventProcessor()

    # --- 1. SALE_CREATED ---
    with tab_sale:
        st.markdown("##### 🛒 Record a New Sale (SALE_CREATED)")
        customers = fetch_customers()
        products = fetch_products()

        if customers and products:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                c_map = {f"{c[0]} - {c[1]} ({c[2]})": c[0] for c in customers}
                sel_c_label = st.selectbox("Customer", options=list(c_map.keys()), key="sale_cust")
                sel_customer_id = c_map[sel_c_label]

                qty = st.number_input("Quantity Sold", min_value=1, value=5, step=1, key="sale_quantity")

            with col_s2:
                p_map = {f"{p[0]} - {p[1]} (₹{float(p[2]):,.0f})": (p[0], float(p[2])) for p in products}
                sel_p_label = st.selectbox("Product", options=list(p_map.keys()), key="sale_prod")
                sel_product_id, def_price = p_map[sel_p_label]

                calculated_amount = float(qty * def_price)
                sale_amount = st.number_input(
                    "Total Sale Amount (₹)",
                    min_value=1.0,
                    value=calculated_amount,
                    step=100.0,
                    help="Defaulted to quantity × product selling price. Can be adjusted for discounts.",
                    key="sale_amount_input"
                )

            sale_date = st.date_input("Sale Date", value=date.today(), key="sale_date_input")

            if st.button("🚀 Record SALE_CREATED Event", type="primary", use_container_width=True, key="btn_sale"):
                with st.spinner("Processing SALE_CREATED event..."):
                    res = processor.process_event(
                        EventType.SALE_CREATED,
                        {
                            "customer_id": sel_customer_id,
                            "product_id": sel_product_id,
                            "quantity": qty,
                            "sale_amount": sale_amount,
                            "sale_date": str(sale_date),
                        }
                    )
                if res.success:
                    st.success(f"✅ Sale event successfully recorded! Sale ID: `{res.details.get('sale_id')}` (Event ID: `{res.event_id}`). Digital Twin refreshed.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"❌ Failed to process sale event: {res.error}")
        else:
            st.warning("Customer or product data unavailable from database.")

    # --- 2. PAYMENT_RECEIVED ---
    with tab_pm:
        st.markdown("##### 💵 Record Customer Payment (PAYMENT_RECEIVED)")
        pending_sales = fetch_pending_sales()

        if pending_sales:
            ps_map = {
                f"{s[0]} | {s[2]} | Prod: {s[3]} (Qty: {s[4]}) | Balance Due: ₹{float(s[6]):,.0f}": (s[0], float(s[6]))
                for s in pending_sales
            }
            sel_ps_label = st.selectbox("Select Pending Sale", options=list(ps_map.keys()), key="pm_sale_sel")
            sel_sale_id, due_amount = ps_map[sel_ps_label]

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                pm_amount = st.number_input(
                    "Payment Amount (₹)",
                    min_value=1.0,
                    value=float(due_amount),
                    step=100.0,
                    key="pm_amount_input"
                )
                pm_date = st.date_input("Payment Date", value=date.today(), key="pm_date_input")

            with col_p2:
                pm_method = st.selectbox(
                    "Payment Method",
                    ["Bank Transfer", "UPI", "Card", "Cheque", "Cash"],
                    key="pm_method_input"
                )

            if st.button("🚀 Record PAYMENT_RECEIVED Event", type="primary", use_container_width=True, key="btn_pm"):
                with st.spinner("Processing PAYMENT_RECEIVED event..."):
                    res = processor.process_event(
                        EventType.PAYMENT_RECEIVED,
                        {
                            "sale_id": sel_sale_id,
                            "amount": pm_amount,
                            "payment_date": str(pm_date),
                            "payment_method": pm_method,
                        }
                    )
                if res.success:
                    status_text = res.details.get("sale_payment_status")
                    st.success(
                        f"✅ Payment of {money(pm_amount)} recorded for Sale `{sel_sale_id}`! "
                        f"Sale Status: `{status_text}`. Cash increased and AR reduced (Event ID: `{res.event_id}`)."
                    )
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"❌ Failed to process payment event: {res.error}")
        else:
            st.info("No pending sales invoices found in the database. Create a sale first.")

    # --- 3. INVENTORY_UPDATED ---
    with tab_inv:
        st.markdown("##### 📦 Update Product Inventory (INVENTORY_UPDATED)")
        inv_data = fetch_inventory_status()

        if inv_data:
            inv_map = {
                f"{i[0]} - {i[1]} (Warehouse: {i[3]}) | Stock: {i[2]} units": (i[0], int(i[2]))
                for i in inv_data
            }
            sel_inv_label = st.selectbox("Product", options=list(inv_map.keys()), key="inv_prod_sel")
            sel_prod_id, current_stock = inv_map[sel_inv_label]

            col_i1, col_i2 = st.columns(2)
            with col_i1:
                qty_change = st.number_input(
                    "Stock Quantity Change (± units)",
                    value=20,
                    step=1,
                    help="Enter a positive number for restock/returns, or negative for shrinkage/scrap.",
                    key="inv_qty_change"
                )
            with col_i2:
                inv_update_date = st.date_input("Update Date", value=date.today(), key="inv_date_input")

            st.caption(f"Current Stock: **{current_stock}** units → Projected Stock: **{current_stock + qty_change}** units")

            if st.button("🚀 Record INVENTORY_UPDATED Event", type="primary", use_container_width=True, key="btn_inv"):
                with st.spinner("Processing INVENTORY_UPDATED event..."):
                    res = processor.process_event(
                        EventType.INVENTORY_UPDATED,
                        {
                            "product_id": sel_prod_id,
                            "quantity_change": qty_change,
                            "update_date": str(inv_update_date),
                        }
                    )
                if res.success:
                    prev_s = res.details.get("previous_stock")
                    new_s = res.details.get("new_stock")
                    st.success(f"✅ Inventory updated for `{sel_prod_id}`: {prev_s} → {new_s} units (Event ID: `{res.event_id}`).")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"❌ Failed to update inventory: {res.error}")
        else:
            st.warning("Inventory records unavailable from database.")

    # --- 4. EXPENSE_RECORDED ---
    with tab_exp:
        st.markdown("##### 🧾 Record Operating Expense (EXPENSE_RECORDED)")

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            exp_category = st.selectbox(
                "Expense Category",
                ["Rent", "Salaries", "Marketing", "Utilities", "Logistics", "Maintenance", "Office Supplies", "Legal"],
                key="exp_category_input"
            )
            exp_amount = st.number_input("Expense Amount (₹)", min_value=1.0, value=25000.0, step=1000.0, key="exp_amt_input")

        with col_e2:
            exp_date = st.date_input("Expense Date", value=date.today(), key="exp_date_input")
            exp_desc = st.text_input("Description", placeholder="e.g. Monthly cloud hosting & internet", key="exp_desc_input")

        if st.button("🚀 Record EXPENSE_RECORDED Event", type="primary", use_container_width=True, key="btn_exp"):
            with st.spinner("Processing EXPENSE_RECORDED event..."):
                res = processor.process_event(
                    EventType.EXPENSE_RECORDED,
                    {
                        "category": exp_category,
                        "amount": exp_amount,
                        "expense_date": str(exp_date),
                        "description": exp_desc,
                    }
                )
            if res.success:
                st.success(f"✅ Operating expense recorded! Expense ID: `{res.details.get('expense_id')}` (Event ID: `{res.event_id}`). Cash and net profit updated.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(f"❌ Failed to record expense: {res.error}")

    # --- 5. SUPPLIER_PAYMENT_UPDATED ---
    with tab_supp:
        st.markdown("##### 🏭 Supplier Payment Event (SUPPLIER_PAYMENT_UPDATED)")

        supp_expenses = fetch_supplier_expenses()
        supp_mode = st.radio(
            "Action Type",
            ["Record New Supplier Payment", "Update Existing Supplier Payment"],
            horizontal=True,
            key="supp_mode_radio"
        )

        target_expense_id = None
        default_supp_amt = 50000.0
        default_supp_desc = "Payment to raw material supplier"

        if supp_mode == "Update Existing Supplier Payment":
            if supp_expenses:
                supp_map = {
                    f"{e[0]} | Date: {e[1]} | Amount: ₹{float(e[2]):,.0f} | {e[3]}": (e[0], float(e[2]), str(e[3]))
                    for e in supp_expenses
                }
                sel_supp_label = st.selectbox("Select Existing Supplier Payment", options=list(supp_map.keys()), key="supp_sel_exp")
                target_expense_id, default_supp_amt, default_supp_desc = supp_map[sel_supp_label]
            else:
                st.info("No existing supplier payments found. Record a new payment below.")
                supp_mode = "Record New Supplier Payment"

        col_sp1, col_sp2 = st.columns(2)
        with col_sp1:
            supp_amount = st.number_input("Supplier Payment Amount (₹)", min_value=1.0, value=float(default_supp_amt), step=5000.0, key="supp_amt_input")
            supp_date = st.date_input("Payment Date", value=date.today(), key="supp_date_input")

        with col_sp2:
            supp_desc = st.text_input("Description / Supplier Reference", value=default_supp_desc, key="supp_desc_input")

        if st.button("🚀 Submit SUPPLIER_PAYMENT_UPDATED Event", type="primary", use_container_width=True, key="btn_supp"):
            with st.spinner("Processing SUPPLIER_PAYMENT_UPDATED event..."):
                payload = {
                    "amount": supp_amount,
                    "payment_date": str(supp_date),
                    "description": supp_desc,
                }
                if supp_mode == "Update Existing Supplier Payment" and target_expense_id:
                    payload["expense_id"] = target_expense_id

                res = processor.process_event(EventType.SUPPLIER_PAYMENT_UPDATED, payload)

            if res.success:
                act = res.details.get("action", "processed")
                st.success(f"✅ Supplier payment {act}! Expense ID: `{res.details.get('expense_id')}` (Event ID: `{res.event_id}`). Digital Twin refreshed.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(f"❌ Failed to process supplier payment: {res.error}")

    # --------------------------------------------------------
    # RECENT EVENTS TABLE
    # --------------------------------------------------------
    st.divider()
    st.subheader("📋 Recent Business Events")
    st.caption("Immutable audit log of business transactions stored in PostgreSQL (`business_events`).")

    event_store = EventStore()
    recent_events = event_store.get_recent_events(limit=50)

    if recent_events:
        events_table = []
        for ev in recent_events:
            status_icon = "🟢" if ev["status"] == "SUCCESS" else ("🔴" if ev["status"] == "FAILED" else "🟡")
            events_table.append({
                "Event ID": ev["event_id"],
                "Event Type": ev["event_type"],
                "Timestamp": ev["timestamp"],
                "Status": f"{status_icon} {ev['status']}",
            })
        st.dataframe(pd.DataFrame(events_table), use_container_width=True, hide_index=True)
    else:
        st.info("No business events recorded yet. Submit an event above to start tracking.")


# ============================================================
# WHAT-IF LAB
# ============================================================

elif page in ["🔬 What-If Lab", "What-If Lab"]:

    st.header("🔬 What-If Simulation Lab")

    st.write(
        "Change business assumptions and simulate their impact "
        "on revenue, cash, receivables and inventory."
    )

    st.divider()

    # --------------------------------------------------------
    # SCENARIO CONTROLS
    # --------------------------------------------------------

    def _execute_pipeline(input_data):
        with st.spinner("Executing Digital Twin Pipeline (Planner → Simulator → Analysis → Red-Team → Decision)..."):
            try:
                pipe_res = run_digital_twin_pipeline(
                    input_data,
                    twin=twin,
                    planner=planner,
                    analyst=analyst,
                    red_team=red_team,
                    decision_agent=decision_agent
                )
                st.session_state["pipeline_result"] = pipe_res
                st.session_state["simulation_result"] = pipe_res.simulation
                st.session_state["scenario"] = pipe_res.scenario
                st.session_state["ai_plan"] = pipe_res.plan
                st.session_state["analysis_result"] = pipe_res.analysis
                st.session_state["red_team_result"] = pipe_res.red_team
                st.session_state["decision_result"] = pipe_res.decision

                if pipe_res.errors:
                    for err in pipe_res.errors:
                        st.error(err)
                if pipe_res.warnings:
                    for warn in pipe_res.warnings:
                        st.warning(warn)
            except Exception as e:
                st.error(f"Pipeline execution failed: {e}")
                st.exception(e)

    tab_ai, tab_manual = st.tabs(["AI Planner 🤖", "Manual Sliders 🎛️"])
    
    with tab_ai:
        st.subheader("Natural Language Simulation")
        user_query = st.text_area(
            "Describe the business scenario you want to simulate:",
            placeholder="e.g., What if sales grow by 30% and customers take 15 extra days to pay?",
            height=100
        )
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            plan_btn = st.button("✨ Plan Scenario", use_container_width=True)
        with col_btn2:
            run_direct_btn = st.button("🚀 Run Full AI Pipeline", type="primary", use_container_width=True)

        if plan_btn:
            if user_query.strip():
                with st.spinner("Extracting parameters via AI Planner..."):
                    try:
                        plan_dict = planner.plan(user_query)
                        st.session_state["ai_plan"] = plan_dict
                        st.session_state["ai_scenario"] = planner.to_scenario(plan_dict)
                    except Exception as e:
                        st.error("Failed to plan scenario.")
                        st.exception(e)
            else:
                st.warning("Please enter a business scenario.")

        if run_direct_btn:
            if user_query.strip():
                _execute_pipeline(user_query.strip())
            else:
                st.warning("Please enter a business scenario.")
        
        if "ai_plan" in st.session_state and not run_direct_btn:
            st.divider()
            plan = st.session_state["ai_plan"]
            st.info(f"**AI Interpretation:** {plan.get('interpretation', '')}")
            st.markdown("#### Extracted Parameters")
            st.code(planner.explain_plan(plan), language="text")
            if st.button("🚀 Run AI Simulation", type="primary", use_container_width=True):
                target = user_query.strip() if user_query.strip() else st.session_state["ai_scenario"]
                _execute_pipeline(target)

    with tab_manual:
            st.subheader("Scenario Parameters")

            col1, col2 = st.columns(2)

            with col1:

                sales_growth = st.slider(
                    "Sales Growth",
                    min_value=-50,
                    max_value=100,
                    value=0,
                    step=5,
                    format="%d%%"
                )

                price_change = st.slider(
                    "Price Change",
                    min_value=-30,
                    max_value=30,
                    value=0,
                    step=5,
                    format="%d%%"
                )

                cost_change = st.slider(
                    "Cost Change",
                    min_value=-30,
                    max_value=50,
                    value=0,
                    step=5,
                    format="%d%%"
                )

                expense_change = st.slider(
                    "Operating Expense Change",
                    min_value=-30,
                    max_value=50,
                    value=0,
                    step=5,
                    format="%d%%"
                )

            with col2:

                payment_delay = st.slider(
                    "Customer Payment Delay",
                    min_value=0,
                    max_value=90,
                    value=0,
                    step=5,
                    help="Additional customer payment delay in days."
                )

                inventory_change = st.slider(
                    "Inventory Policy Change",
                    min_value=-50,
                    max_value=50,
                    value=0,
                    step=5,
                    format="%d%%"
                )

                supplier_delay = st.slider(
                    "Supplier Payment Delay",
                    min_value=0,
                    max_value=90,
                    value=0,
                    step=5
                )

                months = st.slider(
                    "Simulation Horizon",
                    min_value=1,
                    max_value=36,
                    value=12,
                    step=1
                )

            # --------------------------------------------------------
            # SCENARIO SUMMARY
            # --------------------------------------------------------

            st.markdown("### Scenario")

            scenario_data = {
                "Parameter": [
                    "Sales Growth",
                    "Price Change",
                    "Cost Change",
                    "Expense Change",
                    "Payment Delay",
                    "Inventory Policy",
                    "Supplier Delay",
                    "Horizon"
                ],
                "Value": [
                    f"{sales_growth}%",
                    f"{price_change}%",
                    f"{cost_change}%",
                    f"{expense_change}%",
                    f"{payment_delay} days",
                    f"{inventory_change}%",
                    f"{supplier_delay} days",
                    f"{months} months"
                ]
            }

            st.dataframe(
                pd.DataFrame(scenario_data),
                use_container_width=True,
                hide_index=True
            )

            # --------------------------------------------------------
            # RUN
            # --------------------------------------------------------

            run = st.button(
                "🚀 Run Simulation",
                type="primary",
                use_container_width=True
            )

            if run:

                scenario = Scenario(
                    sales_growth=sales_growth / 100,
                    price_change=price_change / 100,
                    cost_change=cost_change / 100,
                    payment_delay_days=payment_delay,
                    inventory_change=inventory_change / 100,
                    expense_change=expense_change / 100,
                    supplier_payment_delay_days=supplier_delay,
                    months=months
                )

                _execute_pipeline(scenario)


    # ========================================================
    # DISPLAY RESULT
    # ========================================================


    if "simulation_result" in st.session_state and st.session_state["simulation_result"] is not None:

        result = st.session_state["simulation_result"]
        pipe_res = st.session_state.get("pipeline_result")

        st.header("📊 Simulation Result")

        indicator_str = pipe_res.get_status_indicator() if pipe_res else "Planner ✓ → Simulator ✓ → Analysis ✓ → Red-Team ✓ → Decision ✓"
        st.markdown(f"**Pipeline Status:** `{indicator_str}`")

        active_scen = st.session_state.get("scenario")
        if active_scen:
            params = []
            sg = getattr(active_scen, "sales_growth", 0) or 0
            if sg != 0:
                params.append(f"Sales Growth = {sg * 100:.1f}%")
            pc = getattr(active_scen, "price_change", 0) or 0
            if pc != 0:
                params.append(f"Price Change = {pc * 100:.1f}%")
            cc = getattr(active_scen, "cost_change", 0) or 0
            if cc != 0:
                params.append(f"Cost Change = {cc * 100:.1f}%")
            pd_days = getattr(active_scen, "payment_delay_days", 0) or 0
            if pd_days != 0:
                params.append(f"Payment Delay = {pd_days:.0f} days")
            ic = getattr(active_scen, "inventory_change", 0) or 0
            if ic != 0:
                params.append(f"Inventory Policy = {ic * 100:.1f}%")
            ec = getattr(active_scen, "expense_change", 0) or 0
            if ec != 0:
                params.append(f"Expense Change = {ec * 100:.1f}%")
            spd = getattr(active_scen, "supplier_payment_delay_days", 0) or 0
            if spd != 0:
                params.append(f"Supplier Delay = {spd:.0f} days")
            m = getattr(active_scen, "months", 12) or 12
            if not params:
                params.append("Default Baseline Parameters")
            st.info(f"🎯 **Active Scenario:** {' | '.join(params)} (Horizon: {m} months)")

        # ----------------------------------------------------
        # RESULT METRICS
        # ----------------------------------------------------
        proj = result.get("projected", {})
        base = result.get("baseline", {})
        risk = result.get("risk", {})

        baseline_revenue = base.get("revenue", 0)
        projected_revenue = proj.get("revenue", 0)

        baseline_cash = base.get("current_cash", 0)
        ending_cash = proj.get("ending_cash", 0)

        baseline_ar = base.get("accounts_receivable", 0)
        ending_ar = proj.get("ending_accounts_receivable", 0)

        baseline_inventory = base.get("inventory_value", 0)
        ending_inventory = proj.get("ending_inventory", 0)

        baseline_gp = base.get("gross_profit", 0)
        projected_gp = proj.get("gross_profit", 0)

        baseline_np = base.get("gross_profit", 0) - base.get("operating_expenses", 0)
        projected_np = proj.get("net_profit", 0)

        # Basic 4 Columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Projected Revenue", money(projected_revenue), money_delta(projected_revenue - baseline_revenue), delta_color="normal")
        with col2:
            st.metric("Ending Cash", money(ending_cash), money_delta(ending_cash - baseline_cash), delta_color="normal")
        with col3:
            st.metric("Ending AR", money(ending_ar), money_delta(ending_ar - baseline_ar), delta_color="inverse")
        with col4:
            st.metric("Ending Inventory", money(ending_inventory), money_delta(ending_inventory - baseline_inventory), delta_color="off")

        st.markdown("### Profitability & Risk")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Gross Profit", money(projected_gp), money_delta(projected_gp - baseline_gp), delta_color="normal")
        with col2:
            st.metric("Net Profit", money(projected_np), money_delta(projected_np - baseline_np), delta_color="normal")
        with col3:
            st.metric(
                "Overall Risk",
                f"{risk_color(risk.get('overall', 'Unknown'))} {risk.get('overall', 'Unknown')}",
                f"Score: {proj.get('average_risk_score', 0):.1f}"
            )
        with col4:
            st.metric(
                "Inventory Risk",
                f"{risk_color(risk.get('inventory', 'Unknown'))} {risk.get('inventory', 'Unknown')}",
                f"Coverage: {proj.get('minimum_inventory_coverage', 0):.1f}m"
            )

        # ----------------------------------------------------
        # RISK BREAKDOWN
        # ----------------------------------------------------
        st.markdown("### Risk Breakdown")

        overall_risk_label   = risk.get("overall",   "Unknown")
        inventory_risk_label = risk.get("inventory", "Unknown")
        inv_cov   = proj.get("minimum_inventory_coverage", 0)
        lost_sales = proj.get("total_lost_sales", 0)

        # Overall risk sentence — sourced directly from simulator
        explanation = (
            f"Overall risk is {overall_risk_label}. "
            f"Inventory risk is {inventory_risk_label} "
            f"with minimum inventory coverage of {inv_cov:.1f} months."
        )

        # Append authoritative facts for notable conditions
        if lost_sales > 0:
            explanation += (
                f" Lost sales of {money(lost_sales)} occurred due to stockouts."
            )

        if ending_cash <= 0:
            explanation += " Cash reserves are fully exhausted."

        st.info(f"**Why This Risk?** {explanation}")

        # ----------------------------------------------------
        # AI BUSINESS ANALYSIS
        # ----------------------------------------------------
        st.divider()
        st.markdown("### 🤖 AI Business Analysis")

        analysis = st.session_state.get("analysis_result")

        if analysis:
            # Executive Summary
            st.markdown(
                f"> **Executive Summary** — {analysis.get('executive_summary', '')}"
            )

            overall = analysis.get("overall_assessment", "Neutral")
            confidence = analysis.get("confidence", 0)
            assessment_color = {
                "Positive": "🟢", "Neutral": "🟡", "Negative": "🔴"
            }.get(overall, "⚪")

            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.metric(
                    "Overall Assessment",
                    f"{assessment_color} {overall}",
                    f"Confidence: {confidence:.0%}"
                )

            with col_b:
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                    "Key Drivers", "Financial", "Cash Flow",
                    "Working Capital", "Inventory", "Risks",
                    "Trade-offs", "Recommendations"
                ])

                def render_list(items):
                    for item in items:
                        st.markdown(f"- {item}")

                with tab1:
                    render_list(analysis.get("key_drivers", []))
                with tab2:
                    render_list(analysis.get("financial_impact", []))
                with tab3:
                    render_list(analysis.get("cash_flow_impact", []))
                with tab4:
                    render_list(analysis.get("working_capital_impact", []))
                with tab5:
                    render_list(analysis.get("inventory_analysis", []))
                with tab6:
                    render_list(analysis.get("risk_analysis", []))
                with tab7:
                    render_list(analysis.get("tradeoffs", []))
                with tab8:
                    render_list(analysis.get("recommendations", []))
        else:
            st.info("AI Analysis is not available for this run.")

        # ----------------------------------------------------
        # RED-TEAM VALIDATION
        # ----------------------------------------------------
        st.divider()
        st.markdown("### 🛡️ Red-Team Validation")

        rt_verdict = st.session_state.get("red_team_result")

        if rt_verdict:
            overall_verdict = rt_verdict.get("overall_verdict", "UNKNOWN")
            confidence = rt_verdict.get("confidence", 0)

            # Determine color and emoji based on verdict
            if overall_verdict == "PASS":
                verdict_color = "🟢"
                st.success(f"**Verdict:** {verdict_color} {overall_verdict} (Confidence: {confidence:.0%})")
            elif overall_verdict == "PASS_WITH_WARNINGS":
                verdict_color = "🟡"
                st.warning(f"**Verdict:** {verdict_color} {overall_verdict} (Confidence: {confidence:.0%})")
            elif overall_verdict == "FAIL":
                verdict_color = "🔴"
                st.error(f"**Verdict:** {verdict_color} {overall_verdict} (Confidence: {confidence:.0%})")
            else:
                verdict_color = "⚪"
                st.info(f"**Verdict:** {verdict_color} {overall_verdict} (Confidence: {confidence:.0%})")

            # Create an expander to show detailed issues
            with st.expander("View Red-Team Audit Details", expanded=(overall_verdict != "PASS")):
                rt_tabs = st.tabs([
                    "Critical", "Numerical", "Logic", 
                    "Unsupported", "Missing Risks", "Rec. Issues", "Verified Claims"
                ])
                
                def render_rt_list(items):
                    if not items:
                        st.markdown("_No issues found._")
                    else:
                        for item in items:
                            st.markdown(f"- {item}")

                with rt_tabs[0]:
                    render_rt_list(rt_verdict.get("critical_issues", []))
                with rt_tabs[1]:
                    render_rt_list(rt_verdict.get("numerical_issues", []))
                with rt_tabs[2]:
                    render_rt_list(rt_verdict.get("logic_issues", []))
                with rt_tabs[3]:
                    render_rt_list(rt_verdict.get("unsupported_claims", []))
                with rt_tabs[4]:
                    render_rt_list(rt_verdict.get("missing_risks", []))
                with rt_tabs[5]:
                    render_rt_list(rt_verdict.get("recommendation_issues", []))
                with rt_tabs[6]:
                    render_rt_list(rt_verdict.get("verified_claims", []))
        else:
            st.info("Red-Team validation is not available for this run.")

        # ----------------------------------------------------
        # BUSINESS DECISION
        # ----------------------------------------------------
        st.divider()
        st.markdown("### 🎯 Business Decision")

        decision = st.session_state.get("decision_result")

        if decision:
            dec_val = decision.get("decision", "UNKNOWN")
            priority = decision.get("priority", "MEDIUM")
            confidence = decision.get("confidence", 0.0)

            norm_dec = dec_val.upper().replace(" ", "_")

            # Status indicator matching requirements:
            # - PROCEED -> success
            # - PROCEED_WITH_CONDITIONS -> warning
            # - DO_NOT_PROCEED / REVIEW_REQUIRED -> error/warning
            if norm_dec == "PROCEED":
                st.success(f"### Decision: {dec_val} (Confidence: {confidence:.0%})")
            elif "CONDITION" in norm_dec:
                st.warning(f"### Decision: {dec_val} (Confidence: {confidence:.0%})")
            elif any(k in norm_dec for k in ["DO_NOT_PROCEED", "FAIL", "REVIEW", "REJECT"]):
                st.error(f"### Decision: {dec_val} (Confidence: {confidence:.0%})")
            else:
                st.info(f"### Decision: {dec_val} (Confidence: {confidence:.0%})")

            col_p1, col_p2 = st.columns([1, 1])
            with col_p1:
                st.markdown(f"**Priority:** `{priority}`")
            with col_p2:
                st.markdown(f"**Confidence:** `{confidence:.0%}`")

            # Rationale
            st.markdown("#### 🔍 Rationale")
            for rat in decision.get("rationale", []):
                st.markdown(f"- {rat}")

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("#### 📋 Recommended Actions")
                for act in decision.get("recommended_actions", []):
                    st.markdown(f"- {act}")

                st.markdown("#### ⚖️ Conditions")
                for cond in decision.get("conditions", []):
                    st.markdown(f"- {cond}")

            with col_d2:
                st.markdown("#### 💡 Expected Benefits")
                for ben in decision.get("expected_benefits", []):
                    st.markdown(f"- {ben}")

                st.markdown("#### ⚠️ Risks")
                for rsk in decision.get("risks", []):
                    st.markdown(f"- {rsk}")
        else:
            st.info("Business decision is not available for this run.")

        # ----------------------------------------------------
        # SCENARIO VS BASELINE COMPARISON
        # ----------------------------------------------------
        st.markdown("### Scenario vs Baseline")
        
        comp_data = {
            "Metric": [
                "Revenue", "Gross Profit", "Net Profit", "Cash", "AR", "Inventory"
            ],
            "Baseline": [
                money(baseline_revenue), money(baseline_gp), money(baseline_np),
                money(baseline_cash), money(baseline_ar), money(baseline_inventory)
            ],
            "Scenario": [
                money(projected_revenue), money(projected_gp), money(projected_np),
                money(ending_cash), money(ending_ar), money(ending_inventory)
            ],
            "Change": [
                money(projected_revenue - baseline_revenue),
                money(projected_gp - baseline_gp),
                money(projected_np - baseline_np),
                money(ending_cash - baseline_cash),
                money(ending_ar - baseline_ar),
                money(ending_inventory - baseline_inventory)
            ]
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        st.markdown(f"**Total Lost Sales:** {money(proj.get('total_lost_sales', 0))}")

        # ----------------------------------------------------
        # MONTHLY DATA
        # ----------------------------------------------------
        monthly = result.get("monthly", [])

        if monthly:
            st.markdown("### Monthly Projection")
            df = pd.DataFrame(monthly)

            # Map the exact keys returned by simulation to desired column names
            display_columns = {
                "month": "Month",
                "fulfilled_revenue": "Revenue",
                "cogs_consumption": "COGS",
                "gross_profit": "Gross Profit",
                "net_profit": "Net Profit",
                "customer_collections": "Collections",
                "ending_ar": "Ending AR",
                "inventory_purchases": "Inventory Purchases",
                "ending_inventory": "Ending Inventory",
                "ending_cash": "Ending Cash",
                "inventory_coverage_months": "Inventory Coverage",
                "risk_level": "Risk"
            }

            # Only select columns that actually exist in the dataframe
            cols_to_show = [c for c in display_columns.keys() if c in df.columns]
            df_display = df[cols_to_show].rename(columns=display_columns)

            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # -----------------------------------------------
            # CHARTS
            # -----------------------------------------------
            
            def create_chart(df, y_col, title, name):
                if y_col in df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df["month"], y=df[y_col], mode="lines+markers", name=name))
                    fig.update_layout(title=title, xaxis_title="Month", yaxis_title="₹", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                create_chart(df, "fulfilled_revenue", "Monthly Revenue", "Revenue")
                create_chart(df, "ending_ar", "Accounts Receivable", "AR")
            with col2:
                create_chart(df, "ending_cash", "Cash Position", "Cash")
                create_chart(df, "ending_inventory", "Inventory Position", "Inventory")
            
            create_chart(df, "net_profit", "Monthly Net Profit", "Net Profit")

            # -----------------------------------------------
            # SAVE SCENARIO
            # -----------------------------------------------
            st.divider()
            st.subheader("💾 Save Scenario")
            st.write("Save this scenario to compare it against others.")
            
            scen_name = st.text_input("Scenario Name", placeholder="e.g. Aggressive Growth")
            
            if st.button("Save Scenario", type="secondary"):
                if scen_name.strip():
                    st.session_state["saved_scenarios"][scen_name.strip()] = {
                        "scenario": st.session_state["scenario"],
                        "result": result
                    }
                    st.success(f"Scenario '{scen_name.strip()}' saved successfully!")
                else:
                    st.warning("Please provide a name for the scenario.")


# ============================================================
# SCENARIO COMPARISON
# ============================================================

elif page in ["⚖️ Scenario Comparison", "Scenario Comparison"]:
    st.header("⚖️ Scenario Comparison")
    
    saved = st.session_state.get("saved_scenarios", {})
    
    if not saved:
        st.warning("No saved scenarios found. Go to the 'What-If Lab' and save a simulation first.")
    else:
        st.write("Compare saved what-if scenarios against the baseline.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_names = st.multiselect(
                "Select Scenarios to Compare",
                options=list(saved.keys()),
                default=list(saved.keys())
            )
        with col2:
            st.write("") # padding
            st.write("")
            if st.button("🗑️ Clear Saved Scenarios"):
                st.session_state["saved_scenarios"] = {}
                st.rerun()
                
        st.divider()
        
        if selected_names:
            first_res = saved[selected_names[0]]["result"]
            base = first_res.get("baseline", {})
            
            metrics = [
                "Revenue", "Gross Profit", "Net Profit", "Ending Cash", 
                "Ending AR", "Ending Inventory", "Inventory Coverage", 
                "Lost Sales", "Overall Risk", "Inventory Risk"
            ]
            
            def extract_comp_metric(res, metric_name, is_baseline=False):
                proj = res.get("projected", {})
                b = res.get("baseline", {})
                rsk = res.get("risk", {})
                
                if is_baseline:
                    if metric_name == "Revenue": return b.get("revenue", 0)
                    if metric_name == "Gross Profit": return b.get("gross_profit", 0)
                    if metric_name == "Net Profit": return b.get("gross_profit", 0) - b.get("operating_expenses", 0)
                    if metric_name == "Ending Cash": return b.get("current_cash", 0)
                    if metric_name == "Ending AR": return b.get("accounts_receivable", 0)
                    if metric_name == "Ending Inventory": return b.get("inventory_value", 0)
                    if metric_name == "Inventory Coverage": return b.get("inventory_value", 0) / (b.get("cogs", 1) / 12) if b.get("cogs", 0) > 0 else 0
                    if metric_name == "Lost Sales": return 0
                    if metric_name == "Overall Risk": return "Baseline"
                    if metric_name == "Inventory Risk": return "Baseline"
                    
                if metric_name == "Revenue": return proj.get("revenue", 0)
                if metric_name == "Gross Profit": return proj.get("gross_profit", 0)
                if metric_name == "Net Profit": return proj.get("net_profit", 0)
                if metric_name == "Ending Cash": return proj.get("ending_cash", 0)
                if metric_name == "Ending AR": return proj.get("ending_accounts_receivable", 0)
                if metric_name == "Ending Inventory": return proj.get("ending_inventory", 0)
                if metric_name == "Inventory Coverage": return proj.get("minimum_inventory_coverage", 0)
                if metric_name == "Lost Sales": return proj.get("total_lost_sales", 0)
                if metric_name == "Overall Risk": return rsk.get("overall", "Unknown")
                if metric_name == "Inventory Risk": return rsk.get("inventory", "Unknown")
                
                return 0

            comp_data = {"Metric": metrics}
            comp_data["Baseline"] = [
                money(extract_comp_metric(first_res, m, True)) if "Risk" not in m and "Coverage" not in m
                else (f"{extract_comp_metric(first_res, m, True):.1f}m" if "Coverage" in m else extract_comp_metric(first_res, m, True))
                for m in metrics
            ]
            
            for s_name in selected_names:
                r = saved[s_name]["result"]
                comp_data[s_name] = [
                    money(extract_comp_metric(r, m)) if "Risk" not in m and "Coverage" not in m
                    else (f"{extract_comp_metric(r, m):.1f}m" if "Coverage" in m else extract_comp_metric(r, m))
                    for m in metrics
                ]
                
            st.markdown("### Comparison Table")
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
            
            st.markdown("### Visual Comparison")
            
            chart_scenarios = ["Baseline"] + selected_names
            
            def get_raw_series(metric_name):
                vals = [extract_comp_metric(first_res, metric_name, True)]
                for s in selected_names:
                    vals.append(extract_comp_metric(saved[s]["result"], metric_name))
                return vals
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_prof = go.Figure()
                fig_prof.add_trace(go.Bar(name='Revenue', x=chart_scenarios, y=get_raw_series("Revenue")))
                fig_prof.add_trace(go.Bar(name='Gross Profit', x=chart_scenarios, y=get_raw_series("Gross Profit")))
                fig_prof.add_trace(go.Bar(name='Net Profit', x=chart_scenarios, y=get_raw_series("Net Profit")))
                fig_prof.update_layout(title="Profitability Comparison", barmode='group', yaxis_title="₹")
                st.plotly_chart(fig_prof, use_container_width=True)
                
                fig_ar = go.Figure()
                fig_ar.add_trace(go.Bar(name='Ending AR', x=chart_scenarios, y=get_raw_series("Ending AR"), marker_color="orange"))
                fig_ar.update_layout(title="Accounts Receivable Pressure", yaxis_title="₹")
                st.plotly_chart(fig_ar, use_container_width=True)

            with col2:
                fig_cash = go.Figure()
                fig_cash.add_trace(go.Bar(name='Ending Cash', x=chart_scenarios, y=get_raw_series("Ending Cash"), marker_color="green"))
                fig_cash.update_layout(title="Cash Position Comparison", yaxis_title="₹")
                st.plotly_chart(fig_cash, use_container_width=True)
                
                fig_inv = go.Figure()
                fig_inv.add_trace(go.Bar(name='Ending Inventory', x=chart_scenarios, y=get_raw_series("Ending Inventory"), marker_color="purple"))
                fig_inv.update_layout(title="Inventory Accumulation", yaxis_title="₹")
                st.plotly_chart(fig_inv, use_container_width=True)
                
            st.divider()
            st.markdown("### Best Scenario Insights")
            
            if len(selected_names) > 0:
                rev_vals = [(s, extract_comp_metric(saved[s]["result"], "Revenue")) for s in selected_names]
                prof_vals = [(s, extract_comp_metric(saved[s]["result"], "Net Profit")) for s in selected_names]
                cash_vals = [(s, extract_comp_metric(saved[s]["result"], "Ending Cash")) for s in selected_names]
                ar_vals = [(s, extract_comp_metric(saved[s]["result"], "Ending AR")) for s in selected_names]
                
                best_rev = max(rev_vals, key=lambda x: x[1])
                best_prof = max(prof_vals, key=lambda x: x[1])
                best_cash = max(cash_vals, key=lambda x: x[1])
                best_ar = min(ar_vals, key=lambda x: x[1])
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Highest Revenue", best_rev[0], money(best_rev[1]), delta_color="off")
                c2.metric("Highest Profit", best_prof[0], money(best_prof[1]), delta_color="off")
                c3.metric("Highest Cash", best_cash[0], money(best_cash[1]), delta_color="off")
                c4.metric("Lowest AR", best_ar[0], money(best_ar[1]), delta_color="off")
                
                st.markdown("### Trade-offs Analysis")
                
                if best_prof[0] != best_cash[0]:
                    st.info(f"💡 **Growth vs Liquidity:** **{best_prof[0]}** produces the highest net profit, but **{best_cash[0]}** preserves the most ending cash.")
                else:
                    st.info(f"💡 **Dominant Strategy:** **{best_prof[0]}** maximizes both net profit and cash preservation.")
                    
                if best_rev[0] != best_ar[0] and best_rev[1] > extract_comp_metric(saved[best_ar[0]]["result"], "Revenue"):
                    st.warning(f"⚠️ **Working Capital Trap:** **{best_rev[0]}** drives top-line revenue, but creates significantly more Accounts Receivable burden compared to **{best_ar[0]}**.")
                    
                high_inv_risk = [s for s in selected_names if extract_comp_metric(saved[s]["result"], "Inventory Risk") in ["High", "Critical"]]
                if high_inv_risk:
                    st.error(f"🚨 **Supply Chain Risk:** The following scenarios run high or critical inventory risk due to thin safety coverage: {', '.join(high_inv_risk)}")