"""
MCP Tool Implementations for Business Digital Twin.
Wraps existing business logic, Digital Twin queries, RAG policies, deterministic simulation,
and event-driven transactional mutations.
"""

from typing import Any, Dict, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

from backend.digital_twin.digital_twin import (
    build_digital_twin,
    decimal_to_float,
    get_connection,
)
from backend.digital_twin.simulation import Scenario, simulate
from events.event_processor import EventProcessor
from events.event_types import EventType
from rag.rag_service import retrieve_business_context


class MCPBusinessTools:
    """
    Executes controlled business operations through standard tool signatures.
    Never exposes raw SQL or arbitrary execution to the LLM.
    """

    def __init__(self):
        self._event_processor: Optional[EventProcessor] = None

    @property
    def event_processor(self) -> EventProcessor:
        if self._event_processor is None:
            self._event_processor = EventProcessor()
        return self._event_processor

    # -----------------------------------------------------------------------
    # 1. READ / OBSERVATION TOOLS
    # -----------------------------------------------------------------------

    def get_business_snapshot(self) -> Dict[str, Any]:
        """Returns the authoritative digital twin state."""
        twin = build_digital_twin()
        return {
            "data_authority": "AUTHORITATIVE",
            "company": twin["metadata"]["company"],
            "currency": twin["metadata"]["currency"],
            "financials": twin["financials"],
            "cash_flow": twin["cash_flow"],
            "working_capital": twin["working_capital"],
            "risk": twin["risk"],
            "operations": twin["operations"],
            "payment_behavior": twin["payment_behavior"],
        }

    def get_inventory_status(self) -> Dict[str, Any]:
        """Queries granular inventory items and products currently at or below reorder level."""
        conn = get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        i.product_id,
                        p.product_name,
                        p.category,
                        i.stock_quantity,
                        i.reorder_level,
                        p.cost_price,
                        (i.stock_quantity * p.cost_price) AS holding_value,
                        (i.stock_quantity <= i.reorder_level) AS is_at_risk
                    FROM inventory i
                    JOIN products p ON i.product_id = p.product_id
                    ORDER BY (i.stock_quantity <= i.reorder_level) DESC, (i.stock_quantity * p.cost_price) DESC;
                """)
                rows = cur.fetchall()

            total_items = len(rows)
            at_risk_items = [
                {
                    "product_id": r["product_id"],
                    "product_name": r["product_name"],
                    "category": r["category"],
                    "stock_quantity": int(r["stock_quantity"]),
                    "reorder_level": int(r["reorder_level"]),
                    "cost_price": decimal_to_float(r["cost_price"]),
                    "holding_value": decimal_to_float(r["holding_value"]),
                }
                for r in rows
                if r["is_at_risk"]
            ]

            total_units = sum(int(r["stock_quantity"]) for r in rows)
            total_value = sum(decimal_to_float(r["holding_value"]) for r in rows)

            return {
                "data_authority": "AUTHORITATIVE",
                "total_products": total_items,
                "products_at_risk_count": len(at_risk_items),
                "risk_percentage": round((len(at_risk_items) / total_items * 100) if total_items > 0 else 0.0, 2),
                "total_stock_units": total_units,
                "total_inventory_value": round(total_value, 2),
                "at_risk_products": at_risk_items,
            }
        finally:
            conn.close()

    def get_customer_exposure(self) -> Dict[str, Any]:
        """Queries customer accounts receivable concentration and outstanding payment exposure."""
        conn = get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        c.customer_id,
                        c.customer_name,
                        c.payment_terms_days,
                        c.credit_limit,
                        COUNT(s.sale_id) AS pending_sales_count,
                        COALESCE(SUM(s.quantity * s.unit_price), 0) AS total_pending_ar
                    FROM customers c
                    LEFT JOIN sales s 
                        ON c.customer_id = s.customer_id 
                        AND s.payment_status = 'Pending'
                    GROUP BY c.customer_id, c.customer_name, c.payment_terms_days, c.credit_limit
                    HAVING COALESCE(SUM(s.quantity * s.unit_price), 0) > 0
                    ORDER BY total_pending_ar DESC;
                """)
                customer_rows = cur.fetchall()

                # Get total overall AR
                cur.execute("""
                    SELECT COALESCE(SUM(quantity * unit_price), 0) AS total_ar
                    FROM sales
                    WHERE payment_status = 'Pending';
                """)
                total_ar = decimal_to_float(cur.fetchone()["total_ar"])

            top_exposures = []
            for r in customer_rows:
                pending = decimal_to_float(r["total_pending_ar"])
                credit_limit = decimal_to_float(r["credit_limit"])
                pct_of_limit = round((pending / credit_limit * 100) if credit_limit > 0 else 0.0, 2)
                pct_over_limit = round(((pending - credit_limit) / credit_limit * 100) if credit_limit > 0 else 0.0, 2)
                is_over = pending > credit_limit
                over_amount = round(max(0.0, pending - credit_limit), 2)

                top_exposures.append({
                    "customer_id": r["customer_id"],
                    "customer_name": r["customer_name"],
                    "payment_terms_days": int(r["payment_terms_days"]),
                    "credit_limit": credit_limit,
                    "pending_sales_count": int(r["pending_sales_count"]),
                    "pending_amount": pending,
                    "share_of_ar_percent": round(
                        (pending / total_ar * 100) if total_ar > 0 else 0.0, 2
                    ),
                    "percentage_of_credit_limit": pct_of_limit,
                    "percentage_over_credit_limit": pct_over_limit if is_over else 0.0,
                    "is_over_credit_limit": is_over,
                    "over_limit_amount": over_amount,
                })

            return {
                "data_authority": "AUTHORITATIVE",
                "total_accounts_receivable": total_ar,
                "customers_with_pending_ar_count": len(top_exposures),
                "customer_exposures": top_exposures,
            }
        finally:
            conn.close()

    def get_cash_position(self) -> Dict[str, Any]:
        """Returns cash position and operational liquidity metrics."""
        twin = build_digital_twin()
        cash_flow = twin["cash_flow"]
        fin = twin["financials"]
        avg_monthly_opex = fin["operating_expenses"] / 12 if fin["operating_expenses"] > 0 else 1.0
        runway_months = round(cash_flow["current_cash"] / avg_monthly_opex, 2) if avg_monthly_opex > 0 else 99.0

        return {
            "data_authority": "AUTHORITATIVE",
            "opening_cash": cash_flow["opening_cash"],
            "payments_received": cash_flow["payments_received"],
            "operating_cash_flow": cash_flow["operating_cash_flow"],
            "current_cash": cash_flow["current_cash"],
            "estimated_cash_runway_months": runway_months,
        }

    def get_payment_metrics(self) -> Dict[str, Any]:
        """
        Returns customer payment performance, payment delay, and Days Sales Outstanding (DSO).
        Metric Definitions:
        - dso_days: Days Sales Outstanding = (AR / Revenue) * 365. Velocity of converting sales to cash.
        - average_payment_days: Historical average collection duration on settled invoices.
        - average_payment_terms_days: Nominal contractual terms agreed with customers.
        - inventory_days: Inventory holding duration = 365 / inventory_turnover.
        """
        twin = build_digital_twin()
        dso_val = twin.get("payment_behavior", {}).get("dso_days")
        if dso_val is None:
            rev = twin["financials"]["revenue"]
            ar = twin["working_capital"]["accounts_receivable"]
            dso_val = round((ar / rev) * 365, 2) if rev > 0 else 0.0

        return {
            "data_authority": "AUTHORITATIVE",
            "dso_days": dso_val,
            "average_payment_days": twin["payment_behavior"]["average_payment_days"],
            "average_payment_terms_days": twin["payment_behavior"]["average_payment_terms"],
            "total_payments_processed": twin["payment_behavior"]["total_payments"],
            "accounts_receivable": twin["working_capital"]["accounts_receivable"],
            "receivable_exposure_percentage": twin["working_capital"]["receivable_percentage"],
        }

    def get_sales_summary(self) -> Dict[str, Any]:
        """Queries historical sales statistics and top revenue contributors."""
        conn = get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) AS total_transactions,
                        COALESCE(SUM(quantity), 0) AS total_units_sold,
                        COALESCE(SUM(quantity * unit_price), 0) AS total_revenue,
                        COALESCE(AVG(unit_price), 0) AS average_sale_price
                    FROM sales;
                """)
                stats = cur.fetchone()

                cur.execute("""
                    SELECT 
                        p.product_name,
                        SUM(s.quantity) AS units_sold,
                        SUM(s.quantity * s.unit_price) AS revenue
                    FROM sales s
                    JOIN products p ON s.product_id = p.product_id
                    GROUP BY p.product_name
                    ORDER BY revenue DESC
                    LIMIT 5;
                """)
                top_products = [
                    {
                        "product_name": r["product_name"],
                        "units_sold": int(r["units_sold"]),
                        "revenue": decimal_to_float(r["revenue"]),
                    }
                    for r in cur.fetchall()
                ]

            return {
                "data_authority": "AUTHORITATIVE",
                "total_transactions": int(stats["total_transactions"]),
                "total_units_sold": int(stats["total_units_sold"]),
                "total_revenue": decimal_to_float(stats["total_revenue"]),
                "average_sale_price": round(decimal_to_float(stats["average_sale_price"]), 2),
                "top_products": top_products,
            }
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # 2. KNOWLEDGE TOOLS (RAG)
    # -----------------------------------------------------------------------

    def search_business_policy(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves authoritative company policy from RAG with source attribution."""
        results = retrieve_business_context(query=query, top_k=top_k)
        for r in results:
            if isinstance(r, dict):
                r["data_authority"] = "RETRIEVED_KNOWLEDGE"
        return results

    # -----------------------------------------------------------------------
    # 3. SIMULATION TOOLS
    # -----------------------------------------------------------------------

    def run_business_simulation(self, **scenario_kwargs: Any) -> Dict[str, Any]:
        """
        Runs a deterministic what-if projection without modifying the database.
        Accepts scenario parameters (sales_growth, payment_delay_days, etc.).
        """
        sc = Scenario(
            sales_growth=float(scenario_kwargs.get("sales_growth", 0.0) or 0.0),
            price_change=float(scenario_kwargs.get("price_change", 0.0) or 0.0),
            cost_change=float(scenario_kwargs.get("cost_change", 0.0) or 0.0),
            expense_change=float(scenario_kwargs.get("expense_change", 0.0) or 0.0),
            payment_delay_days=float(scenario_kwargs.get("payment_delay_days", 0.0) or 0.0),
            inventory_change=float(scenario_kwargs.get("inventory_change", 0.0) or 0.0),
            supplier_payment_delay_days=float(scenario_kwargs.get("supplier_payment_delay_days", 0.0) or 0.0),
            months=int(scenario_kwargs.get("months", 12) or 12),
        )

        sim_result = simulate(sc)
        base = sim_result.get("baseline", {})
        proj = sim_result.get("projected", {})
        risk = sim_result.get("risk", {})

        return {
            "data_authority": "SIMULATED",
            "scenario": {
                "sales_growth": f"{round(sc.sales_growth * 100, 2):g}%",
                "price_change": f"{round(sc.price_change * 100, 2):g}%",
                "cost_change": f"{round(sc.cost_change * 100, 2):g}%",
                "expense_change": f"{round(sc.expense_change * 100, 2):g}%",
                "payment_delay_days": f"{sc.payment_delay_days:g} days",
                "inventory_change": f"{round(sc.inventory_change * 100, 2):g}%",
                "supplier_payment_delay_days": f"{sc.supplier_payment_delay_days:g} days",
                "months": f"{sc.months} months",
            },
            "financial_impact": {
                "baseline_revenue": base.get("revenue"),
                "projected_revenue": proj.get("revenue"),
                "revenue_change_pct": round(proj.get("revenue_growth_pct", 0.0) or 0.0, 2),
                "projected_gross_margin": proj.get("gross_margin"),
                "projected_net_profit": proj.get("net_profit"),
                "projected_net_margin": proj.get("net_margin"),
            },
            "liquidity_impact": {
                "baseline_cash": base.get("current_cash"),
                "projected_ending_cash": proj.get("ending_cash"),
                "cash_runway_months": proj.get("cash_runway_months"),
                "ending_accounts_receivable": proj.get("ending_accounts_receivable"),
            },
            "inventory_impact": {
                "projected_ending_inventory": proj.get("ending_inventory"),
                "minimum_inventory_coverage_months": proj.get("minimum_inventory_coverage"),
                "lost_sales": proj.get("total_lost_sales", 0.0),
                "stockout_exposure": proj.get("maximum_stockout_exposure", 0.0),
            },
            "risk_assessment": {
                "inventory_risk": risk.get("inventory"),
                "overall_risk": risk.get("overall"),
            },
        }

    def compare_scenarios(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simulates and compares multiple scenarios."""
        comparisons: List[Dict[str, Any]] = []
        for s in scenarios:
            name = s.get("name", "Unnamed Scenario")
            result = self.run_business_simulation(**s)
            comparisons.append({
                "data_authority": "SIMULATED",
                "name": name,
                "projected_revenue": result["financial_impact"]["projected_revenue"],
                "projected_gross_margin": result["financial_impact"]["projected_gross_margin"],
                "projected_net_profit": result["financial_impact"]["projected_net_profit"],
                "projected_ending_cash": result["liquidity_impact"]["projected_ending_cash"],
                "cash_runway": result["liquidity_impact"]["cash_runway_months"],
                "inventory_risk": result["risk_assessment"]["inventory_risk"],
                "overall_risk": result["risk_assessment"]["overall_risk"],
            })
        return comparisons

    # -----------------------------------------------------------------------
    # 4. ACTION TOOL (GATED HUMAN-IN-THE-LOOP)
    # -----------------------------------------------------------------------

    def create_business_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        approved: bool = False,
    ) -> Dict[str, Any]:
        """
        Gated action interface. If approved=False, returns proposal preview.
        If approved=True, executes via EventProcessor with PostgreSQL transaction boundaries.
        """
        if not approved:
            return {
                "data_authority": "AUTHORITATIVE",
                "status": "PENDING_APPROVAL",
                "message": (
                    f"Action proposed: {event_type}. User approval is REQUIRED before mutating the database."
                ),
                "event_type": event_type,
                "payload": payload,
                "requires_user_approval": True,
            }

        result = self.event_processor.process_event(
            event_type=event_type,
            payload=payload,
        )

        if result.success:
            # Refresh Digital Twin to verify state update
            new_twin = build_digital_twin()
            return {
                "data_authority": "AUTHORITATIVE",
                "status": "EXECUTED",
                "success": True,
                "event_id": result.event_id,
                "event_type": result.event_type,
                "message": result.message,
                "details": result.details,
                "digital_twin_verified": True,
                "updated_business_state": {
                    "cash_balance": new_twin.get("cash_flow", {}).get("current_cash"),
                    "revenue": new_twin.get("financials", {}).get("revenue"),
                    "accounts_receivable": new_twin.get("working_capital", {}).get("accounts_receivable"),
                    "inventory_value": new_twin.get("working_capital", {}).get("inventory_value"),
                },
            }

        return {
            "data_authority": "AUTHORITATIVE",
            "status": "FAILED",
            "success": False,
            "event_id": result.event_id,
            "event_type": result.event_type,
            "message": result.message,
            "details": result.details,
            "error": result.error,
            "digital_twin_verified": False,
        }
