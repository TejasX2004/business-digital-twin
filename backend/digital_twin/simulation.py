import sys
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from .digital_twin import build_digital_twin

# ============================================================
# CONFIGURATION
# ============================================================
MONTHS_IN_YEAR = 12
SUPPLIER_PAYMENT_DAYS = 30
TAX_RATE = 0.0
REPLENISHMENT_CAPACITY_FACTOR = 1.5

@dataclass
class Scenario:
    sales_growth: float = 0.0
    price_change: float = 0.0
    cost_change: float = 0.0
    expense_change: float = 0.0
    payment_delay_days: float = 0.0
    inventory_change: float = 0.0
    supplier_payment_delay_days: float = 0.0
    months: int = 12

def validate_scenario(scenario: Scenario):
    if scenario.months < 1:
        raise ValueError("months must be >= 1")
    if scenario.sales_growth <= -1:
        raise ValueError("sales_growth must be greater than -100%")
    if scenario.price_change <= -1:
        raise ValueError("price_change must be greater than -100%")
    if scenario.cost_change <= -1:
        raise ValueError("cost_change must be greater than -100%")
    if scenario.expense_change <= -1:
        raise ValueError("expense_change must be greater than -100%")
    if scenario.inventory_change <= -1:
        raise ValueError("inventory_change must be greater than -100%")
    if scenario.payment_delay_days < -365:
        raise ValueError("payment_delay_days is too negative")

def safe_divide(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0

def pct_change(current: float, projected: float) -> float:
    return ((projected - current) / abs(current)) * 100 if current != 0 else 0.0

def risk_level(score: float) -> str:
    if score >= 75: return "Critical"
    if score >= 50: return "High"
    if score >= 25: return "Medium"
    return "Low"

def get_baseline_metrics(twin: Dict[str, Any]) -> Dict[str, float]:
    f = twin["financials"]
    wc = twin["working_capital"]
    cf = twin["cash_flow"]
    
    return {
        "revenue": float(f["revenue"]),
        "cogs": float(f["cogs"]),
        "gross_profit": float(f["gross_profit"]),
        "operating_expenses": float(f["operating_expenses"]),
        "accounts_receivable": float(wc["accounts_receivable"]),
        "inventory_value": float(wc["inventory_value"]),
        "current_cash": float(cf["current_cash"])
    }

def simulate_monthly_states(baseline: Dict[str, float], scenario: Scenario) -> List[Dict[str, Any]]:
    base_monthly_revenue = baseline["revenue"] / MONTHS_IN_YEAR
    cogs_ratio = safe_divide(baseline["cogs"], baseline["revenue"])
    base_monthly_expenses = baseline["operating_expenses"] / MONTHS_IN_YEAR
    base_monthly_cogs = baseline["cogs"] / MONTHS_IN_YEAR
    
    baseline_inventory_coverage = safe_divide(baseline["inventory_value"], base_monthly_cogs)
    target_inventory_coverage = baseline_inventory_coverage * (1 + scenario.inventory_change)
    
    current_cash = baseline["current_cash"]
    current_ar = baseline["accounts_receivable"]
    current_inventory = baseline["inventory_value"]
    
    current_ap = base_monthly_cogs * (SUPPLIER_PAYMENT_DAYS / 30)
    
    baseline_dso = safe_divide(baseline["accounts_receivable"], baseline["revenue"]) * 365
    effective_dso = max(1.0, baseline_dso + scenario.payment_delay_days)
    collection_ratio = min(1.0, 30.0 / effective_dso)
    
    effective_supplier_days = max(1.0, SUPPLIER_PAYMENT_DAYS + scenario.supplier_payment_delay_days)
    payment_ratio = min(1.0, 30.0 / effective_supplier_days)
    
    monthly_results = []
    
    for month in range(1, scenario.months + 1):
        # 1. Base Expected Demand
        expected_demand_revenue = base_monthly_revenue * (1 + scenario.sales_growth) * (1 + scenario.price_change)
        expected_demand_cogs = expected_demand_revenue * cogs_ratio * (1 + scenario.cost_change)
        
        # 2. Inventory Replenishment Target & Limits
        target_inventory = expected_demand_cogs * target_inventory_coverage
        inventory_gap = target_inventory - current_inventory
        
        desired_purchases = max(0.0, inventory_gap + expected_demand_cogs)
        max_replenishment_per_month = expected_demand_cogs * REPLENISHMENT_CAPACITY_FACTOR
        inventory_purchases = min(desired_purchases, max_replenishment_per_month)
        
        available_inventory = current_inventory + inventory_purchases
        
        # 3. Fulfillment & Stockouts
        if available_inventory >= expected_demand_cogs:
            fulfilled_revenue = expected_demand_revenue
            cogs_consumption = expected_demand_cogs
            lost_sales = 0.0
            stockout_units_or_value = 0.0
            ending_inventory = available_inventory - expected_demand_cogs
        else:
            fulfilled_ratio = safe_divide(available_inventory, expected_demand_cogs)
            fulfilled_revenue = expected_demand_revenue * fulfilled_ratio
            lost_sales = expected_demand_revenue - fulfilled_revenue
            cogs_consumption = available_inventory
            stockout_units_or_value = expected_demand_cogs - available_inventory
            ending_inventory = 0.0
            
        # 4. P&L Items
        gross_profit = fulfilled_revenue - cogs_consumption
        expenses = base_monthly_expenses * (1 + scenario.expense_change)
        operating_profit = gross_profit - expenses
        taxes = max(0.0, operating_profit * TAX_RATE)
        net_profit = operating_profit - taxes
        
        # 5. Accounts Receivable & Collections
        customer_collections = current_ar * collection_ratio
        ending_ar = current_ar + fulfilled_revenue - customer_collections
        
        # 6. Accounts Payable & Supplier Payments
        supplier_payments = current_ap * payment_ratio
        ending_ap = current_ap + inventory_purchases - supplier_payments
        
        # 7. Cash Flow
        operating_cash_flow = customer_collections - supplier_payments - expenses - taxes
        ending_cash = current_cash + operating_cash_flow
        
        # Recon Checks & Assertions
        assert ending_inventory >= 0, f"Month {month}: Ending inventory cannot be negative."
        assert abs((current_inventory + inventory_purchases - cogs_consumption) - ending_inventory) < 0.01, "Inventory Recon Failed"
        if month > 1:
            assert current_inventory == monthly_results[-1]["ending_inventory"], "State continuation failed"
            
        # 8. Risk Scoring
        inventory_coverage_months = safe_divide(ending_inventory, expected_demand_cogs)
        
        if inventory_coverage_months >= 6.0:
            inv_risk_score = 0
        elif inventory_coverage_months >= 4.0:
            inv_risk_score = 50
        elif inventory_coverage_months >= 2.0:
            inv_risk_score = 75
        else:
            inv_risk_score = 100
            
        if lost_sales > 0:
            inv_risk_score = 100

        profit_margin = safe_divide(net_profit, fulfilled_revenue)
        prof_risk_score = 100 if profit_margin < 0 else (75 if profit_margin < 0.05 else 0)
        
        cash_risk_score = 100 if ending_cash <= 0 else (75 if ending_cash < baseline["current_cash"] * 0.25 else (25 if ending_cash < baseline["current_cash"] * 0.50 else 0))
        
        overall_score = (prof_risk_score * 0.4) + (cash_risk_score * 0.4) + (inv_risk_score * 0.2)
        
        monthly_results.append({
            "month": month,
            "demand_revenue": expected_demand_revenue,
            "fulfilled_revenue": fulfilled_revenue,
            "cogs_consumption": cogs_consumption,
            "gross_profit": gross_profit,
            "operating_expenses": expenses,
            "taxes": taxes,
            "net_profit": net_profit,
            "opening_cash": current_cash,
            "customer_collections": customer_collections,
            "inventory_purchases": inventory_purchases,
            "supplier_payments": supplier_payments,
            "operating_expense_cash": expenses,
            "operating_cash_flow": operating_cash_flow,
            "ending_cash": ending_cash,
            "cash_change": ending_cash - current_cash,
            "opening_ar": current_ar,
            "ending_ar": ending_ar,
            "ar_change": ending_ar - current_ar,
            "opening_inventory": current_inventory,
            "ending_inventory": ending_inventory,
            "target_inventory": target_inventory,
            "inventory_gap": inventory_gap,
            "inventory_coverage_months": inventory_coverage_months,
            "inventory_pressure": "Critical" if inv_risk_score >= 100 else ("High" if inv_risk_score >= 75 else ("Medium" if inv_risk_score >= 50 else "Low")),
            "stockout_units_or_value": stockout_units_or_value,
            "lost_sales": lost_sales,
            "inventory_risk_score": inv_risk_score,
            "risk_score": overall_score,
            "risk_level": risk_level(overall_score)
        })
        
        current_cash = ending_cash
        current_ar = ending_ar
        current_inventory = ending_inventory
        current_ap = ending_ap
        
    return monthly_results

def aggregate_results(monthly_results: List[Dict[str, Any]], scenario_months: int) -> Dict[str, Any]:
    if not monthly_results: return {}
    
    annualized_factor = 12.0 / scenario_months
    
    total_demand = sum(m["demand_revenue"] for m in monthly_results)
    total_revenue = sum(m["fulfilled_revenue"] for m in monthly_results)
    total_cogs = sum(m["cogs_consumption"] for m in monthly_results)
    total_gp = sum(m["gross_profit"] for m in monthly_results)
    total_exp = sum(m["operating_expenses"] for m in monthly_results)
    total_taxes = sum(m["taxes"] for m in monthly_results)
    total_np = sum(m["net_profit"] for m in monthly_results)
    
    total_lost_sales = sum(m["lost_sales"] for m in monthly_results)
    max_stockout = max(m["stockout_units_or_value"] for m in monthly_results)
    
    total_collections = sum(m["customer_collections"] for m in monthly_results)
    total_purchases = sum(m["inventory_purchases"] for m in monthly_results)
    total_supplier_payments = sum(m["supplier_payments"] for m in monthly_results)
    
    final_month = monthly_results[-1]
    
    avg_ar = sum(m["ending_ar"] for m in monthly_results) / scenario_months
    avg_inv = sum(m["ending_inventory"] for m in monthly_results) / scenario_months
    avg_risk = sum(m["risk_score"] for m in monthly_results) / scenario_months
    avg_inv_risk = sum(m["inventory_risk_score"] for m in monthly_results) / scenario_months
    
    min_inv = min(m["ending_inventory"] for m in monthly_results)
    min_inv_coverage = min(m["inventory_coverage_months"] for m in monthly_results)
    
    inventory_turnover = safe_divide(total_cogs * annualized_factor, avg_inv)
    
    exhaustion_month = None
    for m in monthly_results:
        if m["ending_cash"] <= 0:
            exhaustion_month = m["month"]
            break
            
    if exhaustion_month:
        cash_runway_str = f"Exhausted in Month {exhaustion_month}"
    else:
        cash_runway_str = "Not reached in projection"

    return {
        "demand_revenue": total_demand * annualized_factor,
        "revenue": total_revenue * annualized_factor,
        "cogs": total_cogs * annualized_factor,
        "gross_profit": total_gp * annualized_factor,
        "operating_expenses": total_exp * annualized_factor,
        "taxes": total_taxes * annualized_factor,
        "net_profit": total_np * annualized_factor,
        "gross_margin": safe_divide(total_gp, total_revenue) * 100,
        "net_margin": safe_divide(total_np, total_revenue) * 100,
        "ending_cash": final_month["ending_cash"],
        "ending_accounts_receivable": final_month["ending_ar"],
        "ending_inventory": final_month["ending_inventory"],
        "average_accounts_receivable": avg_ar,
        "average_inventory": avg_inv,
        "minimum_inventory": min_inv,
        "minimum_inventory_coverage": min_inv_coverage,
        "maximum_stockout_exposure": max_stockout,
        "total_lost_sales": total_lost_sales,
        "inventory_turnover": inventory_turnover,
        "inventory_risk": risk_level(avg_inv_risk),
        "cash_runway_months": cash_runway_str,
        "average_risk_score": avg_risk,
        "final_risk_level": final_month["risk_level"]
    }

def simulate(scenario: Scenario, baseline_twin: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    validate_scenario(scenario)
    twin = baseline_twin if baseline_twin else build_digital_twin()
    baseline = get_baseline_metrics(twin)
    monthly_results = simulate_monthly_states(baseline, scenario)
    projected = aggregate_results(monthly_results, scenario.months)
    
    avg_score = projected["average_risk_score"]
    
    return {
        "scenario": asdict(scenario),
        "baseline": baseline,
        "projected": projected,
        "monthly": monthly_results,
        "risk": {
            "overall": risk_level(avg_score),
            "inventory": projected["inventory_risk"]
        }
    }

def format_inr(v: float) -> str:
    return f"₹{v:,.2f}"

def print_simulation_report(name: str, res: Dict[str, Any]):
    proj = res["projected"]
    print(f"\n{name}")
    print("-" * 60)
    print(f"Projected Revenue         : {format_inr(proj['revenue'])}")
    print(f"Projected COGS            : {format_inr(proj['cogs'])}")
    print(f"Ending Cash               : {format_inr(proj['ending_cash'])}")
    print(f"Ending AR                 : {format_inr(proj['ending_accounts_receivable'])}")
    print(f"Ending Inventory          : {format_inr(proj['ending_inventory'])}")
    print(f"Target Inventory (M12)    : {format_inr(res['monthly'][-1]['target_inventory'])}")
    print(f"Minimum Inventory         : {format_inr(proj['minimum_inventory'])}")
    print(f"Minimum Inventory Coverage: {proj['minimum_inventory_coverage']:.2f} months")
    print(f"Total Lost Sales          : {format_inr(proj['total_lost_sales'])}")
    print(f"Inventory Risk Score      : {res['monthly'][-1]['inventory_risk_score']:.1f}")
    print(f"Inventory Risk            : {proj['inventory_risk']}")
    print(f"Overall Risk Score        : {proj['average_risk_score']:.1f}")
    print(f"Overall Risk Level        : {res['risk']['overall']}")
    
    m1 = res["monthly"][0]
    m6 = res["monthly"][5] if len(res["monthly"]) > 5 else None
    m12 = res["monthly"][-1]
    
    print("\nInventory States (M1, M6, M12):")
    for m in [m1, m6, m12]:
        if m:
            print(f"Month {m['month']}:")
            print(f"  opening_inventory: {m['opening_inventory']:,.2f}")
            print(f"  inventory_purchases: {m['inventory_purchases']:,.2f}")
            print(f"  cogs_consumption: {m['cogs_consumption']:,.2f}")
            print(f"  ending_inventory: {m['ending_inventory']:,.2f}")
            print(f"  target_inventory: {m['target_inventory']:,.2f}")
            print(f"  inventory_gap: {m['inventory_gap']:,.2f}")
            print(f"  inventory_coverage_months: {m['inventory_coverage_months']:.2f}")
            print(f"  inventory_risk_score: {m['inventory_risk_score']}")
    print("=" * 60)


if __name__ == "__main__":
    baseline = build_digital_twin()
    
    res_a = simulate(Scenario(sales_growth=0.0, inventory_change=0.0), baseline)
    res_b = simulate(Scenario(sales_growth=0.30, inventory_change=0.0), baseline)
    res_c = simulate(Scenario(sales_growth=0.30, inventory_change=-0.20), baseline)
    res_d = simulate(Scenario(sales_growth=0.30, inventory_change=0.20), baseline)
    
    print_simulation_report("TEST A: Baseline", res_a)
    print_simulation_report("TEST B: +30% Sales, 0% Inventory", res_b)
    print_simulation_report("TEST C: +30% Sales, -20% Inventory", res_c)
    print_simulation_report("TEST D: +30% Sales, +20% Inventory", res_d)
    
    # Assertions
    assert res_c["projected"]["minimum_inventory"] <= res_b["projected"]["minimum_inventory"], "Test C minimum inventory is not <= Test B"
    assert res_d["projected"]["minimum_inventory"] >= res_b["projected"]["minimum_inventory"], "Test D minimum inventory is not >= Test B"
    
    print("\nAll Reconciliations and Cross-Scenario Assertions Passed!")