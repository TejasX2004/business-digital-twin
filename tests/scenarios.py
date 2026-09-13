"""
Canonical Test Scenarios for Business Digital Twin.
Defines 20 representative business scenarios covering baseline,
single-parameter changes, multi-parameter changes, and extreme boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScenarioDefinition:
    id: int
    name: str
    query: str
    expected_params: Dict[str, Any]
    description: str
    tolerance: float = 0.05
    inject_fault_for_red_team: bool = False
    injected_fault: Optional[Dict[str, Any]] = None


SCENARIOS: List[ScenarioDefinition] = [
    # 1. Baseline / no changes
    ScenarioDefinition(
        id=1,
        name="Baseline / No Changes",
        query="Simulate baseline performance for 12 months with no changes.",
        expected_params={
            "sales_growth": 0.0,
            "price_change": 0.0,
            "cost_change": 0.0,
            "expense_change": 0.0,
            "payment_delay_days": 0.0,
            "inventory_change": 0.0,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="Standard 12-month projection with default baseline assumptions.",
    ),

    # 2. Sales +30%
    ScenarioDefinition(
        id=2,
        name="Sales +30%",
        query="What if sales increase by 30%?",
        expected_params={
            "sales_growth": 0.30,
            "price_change": 0.0,
            "cost_change": 0.0,
            "expense_change": 0.0,
            "payment_delay_days": 0.0,
            "inventory_change": 0.0,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="Moderate demand expansion without pricing, cost, or working capital changes.",
    ),

    # 3. Sales -20%
    ScenarioDefinition(
        id=3,
        name="Sales -20%",
        query="What if sales decrease by 20%?",
        expected_params={
            "sales_growth": -0.20,
            "price_change": 0.0,
            "cost_change": 0.0,
            "expense_change": 0.0,
            "payment_delay_days": 0.0,
            "inventory_change": 0.0,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="Demand contraction without cost or price adjustments.",
    ),

    # 4. Sales +70%
    ScenarioDefinition(
        id=4,
        name="Sales +70%",
        query="What if sales grow by 70%?",
        expected_params={
            "sales_growth": 0.70,
            "price_change": 0.0,
            "cost_change": 0.0,
            "expense_change": 0.0,
            "payment_delay_days": 0.0,
            "inventory_change": 0.0,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="High growth scenario testing working capital and inventory scaling.",
    ),

    # 5. Price +10%
    ScenarioDefinition(
        id=5,
        name="Price +10%",
        query="What if we increase prices by 10%?",
        expected_params={
            "sales_growth": 0.0,
            "price_change": 0.10,
            "cost_change": 0.0,
            "expense_change": 0.0,
            "payment_delay_days": 0.0,
            "inventory_change": 0.0,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="Pure price increase expanding gross margins with constant volume.",
    ),

    # 6. Price -10%
    ScenarioDefinition(
        id=6,
        name="Price -10%",
        query="What if we reduce prices by 10%?",
        expected_params={
            "sales_growth": 0.0,
            "price_change": -0.10,
            "cost_change": 0.0,
            "expense_change": 0.0,
            "payment_delay_days": 0.0,
            "inventory_change": 0.0,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="Discounting scenario compressing margins without volume lift.",
    ),

    # 7. Cost +10%
    ScenarioDefinition(
        id=7,
        name="Cost +10%",
        query="What if COGS costs increase by 10%?",
        expected_params={
            "sales_growth": 0.0,
            "price_change": 0.0,
            "cost_change": 0.10,
            "expense_change": 0.0,
            "payment_delay_days": 0.0,
            "inventory_change": 0.0,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="Direct cost inflation affecting gross margins.",
    ),

    # 8. Expense +10%
    ScenarioDefinition(
        id=8,
        name="Expense +10%",
        query="What if operating expenses increase by 10%?",
        expected_params={
            "sales_growth": 0.0,
            "price_change": 0.0,
            "cost_change": 0.0,
            "expense_change": 0.10,
            "payment_delay_days": 0.0,
            "inventory_change": 0.0,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="Operational overhead inflation compressing net margins.",
    ),

    # 9. Sales +30% + payment delay +15 days
    ScenarioDefinition(
        id=9,
        name="Sales +30% & Payment Delay 15d",
        query="What if sales grow by 30% and customers take 15 extra days to pay?",
        expected_params={
            "sales_growth": 0.30,
            "price_change": 0.0,
            "cost_change": 0.0,
            "expense_change": 0.0,
            "payment_delay_days": 15.0,
            "inventory_change": 0.0,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="Growth accompanied by customer collection lag, straining cash.",
    ),

    # 10. Sales +30% + inventory -20%
    ScenarioDefinition(
        id=10,
        name="Sales +30% & Inventory -20%",
        query="What if sales grow by 30% and we reduce inventory buffer by 20%?",
        expected_params={
            "sales_growth": 0.30,
            "price_change": 0.0,
            "cost_change": 0.0,
            "expense_change": 0.0,
            "payment_delay_days": 0.0,
            "inventory_change": -0.20,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="Aggressive lean inventory policy under rising sales volume.",
    ),

    # 11. Sales +30% + inventory +20%
    ScenarioDefinition(
        id=11,
        name="Sales +30% & Inventory +20%",
        query="What if sales increase by 30% and we increase inventory buffer by 20%?",
        expected_params={
            "sales_growth": 0.30,
            "price_change": 0.0,
            "cost_change": 0.0,
            "expense_change": 0.0,
            "payment_delay_days": 0.0,
            "inventory_change": 0.20,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="High safety stock approach during growth to safeguard against stockouts.",
    ),

    # 12. Sales -20% + payment delay +30 days
    ScenarioDefinition(
        id=12,
        name="Sales -20% & Payment Delay 30d",
        query="What if sales drop by 20% and customers take 30 extra days to pay?",
        expected_params={
            "sales_growth": -0.20,
            "price_change": 0.0,
            "cost_change": 0.0,
            "expense_change": 0.0,
            "payment_delay_days": 30.0,
            "inventory_change": 0.0,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="Stagflation/downturn shock: falling top-line paired with slower collections.",
    ),

    # 13. Cost +20% + expense +10%
    ScenarioDefinition(
        id=13,
        name="Cost +20% & Expense +10%",
        query="What if costs increase by 20% and operating expenses increase by 10%?",
        expected_params={
            "sales_growth": 0.0,
            "price_change": 0.0,
            "cost_change": 0.20,
            "expense_change": 0.10,
            "payment_delay_days": 0.0,
            "inventory_change": 0.0,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="Combined inflationary squeeze across both COGS and operating expenses.",
    ),

    # 14. Sales +50% + payment delay +30 days + inventory -20%
    ScenarioDefinition(
        id=14,
        name="Sales +50%, Delay 30d, Inv -20%",
        query="What if sales grow by 50%, customers pay 30 days late, and inventory is reduced by 20%?",
        expected_params={
            "sales_growth": 0.50,
            "price_change": 0.0,
            "cost_change": 0.0,
            "expense_change": 0.0,
            "payment_delay_days": 30.0,
            "inventory_change": -0.20,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="High working capital friction with tight inventory during rapid expansion.",
    ),

    # 15. Supplier payment delay +30 days
    ScenarioDefinition(
        id=15,
        name="Supplier Delay 30d",
        query="What if we delay payments to suppliers by 30 days?",
        expected_params={
            "sales_growth": 0.0,
            "price_change": 0.0,
            "cost_change": 0.0,
            "expense_change": 0.0,
            "payment_delay_days": 0.0,
            "inventory_change": 0.0,
            "supplier_payment_delay_days": 30.0,
            "months": 12,
        },
        description="Working capital optimization through supplier credit terms expansion.",
    ),

    # 16. Sales +30% + price +10%
    ScenarioDefinition(
        id=16,
        name="Sales +30% & Price +10%",
        query="What if sales grow by 30% and we increase prices by 10%?",
        expected_params={
            "sales_growth": 0.30,
            "price_change": 0.10,
            "cost_change": 0.0,
            "expense_change": 0.0,
            "payment_delay_days": 0.0,
            "inventory_change": 0.0,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="Strong market power scenario with volume and price growth compounding.",
    ),

    # 17. Sales +30% + cost +10%
    ScenarioDefinition(
        id=17,
        name="Sales +30% & Cost +10%",
        query="What if sales grow by 30% but product costs increase by 10%?",
        expected_params={
            "sales_growth": 0.30,
            "price_change": 0.0,
            "cost_change": 0.10,
            "expense_change": 0.0,
            "payment_delay_days": 0.0,
            "inventory_change": 0.0,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="Revenue growth partially offset by input cost inflation.",
    ),

    # 18. Sales -20% + cost +20%
    ScenarioDefinition(
        id=18,
        name="Sales -20% & Cost +20%",
        query="What if sales decrease by 20% and costs increase by 20%?",
        expected_params={
            "sales_growth": -0.20,
            "price_change": 0.0,
            "cost_change": 0.20,
            "expense_change": 0.0,
            "payment_delay_days": 0.0,
            "inventory_change": 0.0,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="Severe profit margin erosion under falling revenue and escalating unit costs.",
    ),

    # 19. Multiple simultaneous changes
    ScenarioDefinition(
        id=19,
        name="Multiple Simultaneous Changes",
        query="What if sales grow by 20%, prices increase by 5%, costs increase by 10%, operating expenses decrease by 5%, and customer payment delay is 10 days?",
        expected_params={
            "sales_growth": 0.20,
            "price_change": 0.05,
            "cost_change": 0.10,
            "expense_change": -0.05,
            "payment_delay_days": 10.0,
            "inventory_change": 0.0,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="Comprehensive business adjustment touching volume, price, costs, opex, and DSO.",
    ),

    # 20. Extreme but valid scenario
    ScenarioDefinition(
        id=20,
        name="Extreme Valid Scenario (+200% sales, 60d delay)",
        query="What if sales surge by 200% and customer payment delay increases by 60 days?",
        expected_params={
            "sales_growth": 2.0,
            "price_change": 0.0,
            "cost_change": 0.0,
            "expense_change": 0.0,
            "payment_delay_days": 60.0,
            "inventory_change": 0.0,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        },
        description="Extreme stress test pushing liquidity, receivables, and working capital buffers.",
    ),
]


def get_scenario_by_id(scenario_id: int) -> Optional[ScenarioDefinition]:
    for s in SCENARIOS:
        if s.id == scenario_id:
            return s
    return None
