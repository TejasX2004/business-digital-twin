# Business Digital Twin — Automated Test Suite

## Overview
This directory contains the automated end-to-end test suite for the **Business Digital Twin**, validating the complete 5-stage decision intelligence pipeline:

$$\text{Planner} \longrightarrow \text{Simulator} \longrightarrow \text{Analysis} \longrightarrow \text{Red-Team} \longrightarrow \text{Decision}$$

The test suite ensures strict parameter fidelity, mathematical reconciliation, semantic consistency, hallucination-free reasoning, and audit robustness.

---

## 1. Directory Structure

```
tests/
├── __init__.py           # Package initializer
├── scenarios.py          # 20 canonical business scenarios with expected parameters
├── test_pipeline.py      # End-to-end pipeline verifier & test runner
└── README.md             # Documentation and usage guide
```

---

## 2. The 20 Representative Test Scenarios

The suite defines 20 business scenarios in `tests/scenarios.py`:

| ID | Scenario | Natural Language Query | Core Parameters Tested |
|---|---|---|---|
| **1** | Baseline / No Changes | *"Simulate baseline performance for 12 months with no changes."* | `sales_growth=0.0`, all defaults |
| **2** | Sales +30% | *"What if sales increase by 30%?"* | `sales_growth = 0.30` |
| **3** | Sales -20% | *"What if sales decrease by 20%?"* | `sales_growth = -0.20` |
| **4** | Sales +70% | *"What if sales grow by 70%?"* | `sales_growth = 0.70` (tests 70% vs 0.7% scaling) |
| **5** | Price +10% | *"What if we increase prices by 10%?"* | `price_change = 0.10` |
| **6** | Price -10% | *"What if we reduce prices by 10%?"* | `price_change = -0.10` |
| **7** | Cost +10% | *"What if COGS costs increase by 10%?"* | `cost_change = 0.10` |
| **8** | Expense +10% | *"What if operating expenses increase by 10%?"* | `expense_change = 0.10` |
| **9** | Sales +30% & Delay 15d | *"What if sales grow by 30% and customers take 15 extra days to pay?"* | `sales_growth = 0.30`, `payment_delay_days = 15` |
| **10** | Sales +30% & Inv -20% | *"What if sales grow by 30% and we reduce inventory buffer by 20%?"* | `sales_growth = 0.30`, `inventory_change = -0.20` |
| **11** | Sales +30% & Inv +20% | *"What if sales increase by 30% and we increase inventory buffer by 20%?"* | `sales_growth = 0.30`, `inventory_change = 0.20` |
| **12** | Sales -20% & Delay 30d | *"What if sales drop by 20% and customers take 30 extra days to pay?"* | `sales_growth = -0.20`, `payment_delay_days = 30` |
| **13** | Cost +20% & Exp +10% | *"What if costs increase by 20% and operating expenses increase by 10%?"* | `cost_change = 0.20`, `expense_change = 0.10` |
| **14** | Sales +50%, Delay 30d, Inv -20% | *"What if sales grow by 50%, customers pay 30 days late, and inventory is reduced by 20%?"* | `sales_growth = 0.50`, `payment_delay = 30`, `inv = -0.20` |
| **15** | Supplier Delay 30d | *"What if we delay payments to suppliers by 30 days?"* | `supplier_payment_delay_days = 30` |
| **16** | Sales +30% & Price +10% | *"What if sales grow by 30% and we increase prices by 10%?"* | `sales_growth = 0.30`, `price_change = 0.10` |
| **17** | Sales +30% & Cost +10% | *"What if sales grow by 30% but product costs increase by 10%?"* | `sales_growth = 0.30`, `cost_change = 0.10` |
| **18** | Sales -20% & Cost +20% | *"What if sales decrease by 20% and costs increase by 20%?"* | `sales_growth = -0.20`, `cost_change = 0.20` |
| **19** | Multiple Simultaneous Changes | *"What if sales grow by 20%, prices increase by 5%, costs increase by 10%, operating expenses decrease by 5%, and customer payment delay is 10 days?"* | 5 simultaneous parameters |
| **20** | Extreme Valid Scenario | *"What if sales surge by 200% and customer payment delay increases by 60 days?"* | `sales_growth = 2.0`, `payment_delay_days = 60` |

---

## 3. Stage Verification Rules

### A. Planner Agent
- **Schema**: Valid dictionary containing all 8 parameters.
- **Safety Bounds**:
  - `sales_growth`, `price_change`, `cost_change`, `inventory_change`, `expense_change` $\in [-0.90, 5.0]$
  - `payment_delay_days` $\ge 0$, `supplier_payment_delay_days` $\ge 0$
  - $1 \le \text{months} \le 60$
- **Semantic Extraction**: Parameters match expected values within tolerance ($\pm 0.05$).

### B. Simulator
- **Mathematical Reconciliation**: $\text{Gross Profit} = \text{Revenue} - \text{COGS}$.
- **Non-negativity**: Revenue, COGS, ending inventory, and lost sales cannot be negative.
- **No NaN / Infinity**: All numeric fields must be finite numbers.
- **Risk Invariants**: `overall` and `inventory` risk levels must be $\in \{\text{Low}, \text{Medium}, \text{High}, \text{Critical}\}$.

### C. Analysis Agent
- **JSON Schema**: All 11 required fields present.
- **Direction Consistency**: If revenue/cash moved in a specific direction, the analysis must never state the opposite direction.
- **Grounding & Forbidden Language**:
  - Currency strictly INR (no `$`, `USD`, `EUR`, `GBP`).
  - No `"sustainable revenue expansion"` or `"sustainable revenue"`.
  - No stockout risk claims if `lost_sales == 0` and `stockout_exposure == 0`.
  - No percentage scaling errors (e.g. 70% growth must never be reported as `0.7%`).

### D. Red-Team Agent
- **Verdict Validity**: `overall_verdict` $\in \{\text{PASS}, \text{PASS_WITH_WARNINGS}, \text{FAIL}\}$.
- **No False Positives**: Valid simulator facts (e.g. `"Not reached in projection"`, simulator risk labels) must not be flagged as issues.
- **Fault Injection Test**: Includes dedicated verification that intentionally injected contradictions (e.g. stating revenue dropped when it increased) are correctly detected.

### E. Decision Agent
- **Deterministic Rules**:
  - If Red-Team verdict is `FAIL` $\implies$ Decision cannot be `PROCEED`.
  - If overall risk is `Critical` $\implies$ Decision must be `DO_NOT_PROCEED`.
- **Actionable Grounding**:
  - No invented cadences (`daily`, `within 30 days`, `every month`).
  - No invented minimum thresholds.
  - Consistent INR currency.

### F. Cross-Pipeline Integrity
- $\text{Planner scenario} \equiv \text{Simulator scenario}$.
- $\text{Simulator facts} \equiv \text{Analysis facts}$.
- Decision rationale cannot contradict simulator facts.

---

## 4. Running the Tests

### Fast Deterministic Mode (Tests all 20 scenarios without LLM calls)
Runs parameter verification, simulator execution, mathematical reconciliation, and financial invariants for all 20 scenarios:
```bash
python -m tests.test_pipeline --deterministic
```

### Run a Single Scenario (Live LLM)
Runs the full 5-stage pipeline for a specific scenario (e.g. Scenario 4: Sales +70%):
```bash
python -m tests.test_pipeline --scenario 4
```

### Run Representative Subset
```bash
python -m tests.test_pipeline --limit 3
```

### Run the Complete 20-Scenario End-to-End Suite
Runs all 20 scenarios through all stages with live LLM calls:
```bash
python -m tests.test_pipeline --all
```

### Run via Standard `unittest`
```bash
python -m unittest discover tests
```
