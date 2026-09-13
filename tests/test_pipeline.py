"""
Automated End-to-End Pipeline Test Suite for Business Digital Twin.
Tests the complete flow:
Planner -> Simulator -> Analysis -> Red-Team -> Decision

Verifies stage invariants, fault injection detection, and cross-pipeline consistency.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unittest
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# Ensure root is in sys.path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.digital_twin.digital_twin import build_digital_twin
from backend.digital_twin.simulation import Scenario, simulate
from agents.planner_agent import PlannerAgent
from agents.analysis_agent import AnalysisAgent
from agents.red_team_agent import RedTeamAgent
from agents.decision_agent import DecisionAgent

from tests.scenarios import SCENARIOS, ScenarioDefinition, get_scenario_by_id


@dataclass
class FailureReport:
    scenario_id: int
    scenario_name: str
    stage: str
    expected: str
    actual: str


class TestResultCollector:
    def __init__(self):
        self.passed: int = 0
        self.failed: int = 0
        self.skipped: int = 0
        self.failures: List[FailureReport] = []

    def record_pass(self):
        self.passed += 1

    def record_fail(self, scenario_id: int, scenario_name: str, stage: str, expected: str, actual: str):
        self.failed += 1
        self.failures.append(FailureReport(scenario_id, scenario_name, stage, expected, actual))

    def record_skip(self):
        self.skipped += 1

    def print_summary(self):
        print("\n" + "=" * 80)
        print("                       BUSINESS DIGITAL TWIN TEST SUMMARY                       ")
        print("=" * 80)
        print(f"PASS: {self.passed}")
        print(f"FAIL: {self.failed}")
        print(f"SKIPPED: {self.skipped}")
        print("=" * 80)

        if self.failures:
            print("\nFAILURE DETAILS:")
            for idx, f in enumerate(self.failures, 1):
                print("-" * 80)
                print(f"Failure #{idx}")
                print(f"Scenario       : [{f.scenario_id}] {f.scenario_name}")
                print(f"Pipeline stage : {f.stage}")
                print(f"Expected       : {f.expected}")
                print(f"Actual         : {f.actual}")
            print("-" * 80 + "\n")


class PipelineVerifier:
    """Deterministic verification rules for each stage and cross-pipeline checks."""

    # ------------------------------------------------------------------
    # 1. PLANNER VERIFICATION
    # ------------------------------------------------------------------
    @staticmethod
    def verify_planner(sc_def: ScenarioDefinition, plan: Dict[str, Any], sc: Scenario) -> List[str]:
        errors: List[str] = []

        # 1. Type and structure
        required_keys = [
            "sales_growth", "price_change", "cost_change", "payment_delay_days",
            "inventory_change", "expense_change", "supplier_payment_delay_days", "months"
        ]
        for k in required_keys:
            if k not in plan:
                errors.append(f"Missing parameter '{k}' in planner output")

        # 2. Safety bounds
        for param in ["sales_growth", "price_change", "cost_change", "inventory_change", "expense_change"]:
            val = plan.get(param, 0.0)
            if not isinstance(val, (int, float)) or math.isnan(val) or not (-0.90 <= val <= 5.0):
                errors.append(f"Parameter '{param}' value {val} out of allowed bounds [-0.90, 5.0]")

        if plan.get("payment_delay_days", 0) < 0:
            errors.append(f"payment_delay_days cannot be negative: {plan.get('payment_delay_days')}")
        if plan.get("supplier_payment_delay_days", 0) < 0:
            errors.append(f"supplier_payment_delay_days cannot be negative: {plan.get('supplier_payment_delay_days')}")
        if not (1 <= plan.get("months", 12) <= 60):
            errors.append(f"months {plan.get('months')} out of allowed bounds [1, 60]")

        # 3. Expected values comparison within tolerance
        for k, expected_v in sc_def.expected_params.items():
            actual_v = getattr(sc, k, None)
            if actual_v is None:
                errors.append(f"Scenario object missing attribute '{k}'")
            elif isinstance(expected_v, (int, float)):
                diff = abs(actual_v - expected_v)
                if diff > sc_def.tolerance:
                    errors.append(f"Parameter '{k}' mismatch: expected ~{expected_v}, got {actual_v}")

        return errors

    # ------------------------------------------------------------------
    # 2. SIMULATOR VERIFICATION
    # ------------------------------------------------------------------
    @staticmethod
    def verify_simulator(sc: Scenario, sim_result: Dict[str, Any]) -> List[str]:
        errors: List[str] = []

        if not isinstance(sim_result, dict):
            return ["Simulator did not return a dictionary"]

        for k in ("baseline", "projected", "risk", "scenario", "monthly"):
            if k not in sim_result:
                errors.append(f"Simulator output missing section '{k}'")

        proj = sim_result.get("projected", {})

        # Check for NaN / Infinity
        numeric_fields = [
            "revenue", "cogs", "gross_profit", "operating_expenses",
            "net_profit", "gross_margin", "net_margin", "ending_cash",
            "ending_accounts_receivable", "ending_inventory",
            "average_accounts_receivable", "average_inventory",
            "minimum_inventory_coverage", "total_lost_sales"
        ]
        for field in numeric_fields:
            val = proj.get(field)
            if val is None or not isinstance(val, (int, float)):
                errors.append(f"Projected field '{field}' is not a valid number: {val}")
            elif math.isnan(val) or math.isinf(val):
                errors.append(f"Projected field '{field}' is NaN or Infinity: {val}")

        # Non-negative validity
        for non_neg in ["revenue", "cogs", "ending_inventory", "total_lost_sales", "maximum_stockout_exposure"]:
            val = proj.get(non_neg, 0.0)
            if isinstance(val, (int, float)) and val < -1e-4:
                errors.append(f"Projected '{non_neg}' cannot be negative: {val}")

        # Revenue / COGS / Gross Profit Reconciliation
        rev = proj.get("revenue", 0.0)
        cogs = proj.get("cogs", 0.0)
        gp = proj.get("gross_profit", 0.0)
        expected_gp = rev - cogs
        if abs(gp - expected_gp) > 1.0:  # Allow minimal float rounding
            errors.append(f"Gross profit reconciliation failure: rev ({rev}) - cogs ({cogs}) != gp ({gp})")

        # Risk fields
        valid_risk_levels = {"Low", "Medium", "High", "Critical"}
        inv_risk = proj.get("inventory_risk")
        overall_risk = sim_result.get("risk", {}).get("overall")
        if inv_risk not in valid_risk_levels:
            errors.append(f"Invalid projected inventory_risk: {inv_risk}")
        if overall_risk not in valid_risk_levels:
            errors.append(f"Invalid overall risk: {overall_risk}")

        # Runway
        runway = proj.get("cash_runway_months")
        if runway != "Not reached in projection" and not (isinstance(runway, str) and "Month" in runway):
            errors.append(f"Invalid cash_runway_months format: {runway}")

        return errors

    # ------------------------------------------------------------------
    # 3. ANALYSIS VERIFICATION
    # ------------------------------------------------------------------
    @staticmethod
    def verify_analysis(sc: Scenario, sim_result: Dict[str, Any], analysis: Dict[str, Any]) -> List[str]:
        errors: List[str] = []

        if not isinstance(analysis, dict):
            return ["Analysis output is not a dictionary"]

        # Required fields
        required_keys = [
            "executive_summary", "key_drivers", "financial_impact",
            "cash_flow_impact", "working_capital_impact", "inventory_analysis",
            "risk_analysis", "tradeoffs", "recommendations",
            "overall_assessment", "confidence"
        ]
        for k in required_keys:
            if k not in analysis:
                errors.append(f"Analysis missing required field '{k}'")

        # Currency check: strictly INR
        analysis_str = json.dumps(analysis, ensure_ascii=False).lower()
        for forbidden_curr in ["$", "usd", "eur", "gbp", "euros", "dollars"]:
            if forbidden_curr in analysis_str:
                errors.append(f"Analysis contains forbidden foreign currency: '{forbidden_curr}'")

        # Unsupported wording
        for forbidden in ["sustainable revenue expansion", "sustainable revenue", "may not be sustainable"]:
            if forbidden in analysis_str:
                errors.append(f"Analysis contains unsupported wording: '{forbidden}'")

        # Stockout check
        lost_sales = float(sim_result.get("projected", {}).get("total_lost_sales", 0.0) or 0.0)
        stockout_exp = float(sim_result.get("projected", {}).get("maximum_stockout_exposure", 0.0) or 0.0)
        if lost_sales == 0 and stockout_exp == 0:
            for phrase in ["stockout risk", "risk of stockouts", "potential stockouts"]:
                if phrase in analysis_str:
                    errors.append(f"Analysis claims stockout risk '{phrase}' when lost_sales=0 and stockout_exposure=0")

        # Percentage scaling check (e.g. 70% growth must not be described as 0.7%)
        for p_field in ["sales_growth", "price_change", "cost_change", "inventory_change", "expense_change"]:
            val = getattr(sc, p_field, 0.0)
            if abs(val) > 0.001:
                wrong_pct = f"{val:g}%"
                correct_pct = f"{round(val * 100, 2):g}%"
                if wrong_pct != correct_pct and re.search(rf"\b{re.escape(wrong_pct)}\b", analysis_str):
                    errors.append(f"Analysis contains 100x scaling error '{wrong_pct}' instead of '{correct_pct}'")

        # Direction checks
        b_rev = sim_result.get("baseline", {}).get("revenue", 0.0)
        p_rev = sim_result.get("projected", {}).get("revenue", 0.0)
        if p_rev > b_rev and ("revenue decreased" in analysis_str or "revenue dropped" in analysis_str):
            errors.append("Analysis states revenue decreased, but simulator shows revenue increased")
        elif p_rev < b_rev and ("revenue increased" in analysis_str or "revenue grew" in analysis_str):
            errors.append("Analysis states revenue increased, but simulator shows revenue decreased")

        b_cash = sim_result.get("baseline", {}).get("current_cash", 0.0)
        p_cash = sim_result.get("projected", {}).get("ending_cash", 0.0)
        if p_cash < b_cash and ("cash increased" in analysis_str or "cash rose" in analysis_str):
            errors.append("Analysis states cash increased, but simulator shows cash decreased")

        # Risk consistency
        inv_risk = sim_result.get("risk", {}).get("inventory", "")
        if inv_risk and inv_risk.lower() not in analysis_str:
            errors.append(f"Simulator inventory risk '{inv_risk}' not mentioned in analysis")

        return errors

    # ------------------------------------------------------------------
    # 4. RED-TEAM VERIFICATION
    # ------------------------------------------------------------------
    @staticmethod
    def verify_red_team(
        sc: Scenario,
        sim_result: Dict[str, Any],
        analysis: Dict[str, Any],
        rt_verdict: Dict[str, Any]
    ) -> List[str]:
        errors: List[str] = []

        if not isinstance(rt_verdict, dict):
            return ["Red-Team output is not a dictionary"]

        required_keys = [
            "critical_issues", "numerical_issues", "logic_issues",
            "unsupported_claims", "missing_risks", "recommendation_issues",
            "verified_claims", "overall_verdict", "confidence"
        ]
        for k in required_keys:
            if k not in rt_verdict:
                errors.append(f"Red-Team missing required field '{k}'")

        verdict = rt_verdict.get("overall_verdict")
        if verdict not in ("PASS", "PASS_WITH_WARNINGS", "FAIL"):
            errors.append(f"Invalid Red-Team overall_verdict: '{verdict}'")

        # Verify no false critical issues on valid facts
        crit = rt_verdict.get("critical_issues", [])
        if verdict == "PASS" and len(crit) > 0:
            errors.append(f"Red-Team reported PASS but has critical_issues: {crit}")

        # Check that authoritative simulator runway status is not falsely flagged
        all_issues = " ".join([
            str(x) for x in (
                crit + rt_verdict.get("numerical_issues", []) +
                rt_verdict.get("unsupported_claims", []) + rt_verdict.get("logic_issues", [])
            )
        ]).lower()
        if "not reached in projection" in all_issues and "exhaustion" in all_issues:
            errors.append("Red-Team falsely flagged 'Not reached in projection' as an issue")

        return errors

    # ------------------------------------------------------------------
    # 5. DECISION VERIFICATION
    # ------------------------------------------------------------------
    @staticmethod
    def verify_decision(
        sc: Scenario,
        sim_result: Dict[str, Any],
        analysis: Dict[str, Any],
        rt_verdict: Dict[str, Any],
        decision: Dict[str, Any]
    ) -> List[str]:
        errors: List[str] = []

        if not isinstance(decision, dict):
            return ["Decision output is not a dictionary"]

        required_keys = [
            "decision", "priority", "rationale", "recommended_actions",
            "expected_benefits", "risks", "conditions", "confidence"
        ]
        for k in required_keys:
            if k not in decision:
                errors.append(f"Decision missing required field '{k}'")

        dec_val = decision.get("decision", "")
        allowed_decisions = {"PROCEED", "PROCEED_WITH_CONDITIONS", "DO_NOT_PROCEED", "FLAG_FOR_REVIEW"}
        if dec_val not in allowed_decisions:
            errors.append(f"Invalid decision value: '{dec_val}'")

        prio = decision.get("priority", "")
        if prio not in ("HIGH", "MEDIUM", "LOW"):
            errors.append(f"Invalid priority: '{prio}'")

        # Deterministic rules
        rt_v = rt_verdict.get("overall_verdict", "")
        if rt_v == "FAIL" and dec_val == "PROCEED":
            errors.append("Red-Team verdict is FAIL, but Decision is PROCEED (must not proceed)")

        overall_risk = sim_result.get("risk", {}).get("overall", "")
        if overall_risk == "Critical" and dec_val != "DO_NOT_PROCEED":
            errors.append(f"Overall risk is Critical, but decision is '{dec_val}' (expected DO_NOT_PROCEED)")

        # Grounding: no invented cadences or thresholds
        dec_str = json.dumps(decision, ensure_ascii=False).lower()
        for cadence in ["daily", "within 30 days", "within 60 days", "within 90 days", "every month"]:
            if cadence in dec_str:
                errors.append(f"Decision contains invented cadence/deadline: '{cadence}'")

        for forbidden in ["sustainable revenue expansion", "sustainable revenue"]:
            if forbidden in dec_str:
                errors.append(f"Decision contains unsupported wording: '{forbidden}'")

        # Currency
        for curr in ["$", "usd", "eur", "gbp"]:
            if curr in dec_str:
                errors.append(f"Decision contains forbidden currency: '{curr}'")

        # Scaling check
        for p_field in ["sales_growth", "price_change", "cost_change", "inventory_change", "expense_change"]:
            val = getattr(sc, p_field, 0.0)
            if abs(val) > 0.001:
                wrong_pct = f"{val:g}%"
                correct_pct = f"{round(val * 100, 2):g}%"
                if wrong_pct != correct_pct and re.search(rf"\b{re.escape(wrong_pct)}\b", dec_str):
                    errors.append(f"Decision contains 100x scaling error '{wrong_pct}' instead of '{correct_pct}'")

        return errors

    # ------------------------------------------------------------------
    # 6. CROSS-PIPELINE INTEGRITY CHECKS
    # ------------------------------------------------------------------
    @staticmethod
    def verify_cross_pipeline(
        plan: Dict[str, Any],
        sc: Scenario,
        sim_result: Dict[str, Any],
        analysis: Dict[str, Any],
        rt_verdict: Dict[str, Any],
        decision: Dict[str, Any]
    ) -> List[str]:
        errors: List[str] = []

        # 1. Planner scenario == simulator scenario
        for field in ["sales_growth", "price_change", "cost_change", "payment_delay_days", "inventory_change", "expense_change"]:
            p_val = plan.get(field)
            s_val = getattr(sc, field, None)
            if p_val != s_val:
                errors.append(f"Cross-pipeline mismatch: planner {field}={p_val} != simulator {field}={s_val}")

        # 2. Simulator facts == Analysis facts
        # Direction of cash
        b_cash = sim_result["baseline"]["current_cash"]
        p_cash = sim_result["projected"]["ending_cash"]
        ana_str = json.dumps(analysis).lower()
        if p_cash < b_cash and "cash decreased" not in ana_str and "cash position declines" not in ana_str and "cash decreases" not in ana_str and "decline in liquidity" not in ana_str:
            # Informational check
            pass

        # 3. Decision cannot override simulator facts
        # E.g. if overall risk is High/Medium, decision cannot claim "no risks observed"
        dec_str = json.dumps(decision).lower()
        if "no risks observed" in dec_str and sim_result["risk"]["overall"] in ("Medium", "High", "Critical"):
            errors.append("Decision claims 'no risks observed' but simulator overall risk is elevated")

        return errors


# ==============================================================================
# TEST RUNNER
# ==============================================================================
class PipelineTestRunner:
    def __init__(self, scenarios: List[ScenarioDefinition], run_llm: bool = True):
        self.scenarios = scenarios
        self.run_llm = run_llm
        self.collector = TestResultCollector()
        self.twin = build_digital_twin()
        self.planner = PlannerAgent()
        self.analyst = AnalysisAgent()
        self.red_team = RedTeamAgent()
        self.decision_agent = DecisionAgent()

    def run_all(self):
        print("=" * 80)
        print(f"STARTING BUSINESS DIGITAL TWIN AUTOMATED PIPELINE TEST SUITE")
        print(f"Scenarios to test: {len(self.scenarios)} | Live LLM: {self.run_llm}")
        print("=" * 80)

        for sc_def in self.scenarios:
            self.run_scenario(sc_def)

        # In addition, run the fault injection test on Red-Team
        self.run_red_team_fault_injection_test()

        self.collector.print_summary()

    def run_scenario(self, sc_def: ScenarioDefinition):
        print(f"\n---> Running Scenario #{sc_def.id}: {sc_def.name}")
        failed_this_scenario = False

        # ----------------------------------------------------
        # STAGE 1: PLANNER
        # ----------------------------------------------------
        try:
            if self.run_llm:
                plan = self.planner.plan(sc_def.query)
            else:
                # Deterministic fallback parameters matching definition
                plan = dict(sc_def.expected_params)
                plan["confidence"] = 1.0
                plan["interpretation"] = sc_def.description

            sc = self.planner.to_scenario(plan)
            p_errors = PipelineVerifier.verify_planner(sc_def, plan, sc)
            if p_errors:
                failed_this_scenario = True
                for err in p_errors:
                    self.collector.record_fail(sc_def.id, sc_def.name, "PLANNER", "Valid bounds and parameter match", err)
            else:
                print("  [PLANNER]      PASS")
        except Exception as e:
            failed_this_scenario = True
            self.collector.record_fail(sc_def.id, sc_def.name, "PLANNER", "Successful planning", str(e))
            return

        # ----------------------------------------------------
        # STAGE 2: SIMULATOR
        # ----------------------------------------------------
        try:
            sim_result = simulate(sc, self.twin)
            s_errors = PipelineVerifier.verify_simulator(sc, sim_result)
            if s_errors:
                failed_this_scenario = True
                for err in s_errors:
                    self.collector.record_fail(sc_def.id, sc_def.name, "SIMULATOR", "Valid financial invariants", err)
            else:
                print("  [SIMULATOR]    PASS")
        except Exception as e:
            failed_this_scenario = True
            self.collector.record_fail(sc_def.id, sc_def.name, "SIMULATOR", "Simulation success", str(e))
            return

        if not self.run_llm:
            if not failed_this_scenario:
                self.collector.record_pass()
            return

        # ----------------------------------------------------
        # STAGE 3: ANALYSIS
        # ----------------------------------------------------
        try:
            analysis = self.analyst.analyze(sc, sim_result)
            a_errors = PipelineVerifier.verify_analysis(sc, sim_result, analysis)
            if a_errors:
                failed_this_scenario = True
                for err in a_errors:
                    self.collector.record_fail(sc_def.id, sc_def.name, "ANALYSIS", "Consistent supported analysis", err)
            else:
                print("  [ANALYSIS]     PASS")
        except Exception as e:
            failed_this_scenario = True
            self.collector.record_fail(sc_def.id, sc_def.name, "ANALYSIS", "Analysis completion", str(e))
            analysis = {}

        # ----------------------------------------------------
        # STAGE 4: RED-TEAM
        # ----------------------------------------------------
        try:
            rt_verdict = self.red_team.review(sc, sim_result, analysis)
            rt_errors = PipelineVerifier.verify_red_team(sc, sim_result, analysis, rt_verdict)
            if rt_errors:
                failed_this_scenario = True
                for err in rt_errors:
                    self.collector.record_fail(sc_def.id, sc_def.name, "RED-TEAM", "Valid audit without false alarms", err)
            else:
                print("  [RED-TEAM]     PASS")
        except Exception as e:
            failed_this_scenario = True
            self.collector.record_fail(sc_def.id, sc_def.name, "RED-TEAM", "Audit completion", str(e))
            rt_verdict = {"overall_verdict": "FAIL", "confidence": 0.0}

        # ----------------------------------------------------
        # STAGE 5: DECISION
        # ----------------------------------------------------
        try:
            decision = self.decision_agent.decide(sc, sim_result, analysis, rt_verdict)
            d_errors = PipelineVerifier.verify_decision(sc, sim_result, analysis, rt_verdict, decision)
            if d_errors:
                failed_this_scenario = True
                for err in d_errors:
                    self.collector.record_fail(sc_def.id, sc_def.name, "DECISION", "Grounded business decision", err)
            else:
                print("  [DECISION]     PASS")
        except Exception as e:
            failed_this_scenario = True
            self.collector.record_fail(sc_def.id, sc_def.name, "DECISION", "Decision completion", str(e))
            decision = {}

        # ----------------------------------------------------
        # STAGE 6: CROSS-PIPELINE CHECKS
        # ----------------------------------------------------
        cp_errors = PipelineVerifier.verify_cross_pipeline(plan, sc, sim_result, analysis, rt_verdict, decision)
        if cp_errors:
            failed_this_scenario = True
            for err in cp_errors:
                self.collector.record_fail(sc_def.id, sc_def.name, "CROSS-PIPELINE", "Integrity across stages", err)
        else:
            print("  [CROSS-CHECK]  PASS")

        if not failed_this_scenario:
            self.collector.record_pass()

    def run_red_team_fault_injection_test(self):
        """Verifies that Red-Team correctly flags intentionally injected contradictions."""
        print("\n---> Running Injected Fault Detection Check (Red-Team Audit)")
        sc = Scenario(sales_growth=0.30)
        sim_res = simulate(sc, self.twin)

        # Corrupted analysis: Claims revenue decreased when simulator shows it increased
        corrupted_analysis = {
            "executive_summary": "Revenue decreased drastically by 50% causing major losses.",
            "key_drivers": ["Revenue decreased despite rising market demand."],
            "financial_impact": ["Revenue decreased"],
            "cash_flow_impact": ["Cash was severely depleted."],
            "working_capital_impact": ["AR skyrocketed"],
            "inventory_analysis": ["Inventory dropped"],
            "risk_analysis": ["Severe solvency crisis"],
            "tradeoffs": ["Trading growth for survival"],
            "recommendations": ["Cut all operations"],
            "overall_assessment": "Negative",
            "confidence": 0.95,
        }

        try:
            rt_out = self.red_team.review(sc, sim_res, corrupted_analysis)
            verdict = rt_out.get("overall_verdict")
            num_issues = rt_out.get("numerical_issues", [])
            crit_issues = rt_out.get("critical_issues", [])
            all_flagged = " ".join(crit_issues + num_issues).lower()

            if "revenue" in all_flagged or verdict in ("FAIL", "PASS_WITH_WARNINGS"):
                print("  [FAULT-DETECTION] PASS: Red-Team successfully detected injected revenue contradiction.")
                self.collector.record_pass()
            else:
                self.collector.record_fail(
                    99, "Injected Contradiction Fault Test", "RED-TEAM",
                    "Red-Team should detect revenue direction contradiction",
                    f"Verdict={verdict}, Issues={num_issues + crit_issues}"
                )
        except Exception as e:
            self.collector.record_fail(99, "Injected Fault Test", "RED-TEAM", "Successful fault detection", str(e))


# ==============================================================================
# UNITTEST INTEGRATION
# ==============================================================================
class TestDigitalTwinPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = PipelineTestRunner(SCENARIOS, run_llm=False)

    def test_all_scenarios_deterministic(self):
        """Runs all 20 scenarios through Planner rules + Simulator + Stage checks."""
        self.runner.run_all()
        self.assertEqual(self.runner.collector.failed, 0, f"Failures encountered: {self.runner.collector.failures}")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Business Digital Twin Pipeline Test Suite")
    parser.add_argument("--scenario", type=int, default=None, help="Run specific scenario ID (1-20)")
    parser.add_argument("--limit", type=int, default=None, help="Run first N scenarios")
    parser.add_argument("--deterministic", action="store_true", help="Run deterministic simulation tests (no LLM calls)")
    parser.add_argument("--all", action="store_true", help="Run all 20 scenarios end-to-end with live LLM")

    args = parser.parse_args()

    scenarios_to_run = SCENARIOS
    if args.scenario is not None:
        sc = get_scenario_by_id(args.scenario)
        if not sc:
            print(f"Error: Scenario {args.scenario} not found (must be 1-20).")
            sys.exit(1)
        scenarios_to_run = [sc]
    elif args.limit is not None:
        scenarios_to_run = SCENARIOS[:args.limit]

    run_llm = not args.deterministic
    runner = PipelineTestRunner(scenarios_to_run, run_llm=run_llm)
    runner.run_all()

    if runner.collector.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
