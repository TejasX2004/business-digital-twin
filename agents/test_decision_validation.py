"""
agents/test_decision_validation.py

Unit tests for deterministic validation rules in DecisionAgent:
- allowed priority values
- valid JSON/schema
- confidence range 0-1
- forbidden currency detection
- no invented simulator values
- recommendations consistent with risk
- no contradiction with Red-Team verdict
- deterministic decision framework (FAIL -> FLAG_FOR_REVIEW, Critical -> DO_NOT_PROCEED, High/Medium -> PROCEED_WITH_CONDITIONS, Low -> PROCEED)
- evidence-based conditions cleaning (no cadences, no thresholds, no missing net profit)
- cash runway not treated as risk
- stable profitability replaced with stable gross margin
"""

import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.decision_agent import DecisionAgent


class TestDecisionValidation(unittest.TestCase):

    def setUp(self):
        self.agent = DecisionAgent()
        self.scenario = {"sales_growth": 0.3, "payment_delay_days": 15.0}
        self.facts = {
            "currency": "INR",
            "revenue": {"baseline": 25841569.0, "projected": 33594040.0, "direction": "increased"},
            "gross_profit": {"baseline": 7028569.0, "projected": 9137140.0, "direction": "increased"},
            "cash": {"baseline": 12148764.0, "projected": 6906899.0, "direction": "decreased"},
            "accounts_receivable": {"baseline": 11102805.0, "projected": 15540785.0, "direction": "increased"},
            "inventory_value": {"baseline": 9404500.0, "projected": 12225850.0, "direction": "increased"},
            "inventory_risk": "High",
            "overall_risk": "Medium",
            "inventory_coverage_months": 5.11,
            "lost_sales": 0.0,
            "cash_runway_raw": "Not reached in projection",
        }
        self.analysis = {
            "executive_summary": "Revenue increases by 30% while cash decreases to Rs. 69.07 lakh.",
            "overall_assessment": "Neutral",
            "confidence": 0.95,
        }
        self.red_team_pass = {
            "overall_verdict": "PASS",
            "confidence": 0.95,
            "critical_issues": [],
            "numerical_issues": [],
        }
        self.red_team_fail = {
            "overall_verdict": "FAIL",
            "confidence": 0.9,
            "critical_issues": ["Gross profit calculation contradicts simulator."],
        }
        self.allowed_numbers = self.agent._build_allowed_numbers(
            self.scenario, self.facts, self.analysis, self.red_team_pass
        )

    def test_valid_decision(self):
        valid = {
            "decision": "PROCEED_WITH_CONDITIONS",
            "priority": "MEDIUM",
            "rationale": ["Revenue grows by 30% to INR 33,594,040 while cash decreases."],
            "recommended_actions": [
                "Priority 1: Arrange working capital buffer for receivables.",
                "Priority 2: Review inventory buffer policies to manage high inventory risk."
            ],
            "expected_benefits": ["Gross profit rises to INR 9,137,140."],
            "risks": ["Inventory risk is High with coverage at 5.11 months."],
            "conditions": ["Review inventory policy given High inventory risk."],
            "confidence": 0.85,
        }
        validated = self.agent._validate_decision(
            valid, self.facts, self.analysis, self.red_team_pass, self.allowed_numbers
        )
        self.assertEqual(validated["priority"], "MEDIUM")
        self.assertEqual(validated["confidence"], 0.85)

    def test_invalid_priority(self):
        invalid = {
            "decision": "PROCEED_WITH_CONDITIONS",
            "priority": "URGENT",  # Invalid! Must be HIGH, MEDIUM, or LOW
            "rationale": ["Some rationale"],
            "recommended_actions": ["Priority 1: Action on inventory buffer."],
            "expected_benefits": ["Benefits"],
            "risks": ["Inventory risk is High"],
            "conditions": ["Conditions"],
            "confidence": 0.8,
        }
        with self.assertRaises(ValueError) as ctx:
            self.agent._validate_decision(
                invalid, self.facts, self.analysis, self.red_team_pass, self.allowed_numbers
            )
        self.assertIn("Invalid priority", str(ctx.exception))

    def test_invalid_confidence_range(self):
        invalid = {
            "decision": "PROCEED_WITH_CONDITIONS",
            "priority": "HIGH",
            "rationale": ["Some rationale"],
            "recommended_actions": ["Priority 1: Action on inventory."],
            "expected_benefits": ["Benefits"],
            "risks": ["Inventory risk is High"],
            "conditions": ["Conditions"],
            "confidence": 1.5,  # Invalid! Out of range
        }
        with self.assertRaises(ValueError) as ctx:
            self.agent._validate_decision(
                invalid, self.facts, self.analysis, self.red_team_pass, self.allowed_numbers
            )
        self.assertIn("Confidence", str(ctx.exception))

    def test_forbidden_currency_symbol(self):
        invalid = {
            "decision": "PROCEED_WITH_CONDITIONS",
            "priority": "HIGH",
            "rationale": ["Revenue rises to $400,000."],  # Forbidden $
            "recommended_actions": ["Priority 1: Monitor inventory."],
            "expected_benefits": ["Benefits"],
            "risks": ["Inventory risk is High"],
            "conditions": ["Conditions"],
            "confidence": 0.8,
        }
        with self.assertRaises(ValueError) as ctx:
            self.agent._validate_decision(
                invalid, self.facts, self.analysis, self.red_team_pass, self.allowed_numbers
            )
        self.assertIn("Forbidden currency symbol", str(ctx.exception))

    def test_invented_numerical_value(self):
        invalid = {
            "decision": "PROCEED_WITH_CONDITIONS",
            "priority": "MEDIUM",
            "rationale": ["Revenue grows and saves INR 87,654,321 in supply chain."],  # Invented number!
            "recommended_actions": ["Priority 1: Monitor inventory stock."],
            "expected_benefits": ["Extra gains"],
            "risks": ["Inventory risk is High"],
            "conditions": ["Conditions"],
            "confidence": 0.8,
        }
        with self.assertRaises(ValueError) as ctx:
            self.agent._validate_decision(
                invalid, self.facts, self.analysis, self.red_team_pass, self.allowed_numbers
            )
        self.assertIn("Unverified financial or numeric value", str(ctx.exception))

    def test_invented_month_target(self):
        invalid = {
            "decision": "PROCEED_WITH_CONDITIONS",
            "priority": "MEDIUM",
            "rationale": ["Revenue grows by 30%."],
            "recommended_actions": ["Priority 1: Adjust inventory buffer."],
            "expected_benefits": ["Benefits"],
            "risks": ["Inventory risk is High"],
            "conditions": ["Inventory coverage must be improved to at least 6 months."],  # 6 months is invented!
            "confidence": 0.8,
        }
        with self.assertRaises(ValueError) as ctx:
            self.agent._validate_decision(
                invalid, self.facts, self.analysis, self.red_team_pass, self.allowed_numbers
            )
        self.assertIn("Invented month figure", str(ctx.exception))

    def test_forbidden_phrase_obsolescence(self):
        invalid = {
            "decision": "PROCEED_WITH_CONDITIONS",
            "priority": "MEDIUM",
            "rationale": ["Revenue grows by 30%."],
            "recommended_actions": ["Priority 1: Adjust inventory."],
            "expected_benefits": ["Reduced obsolescence risk."],  # Forbidden!
            "risks": ["Inventory risk is High"],
            "conditions": ["Conditions"],
            "confidence": 0.8,
        }
        with self.assertRaises(ValueError) as ctx:
            self.agent._validate_decision(
                invalid, self.facts, self.analysis, self.red_team_pass, self.allowed_numbers
            )
        self.assertIn("Forbidden term 'obsolescence'", str(ctx.exception))

    def test_deterministic_decision_framework(self):
        # 1. Medium overall risk + High inventory risk -> PROCEED_WITH_CONDITIONS
        dec1 = self.agent._determine_expected_decision(self.facts, self.red_team_pass)
        self.assertEqual(dec1, "PROCEED_WITH_CONDITIONS")

        # 2. Critical overall risk -> DO_NOT_PROCEED
        crit_facts = dict(self.facts)
        crit_facts["overall_risk"] = "Critical"
        dec2 = self.agent._determine_expected_decision(crit_facts, self.red_team_pass)
        self.assertEqual(dec2, "DO_NOT_PROCEED")

        # 3. Low overall risk + Low inventory risk + unchanged cash -> PROCEED
        low_facts = dict(self.facts)
        low_facts["overall_risk"] = "Low"
        low_facts["inventory_risk"] = "Low"
        low_facts["cash"] = {"baseline": 100, "projected": 100, "direction": "unchanged"}
        low_facts["accounts_receivable"] = {"baseline": 100, "projected": 100, "direction": "unchanged"}
        dec3 = self.agent._determine_expected_decision(low_facts, self.red_team_pass)
        self.assertEqual(dec3, "PROCEED")

        # 4. Red-Team FAIL -> FLAG_FOR_REVIEW
        dec4 = self.agent._determine_expected_decision(self.facts, self.red_team_fail)
        self.assertEqual(dec4, "FLAG_FOR_REVIEW")

    def test_clean_and_validate_conditions(self):
        raw_conditions = [
            "Inventory coverage must be improved to at least 6 months before full-scale operations proceed.",
            "A formal collections process review and improvement plan must be approved and implemented within 30 days.",
            "Cash position must be monitored daily with a threshold alert at Rs.7,000,000.00 to prevent liquidity risk.",
            "Net profit is not provided in baseline, so financial viability under full operations cannot be confirmed without additional data."
        ]
        cleaned = self.agent._clean_and_validate_conditions(raw_conditions, self.facts)
        # Check no missing net profit claim
        self.assertFalse(any("net profit" in c.lower() for c in cleaned))
        # Check no cadences or deadlines
        self.assertFalse(any("30 days" in c.lower() or "daily" in c.lower() for c in cleaned))
        # Check no invented threshold
        self.assertFalse(any("6 months" in c.lower() or "7,000,000" in c.lower() for c in cleaned))
        # Check valid evidence-based conditions are present
        self.assertTrue(any("inventory" in c.lower() for c in cleaned))
        self.assertTrue(any("collections" in c.lower() or "receivable" in c.lower() for c in cleaned))
        self.assertTrue(any("cash" in c.lower() for c in cleaned))

    def test_clean_risks_never_treats_cash_runway_as_risk(self):
        raw_risks = [
            "Inventory risk is High, indicating potential stockouts or obsolescence.",
            "Ending cash position has decreased from baseline.",
            "Cash runway was not reached in projection, posing a liquidity risk."
        ]
        cleaned = self.agent._clean_risks(raw_risks, self.facts)
        self.assertFalse(any("cash runway" in r.lower() for r in cleaned))
        self.assertFalse(any("obsolescence" in r.lower() for r in cleaned))
        self.assertTrue(any("inventory risk is high" in r.lower() for r in cleaned))

    def test_enforce_deterministic_guardrails_on_fail(self):
        raw = {
            "decision": "PROCEED",
            "priority": "LOW",
            "rationale": ["Initial rationale"],
            "recommended_actions": ["Action 1"],
            "expected_benefits": ["Consistent profitability under current sales"],
            "risks": ["Risks"],
            "conditions": ["Conditions"],
            "confidence": 0.95,
        }
        guarded = self.agent._enforce_deterministic_guardrails(raw, self.facts, self.red_team_fail)
        self.assertEqual(guarded["decision"], "FLAG_FOR_REVIEW")
        self.assertEqual(guarded["priority"], "HIGH")
        self.assertLessEqual(guarded["confidence"], 0.2)
        self.assertTrue(any("audit" in cond.lower() for cond in guarded["conditions"]))
        # Profitability overclaim cleaned
        self.assertTrue(any("stable gross margin" in b for b in guarded["expected_benefits"]))


if __name__ == "__main__":
    unittest.main()
