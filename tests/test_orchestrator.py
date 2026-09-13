"""
Unit and Integration Tests for the Digital Twin Orchestrator.
Validates:
- Successful full pipeline execution
- Planner failure isolation
- Simulator failure isolation
- Analysis / Ollama failure graceful degradation (upstream preservation)
- Red-Team failure graceful degradation (upstream preservation)
- Decision failure graceful degradation (upstream preservation)
- Simulator facts immutability throughout the pipeline
- Status indicator accuracy
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.digital_twin.digital_twin import build_digital_twin
from backend.digital_twin.simulation import Scenario, simulate
from backend.orchestration.state import PipelineResult, PipelineStatus, StageStatus
from backend.orchestration.pipeline import DigitalTwinOrchestrator, run_digital_twin_pipeline


class TestDigitalTwinOrchestrator(unittest.TestCase):

    def setUp(self):
        self.twin = build_digital_twin()
        self.scenario = Scenario(sales_growth=0.30, payment_delay_days=15.0)

        # Mocked valid agent responses
        self.mock_plan = {
            "sales_growth": 0.30,
            "price_change": 0.0,
            "cost_change": 0.0,
            "expense_change": 0.0,
            "payment_delay_days": 15.0,
            "inventory_change": 0.0,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
            "confidence": 0.95,
            "interpretation": "30% sales increase with 15 days payment delay",
        }

        self.mock_analysis = {
            "executive_summary": "Revenue expands by 30% while cash decreases to Rs. 69.07 lakh.",
            "key_drivers": ["30% sales growth drives revenue expansion."],
            "financial_impact": ["Gross profit rises."],
            "cash_flow_impact": ["Cash drops due to AR build."],
            "working_capital_impact": ["AR increases by 15 days."],
            "inventory_analysis": ["Inventory coverage at 5.11 months."],
            "risk_analysis": ["Inventory risk is High."],
            "tradeoffs": ["Top-line growth vs liquidity."],
            "recommendations": ["Review inventory policy.", "Monitor collections."],
            "overall_assessment": "Neutral",
            "confidence": 0.95,
        }

        self.mock_red_team = {
            "critical_issues": [],
            "numerical_issues": [],
            "logic_issues": [],
            "unsupported_claims": [],
            "missing_risks": [],
            "recommendation_issues": [],
            "verified_claims": ["30% revenue growth verified.", "Inventory risk High verified."],
            "overall_verdict": "PASS",
            "confidence": 0.95,
        }

        self.mock_decision = {
            "decision": "PROCEED_WITH_CONDITIONS",
            "priority": "MEDIUM",
            "rationale": ["Revenue grows 30% but cash drops."],
            "recommended_actions": ["Priority 1: Review inventory policy."],
            "expected_benefits": ["Gross profit increases."],
            "risks": ["Inventory risk is High."],
            "conditions": ["Review inventory policy given High inventory risk."],
            "confidence": 0.90,
        }

    # ------------------------------------------------------------------
    # 1. FULL SUCCESSFUL PIPELINE
    # ------------------------------------------------------------------
    def test_full_pipeline_success(self):
        mock_planner = MagicMock()
        mock_planner.plan.return_value = self.mock_plan
        mock_planner.to_scenario.return_value = self.scenario

        mock_analyst = MagicMock()
        mock_analyst.analyze.return_value = self.mock_analysis

        mock_rt = MagicMock()
        mock_rt.review.return_value = self.mock_red_team

        mock_dec = MagicMock()
        mock_dec.decide.return_value = self.mock_decision

        orchestrator = DigitalTwinOrchestrator(
            twin=self.twin,
            planner=mock_planner,
            analyst=mock_analyst,
            red_team=mock_rt,
            decision_agent=mock_dec,
        )

        res = orchestrator.run("What if sales grow by 30% and customers take 15 extra days to pay?")

        self.assertEqual(res.pipeline_status, PipelineStatus.SUCCESS.value)
        self.assertEqual(res.stage_statuses["planner"], StageStatus.SUCCESS.value)
        self.assertEqual(res.stage_statuses["simulator"], StageStatus.SUCCESS.value)
        self.assertEqual(res.stage_statuses["analysis"], StageStatus.SUCCESS.value)
        self.assertEqual(res.stage_statuses["red_team"], StageStatus.SUCCESS.value)
        self.assertEqual(res.stage_statuses["decision"], StageStatus.SUCCESS.value)

        self.assertIsNotNone(res.scenario)
        self.assertIsNotNone(res.simulation)
        self.assertIsNotNone(res.analysis)
        self.assertIsNotNone(res.red_team)
        self.assertIsNotNone(res.decision)
        self.assertEqual(len(res.errors), 0)

        # Verify status indicator
        indicator = res.get_status_indicator()
        self.assertIn("Planner ✓", indicator)
        self.assertIn("Simulator ✓", indicator)
        self.assertIn("Analysis ✓", indicator)
        self.assertIn("Red-Team ✓", indicator)
        self.assertIn("Decision ✓", indicator)

    # ------------------------------------------------------------------
    # 2. PLANNER FAILURE
    # ------------------------------------------------------------------
    def test_planner_failure(self):
        mock_planner = MagicMock()
        mock_planner.plan.side_effect = ValueError("Could not parse ambiguous query.")

        orchestrator = DigitalTwinOrchestrator(
            twin=self.twin,
            planner=mock_planner,
        )

        res = orchestrator.run("Unparseable query")

        self.assertEqual(res.pipeline_status, PipelineStatus.FAILED.value)
        self.assertEqual(res.stage_statuses["planner"], StageStatus.FAILED.value)
        self.assertEqual(res.stage_statuses["simulator"], StageStatus.SKIPPED.value)
        self.assertEqual(res.stage_statuses["analysis"], StageStatus.SKIPPED.value)
        self.assertEqual(res.stage_statuses["red_team"], StageStatus.SKIPPED.value)
        self.assertEqual(res.stage_statuses["decision"], StageStatus.SKIPPED.value)

        self.assertIsNone(res.simulation)
        self.assertIsNone(res.analysis)
        self.assertTrue(any("Planner Error" in e for e in res.errors))

    def test_empty_question_failure(self):
        orchestrator = DigitalTwinOrchestrator(twin=self.twin)
        res = orchestrator.run("   ")
        self.assertEqual(res.pipeline_status, PipelineStatus.FAILED.value)
        self.assertEqual(res.stage_statuses["planner"], StageStatus.FAILED.value)
        self.assertTrue(any("Question cannot be empty" in e for e in res.errors))

    # ------------------------------------------------------------------
    # 3. SIMULATOR FAILURE
    # ------------------------------------------------------------------
    def test_simulator_failure(self):
        # Scenario with illegal parameters that violate simulation constraints
        invalid_sc = Scenario(sales_growth=-1.5)  # Less than -100%

        orchestrator = DigitalTwinOrchestrator(twin=self.twin)
        res = orchestrator.run(invalid_sc)

        self.assertEqual(res.pipeline_status, PipelineStatus.FAILED.value)
        self.assertEqual(res.stage_statuses["planner"], StageStatus.SUCCESS.value)
        self.assertEqual(res.stage_statuses["simulator"], StageStatus.FAILED.value)
        self.assertEqual(res.stage_statuses["analysis"], StageStatus.SKIPPED.value)
        self.assertEqual(res.stage_statuses["red_team"], StageStatus.SKIPPED.value)
        self.assertEqual(res.stage_statuses["decision"], StageStatus.SKIPPED.value)

        self.assertIsNone(res.simulation)
        self.assertTrue(any("Simulator Error" in e for e in res.errors))

    # ------------------------------------------------------------------
    # 4. ANALYSIS / OLLAMA FAILURE PRESERVES UPSTREAM
    # ------------------------------------------------------------------
    def test_analysis_failure_preserves_upstream(self):
        mock_analyst = MagicMock()
        mock_analyst.analyze.side_effect = RuntimeError("Ollama connection timed out")

        orchestrator = DigitalTwinOrchestrator(
            twin=self.twin,
            analyst=mock_analyst,
        )

        res = orchestrator.run(self.scenario)

        # Pipeline status is PARTIAL_SUCCESS because simulation succeeded
        self.assertEqual(res.pipeline_status, PipelineStatus.PARTIAL_SUCCESS.value)
        self.assertEqual(res.stage_statuses["planner"], StageStatus.SUCCESS.value)
        self.assertEqual(res.stage_statuses["simulator"], StageStatus.SUCCESS.value)
        self.assertEqual(res.stage_statuses["analysis"], StageStatus.FAILED.value)
        self.assertEqual(res.stage_statuses["red_team"], StageStatus.SKIPPED.value)
        self.assertEqual(res.stage_statuses["decision"], StageStatus.SKIPPED.value)

        # Upstream results MUST be preserved
        self.assertIsNotNone(res.scenario)
        self.assertIsNotNone(res.simulation)
        self.assertIn("projected", res.simulation)
        self.assertIn("baseline", res.simulation)
        self.assertIsNone(res.analysis)
        self.assertIsNone(res.red_team)
        self.assertIsNone(res.decision)

        # Warning recorded for user awareness
        self.assertTrue(any("preserved" in w.lower() for w in res.warnings))

    # ------------------------------------------------------------------
    # 5. RED-TEAM FAILURE PRESERVES UPSTREAM
    # ------------------------------------------------------------------
    def test_red_team_failure_preserves_upstream(self):
        mock_analyst = MagicMock()
        mock_analyst.analyze.return_value = self.mock_analysis

        mock_rt = MagicMock()
        mock_rt.review.side_effect = RuntimeError("Red-Team model OOM")

        orchestrator = DigitalTwinOrchestrator(
            twin=self.twin,
            analyst=mock_analyst,
            red_team=mock_rt,
        )

        res = orchestrator.run(self.scenario)

        self.assertEqual(res.pipeline_status, PipelineStatus.PARTIAL_SUCCESS.value)
        self.assertEqual(res.stage_statuses["planner"], StageStatus.SUCCESS.value)
        self.assertEqual(res.stage_statuses["simulator"], StageStatus.SUCCESS.value)
        self.assertEqual(res.stage_statuses["analysis"], StageStatus.SUCCESS.value)
        self.assertEqual(res.stage_statuses["red_team"], StageStatus.FAILED.value)
        self.assertEqual(res.stage_statuses["decision"], StageStatus.SKIPPED.value)

        # Upstream results MUST be preserved
        self.assertIsNotNone(res.simulation)
        self.assertIsNotNone(res.analysis)
        self.assertIsNone(res.red_team)
        self.assertIsNone(res.decision)

    # ------------------------------------------------------------------
    # 6. DECISION FAILURE PRESERVES UPSTREAM
    # ------------------------------------------------------------------
    def test_decision_failure_preserves_upstream(self):
        mock_analyst = MagicMock()
        mock_analyst.analyze.return_value = self.mock_analysis

        mock_rt = MagicMock()
        mock_rt.review.return_value = self.mock_red_team

        mock_dec = MagicMock()
        mock_dec.decide.side_effect = RuntimeError("Decision validation failure")

        orchestrator = DigitalTwinOrchestrator(
            twin=self.twin,
            analyst=mock_analyst,
            red_team=mock_rt,
            decision_agent=mock_dec,
        )

        res = orchestrator.run(self.scenario)

        self.assertEqual(res.pipeline_status, PipelineStatus.PARTIAL_SUCCESS.value)
        self.assertEqual(res.stage_statuses["planner"], StageStatus.SUCCESS.value)
        self.assertEqual(res.stage_statuses["simulator"], StageStatus.SUCCESS.value)
        self.assertEqual(res.stage_statuses["analysis"], StageStatus.SUCCESS.value)
        self.assertEqual(res.stage_statuses["red_team"], StageStatus.SUCCESS.value)
        self.assertEqual(res.stage_statuses["decision"], StageStatus.FAILED.value)

        # Upstream results MUST be preserved
        self.assertIsNotNone(res.simulation)
        self.assertIsNotNone(res.analysis)
        self.assertIsNotNone(res.red_team)
        self.assertIsNone(res.decision)

    # ------------------------------------------------------------------
    # 7. SIMULATOR FACTS IMMUTABILITY
    # ------------------------------------------------------------------
    def test_simulator_facts_immutability(self):
        """Ensure malicious or erroneous downstream agent mutation does not alter simulator facts."""
        def malicious_analyze(sc, facts):
            # Attempt to corrupt revenue and cash
            if "projected" in facts:
                facts["projected"]["revenue"] = 999999999.0
                facts["projected"]["ending_cash"] = 0.0
            return self.mock_analysis

        mock_analyst = MagicMock()
        mock_analyst.analyze.side_effect = malicious_analyze

        mock_rt = MagicMock()
        mock_rt.review.return_value = self.mock_red_team

        mock_dec = MagicMock()
        mock_dec.decide.return_value = self.mock_decision

        orchestrator = DigitalTwinOrchestrator(
            twin=self.twin,
            analyst=mock_analyst,
            red_team=mock_rt,
            decision_agent=mock_dec,
        )

        res = orchestrator.run(self.scenario)

        # Pristine simulation result for comparison
        pristine_sim = simulate(self.scenario, self.twin)

        self.assertEqual(res.simulation["projected"]["revenue"], pristine_sim["projected"]["revenue"])
        self.assertEqual(res.simulation["projected"]["ending_cash"], pristine_sim["projected"]["ending_cash"])
        self.assertNotEqual(res.simulation["projected"]["revenue"], 999999999.0)

    # ------------------------------------------------------------------
    # 8. DIRECT ENTRY POINT CONVENIENCE FUNCTION
    # ------------------------------------------------------------------
    def test_run_digital_twin_pipeline_entry_point(self):
        mock_planner = MagicMock()
        mock_planner.plan.return_value = self.mock_plan
        mock_planner.to_scenario.return_value = self.scenario

        mock_analyst = MagicMock()
        mock_analyst.analyze.return_value = self.mock_analysis

        mock_rt = MagicMock()
        mock_rt.review.return_value = self.mock_red_team

        mock_dec = MagicMock()
        mock_dec.decide.return_value = self.mock_decision

        res = run_digital_twin_pipeline(
            "What if sales grow by 30% and customers take 15 extra days to pay?",
            twin=self.twin,
            planner=mock_planner,
            analyst=mock_analyst,
            red_team=mock_rt,
            decision_agent=mock_dec,
        )

        self.assertIsInstance(res, PipelineResult)
        self.assertEqual(res.pipeline_status, PipelineStatus.SUCCESS.value)
        self.assertEqual(res.decision["decision"], "PROCEED_WITH_CONDITIONS")


if __name__ == "__main__":
    unittest.main()
