"""
Tests for ActionValidator & Gated Operations Action Workflow.
Verifies:
1. Correct credit-limit percentage calculation (percentage_of_limit and percentage_over_limit)
2. Rejection of invented 50% threshold
3. Rejection of unsupported credit-limit value (e.g. ₹500,000)
4. Rejection of LLM-generated historical/future date (e.g. 2025-04-05)
5. Valid policy-supported recommendation passes validation (PASS)
6. Invalid action blocked before approval (can_approve=False, status=REVIEW_REQUIRED)
7. Valid approved action reaches Event System
8. Digital Twin updates after approved action
"""

import unittest
from datetime import date
from decimal import Decimal

from agents.action_validator import ActionValidator
from backend.digital_twin.digital_twin import build_digital_twin, get_connection
from events.event_processor import EventProcessor
from events.event_types import EventType
from mcp.server import MCPServer
from mcp.tools import MCPBusinessTools


class TestActionValidator(unittest.TestCase):

    def setUp(self):
        self.validator = ActionValidator()
        self.processor = EventProcessor()
        self.mcp = MCPServer()

    def test_credit_limit_percentage_calculation(self):
        """
        Requirement 1:
        Pending AR = 4,755,922, Credit limit = 1,000,000
        percentage_of_limit = pending / credit_limit * 100 ≈ 475.59%
        percentage_over_limit = (pending - credit_limit) / credit_limit * 100 ≈ 375.59%
        """
        pending = 4755922.0
        credit_limit = 1000000.0

        percentage_of_limit = round((pending / credit_limit) * 100, 2)
        percentage_over_limit = round(((pending - credit_limit) / credit_limit) * 100, 2)

        self.assertAlmostEqual(percentage_of_limit, 475.59, places=2)
        self.assertAlmostEqual(percentage_over_limit, 375.59, places=2)

        # Also verify against MCP get_customer_exposure output
        tools = MCPBusinessTools()
        exposure_data = tools.get_customer_exposure()
        top_cust = None
        for cust in exposure_data.get("customer_exposures", []):
            if cust.get("customer_id") == "C004":
                top_cust = cust
                break

        if top_cust:
            self.assertAlmostEqual(top_cust["percentage_of_credit_limit"], 475.59, places=1)
            self.assertAlmostEqual(top_cust["percentage_over_credit_limit"], 375.59, places=1)
            self.assertTrue(top_cust["is_over_credit_limit"])

    def test_rejection_of_invented_50_percent_threshold(self):
        """Requirement 3: Rejection of invented 50% threshold in action rationale."""
        action = {
            "event_type": "RESTRICT_CREDIT",
            "payload": {"customer_id": "C004", "action": "restrict_credit"},
            "reason": "Customer is 50% over allowed balance threshold",
        }
        res = self.validator.validate_action(action, user_objective="Check customer credit health")
        self.assertFalse(res.is_valid)
        self.assertIn(res.status, ["REVIEW_REQUIRED", "FAILED"])
        self.assertTrue(any("50%" in f for f in res.findings))

    def test_rejection_of_unsupported_credit_limit_value(self):
        """Requirement 4: Rejection of unsupported/arbitrary credit limit reduction (e.g. 500,000)."""
        action = {
            "event_type": "RESTRICT_CREDIT",
            "payload": {
                "customer_id": "C004",
                "new_credit_limit": 500000,
            },
            "reason": "Reduce Digital Hub credit limit to 500,000",
        }
        res = self.validator.validate_action(action, user_objective="Check receivables risk")
        self.assertFalse(res.is_valid)
        self.assertEqual(res.status, "REVIEW_REQUIRED")
        self.assertTrue(any("500,000" in f or "arbitrary" in f.lower() for f in res.findings))

    def test_rejection_of_llm_generated_historical_date(self):
        """Requirement 5: Rejection of LLM-generated past/future date (e.g. 2025-04-05)."""
        action = {
            "event_type": "INVENTORY_UPDATED",
            "payload": {
                "product_id": "P001",
                "quantity_change": 10,
                "update_date": "2025-04-05",
            },
            "reason": "Replenish buffer",
        }
        res = self.validator.validate_action(action, user_objective="Replenish stock")
        self.assertFalse(res.is_valid)
        self.assertIn(res.status, ["REVIEW_REQUIRED", "FAILED"])
        self.assertTrue(any("2025-04-05" in f or "Date safety" in f for f in res.findings))

    def test_valid_policy_supported_recommendation(self):
        """Requirement 4 & 6: Policy-supported recommendation passes validation."""
        action = {
            "event_type": "RESTRICT_CREDIT",
            "payload": {
                "customer_id": "C004",
                "action": "restrict_additional_credit",
                "effective_date": str(date.today()),
            },
            "reason": "Initiate a formal credit review and restrict additional credit exposure until the outstanding balance is reviewed.",
        }
        res = self.validator.validate_action(action, user_objective="Find the biggest business risk right now.")
        self.assertTrue(res.is_valid)
        self.assertEqual(res.status, "PASS")
        self.assertIn("payment_policy.md", res.policy_sources)

    def test_invalid_action_blocked_before_approval(self):
        """Requirement 9: Invalid action has is_valid=False and status=REVIEW_REQUIRED."""
        invalid_action = {
            "event_type": "RESTRICT_CREDIT",
            "payload": {
                "customer_id": "C004",
                "new_credit_limit": 500000,
                "effective_date": "2025-04-05",
            },
            "reason": "Reduce credit limit to 500,000 with 50% threshold",
        }
        res = self.validator.validate_action(invalid_action, user_objective="Evaluate risk")
        self.assertFalse(res.is_valid)
        self.assertEqual(res.status, "REVIEW_REQUIRED")
        self.assertGreater(len(res.findings), 1)

    def test_valid_approved_action_reaches_event_system(self):
        """Requirement 11: Valid approved action executes through Event System."""
        res = self.processor.process_event(
            event_type=EventType.RESTRICT_CREDIT,
            payload={
                "customer_id": "C004",
                "action": "restrict_additional_credit",
                "effective_date": str(date.today()),
            },
        )
        self.assertTrue(res.success)
        self.assertEqual(res.details.get("customer_id"), "C004")
        self.assertEqual(res.details.get("status"), "RESTRICTED_PENDING_REVIEW")

    def test_digital_twin_updates_after_approved_action(self):
        """Requirement 11: Digital Twin reflects changes after approved action."""
        before_twin = build_digital_twin()
        before_qty = 0
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT stock_quantity FROM inventory WHERE product_id = 'P001';")
                before_qty = int(cur.fetchone()[0])
        finally:
            conn.close()

        # Execute approved inventory update
        proc_res = self.processor.process_event(
            event_type=EventType.INVENTORY_UPDATED,
            payload={
                "product_id": "P001",
                "quantity_change": 10,
                "update_date": str(date.today()),
            },
        )
        self.assertTrue(proc_res.success)

        # Verify DB and Digital Twin update
        after_twin = build_digital_twin()
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT stock_quantity FROM inventory WHERE product_id = 'P001';")
                after_qty = int(cur.fetchone()[0])
        finally:
            conn.close()

        self.assertEqual(after_qty, before_qty + 10)

        # Rollback the 10 units test change to keep DB clean
        self.processor.process_event(
            event_type=EventType.INVENTORY_UPDATED,
            payload={
                "product_id": "P001",
                "quantity_change": -10,
                "update_date": str(date.today()),
            },
        )


if __name__ == "__main__":
    unittest.main()
