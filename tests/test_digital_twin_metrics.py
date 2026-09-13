"""
Tests for Digital Twin Metrics Validation.
Verifies:
1. DSO calculation formula: (AR / Revenue) * 365
2. Distinction between DSO, average payment settlement days, average payment terms, and inventory days
3. Digital Twin reconciliation tests
"""

import unittest
from backend.digital_twin.digital_twin import (
    build_digital_twin,
    calculate_dso,
)
from mcp.tools import MCPBusinessTools


class TestDigitalTwinMetrics(unittest.TestCase):

    def setUp(self):
        self.twin = build_digital_twin()
        self.tools = MCPBusinessTools()

    def test_dso_calculation_formula(self):
        """
        Requirement 2:
        DSO = (Accounts Receivable / Revenue) * 365
        """
        rev = self.twin["financials"]["revenue"]
        ar = self.twin["working_capital"]["accounts_receivable"]
        expected_dso = round((ar / rev) * 365, 2) if rev > 0 else 0.0

        calculated_dso = calculate_dso(ar, rev)
        stored_dso = self.twin.get("payment_behavior", {}).get("dso_days")

        self.assertEqual(calculated_dso, expected_dso)
        self.assertEqual(stored_dso, expected_dso)

        # Verify against MCP payment metrics
        payment_metrics = self.tools.get_payment_metrics()
        self.assertEqual(payment_metrics["dso_days"], expected_dso)

    def test_metrics_distinction(self):
        """
        Requirement 2:
        Ensure the system does not conflate distinct metrics:
        - DSO: days of revenue tied up in receivables
        - average_payment_days: settlement duration of paid invoices
        - average_payment_terms: contractual term duration
        - inventory_days: inventory holding duration (365 / inventory_turnover)
        """
        payment_behavior = self.twin["payment_behavior"]
        operations = self.twin["operations"]

        dso = payment_behavior.get("dso_days")
        avg_payment_days = payment_behavior.get("average_payment_days")
        avg_payment_terms = payment_behavior.get("average_payment_terms")
        inventory_days = operations.get("inventory_days")

        # Verify they are separate metrics
        self.assertIsNotNone(dso)
        self.assertIsNotNone(avg_payment_days)
        self.assertIsNotNone(avg_payment_terms)
        self.assertIsNotNone(inventory_days)

        # DSO (approx 151d) must NOT equal average payment days (approx 42d)
        self.assertNotEqual(dso, avg_payment_days)
        # DSO must NOT be blindly equated with inventory days
        self.assertNotEqual(dso, inventory_days)

        # Check MCP tools return authoritative values
        payment_metrics = self.tools.get_payment_metrics()
        self.assertEqual(payment_metrics["data_authority"], "AUTHORITATIVE")
        self.assertEqual(payment_metrics["dso_days"], dso)
        self.assertEqual(payment_metrics["average_payment_days"], avg_payment_days)
        self.assertEqual(payment_metrics["average_payment_terms_days"], avg_payment_terms)

    def test_digital_twin_reconciliation(self):
        """Run Digital Twin mathematical reconciliation checks."""
        fin = self.twin["financials"]
        wc = self.twin["working_capital"]
        cf = self.twin["cash_flow"]
        pb = self.twin["payment_behavior"]

        rev = fin["revenue"]
        cogs = fin["cogs"]
        gp = fin["gross_profit"]
        opex = fin["operating_expenses"]
        np = fin["net_profit"]
        ar = wc["accounts_receivable"]

        # Gross profit reconciliation: rev - cogs = gp
        self.assertAlmostEqual(gp, rev - cogs, delta=1.0)
        # Net profit reconciliation: gp - opex = np
        self.assertAlmostEqual(np, gp - opex, delta=1.0)
        # DSO reconciliation: (ar / rev) * 365 = dso
        expected_dso = round((ar / rev) * 365, 2)
        self.assertEqual(pb["dso_days"], expected_dso)
        # Receivable % reconciliation: (ar / rev) * 100
        expected_rec_pct = round((ar / rev) * 100, 2)
        self.assertEqual(wc["receivable_percentage"], expected_rec_pct)


if __name__ == "__main__":
    unittest.main()
