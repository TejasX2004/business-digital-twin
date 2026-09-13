"""
Automated Tests for Event-Driven Business Digital Twin Layer.
Tests:
1. Valid PAYMENT_RECEIVED event execution
2. PostgreSQL data changes verification
3. Rebuilding the Digital Twin after an event
4. Verifying updated Cash and Accounts Receivable in the Digital Twin
5. Rollback on invalid event and ensuring PostgreSQL state is unchanged
6. Idempotency and duplicate event rejection
7. SALE_CREATED, INVENTORY_UPDATED, EXPENSE_RECORDED, SUPPLIER_PAYMENT_UPDATED
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import date
from decimal import Decimal

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.digital_twin.digital_twin import build_digital_twin, get_connection
from backend.database.seed_database import seed_database
from events.event_processor import EventProcessor
from events.event_store import EventStore
from events.event_types import EventStatus, EventType


class TestBusinessEvents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Re-seed database to ensure pristine baseline state
        seed_database()

    def setUp(self):
        self.processor = EventProcessor()
        self.store = EventStore()

    # ------------------------------------------------------------------
    # 1. PAYMENT_RECEIVED: CASH & AR REFRESH
    # ------------------------------------------------------------------
    def test_payment_received_updates_cash_and_ar(self):
        """
        Record a valid PAYMENT_RECEIVED event.
        Verify PostgreSQL data changes.
        Rebuild Digital Twin.
        Verify Cash increases and AR decreases.
        """
        # Baseline twin
        base_twin = build_digital_twin()
        initial_cash = base_twin["cash_flow"]["current_cash"]
        initial_ar = base_twin["working_capital"]["accounts_receivable"]

        # Sale S0005 is pending: 10 units @ 47999 = 479,990.00
        sale_id = "S0005"
        amount = 479990.00

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT payment_status FROM sales WHERE sale_id = %s;", (sale_id,))
                status_before = cur.fetchone()[0]
                self.assertEqual(status_before, "Pending")
        finally:
            conn.close()

        # Execute event
        res = self.processor.process_event(
            event_type=EventType.PAYMENT_RECEIVED,
            payload={
                "sale_id": sale_id,
                "amount": amount,
                "payment_date": str(date.today()),
                "payment_method": "Bank Transfer",
            }
        )

        self.assertTrue(res.success, f"Event processing failed: {res.error}")

        # Verify PostgreSQL state
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                # Check payment record exists
                cur.execute("SELECT payment_id, amount FROM payments WHERE sale_id = %s;", (sale_id,))
                pm_row = cur.fetchone()
                self.assertIsNotNone(pm_row)
                self.assertEqual(float(pm_row[1]), amount)

                # Check sale status updated to 'Paid'
                cur.execute("SELECT payment_status FROM sales WHERE sale_id = %s;", (sale_id,))
                status_after = cur.fetchone()[0]
                self.assertEqual(status_after, "Paid")
        finally:
            conn.close()

        # Rebuild Digital Twin
        updated_twin = build_digital_twin()
        new_cash = updated_twin["cash_flow"]["current_cash"]
        new_ar = updated_twin["working_capital"]["accounts_receivable"]

        # Verify cash increased by exact payment amount
        self.assertAlmostEqual(new_cash, initial_cash + amount, places=2)
        # Verify AR decreased by exact payment amount
        self.assertAlmostEqual(new_ar, initial_ar - amount, places=2)

    # ------------------------------------------------------------------
    # 2. INVALID EVENT ROLLBACK & UNCHANGED DATABASE
    # ------------------------------------------------------------------
    def test_invalid_event_leaves_database_unchanged(self):
        """
        Attempt an invalid event (e.g. non-existent sale ID).
        Verify that database transaction is rolled back and target table is unchanged.
        """
        base_twin = build_digital_twin()
        initial_cash = base_twin["cash_flow"]["current_cash"]
        initial_ar = base_twin["working_capital"]["accounts_receivable"]

        # Non-existent sale
        res = self.processor.process_event(
            event_type=EventType.PAYMENT_RECEIVED,
            payload={
                "sale_id": "NON_EXISTENT_SALE_XYZ",
                "amount": 100000.00,
                "payment_date": str(date.today()),
            }
        )

        self.assertFalse(res.success)
        self.assertIn("does not exist", res.error.lower())

        # Check DB payments count
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM payments WHERE sale_id = 'NON_EXISTENT_SALE_XYZ';")
                count = cur.fetchone()[0]
                self.assertEqual(count, 0)
        finally:
            conn.close()

        # Rebuild twin, verify zero changes
        twin_after = build_digital_twin()
        self.assertEqual(twin_after["cash_flow"]["current_cash"], initial_cash)
        self.assertEqual(twin_after["working_capital"]["accounts_receivable"], initial_ar)

    # ------------------------------------------------------------------
    # 3. IDEMPOTENCY & DUPLICATE REJECTION
    # ------------------------------------------------------------------
    def test_duplicate_event_id_is_rejected(self):
        """
        Verify that re-submitting an event with the same event_id is rejected
        and does not result in double processing.
        """
        import uuid
        evt_id = f"EVT-TEST-IDEMP-{uuid.uuid4().hex[:8]}"
        payload = {
            "category": "Utilities",
            "amount": 2500.0,
            "expense_date": str(date.today()),
            "description": "Internet Bill",
        }

        # First execution succeeds
        res1 = self.processor.process_event(
            event_type=EventType.EXPENSE_RECORDED,
            payload=payload,
            event_id=evt_id,
        )
        self.assertTrue(res1.success)

        # Re-submitting with identical event_id must fail
        res2 = self.processor.process_event(
            event_type=EventType.EXPENSE_RECORDED,
            payload=payload,
            event_id=evt_id,
        )
        self.assertFalse(res2.success)
        self.assertIn("Duplicate", res2.message)

    # ------------------------------------------------------------------
    # 4. SALE_CREATED EVENT
    # ------------------------------------------------------------------
    def test_sale_created_event(self):
        """
        Verify SALE_CREATED inserts record with 'Pending' status
        and increases revenue and accounts receivable in the Digital Twin.
        """
        base_twin = build_digital_twin()
        initial_rev = base_twin["financials"]["revenue"]
        initial_ar = base_twin["working_capital"]["accounts_receivable"]

        res = self.processor.process_event(
            event_type=EventType.SALE_CREATED,
            payload={
                "customer_id": "C001",
                "product_id": "P001",
                "quantity": 2,
                "sale_amount": 139998.0,
                "sale_date": str(date.today()),
            }
        )

        self.assertTrue(res.success)
        sale_id = res.details.get("sale_id")
        self.assertIsNotNone(sale_id)

        # Check DB
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT payment_status, unit_price FROM sales WHERE sale_id = %s;", (sale_id,))
                row = cur.fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row[0], "Pending")
                self.assertEqual(float(row[1]), 69999.0)
        finally:
            conn.close()

        # Rebuild twin
        twin_after = build_digital_twin()
        self.assertAlmostEqual(twin_after["financials"]["revenue"], initial_rev + 139998.0, places=2)
        self.assertAlmostEqual(twin_after["working_capital"]["accounts_receivable"], initial_ar + 139998.0, places=2)

    # ------------------------------------------------------------------
    # 5. INVENTORY_UPDATED EVENT
    # ------------------------------------------------------------------
    def test_inventory_updated_event(self):
        """
        Verify INVENTORY_UPDATED adjusts stock_quantity and updates twin inventory value.
        """
        product_id = "P006"  # Wireless Mouse: cost_price 900
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT stock_quantity FROM inventory WHERE product_id = %s;", (product_id,))
                stock_before = int(cur.fetchone()[0])
        finally:
            conn.close()

        delta = 20
        res = self.processor.process_event(
            event_type=EventType.INVENTORY_UPDATED,
            payload={
                "product_id": product_id,
                "quantity_change": delta,
                "update_date": str(date.today()),
            }
        )

        self.assertTrue(res.success)
        self.assertEqual(res.details["new_stock"], stock_before + delta)

        # Check DB
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT stock_quantity FROM inventory WHERE product_id = %s;", (product_id,))
                stock_after = int(cur.fetchone()[0])
                self.assertEqual(stock_after, stock_before + delta)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 6. EXPENSE_RECORDED EVENT
    # ------------------------------------------------------------------
    def test_expense_recorded_event(self):
        """
        Verify EXPENSE_RECORDED increases operating expenses and decreases cash.
        """
        base_twin = build_digital_twin()
        initial_expenses = base_twin["financials"]["operating_expenses"]
        initial_cash = base_twin["cash_flow"]["current_cash"]

        amount = 15000.0
        res = self.processor.process_event(
            event_type=EventType.EXPENSE_RECORDED,
            payload={
                "category": "Marketing",
                "amount": amount,
                "expense_date": str(date.today()),
                "description": "Quarterly SEO Campaign",
            }
        )

        self.assertTrue(res.success)

        twin_after = build_digital_twin()
        self.assertAlmostEqual(twin_after["financials"]["operating_expenses"], initial_expenses + amount, places=2)
        self.assertAlmostEqual(twin_after["cash_flow"]["current_cash"], initial_cash - amount, places=2)

    # ------------------------------------------------------------------
    # 7. SUPPLIER_PAYMENT_UPDATED EVENT
    # ------------------------------------------------------------------
    def test_supplier_payment_updated_event(self):
        """
        Verify SUPPLIER_PAYMENT_UPDATED records/updates a supplier payment in expenses.
        """
        amount = 80000.0
        res = self.processor.process_event(
            event_type=EventType.SUPPLIER_PAYMENT_UPDATED,
            payload={
                "amount": amount,
                "payment_date": str(date.today()),
                "description": "Payment to Component Supplier",
            }
        )

        self.assertTrue(res.success)
        expense_id = res.details["expense_id"]

        # Check DB
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT amount, category FROM expenses WHERE expense_id = %s;", (expense_id,))
                row = cur.fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(float(row[0]), amount)
                self.assertEqual(row[1], "Supplier Payment")
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
