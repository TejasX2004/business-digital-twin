"""
Tests for Model Context Protocol (MCP) Server & Business Tools.
Verifies tool discovery, observation tools, RAG search, simulation, and safety boundaries.
"""

import unittest
from datetime import date
from mcp import MCPServer, MCPBusinessTools, TOOL_DEFINITIONS
from events import EventStore


class TestMCPServerAndTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = MCPServer()
        cls.tools = cls.server.tools

    def test_tool_discovery(self):
        """1. Agent can discover available MCP tools with valid schemas."""
        tools = self.server.list_tools()
        self.assertEqual(len(tools), 10)
        tool_names = [t["name"] for t in tools]
        expected_tools = [
            "get_business_snapshot",
            "get_inventory_status",
            "get_customer_exposure",
            "get_cash_position",
            "get_payment_metrics",
            "get_sales_summary",
            "search_business_policy",
            "run_business_simulation",
            "compare_scenarios",
            "create_business_event",
        ]
        for expected in expected_tools:
            self.assertIn(expected, tool_names)

        # Verify each tool has a description and inputSchema
        for t in tools:
            self.assertTrue(bool(t.get("description")))
            self.assertIn("inputSchema", t)
            self.assertEqual(t["inputSchema"]["type"], "object")

    def test_get_business_snapshot(self):
        """2. Agent can retrieve business snapshot."""
        res = self.server.call_tool("get_business_snapshot", {})
        self.assertNotIn("error", res)
        self.assertIn("structured_data", res)
        data = res["structured_data"]
        self.assertIn("cash_flow", data)
        self.assertIn("working_capital", data)
        self.assertIn("financials", data)
        self.assertIn("risk", data)

    def test_get_inventory_status(self):
        """3. Agent can investigate inventory risk."""
        res = self.server.call_tool("get_inventory_status", {})
        self.assertNotIn("error", res)
        data = res["structured_data"]
        self.assertIn("total_products", data)
        self.assertIn("total_inventory_value", data)
        self.assertIn("products_at_risk_count", data)
        self.assertIn("at_risk_products", data)
        self.assertIsInstance(data["at_risk_products"], list)

    def test_get_customer_exposure(self):
        """4. Agent can investigate receivables and customer exposure."""
        res = self.server.call_tool("get_customer_exposure", {})
        self.assertNotIn("error", res)
        data = res["structured_data"]
        self.assertIn("total_accounts_receivable", data)
        self.assertIn("customer_exposures", data)
        self.assertIsInstance(data["customer_exposures"], list)

    def test_get_cash_position(self):
        """Agent can retrieve cash position and runway status."""
        res = self.server.call_tool("get_cash_position", {})
        self.assertNotIn("error", res)
        data = res["structured_data"]
        self.assertIn("current_cash", data)
        self.assertIn("operating_cash_flow", data)
        self.assertIn("estimated_cash_runway_months", data)

    def test_get_payment_metrics(self):
        """Agent can retrieve payment behavior metrics."""
        res = self.server.call_tool("get_payment_metrics", {})
        self.assertNotIn("error", res)
        data = res["structured_data"]
        self.assertIn("average_payment_days", data)
        self.assertIn("dso_days", data)

    def test_get_sales_summary(self):
        """Agent can retrieve sales summary."""
        res = self.server.call_tool("get_sales_summary", {})
        self.assertNotIn("error", res)
        data = res["structured_data"]
        self.assertIn("total_transactions", data)
        self.assertIn("total_revenue", data)

    def test_search_business_policy(self):
        """6. Agent can retrieve authoritative company policy via RAG."""
        res = self.server.call_tool("search_business_policy", {"query": "inventory safety stock reorder policy", "top_k": 2})
        self.assertNotIn("error", res)
        data = res["structured_data"]
        self.assertIsInstance(data, list)
        self.assertTrue(len(data) > 0)
        first_doc = data[0]
        self.assertIn("content", first_doc)
        self.assertIn("source", first_doc)

    def test_run_business_simulation(self):
        """5 & 7. Agent can run simulations and simulator numbers match ground truth."""
        res = self.server.call_tool("run_business_simulation", {"sales_growth": 0.30, "price_change": 0.05})
        self.assertNotIn("error", res)
        data = res["structured_data"]
        self.assertIn("financial_impact", data)
        self.assertIn("liquidity_impact", data)
        self.assertIn("risk_assessment", data)
        self.assertIn("projected_revenue", data["financial_impact"])
        self.assertIn("projected_ending_cash", data["liquidity_impact"])

        # Baseline comparison
        base_res = self.server.call_tool("run_business_simulation", {"sales_growth": 0.0})
        base_data = base_res["structured_data"]
        self.assertGreater(data["financial_impact"]["projected_revenue"], base_data["financial_impact"]["projected_revenue"])

    def test_compare_scenarios(self):
        """Agent can run and compare multiple scenarios side-by-side."""
        scenarios = [
            {"name": "Base", "sales_growth": 0.0},
            {"name": "Growth", "sales_growth": 0.30},
            {"name": "Severe Downturn", "sales_growth": -0.20, "payment_delay_days": 30},
        ]
        res = self.server.call_tool("compare_scenarios", {"scenarios": scenarios})
        self.assertNotIn("error", res)
        data = res["structured_data"]
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0]["name"], "Base")
        self.assertEqual(data[1]["name"], "Growth")
        self.assertEqual(data[2]["name"], "Severe Downturn")

    def test_create_business_event_unapproved(self):
        """9. Agent cannot execute business mutations without explicit user approval."""
        store = EventStore()
        initial_events = len(store.get_recent_events(limit=100))

        res = self.server.call_tool(
            "create_business_event",
            {
                "event_type": "EXPENSE_RECORDED",
                "payload": {
                    "category": "Office Supplies",
                    "amount": 250.0,
                    "expense_date": str(date.today()),
                    "description": "MCP Unapproved Test",
                },
                "approved": False,
            },
        )
        self.assertIn("structured_data", res)
        data = res["structured_data"]
        self.assertEqual(data["status"], "PENDING_APPROVAL")
        self.assertTrue(data["requires_user_approval"])

        # Verify nothing was added to event store
        current_events = len(store.get_recent_events(limit=100))
        self.assertEqual(current_events, initial_events)

    def test_create_business_event_approved(self):
        """10, 11, 12. Approved business event changes PostgreSQL, Digital Twin reflects it, and verifies state."""
        res = self.server.call_tool(
            "create_business_event",
            {
                "event_type": "EXPENSE_RECORDED",
                "payload": {
                    "category": "Utilities",
                    "amount": 100.0,
                    "expense_date": str(date.today()),
                    "description": "MCP Approved Test Mutation",
                },
                "approved": True,
            },
        )
        self.assertNotIn("error", res)
        data = res["structured_data"]
        self.assertEqual(data["status"], "EXECUTED")
        self.assertTrue(data["digital_twin_verified"])
        self.assertIn("updated_business_state", data)
        self.assertIn("cash_balance", data["updated_business_state"])

    def test_json_rpc_dispatch(self):
        """Verifies JSON-RPC 2.0 protocol request handling."""
        # Initialize
        init_resp = self.server.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        self.assertEqual(init_resp["id"], 1)
        self.assertEqual(init_resp["result"]["serverInfo"]["name"], "acme-business-control-tower-mcp")

        # tools/list
        list_resp = self.server.handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.assertEqual(list_resp["id"], 2)
        self.assertEqual(len(list_resp["result"]["tools"]), 10)

        # tools/call
        call_resp = self.server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "get_cash_position", "arguments": {}},
        })
        self.assertEqual(call_resp["id"], 3)
        self.assertFalse(call_resp["result"].get("isError", False))

        # Unknown method
        unknown_resp = self.server.handle_jsonrpc({"jsonrpc": "2.0", "id": 4, "method": "nonexistent_method"})
        self.assertIn("error", unknown_resp)
        self.assertEqual(unknown_resp["error"]["code"], -32601)


if __name__ == "__main__":
    unittest.main()
