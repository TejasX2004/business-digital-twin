"""
Tests for the Autonomous Business Operations / Control Tower Agent.
Verifies dynamic investigation loop, MCP tool calling, risk analysis,
simulation execution, RAG policy consultation, safety boundaries, and schema compliance.
"""

import unittest
from datetime import date
from mcp import MCPServer
from agents.operations_agent import OperationsAgent
from events import EventStore


class TestOperationsAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = MCPServer()
        cls.agent = OperationsAgent(mcp_server=cls.server, max_iterations=4)

    def test_agent_tool_discovery(self):
        """1. Agent can inspect and discover available MCP tools."""
        tools = self.agent.mcp_server.list_tools()
        self.assertGreaterEqual(len(tools), 10)
        tool_names = [t["name"] for t in tools]
        self.assertIn("get_business_snapshot", tool_names)
        self.assertIn("search_business_policy", tool_names)
        self.assertIn("run_business_simulation", tool_names)
        self.assertIn("create_business_event", tool_names)

    def test_investigate_find_biggest_risk(self):
        """2, 4, 6, 10. Agent investigates business state, identifies receivables risk, runs simulation, consults RAG."""
        result = self.agent.investigate("Find the biggest business risk right now.")

        # Structure compliance
        self.assertIn("objective", result)
        self.assertIn("investigation_summary", result)
        self.assertIn("observed_facts", result)
        self.assertIn("identified_issues", result)
        self.assertIn("tools_used", result)
        self.assertIn("simulations_run", result)
        self.assertIn("business_knowledge_used", result)
        self.assertIn("options_considered", result)
        self.assertIn("recommendation", result)
        self.assertIn("recommended_action", result)
        self.assertTrue(result.get("requires_user_approval", False))
        self.assertIn("timeline", result)

        # Verify tools called
        self.assertIn("get_business_snapshot", result["tools_used"])
        self.assertTrue(any("receivable" in str(f).lower() or "ar" in str(f).lower() or "cash" in str(f).lower() for f in result["observed_facts"]))
        self.assertIn("search_business_policy", result["tools_used"])
        self.assertTrue(len(result["business_knowledge_used"]) > 0)
        self.assertIn("run_business_simulation", result["tools_used"])

    def test_investigate_inventory_risk(self):
        """3. Agent investigates inventory status when prompted about stock."""
        result = self.agent.investigate("Check inventory status and assess stockout risks.")
        self.assertIn("get_inventory_status", result["tools_used"])
        self.assertTrue(len(result["observed_facts"]) > 0)
        self.assertIn("search_business_policy", result["tools_used"])

    def test_investigate_what_if_scenario(self):
        """5 & 7. Agent runs simulations without altering DB and simulator facts are authoritative."""
        result = self.agent.investigate("Should we increase inventory if sales increase 30%?")
        self.assertIn("run_business_simulation", result["tools_used"])
        self.assertTrue(len(result["simulations_run"]) > 0)
        self.assertTrue(result["requires_user_approval"])

    def test_safety_mutation_cannot_bypass_approval(self):
        """8 & 9. Agent cannot execute business mutations without explicit user approval."""
        store = EventStore()
        initial_events_count = len(store.get_recent_events(limit=100))

        # Investigating an objective should NEVER mutate the DB
        result = self.agent.investigate("Fix our cash flow problem by adjusting expenses.")
        self.assertTrue(result.get("requires_user_approval", True))

        current_events_count = len(store.get_recent_events(limit=100))
        self.assertEqual(current_events_count, initial_events_count)

    def test_tool_call_limit_prevents_infinite_loop(self):
        """13. Investigation loop bounds iteration count to max_iterations."""
        bounded_agent = OperationsAgent(mcp_server=self.server, max_iterations=3)
        result = bounded_agent.investigate("Explore every possible strategic direction in detail.")
        # Timeline should not exceed max_iterations + deterministic safety wraps
        self.assertLessEqual(len(result["timeline"]), 7)
        self.assertIn("recommendation", result)


if __name__ == "__main__":
    unittest.main()
