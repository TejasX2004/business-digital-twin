"""
MCP Tool Schemas for Business Digital Twin.
Defines explicit JSON schemas for tool definitions and parameters adhering to the Model Context Protocol (MCP).
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Tool Definitions with MCP-compliant JSON Schema
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    # --- READ / OBSERVATION TOOLS ---
    {
        "name": "get_business_snapshot",
        "description": (
            "Retrieves an authoritative holistic snapshot of the business from the Digital Twin, "
            "including revenue, gross/net margins, operating expenses, current cash, accounts receivable, "
            "inventory value/units, and overall risk score/level."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_inventory_status",
        "description": (
            "Retrieves detailed inventory metrics and stock levels across product catalog. "
            "Identifies products at risk (stock <= reorder level), current inventory value, "
            "inventory turnover ratio, and estimated inventory days."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_customer_exposure",
        "description": (
            "Retrieves accounts receivable customer concentration and payment exposures. "
            "Identifies customers with pending or overdue invoices, credit terms, and total outstanding balances."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_cash_position",
        "description": (
            "Retrieves current cash and liquidity breakdown: opening cash, total collections received, "
            "operating expenses disbursements, ending cash balance, and operating cash flow."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_payment_metrics",
        "description": (
            "Retrieves customer payment behavior metrics, including historical average payment delay days, "
            "standard payment terms, and Days Sales Outstanding (DSO)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_sales_summary",
        "description": (
            "Retrieves historical sales volume, transaction counts, average order values, and revenue contribution."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },

    # --- KNOWLEDGE TOOLS ---
    {
        "name": "search_business_policy",
        "description": (
            "Searches authoritative company policy documents via the RAG Knowledge Layer. "
            "Returns relevant policy chunks for inventory coverage, credit/payment terms, supplier rules, "
            "expense authorizations, and business benchmarks with similarity scores."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Business policy question or topic to search (e.g., 'inventory safety stock policy').",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top matching policy excerpts to return (default 3).",
                    "default": 3,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },

    # --- SIMULATION TOOLS ---
    {
        "name": "run_business_simulation",
        "description": (
            "Runs a deterministic forward-looking what-if simulation against the Digital Twin without altering database state. "
            "Simulates the financial, cash flow, working capital, inventory risk, and cash runway impacts of scenario adjustments."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sales_growth": {
                    "type": "number",
                    "description": "Projected sales growth as a decimal (e.g. 0.3 for +30%, -0.2 for -20%). Default 0.0.",
                    "default": 0.0,
                },
                "price_change": {
                    "type": "number",
                    "description": "Projected product price change as a decimal (e.g. 0.1 for +10%). Default 0.0.",
                    "default": 0.0,
                },
                "cost_change": {
                    "type": "number",
                    "description": "Projected COGS/cost change as a decimal (e.g. 0.1 for +10%). Default 0.0.",
                    "default": 0.0,
                },
                "expense_change": {
                    "type": "number",
                    "description": "Projected operating expense change as a decimal (e.g. 0.1 for +10%). Default 0.0.",
                    "default": 0.0,
                },
                "payment_delay_days": {
                    "type": "number",
                    "description": "Additional customer payment delay in days (e.g. 15.0 for +15 days). Default 0.0.",
                    "default": 0.0,
                },
                "inventory_change": {
                    "type": "number",
                    "description": "Projected inventory stock change as a decimal (e.g. 0.2 for +20%). Default 0.0.",
                    "default": 0.0,
                },
                "supplier_payment_delay_days": {
                    "type": "number",
                    "description": "Additional supplier payment delay in days (e.g. 30.0 for +30 days). Default 0.0.",
                    "default": 0.0,
                },
                "months": {
                    "type": "integer",
                    "description": "Simulation horizon in months (default 12).",
                    "default": 12,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_scenarios",
        "description": (
            "Simulates and compares multiple what-if scenarios side-by-side, summarizing revenue, ending cash, "
            "gross margin, ending inventory, inventory risk, and overall risk level across options."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "scenarios": {
                    "type": "array",
                    "description": "List of scenario specification dicts (with 'name' and scenario parameter fields).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "sales_growth": {"type": "number"},
                            "price_change": {"type": "number"},
                            "cost_change": {"type": "number"},
                            "expense_change": {"type": "number"},
                            "payment_delay_days": {"type": "number"},
                            "inventory_change": {"type": "number"},
                            "supplier_payment_delay_days": {"type": "number"},
                            "months": {"type": "integer"},
                        },
                        "required": ["name"],
                    },
                },
            },
            "required": ["scenarios"],
            "additionalProperties": False,
        },
    },

    # --- ACTION TOOL ---
    {
        "name": "create_business_event",
        "description": (
            "Gated action interface: Submits a business event to the Event System for execution against PostgreSQL. "
            "CRITICAL: Requires explicit user approval ('approved': true). If approved=false, returns validation preview only without mutating state."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "event_type": {
                    "type": "string",
                    "enum": [
                        "SALE_CREATED",
                        "PAYMENT_RECEIVED",
                        "INVENTORY_UPDATED",
                        "EXPENSE_RECORDED",
                        "SUPPLIER_PAYMENT_UPDATED",
                        "RESTRICT_CREDIT",
                        "INITIATE_CREDIT_REVIEW",
                    ],
                    "description": "Type of business event to create.",
                },
                "payload": {
                    "type": "object",
                    "description": "Event-specific parameters according to domain schema.",
                },
                "approved": {
                    "type": "boolean",
                    "description": "Must be true to execute mutation in PostgreSQL. False returns a preview.",
                    "default": False,
                },
            },
            "required": ["event_type", "payload"],
            "additionalProperties": False,
        },
    },
]
