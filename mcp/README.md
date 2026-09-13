# Model Context Protocol (MCP) - Business Digital Twin Server

This package implements the **Model Context Protocol (MCP)** interface for the Acme Retail Pvt. Ltd. Business Digital Twin. It provides an autonomous Operations Agent and external AI tools with safe, controlled access to business state observation, forward-looking simulations, company policy knowledge (RAG), and human-in-the-loop transactional event processing.

---

## 1. Architectural Safeguards

- **No Raw SQL**: Agents cannot write or execute arbitrary SQL queries against PostgreSQL.
- **No Arbitrary Code Execution**: No `eval`, `exec`, or generic shell command execution.
- **Read-Only / Simulation Isolation**: Observational tools and simulation tools are completely non-mutating. Simulations run deterministically in memory.
- **Human-in-the-Loop Action Gating**: `create_business_event()` enforces explicit human approval (`approved: true`). Without approval, it generates a non-executing proposal preview.
- **Transactional Audit Log**: All executed events pass through the transactional `EventProcessor` and are recorded in `business_events_log`.

---

## 2. Available MCP Tools

### Read / Observation Tools
| Tool Name | Description | Return Values |
| :--- | :--- | :--- |
| `get_business_snapshot` | Authoritative holistic state from Digital Twin | Revenue, COGS, margins, cash, AR, inventory, risk score |
| `get_inventory_status` | Detailed SKU stock levels & reorder alerts | Total units/value, items at risk (`stock <= reorder_level`) |
| `get_customer_exposure` | Accounts receivable concentration & credit limits | Pending invoices, customer exposure percentages |
| `get_cash_position` | Liquidity breakdown & operational cash flow | Opening cash, collections, OpEx outflows, current cash |
| `get_payment_metrics` | Customer payment behavior & collection velocity | DSO, average payment delay days, terms |
| `get_sales_summary` | Historical sales volume & top revenue items | Transactions, units sold, top 5 revenue products |

### Knowledge Tools (RAG)
| Tool Name | Description | Return Values |
| :--- | :--- | :--- |
| `search_business_policy` | Semantic search over company policies | Relevant policy chunks, source attribution, similarity score |

### Simulation Tools
| Tool Name | Description | Return Values |
| :--- | :--- | :--- |
| `run_business_simulation` | Deterministic what-if simulation (0 DB changes) | Projected revenue, margin, cash runway, inventory risk |
| `compare_scenarios` | Side-by-side comparative simulation of options | Comparative matrix of revenue, cash, margin, and risk |

### Action Tools (Gated)
| Tool Name | Description | Safety Gate |
| :--- | :--- | :--- |
| `create_business_event` | Submits event to PostgreSQL via EventProcessor | Requires `approved: true` parameter |

---

## 3. Usage Examples

### In-Process Python Client
```python
from mcp.server import MCPServer

server = MCPServer()

# 1. List available tools
tools = server.list_tools()

# 2. Get business snapshot
snapshot = server.call_tool("get_business_snapshot")

# 3. Simulate sales growth with inventory adjustment
sim = server.call_tool("run_business_simulation", {
    "sales_growth": 0.3,
    "inventory_change": 0.1
})

# 4. Search policy
policy = server.call_tool("search_business_policy", {
    "query": "inventory safety stock and reorder rules"
})
```

### Running Over Stdio (Standard MCP)
```bash
python -m mcp.server
```
Clients can send line-delimited JSON-RPC 2.0 messages:
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "get_business_snapshot", "arguments": {}}}
```
