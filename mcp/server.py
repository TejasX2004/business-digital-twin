"""
MCP Server for Business Digital Twin.
Provides standard Model Context Protocol (MCP) JSON-RPC 2.0 interface
and direct in-process tool dispatch for autonomous agents.
"""

import json
import sys
from typing import Any, Dict, List, Optional

from mcp.schemas import TOOL_DEFINITIONS
from mcp.tools import MCPBusinessTools


class MCPServer:
    """
    Model Context Protocol (MCP) server managing tool registration,
    schema validation, and execution.
    """

    SERVER_INFO = {
        "name": "acme-business-control-tower-mcp",
        "version": "1.0.0",
    }

    def __init__(self):
        self.tools = MCPBusinessTools()
        self._handlers = {
            "get_business_snapshot": lambda args: self.tools.get_business_snapshot(),
            "get_inventory_status": lambda args: self.tools.get_inventory_status(),
            "get_customer_exposure": lambda args: self.tools.get_customer_exposure(),
            "get_cash_position": lambda args: self.tools.get_cash_position(),
            "get_payment_metrics": lambda args: self.tools.get_payment_metrics(),
            "get_sales_summary": lambda args: self.tools.get_sales_summary(),
            "search_business_policy": lambda args: self.tools.search_business_policy(
                query=args.get("query", ""),
                top_k=int(args.get("top_k", 3)),
            ),
            "run_business_simulation": lambda args: self.tools.run_business_simulation(**args),
            "compare_scenarios": lambda args: self.tools.compare_scenarios(
                scenarios=args.get("scenarios", [])
            ),
            "create_business_event": lambda args: self.tools.create_business_event(
                event_type=args.get("event_type", ""),
                payload=args.get("payload", {}),
                approved=bool(args.get("approved", False)),
            ),
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns the list of available MCP tools with their schemas."""
        return TOOL_DEFINITIONS

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes an MCP tool by name with arguments.
        Returns a standard MCP tool call result object:
        {
            "content": [{"type": "text", "text": "..."}],
            "isError": bool
        }
        """
        args = arguments or {}
        if name not in self._handlers:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: Tool '{name}' not found. Available tools: {list(self._handlers.keys())}",
                    }
                ],
                "isError": True,
            }

        authority_map = {
            "get_business_snapshot": "AUTHORITATIVE",
            "get_inventory_status": "AUTHORITATIVE",
            "get_customer_exposure": "AUTHORITATIVE",
            "get_cash_position": "AUTHORITATIVE",
            "get_payment_metrics": "AUTHORITATIVE",
            "get_sales_summary": "AUTHORITATIVE",
            "search_business_policy": "RETRIEVED_KNOWLEDGE",
            "run_business_simulation": "SIMULATED",
            "compare_scenarios": "SIMULATED",
            "create_business_event": "AUTHORITATIVE",
        }

        try:
            handler = self._handlers[name]
            raw_result = handler(args)
            data_authority = authority_map.get(name, "AUTHORITATIVE")

            # Stamp data authority if result is dict
            if isinstance(raw_result, dict) and "data_authority" not in raw_result:
                raw_result["data_authority"] = data_authority
            elif isinstance(raw_result, list):
                for item in raw_result:
                    if isinstance(item, dict) and "data_authority" not in item:
                        item["data_authority"] = data_authority

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(raw_result, indent=2, default=str),
                    }
                ],
                "structured_data": raw_result,
                "data_authority": data_authority,
                "isError": False,
            }
        except Exception as exc:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Execution error in tool '{name}': {str(exc)}",
                    }
                ],
                "isError": True,
            }

    def handle_jsonrpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handles a single JSON-RPC 2.0 MCP request."""
        method = request.get("method")
        msg_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": self.SERVER_INFO,
                },
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": self.list_tools()},
            }

        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            call_res = self.call_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": call_res,
            }

        elif method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {},
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not found.",
                },
            }

    def run_stdio(self) -> None:
        """Runs the MCP server over standard input/output (line-delimited JSON-RPC)."""
        sys.stderr.write(f"Starting {self.SERVER_INFO['name']} on stdio...\n")
        sys.stderr.flush()

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = self.handle_jsonrpc(req)
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
            except json.JSONDecodeError:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


def main():
    server = MCPServer()
    server.run_stdio()


if __name__ == "__main__":
    main()
