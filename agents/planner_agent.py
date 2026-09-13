import json
import re
from dataclasses import asdict
from typing import Any

import requests

from backend.digital_twin.simulation import Scenario


class PlannerAgent:
    """
    Converts a natural-language business question into
    a validated Scenario object.

    The Planner Agent does NOT perform financial calculations.
    """

    def __init__(
        self,
        model: str = "qwen3:4b-instruct-2507-q4_K_M",
        ollama_url: str = "http://localhost:11434/api/generate",
    ):
        self.model = model
        self.ollama_url = ollama_url

    # ---------------------------------------------------------
    # SYSTEM PROMPT
    # ---------------------------------------------------------

    SYSTEM_PROMPT = """
You are the Planner Agent for a Business Digital Twin.

Your ONLY responsibility is to convert a user's natural-language
business scenario into structured simulation parameters.

You MUST NOT:
- calculate revenue
- calculate profit
- calculate cash
- calculate AR
- calculate inventory
- predict financial results
- invent business data
- provide financial recommendations

You ONLY extract scenario parameters.

The available parameters are:

sales_growth
price_change
cost_change
payment_delay_days
inventory_change
expense_change
supplier_payment_delay_days
months

Interpret percentages as decimal values.

Examples:

"sales increase by 30%"
=> sales_growth = 0.30

"sales decrease by 10%"
=> sales_growth = -0.10

"increase prices by 5%"
=> price_change = 0.05

"reduce prices by 10%"
=> price_change = -0.10

"costs increase by 15%"
=> cost_change = 0.15

"customers pay 20 days late"
=> payment_delay_days = 20

"customers take 15 extra days to pay"
=> payment_delay_days = 15

"reduce inventory by 20%"
=> inventory_change = -0.20

"increase inventory by 30%"
=> inventory_change = 0.30

"reduce operating expenses by 10%"
=> expense_change = -0.10

"increase operating expenses by 15%"
=> expense_change = 0.15

"delay supplier payments by 30 days"
=> supplier_payment_delay_days = 30

"simulate for 24 months"
=> months = 24

DEFAULTS:

sales_growth = 0.0
price_change = 0.0
cost_change = 0.0
payment_delay_days = 0.0
inventory_change = 0.0
expense_change = 0.0
supplier_payment_delay_days = 0.0
months = 12

IMPORTANT:

If a parameter is not mentioned, use its default.

Return ONLY valid JSON.

The JSON must contain exactly these fields:

{
  "sales_growth": number,
  "price_change": number,
  "cost_change": number,
  "payment_delay_days": number,
  "inventory_change": number,
  "expense_change": number,
  "supplier_payment_delay_days": number,
  "months": integer,
  "confidence": number,
  "assumptions": array,
  "interpretation": string
}

confidence must be between 0 and 1.

Do not put currency calculations in the response.

Do not add markdown.

Do not add explanations outside the JSON.
"""

    # ---------------------------------------------------------
    # PUBLIC METHOD
    # ---------------------------------------------------------

    def plan(self, user_query: str) -> dict[str, Any]:
        """
        Convert natural-language query into a validated plan.
        """

        if not user_query or not user_query.strip():
            raise ValueError("Business scenario cannot be empty.")

        raw = self._call_ollama(user_query)

        parsed = self._parse_json(raw)

        validated = self._validate(parsed)

        return validated

    # ---------------------------------------------------------
    # OLLAMA
    # ---------------------------------------------------------

    def _call_ollama(self, user_query: str) -> str:

        prompt = f"""
{self.SYSTEM_PROMPT}

USER BUSINESS QUESTION:

{user_query}

Return ONLY the JSON object.
"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0
            }
        }

        try:

            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=120
            )

            response.raise_for_status()

        except requests.RequestException as exc:

            raise RuntimeError(
                f"Unable to connect to Ollama at {self.ollama_url}. "
                f"Make sure Ollama is running and the model "
                f"'{self.model}' is available."
            ) from exc

        data = response.json()

        if "response" not in data:
            raise RuntimeError(
                "Ollama returned an unexpected response."
            )

        return data["response"]

    # ---------------------------------------------------------
    # JSON PARSER
    # ---------------------------------------------------------

    def _parse_json(self, raw: str) -> dict:

        raw = raw.strip()

        # First attempt: direct JSON
        try:
            return json.loads(raw)

        except json.JSONDecodeError:
            pass

        # Fallback: extract first JSON object
        match = re.search(
            r"\{.*\}",
            raw,
            flags=re.DOTALL
        )

        if not match:
            raise ValueError(
                "Planner Agent did not return valid JSON."
            )

        try:

            return json.loads(match.group(0))

        except json.JSONDecodeError as exc:

            raise ValueError(
                "Planner Agent returned malformed JSON."
            ) from exc

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    def _validate(self, data: dict) -> dict:

        required_fields = [
            "sales_growth",
            "price_change",
            "cost_change",
            "payment_delay_days",
            "inventory_change",
            "expense_change",
            "supplier_payment_delay_days",
            "months",
        ]

        defaults = {
            "sales_growth": 0.0,
            "price_change": 0.0,
            "cost_change": 0.0,
            "payment_delay_days": 0.0,
            "inventory_change": 0.0,
            "expense_change": 0.0,
            "supplier_payment_delay_days": 0.0,
            "months": 12,
        }

        # Apply defaults
        for field in required_fields:

            if field not in data or data[field] is None:
                data[field] = defaults[field]

        # Convert numeric values
        numeric_fields = [
            "sales_growth",
            "price_change",
            "cost_change",
            "payment_delay_days",
            "inventory_change",
            "expense_change",
            "supplier_payment_delay_days",
        ]

        for field in numeric_fields:

            try:
                data[field] = float(data[field])

            except (TypeError, ValueError):

                raise ValueError(
                    f"Invalid value for {field}: {data[field]}"
                )

        # Months must be integer
        try:
            data["months"] = int(data["months"])

        except (TypeError, ValueError):

            raise ValueError(
                f"Invalid simulation horizon: {data['months']}"
            )

        # -----------------------------------------------------
        # Safety / sanity bounds
        # -----------------------------------------------------

        if not -0.90 <= data["sales_growth"] <= 5.0:

            raise ValueError(
                "Sales growth must be between -90% and +500%."
            )

        if not -0.90 <= data["price_change"] <= 5.0:

            raise ValueError(
                "Price change must be between -90% and +500%."
            )

        if not -0.90 <= data["cost_change"] <= 5.0:

            raise ValueError(
                "Cost change must be between -90% and +500%."
            )

        if not -0.90 <= data["inventory_change"] <= 5.0:

            raise ValueError(
                "Inventory change must be between -90% and +500%."
            )

        if not -0.90 <= data["expense_change"] <= 5.0:

            raise ValueError(
                "Expense change must be between -90% and +500%."
            )

        if data["payment_delay_days"] < 0:

            raise ValueError(
                "Customer payment delay cannot be negative."
            )

        if data["supplier_payment_delay_days"] < 0:

            raise ValueError(
                "Supplier payment delay cannot be negative."
            )

        if not 1 <= data["months"] <= 60:

            raise ValueError(
                "Simulation horizon must be between 1 and 60 months."
            )

        # Confidence
        confidence = float(
            data.get("confidence", 0.0)
        )

        data["confidence"] = max(
            0.0,
            min(1.0, confidence)
        )

        # Assumptions
        assumptions = data.get("assumptions", [])

        if not isinstance(assumptions, list):
            assumptions = []

        data["assumptions"] = [
            str(item) for item in assumptions
        ]

        data["interpretation"] = str(
            data.get(
                "interpretation",
                "Scenario extracted from user request."
            )
        )

        return data

    # ---------------------------------------------------------
    # SCENARIO OBJECT
    # ---------------------------------------------------------

    def to_scenario(self, plan: dict) -> Scenario:

        return Scenario(
            sales_growth=plan["sales_growth"],
            price_change=plan["price_change"],
            cost_change=plan["cost_change"],
            payment_delay_days=plan["payment_delay_days"],
            inventory_change=plan["inventory_change"],
            expense_change=plan["expense_change"],
            supplier_payment_delay_days=plan[
                "supplier_payment_delay_days"
            ],
            months=plan["months"],
        )

    # ---------------------------------------------------------
    # DEBUG / DISPLAY
    # ---------------------------------------------------------

    def explain_plan(self, plan: dict) -> str:

        return (
            f"Sales Growth: "
            f"{plan['sales_growth'] * 100:.1f}%\n"
            f"Price Change: "
            f"{plan['price_change'] * 100:.1f}%\n"
            f"Cost Change: "
            f"{plan['cost_change'] * 100:.1f}%\n"
            f"Customer Payment Delay: "
            f"{plan['payment_delay_days']:.0f} days\n"
            f"Inventory Policy: "
            f"{plan['inventory_change'] * 100:.1f}%\n"
            f"Operating Expense Change: "
            f"{plan['expense_change'] * 100:.1f}%\n"
            f"Supplier Payment Delay: "
            f"{plan['supplier_payment_delay_days']:.0f} days\n"
            f"Horizon: "
            f"{plan['months']} months"
        )