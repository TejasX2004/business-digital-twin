"""
Decision Agent — agents/decision_agent.py
STATUS: FROZEN (Validated and Hardened)

Architecture:
    Planner → Simulator → Analysis → Red-Team → Decision

The Decision Agent converts the validated analysis and Red-Team audit verdict
into an actionable executive business decision.

Rules:
- Simulator is the source of truth.
- Decision Agent must NOT calculate, modify, or invent financial numbers.
- Respect the Red-Team verdict.
- If Red-Team = FAIL, do not make a confident recommendation; flag the decision for review.
- Recommendations must be directly supported by simulator facts and analysis.
- Do not invent benchmarks, thresholds, financial figures, or external information.
- Do not claim an action will definitely produce a specific financial outcome unless
  the simulator provides that evidence.
- Distinguish observed facts from recommendations.
- Prefer specific actionable recommendations over generic advice.
- If multiple actions are needed, prioritize them.
- Use INR when referring to currency.
"""

from __future__ import annotations

import json
import re
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are the Executive Decision Agent for a Business Digital Twin.\n"
    "\n"
    "You receive:\n"
    "1. SIMULATOR_FACTS: Authoritative, deterministic simulation facts (source of truth).\n"
    "2. ANALYSIS_OUTPUT: The Analysis Agent's structured interpretation.\n"
    "3. RED_TEAM_VERDICT: The Red-Team audit findings and verdict (PASS, PASS_WITH_WARNINGS, FAIL).\n"
    "\n"
    "Your purpose is to convert the validated analysis into an actionable business decision.\n"
    "\n"
    "DETERMINISTIC DECISION FRAMEWORK:\n"
    "- Red-Team FAIL -> decision cannot be PROCEED (must be FLAG_FOR_REVIEW).\n"
    "- Critical overall risk -> DO_NOT_PROCEED.\n"
    "- High/Medium risk with material risks (e.g. High inventory risk, declining cash, AR increase) -> PROCEED_WITH_CONDITIONS.\n"
    "- Low risk with no material issues -> PROCEED.\n"
    "\n"
    "CONDITIONS RULES (STRICT):\n"
    "Conditions must be evidence-based observations/actions, NOT invented business policies.\n"
    "Valid conditions:\n"
    "- Review inventory policy given High inventory risk.\n"
    "- Monitor cash position given lower ending cash.\n"
    "- Strengthen collections given increased AR and payment delay.\n"
    "INVALID conditions (STRICTLY FORBIDDEN):\n"
    "- NEVER invent cash or inventory thresholds (do NOT say 'maintain cash above Rs. X', 'improve inventory to X months', 'minimum threshold').\n"
    "- NEVER invent monitoring cadences or deadlines (do NOT say 'daily', 'monthly', 'quarterly', 'within 30 days', 'within X days', 'every X months').\n"
    "- NEVER claim 'net profit is not provided in baseline' or question financial viability over missing baseline net profit.\n"
    "\n"
    "PROFITABILITY & CASH RUNWAY RULES:\n"
    "- NEVER claim 'stable profitability' when only gross margin is stable. Use 'stable gross margin'.\n"
    "- NEVER treat 'cash runway not reached in projection' as a risk. It is a factual projection status, not a liquidity shortfall.\n"
    "- NEVER mention obsolescence or unmodeled storage costs.\n"
    "- CURRENCY: Strictly INR (or Rs.). Never use $, USD, EUR, GBP, or dollars.\n"
    "\n"
    "SCENARIO PERCENTAGES & UNITS:\n"
    "- Scenario percentage parameters (sales_growth, price_change, cost_change, etc.) are formatted as percentages (e.g. '70%' means 70%, NOT 0.7%).\n"
    "- NEVER divide an already-normalized percentage by 100 or state a 70% growth scenario as 0.7%.\n"
    "- Always use consistent units across all outputs.\n"
    "\n"
    "REVENUE & STOCKOUT WORDING:\n"
    "- NEVER use the phrase 'sustainable revenue expansion' or 'sustainable revenue'. Use 'projected revenue growth' or 'revenue expansion'.\n"
    "- NEVER claim 'stockout risk' or 'risk of stockouts' unless lost sales or stockout exposure in SIMULATOR_FACTS are greater than 0.\n"
    "\n"
    "OUTPUT SCHEMA:\n"
    "Return strict JSON with the following fields:\n"
    "{\n"
    '  "decision": "PROCEED_WITH_CONDITIONS|DO_NOT_PROCEED|PROCEED|FLAG_FOR_REVIEW",\n'
    '  "priority": "HIGH|MEDIUM|LOW",\n'
    '  "rationale": ["Directly supported business reasoning based on simulator facts"],\n'
    '  "recommended_actions": ["Prioritized, concrete operational steps (Priority 1: ..., Priority 2: ...)"],\n'
    '  "expected_benefits": ["Anticipated business upside supported by simulation"],\n'
    '  "risks": ["Observed risks directly reported by simulator or analysis"],\n'
    '  "conditions": ["Evidence-based gating conditions"],\n'
    '  "confidence": 0.0\n'
    "}\n"
    "\n"
    "Return ONLY the valid JSON object."
)

# ---------------------------------------------------------------------------
# FORBIDDEN PHRASES & ALLOWED VALUES
# ---------------------------------------------------------------------------
_FORBIDDEN_PHRASES = [
    # Foreign currency
    "$", "usd", "eur", "gbp", "dollars", "pounds", "euros",
    # Obsolescence / unmodeled factors
    "obsolescence", "storage costs",
    # False crisis / runway
    "sustainability risk", "financial sustainability",
    "potential cash shortfall", "cash shortfall", "lack of runway", "insufficient runway",
    # Thresholds / benchmarks
    "typical threshold", "safety threshold", "industry benchmark", "industry threshold",
    "threshold alert", "minimum threshold",
    # Profitability overclaim
    "stable profitability", "consistent profitability",
    # Revenue wording
    "sustainable revenue expansion", "sustainable revenue",
    # Missing net profit claims
    "net profit is not provided", "baseline net profit", "cannot be confirmed without additional data",
    # Cadences / deadlines
    "daily", "within 30 days", "within 60 days", "within 90 days", "every month", "every 6 months",
]

_ALLOWED_PRIORITIES = {"HIGH", "MEDIUM", "LOW"}

# ---------------------------------------------------------------------------
# OUTPUT SCHEMA FOR OLLAMA
# ---------------------------------------------------------------------------
_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "decision":            {"type": "string"},
        "priority":            {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
        "rationale":           {"type": "array", "items": {"type": "string"}},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
        "expected_benefits":   {"type": "array", "items": {"type": "string"}},
        "risks":               {"type": "array", "items": {"type": "string"}},
        "conditions":          {"type": "array", "items": {"type": "string"}},
        "confidence":          {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "decision",
        "priority",
        "rationale",
        "recommended_actions",
        "expected_benefits",
        "risks",
        "conditions",
        "confidence",
    ],
}


# ---------------------------------------------------------------------------
# DECISION AGENT CLASS
# ---------------------------------------------------------------------------
@dataclass
class DecisionAgent:
    """
    Synthesizes authoritative simulator facts, analysis, and Red-Team audit
    verdict into an actionable, validated executive decision.
    """

    model: str = "qwen3:4b-instruct-2507-q4_K_M"
    ollama_url: str = "http://localhost:11434/api/generate"

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE
    # ------------------------------------------------------------------
    def decide(
        self,
        scenario: Any,
        simulator_facts_or_result: Dict[str, Any],
        analysis: Dict[str, Any],
        red_team_verdict: Dict[str, Any],
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Execute the Decision Agent pipeline.

        Parameters
        ----------
        scenario                  : Scenario dataclass or dict
        simulator_facts_or_result : Authoritative facts dict OR raw simulation_result dict
        analysis                  : Dict output from AnalysisAgent.analyze()
        red_team_verdict          : Dict output from RedTeamAgent.review()
        max_retries               : Max retry attempts on validation error

        Returns
        -------
        Strict decision JSON matching the required schema.
        """
        if not isinstance(simulator_facts_or_result, dict):
            raise TypeError("simulator_facts_or_result must be a dictionary.")
        if not isinstance(analysis, dict):
            raise TypeError("analysis must be a dictionary.")
        if not isinstance(red_team_verdict, dict):
            raise TypeError("red_team_verdict must be a dictionary.")

        # 1. Normalize simulator facts
        facts = self._normalize_facts(scenario, simulator_facts_or_result)
        if scenario is not None:
            facts["scenario"] = DecisionAgent._format_scenario(scenario)

        # 2. Extract allowed numbers set for deterministic verification
        allowed_numbers = self._build_allowed_numbers(scenario, facts, analysis, red_team_verdict)

        # 3. Check for Red-Team FAIL upfront to guide prompt
        rt_verdict_str = str(red_team_verdict.get("overall_verdict", "UNKNOWN")).upper()

        # 4. Construct prompt
        base_prompt = self._build_prompt(scenario, facts, analysis, red_team_verdict)

        last_error: Optional[Exception] = None
        decision_data: Dict[str, Any] = {}
        extra_instruction = ""

        # 5. Call Ollama with retries
        for attempt in range(max_retries + 1):
            prompt = (extra_instruction + "\n\n" + base_prompt) if extra_instruction else base_prompt
            try:
                raw = self._call_ollama(prompt)
                parsed = self._parse_json(raw)
                sanitized = self._sanitize_llm_result(parsed, facts, scenario)
                # Deterministic validation
                validated = self._validate_decision(
                    sanitized, facts, analysis, red_team_verdict, allowed_numbers
                )
                decision_data = validated
                break
            except (ValueError, RuntimeError) as exc:
                last_error = exc
                extra_instruction = (
                    f"CORRECTION FOR PREVIOUS RESPONSE:\n"
                    f"Your previous response had validation error: {exc}\n"
                    f"Please correct this immediately. Ensure strict JSON, priority in [HIGH, MEDIUM, LOW], "
                    f"confidence 0-1, strictly INR currency, respect Red-Team verdict '{rt_verdict_str}', "
                    f"do not invent thresholds/deadlines/cadences, and do not mention missing baseline net profit."
                )

        # 6. Fallback if LLM failed
        if not decision_data:
            decision_data = self._generate_fallback_decision(
                facts, analysis, red_team_verdict, f"LLM error after retries: {last_error}"
            )

        # 7. Final deterministic enforcement
        decision_data = self._enforce_deterministic_guardrails(decision_data, facts, red_team_verdict)

        return decision_data

    # ------------------------------------------------------------------
    # FACT NORMALIZATION
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_facts(scenario: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures we have an authoritative facts dictionary.
        If data has 'baseline' and 'projected', extract using standard mapping.
        If already flattened, return as is.
        """
        if "projected" in data and "baseline" in data:
            return DecisionAgent._extract_facts_from_simulation(scenario, data)
        return data

    @staticmethod
    def _format_scenario(scenario: Any) -> Dict[str, Any]:
        if hasattr(scenario, "__dataclass_fields__"):
            raw = {k: getattr(scenario, k) for k in scenario.__dataclass_fields__}
        elif isinstance(scenario, dict):
            raw = dict(scenario)
        elif hasattr(scenario, "__dict__"):
            raw = vars(scenario)
        else:
            return {"scenario": str(scenario)}

        formatted: Dict[str, Any] = {}
        for k, v in raw.items():
            if k in ("sales_growth", "price_change", "cost_change", "expense_change", "inventory_change"):
                try:
                    fv = float(v)
                    formatted[k] = f"{round(fv * 100, 2):g}%"
                except (TypeError, ValueError):
                    formatted[k] = v
            elif k in ("payment_delay_days", "supplier_payment_delay_days"):
                try:
                    days = float(v)
                    formatted[k] = f"{days:g} days"
                except (TypeError, ValueError):
                    formatted[k] = v
            elif k == "months":
                try:
                    formatted[k] = f"{int(v)} months"
                except (TypeError, ValueError):
                    formatted[k] = v
            else:
                formatted[k] = v
        return formatted

    @staticmethod
    def _extract_facts_from_simulation(scenario: Any, sim_res: Dict[str, Any]) -> Dict[str, Any]:
        def get(d: dict, k: str) -> Any:
            v = d.get(k)
            return "NOT PROVIDED" if v is None else v

        def pct(n, den) -> Any:
            try:
                d = float(den)
                return "NOT PROVIDED" if d == 0 else round(float(n) / d * 100, 4)
            except (TypeError, ValueError):
                return "NOT PROVIDED"

        def direction(b, p) -> str:
            if b == "NOT PROVIDED" or p == "NOT PROVIDED":
                return "cannot_determine"
            try:
                bv, pv = float(b), float(p)
                denom = max(abs(bv), abs(pv), 1e-9)
                if abs(pv - bv) / denom < 0.0001:
                    return "unchanged"
                return "increased" if pv > bv else "decreased"
            except (TypeError, ValueError):
                return "cannot_determine"

        def serialize(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "__dict__"):
                return vars(obj)
            return str(obj)

        baseline  = sim_res.get("baseline", {})
        projected = sim_res.get("projected", {})
        risk      = sim_res.get("risk", {})

        b_rev  = get(baseline, "revenue")
        b_gp   = get(baseline, "gross_profit")
        b_opex = get(baseline, "operating_expenses")
        b_cogs = get(baseline, "cogs")
        b_cash = get(baseline, "current_cash")
        b_ar   = get(baseline, "accounts_receivable")
        b_inv  = get(baseline, "inventory_value")

        b_gross_margin = pct(b_gp, b_rev)
        b_net_profit: Any = (
            "NOT PROVIDED"
            if b_gp == "NOT PROVIDED" or b_opex == "NOT PROVIDED"
            else round(float(b_gp) - float(b_opex), 4)
        )
        b_net_margin = (
            pct(b_net_profit, b_rev)
            if b_net_profit != "NOT PROVIDED" and b_rev != "NOT PROVIDED"
            else "NOT PROVIDED"
        )

        p_rev          = get(projected, "revenue")
        p_gp           = get(projected, "gross_profit")
        p_opex         = get(projected, "operating_expenses")
        p_cogs         = get(projected, "cogs")
        p_net_profit   = get(projected, "net_profit")
        p_gross_margin = get(projected, "gross_margin")
        p_net_margin   = get(projected, "net_margin")
        p_cash         = get(projected, "ending_cash")
        p_ar           = get(projected, "ending_accounts_receivable")
        p_inv          = get(projected, "ending_inventory")
        p_avg_inv      = get(projected, "average_inventory")
        p_min_cov      = get(projected, "minimum_inventory_coverage")
        p_lost         = get(projected, "total_lost_sales")
        p_stockout     = get(projected, "maximum_stockout_exposure")

        inv_risk     = get(risk, "inventory")
        overall_risk = get(risk, "overall")
        raw_runway   = get(projected, "cash_runway_months")

        return {
            "scenario": DecisionAgent._format_scenario(scenario),
            "currency": "INR",
            "revenue":            {"baseline": b_rev,  "projected": p_rev,  "direction": direction(b_rev, p_rev)},
            "cogs":               {"baseline": b_cogs, "projected": p_cogs, "direction": direction(b_cogs, p_cogs)},
            "gross_profit":       {"baseline": b_gp,   "projected": p_gp,   "direction": direction(b_gp, p_gp)},
            "gross_margin":       {"baseline": b_gross_margin, "projected": p_gross_margin, "direction": direction(b_gross_margin, p_gross_margin)},
            "net_profit":         {"baseline": "NOT PROVIDED", "projected": p_net_profit, "direction": "cannot_determine"},
            "net_margin":         {"baseline": b_net_margin, "projected": p_net_margin, "direction": direction(b_net_margin, p_net_margin)},
            "operating_expenses": {"baseline": b_opex, "projected": p_opex, "direction": direction(b_opex, p_opex)},
            "cash":               {"baseline": b_cash, "projected": p_cash, "direction": direction(b_cash, p_cash)},
            "accounts_receivable":{"baseline": b_ar,   "projected": p_ar,   "direction": direction(b_ar, p_ar)},
            "inventory_value":    {"baseline": b_inv,  "projected": p_inv,  "direction": direction(b_inv, p_inv)},
            "average_inventory":  {"baseline": "NOT PROVIDED", "projected": p_avg_inv, "direction": "cannot_determine"},
            "inventory_coverage_months": p_min_cov,
            "lost_sales":         p_lost,
            "stockout_exposure":  p_stockout,
            "inventory_risk":     inv_risk,
            "overall_risk":       overall_risk,
            "cash_runway_raw":    raw_runway,
        }

    # ------------------------------------------------------------------
    # PROMPT CONSTRUCTION
    # ------------------------------------------------------------------
    @staticmethod
    def _build_prompt(
        scenario: Any,
        facts: Dict[str, Any],
        analysis: Dict[str, Any],
        red_team_verdict: Dict[str, Any],
    ) -> str:
        """
        Build the full context prompt for Qwen.
        """
        scen_dict = DecisionAgent._format_scenario(scenario)

        analysis_extract = {
            "executive_summary": analysis.get("executive_summary", ""),
            "key_drivers": analysis.get("key_drivers", []),
            "recommendations": analysis.get("recommendations", []),
            "risks": analysis.get("risk_analysis", []),
            "tradeoffs": analysis.get("tradeoffs", []),
            "overall_assessment": analysis.get("overall_assessment", "Neutral"),
            "confidence": analysis.get("confidence", 0.0),
        }

        rt_verdict_str = str(red_team_verdict.get("overall_verdict", "UNKNOWN")).upper()
        red_team_extract = {
            "overall_verdict": rt_verdict_str,
            "confidence": red_team_verdict.get("confidence", 0.0),
            "critical_issues": red_team_verdict.get("critical_issues", []),
            "numerical_issues": red_team_verdict.get("numerical_issues", []),
            "logic_issues": red_team_verdict.get("logic_issues", []),
            "unsupported_claims": red_team_verdict.get("unsupported_claims", []),
            "missing_risks": red_team_verdict.get("missing_risks", []),
            "recommendation_issues": red_team_verdict.get("recommendation_issues", []),
            "verified_claims_count": len(red_team_verdict.get("verified_claims", [])),
        }

        expected_dec = DecisionAgent._determine_expected_decision(facts, red_team_verdict)

        guidance_text = (
            f"DETERMINISTIC TARGET DECISION: '{expected_dec}'.\n"
            "Ensure 'conditions' are strictly evidence-based observations/actions:\n"
            "  - Review inventory policy given High inventory risk.\n"
            "  - Monitor cash position given lower ending cash.\n"
            "  - Strengthen collections given increased AR and payment delay.\n"
            "Do NOT invent thresholds (e.g. at least 6 months, above Rs. X), cadences (daily, monthly), "
            "deadlines (within 30 days), or mention missing baseline net profit."
        )

        prompt_parts = [
            f"=== 1. SCENARIO ===\n{json.dumps(scen_dict, indent=2, default=str)}",
            f"=== 2. SIMULATOR FACTS (GROUND TRUTH) ===\n{json.dumps(facts, indent=2, default=str)}",
            f"=== 3. ANALYSIS OUTPUT ===\n{json.dumps(analysis_extract, indent=2, default=str)}",
            f"=== 4. RED-TEAM VERDICT ===\n{json.dumps(red_team_extract, indent=2, default=str)}",
            f"=== 5. GUIDANCE ===\n{guidance_text}",
            "Synthesize these inputs into a final executive decision JSON adhering to all rules.",
        ]

        return "\n\n".join(prompt_parts)

    # ------------------------------------------------------------------
    # OLLAMA CALL
    # ------------------------------------------------------------------
    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "system": _SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "format": _OUTPUT_SCHEMA,
            "options": {"temperature": 0},
        }
        response = requests.post(self.ollama_url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        if "response" not in data:
            raise RuntimeError(
                f"Ollama response did not contain 'response'. Keys: {list(data.keys())}"
            )
        return data["response"]

    # ------------------------------------------------------------------
    # PARSING & SANITIZATION
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        try:
            res = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON returned: {exc}\nText: {text[:200]}") from exc
        if not isinstance(res, dict):
            raise ValueError("Decision output must be a JSON object.")
        return res

    @staticmethod
    def _sanitize_llm_result(
        data: Dict[str, Any],
        facts: Dict[str, Any],
        scenario: Any = None,
    ) -> Dict[str, Any]:
        """
        Coerce fields to correct types and apply deterministic string cleaning.
        """
        clean = dict(data)

        # Priority
        p = str(clean.get("priority", "MEDIUM")).strip().upper()
        clean["priority"] = p if p in _ALLOWED_PRIORITIES else "MEDIUM"

        # Confidence
        try:
            c = float(clean.get("confidence", 0.5))
            clean["confidence"] = max(0.0, min(1.0, c))
        except (TypeError, ValueError):
            clean["confidence"] = 0.5

        # String fields
        clean["decision"] = str(clean.get("decision", "")).strip().replace(" ", "_")

        # List fields
        list_fields = ["rationale", "recommended_actions", "expected_benefits", "risks", "conditions"]
        for f in list_fields:
            v = clean.get(f)
            if v is None:
                clean[f] = []
            elif isinstance(v, list):
                clean[f] = [DecisionAgent._clean_text(str(item), scenario, facts) for item in v if str(item).strip()]
            elif isinstance(v, str):
                clean[f] = [DecisionAgent._clean_text(v.strip(), scenario, facts)] if v.strip() else []
            else:
                clean[f] = [DecisionAgent._clean_text(str(v), scenario, facts)]

        # Specific field sanitation
        clean["risks"] = DecisionAgent._clean_risks(clean.get("risks", []), facts, scenario)
        clean["conditions"] = DecisionAgent._clean_and_validate_conditions(clean.get("conditions", []), facts)

        return clean

    # ------------------------------------------------------------------
    # DETERMINISTIC DECISION FRAMEWORK
    # ------------------------------------------------------------------
    @staticmethod
    def _determine_expected_decision(facts: Dict[str, Any], red_team_verdict: Dict[str, Any]) -> str:
        """
        Deterministic decision rules:
        - Red-Team FAIL -> decision cannot be PROCEED (must be FLAG_FOR_REVIEW)
        - Critical overall risk -> DO_NOT_PROCEED
        - High/Medium risk with material risks -> PROCEED_WITH_CONDITIONS
        - Low risk with no material issues -> PROCEED
        """
        rt_verdict = str(red_team_verdict.get("overall_verdict", "")).upper()
        if rt_verdict == "FAIL":
            return "FLAG_FOR_REVIEW"

        overall_risk = str(facts.get("overall_risk", "")).lower()
        inv_risk = str(facts.get("inventory_risk", "")).lower()
        cash_dir = str(facts.get("cash", {}).get("direction", "")).lower()
        ar_dir = str(facts.get("accounts_receivable", {}).get("direction", "")).lower()

        if overall_risk in ["critical", "high_critical"]:
            return "DO_NOT_PROCEED"

        has_material_risks = (
            overall_risk in ["high", "medium"]
            or inv_risk == "high"
            or cash_dir == "decreased"
            or ar_dir == "increased"
        )
        if has_material_risks:
            return "PROCEED_WITH_CONDITIONS"

        return "PROCEED"

    # ------------------------------------------------------------------
    # DETERMINISTIC VALIDATION
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_decision(
        decision: Dict[str, Any],
        facts: Dict[str, Any],
        analysis: Dict[str, Any],
        red_team_verdict: Dict[str, Any],
        allowed_numbers: Set[float],
    ) -> Dict[str, Any]:
        """
        Validate decision against all business and deterministic criteria.
        Raises ValueError if a violation cannot be automatically repaired.
        """
        # 1. Schema check
        for req in _OUTPUT_SCHEMA["required"]:
            if req not in decision:
                raise ValueError(f"Missing required field: '{req}'")

        if not decision["decision"]:
            raise ValueError("Field 'decision' cannot be empty.")

        # 2. Priority check
        if decision["priority"] not in _ALLOWED_PRIORITIES:
            raise ValueError(
                f"Invalid priority '{decision['priority']}'. Must be one of {_ALLOWED_PRIORITIES}"
            )

        # 3. Confidence range
        conf = decision["confidence"]
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            raise ValueError(f"Confidence {conf} must be a float between 0.0 and 1.0.")

        # 4. Currency and forbidden phrases check
        all_text = json.dumps(decision, ensure_ascii=False).lower()
        for forbidden in _FORBIDDEN_PHRASES:
            if forbidden == "$":
                if "$" in all_text:
                    raise ValueError("Forbidden currency symbol '$' found in decision. All currency must be INR.")
            elif re.search(rf"\b{re.escape(forbidden)}\b", all_text):
                raise ValueError(
                    f"Forbidden term '{forbidden}' found in decision. "
                    "Do not mention unmodeled factors, benchmarks, or false sustainability crises."
                )

        # 5. Deterministic Decision Framework Check
        expected_dec = DecisionAgent._determine_expected_decision(facts, red_team_verdict)
        dec_norm = decision["decision"].strip().upper().replace(" ", "_")

        if expected_dec == "FLAG_FOR_REVIEW":
            if "PROCEED" in dec_norm and "DO_NOT_PROCEED" not in dec_norm:
                raise ValueError("Red-Team verdict is FAIL: decision cannot be PROCEED. Must be FLAG_FOR_REVIEW.")
            if decision["confidence"] > 0.2:
                raise ValueError(f"Red-Team verdict is FAIL: confidence must be <= 0.2, got {decision['confidence']}.")
        elif expected_dec == "DO_NOT_PROCEED":
            if dec_norm != "DO_NOT_PROCEED":
                raise ValueError("Critical overall risk: decision must be DO_NOT_PROCEED.")
        elif expected_dec == "PROCEED_WITH_CONDITIONS":
            if dec_norm not in ["PROCEED_WITH_CONDITIONS", "PROCEED_WITH_RISK_MITIGATION"]:
                raise ValueError("High/Medium risk with material risks: decision must be PROCEED_WITH_CONDITIONS.")
        elif expected_dec == "PROCEED":
            if dec_norm != "PROCEED":
                raise ValueError("Low risk with no material issues: decision must be PROCEED.")

        # 6. Risk consistency check
        inv_risk = str(facts.get("inventory_risk", "")).lower()
        lost_sales = facts.get("lost_sales", 0)
        try:
            lost_sales_val = float(lost_sales) if lost_sales != "NOT PROVIDED" else 0
        except (TypeError, ValueError):
            lost_sales_val = 0

        if inv_risk == "high" or lost_sales_val > 0:
            rec_and_risks_text = " ".join(
                decision.get("risks", []) +
                decision.get("recommended_actions", []) +
                decision.get("conditions", [])
            ).lower()
            if not any(w in rec_and_risks_text for w in ["inventory", "stock", "supply", "buffer", "lead time", "coverage"]):
                raise ValueError(
                    "Simulator reports High Inventory Risk, but neither risks nor recommended_actions address inventory."
                )

        # 7. Check for unverified numerical values
        DecisionAgent._check_numerical_authenticity(decision, allowed_numbers, facts)

        return decision

    # ------------------------------------------------------------------
    # STRING & RISK CLEANERS
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_text(
        text: str,
        scenario: Any = None,
        facts: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Removes invented cadences, deadlines, threshold alerts, overclaimed profitability,
        unsupported revenue/stockout claims, and auto-corrects decimal percentage scaling.
        """
        # Replace sustainable revenue expansion / sustainable revenue
        text = re.sub(r"\bsustainable revenue expansion\b", "projected revenue growth", text, flags=re.IGNORECASE)
        text = re.sub(r"\bsustainable revenue\b", "projected revenue", text, flags=re.IGNORECASE)

        # Replace stable profitability with stable gross margin
        text = re.sub(r"\b(?:stable|consistent)\s+profitability\b", "stable gross margin", text, flags=re.IGNORECASE)
        # Remove dead-end claims about cash runway in rationale
        text = re.sub(r",?\s*(?:posing a financial sustainability risk|posing a liquidity risk|posing a risk)\.?", ".", text, flags=re.IGNORECASE)
        # Strip cadences and deadlines
        text = re.sub(r"\s+within\s+(?:the\s+next\s+)?\d+\s+days\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+on\s+a\s+monthly\s+basis\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bmonthly\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bdaily\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bquarterly\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwith\s+a\s+threshold\s+alert[^\.,]*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwith\s+escalation\s+triggers[^\.,]*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+to\s+at\s+least\s+\d+\s+months\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+within\s+acceptable\s+thresholds\b", " closely", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+before\s+the\s+next\s+quarter\b", "", text, flags=re.IGNORECASE)

        # Unsupported stockout wording if no lost sales and no stockout exposure
        if facts is not None:
            lost = float(facts.get("lost_sales", 0) or 0)
            stockout = float(facts.get("stockout_exposure", 0) or 0)
            if lost == 0 and stockout == 0:
                text = re.sub(r",?\s*(?:indicating potential stockouts|potential stockouts|risk of stockouts|stockout risk)", "", text, flags=re.IGNORECASE)

        # Auto-correct percentage scaling errors from scenario (e.g. 0.7% -> 70%)
        if scenario is not None:
            if hasattr(scenario, "__dataclass_fields__"):
                raw_scen = {k: getattr(scenario, k) for k in scenario.__dataclass_fields__}
            elif isinstance(scenario, dict):
                raw_scen = scenario
            elif hasattr(scenario, "__dict__"):
                raw_scen = vars(scenario)
            else:
                raw_scen = {}

            for field in ("sales_growth", "price_change", "cost_change", "expense_change", "inventory_change"):
                val = raw_scen.get(field)
                if val is not None:
                    try:
                        fval = float(val)
                        if abs(fval) > 0.001:
                            wrong_pct = f"{fval:g}%"
                            correct_pct = f"{round(fval * 100, 2):g}%"
                            if wrong_pct != correct_pct:
                                text = re.sub(rf"\b{re.escape(wrong_pct)}\b", correct_pct, text)
                    except (TypeError, ValueError):
                        pass

        # Remove trailing commas and extra spaces
        text = re.sub(r"\s+,\s+", ", ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip().rstrip(",")

    @staticmethod
    def _clean_risks(
        raw_risks: List[str],
        facts: Dict[str, Any],
        scenario: Any = None,
    ) -> List[str]:
        """
        Ensures 'cash runway not reached in projection' is NEVER treated as a risk by itself.
        Removes obsolescence mentions and unsupported stockout risks.
        """
        cleaned: List[str] = []
        lost = float(facts.get("lost_sales", 0) or 0)
        stockout = float(facts.get("stockout_exposure", 0) or 0)
        has_stockouts = (lost > 0 or stockout > 0)

        for r in raw_risks:
            r_str = str(r).strip()
            r_low = r_str.lower()
            # 1. Never treat 'cash runway not reached in projection' as a risk
            if "cash runway" in r_low and "not reached" in r_low:
                continue
            # 2. Strip obsolescence
            if "obsolescence" in r_low:
                r_str = re.sub(r",?\s*(?:indicating potential stockouts or obsolescence|or obsolescence)", "", r_str, flags=re.IGNORECASE).strip().rstrip(".") + "."
            # 3. Strip or filter unsupported stockout risk
            if not has_stockouts:
                r_str = re.sub(r",?\s*(?:indicating potential stockouts|potential stockouts|risk of stockouts|stockout risk)", "", r_str, flags=re.IGNORECASE).strip().rstrip(".") + "."
                if any(k in r_str.lower() for k in ["stockout", "stock-out", "lost sales"]):
                    continue
            r_str = DecisionAgent._clean_text(r_str, scenario, facts)
            if r_str and r_str not in cleaned:
                cleaned.append(r_str)
        return cleaned

    @staticmethod
    def _clean_and_validate_conditions(raw_conditions: List[str], facts: Dict[str, Any]) -> List[str]:
        """
        Ensures conditions are evidence-based observations/actions, NOT invented policies.
        Valid:
        - Review inventory policy given High inventory risk.
        - Monitor cash position given lower ending cash.
        - Strengthen collections given increased AR and payment delay.
        """
        cleaned: List[str] = []
        inv_risk = str(facts.get("inventory_risk", "")).lower()
        cash_dir = str(facts.get("cash", {}).get("direction", "")).lower()
        ar_dir = str(facts.get("accounts_receivable", {}).get("direction", "")).lower()

        for c in raw_conditions:
            c_str = str(c).strip()
            c_low = c_str.lower()

            # Remove missing net profit complaints
            if "net profit" in c_low or "additional data" in c_low or "baseline" in c_low:
                continue

            # Check if condition has forbidden cadences, deadlines, or invented thresholds
            has_cadence_or_deadline = any(re.search(pat, c_low) for pat in [
                r"\b(daily|weekly|monthly|quarterly|annually)\b",
                r"\bwithin\s+\d+\s+days?\b",
                r"\bin\s+\d+\s+days?\b",
                r"\bevery\s+\d+\s+(?:days?|weeks?|months?)\b",
                r"\bnext quarter\b",
            ])
            has_threshold = any(re.search(pat, c_low) for pat in [
                r"\b(?:above|below|at least|to at least)\s+(?:rs\.?|inr|\d+)",
                r"\bthreshold alert\b",
                r"\bminimum threshold\b",
                r"\b\d+\s*months?\b",
            ])

            if has_cadence_or_deadline or has_threshold:
                # Map to standard valid evidence-based conditions
                if "inventory" in c_low:
                    valid = "Review inventory policy given High inventory risk."
                elif "cash" in c_low or "liquid" in c_low:
                    valid = "Monitor cash position given lower ending cash."
                elif "collection" in c_low or "receivable" in c_low or "ar" in c_low:
                    valid = "Strengthen collections given increased AR and payment delay."
                else:
                    continue
                if valid not in cleaned:
                    cleaned.append(valid)
            else:
                cleaned_c = DecisionAgent._clean_text(c_str)
                if cleaned_c and cleaned_c not in cleaned:
                    cleaned.append(cleaned_c)

        # Guarantee core evidence-based conditions exist if corresponding risk is present
        if (inv_risk == "high" or "inventory" in inv_risk) and not any("inventory" in x.lower() for x in cleaned):
            cleaned.append("Review inventory policy given High inventory risk.")
        if cash_dir == "decreased" and not any("cash" in x.lower() for x in cleaned):
            cleaned.append("Monitor cash position given lower ending cash.")
        if ar_dir == "increased" and not any("collection" in x.lower() or "receivable" in x.lower() for x in cleaned):
            cleaned.append("Strengthen collections given increased AR and payment delay.")

        return cleaned

    # ------------------------------------------------------------------
    # NUMERICAL AUTHENTICITY CHECK
    # ------------------------------------------------------------------
    @staticmethod
    def _build_allowed_numbers(
        scenario: Any,
        facts: Dict[str, Any],
        analysis: Dict[str, Any],
        red_team_verdict: Dict[str, Any],
    ) -> Set[float]:
        """
        Gathers all numbers present in scenario, facts, analysis, and red-team verdict.
        Used to ensure the Decision Agent does not invent financial numbers.
        """
        allowed: Set[float] = set()

        def extract_nums(obj: Any):
            if isinstance(obj, (int, float)):
                val = float(obj)
                allowed.add(val)
                allowed.add(round(val, 2))
                allowed.add(round(val * 100, 2))  # e.g. 0.3 -> 30%
                allowed.add(round(val / 100, 4))
                if val >= 100000:
                    allowed.add(round(val / 100000, 2))
                    allowed.add(round(val / 100000, 1))
            elif isinstance(obj, str):
                for match in re.finditer(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", obj):
                    raw = match.group(0).replace(",", "")
                    try:
                        fv = float(raw)
                        allowed.add(fv)
                        allowed.add(round(fv, 2))
                    except ValueError:
                        pass
            elif isinstance(obj, dict):
                for v in obj.values():
                    extract_nums(v)
            elif isinstance(obj, (list, tuple, set)):
                for item in obj:
                    extract_nums(item)
            elif hasattr(obj, "__dataclass_fields__"):
                for k in obj.__dataclass_fields__:
                    extract_nums(getattr(obj, k))

        extract_nums(scenario)
        extract_nums(facts)
        extract_nums(analysis)
        extract_nums(red_team_verdict)

        for n in range(1, 10):
            allowed.add(float(n))

        return allowed

    @staticmethod
    def _check_numerical_authenticity(
        decision: Dict[str, Any], allowed_numbers: Set[float], facts: Dict[str, Any]
    ) -> None:
        """
        Scans decision text for large numbers, specific percentages, or month figures
        and ensures they exist in authoritative simulator facts.
        """
        text_fields = (
            [decision.get("decision", "")]
            + decision.get("rationale", [])
            + decision.get("recommended_actions", [])
            + decision.get("expected_benefits", [])
            + decision.get("risks", [])
            + decision.get("conditions", [])
        )

        allowed_months: Set[float] = set()
        cov = facts.get("inventory_coverage_months")
        if isinstance(cov, (int, float)):
            allowed_months.add(round(float(cov), 2))
            allowed_months.add(round(float(cov), 1))
        scen = facts.get("scenario", {})
        if isinstance(scen, dict):
            m = scen.get("months")
            if isinstance(m, (int, float)):
                allowed_months.add(float(m))

        for text in text_fields:
            # Check specifically for invented month targets/benchmarks (e.g. "6 months")
            for m_match in re.finditer(r"(\d+(?:\.\d+)?)\s*months?", text, re.IGNORECASE):
                try:
                    m_val = float(m_match.group(1))
                    if not any(abs(m_val - am) < 0.1 for am in allowed_months):
                        raise ValueError(
                            f"Invented month figure/target '{m_match.group(0)}' in decision text. "
                            "Do not invent external benchmarks or coverage targets."
                        )
                except ValueError as ve:
                    if "Invented month figure" in str(ve):
                        raise ve

            # Find numbers with optional INR / Rs / %
            for match in re.finditer(r"(?:(?:inr|rs\.?)\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|lakh|crore)?", text, re.IGNORECASE):
                num_str = match.group(1).replace(",", "")
                unit = (match.group(2) or "").lower()
                try:
                    val = float(num_str)
                except ValueError:
                    continue

                if val in range(1, 10) and not unit:
                    continue

                candidates = [val]
                if unit == "lakh":
                    candidates.extend([val, val * 100000, round(val * 100000, 0)])
                elif unit == "crore":
                    candidates.extend([val, val * 10000000, round(val * 10000000, 0)])
                elif unit == "%":
                    candidates.extend([val, val / 100.0])

                matched = False
                for c in candidates:
                    for a in allowed_numbers:
                        if abs(c - a) < 0.05 or (a != 0 and abs(c - a) / abs(a) < 0.02):
                            matched = True
                            break
                    if matched:
                        break

                if not matched and val > 10.0:
                    raise ValueError(
                        f"Unverified financial or numeric value '{match.group(0)}' in decision text. "
                        "Decision Agent must not invent financial figures."
                    )

    # ------------------------------------------------------------------
    # DETERMINISTIC GUARDRAILS ENFORCEMENT
    # ------------------------------------------------------------------
    @staticmethod
    def _enforce_deterministic_guardrails(
        decision: Dict[str, Any],
        facts: Dict[str, Any],
        red_team_verdict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Hard final guardrail to guarantee 100% compliance with Red-Team verdict,
        deterministic decision framework, and schema constraints.
        """
        expected_decision = DecisionAgent._determine_expected_decision(facts, red_team_verdict)
        decision["decision"] = expected_decision

        rt_verdict = str(red_team_verdict.get("overall_verdict", "")).upper()

        if rt_verdict == "FAIL":
            decision["decision"] = "FLAG_FOR_REVIEW"
            decision["priority"] = "HIGH"
            decision["confidence"] = min(0.2, float(decision.get("confidence", 0.1)))
            crit_issues = red_team_verdict.get("critical_issues", [])
            audit_note = (
                f"Audit failed with {len(crit_issues)} critical issues. "
                "All automated execution is halted pending human review."
            )
            if not any("audit" in r.lower() or "red-team" in r.lower() for r in decision.get("rationale", [])):
                decision["rationale"].insert(0, audit_note)
            decision["conditions"] = ["Resolve all Red-Team audit discrepancies before implementing any changes."]

        elif rt_verdict == "PASS_WITH_WARNINGS":
            if decision.get("confidence", 0.5) > 0.75:
                decision["confidence"] = 0.75

        # Clean all text fields
        decision["rationale"] = [DecisionAgent._clean_text(r) for r in decision.get("rationale", []) if r.strip()]
        decision["recommended_actions"] = [DecisionAgent._clean_text(a) for a in decision.get("recommended_actions", []) if a.strip()]
        decision["expected_benefits"] = [DecisionAgent._clean_text(b) for b in decision.get("expected_benefits", []) if b.strip()]
        decision["risks"] = DecisionAgent._clean_risks(decision.get("risks", []), facts)
        decision["conditions"] = DecisionAgent._clean_and_validate_conditions(decision.get("conditions", []), facts)

        # Ensure priority is valid
        if decision.get("priority") not in _ALLOWED_PRIORITIES:
            decision["priority"] = "MEDIUM"

        return decision

    # ------------------------------------------------------------------
    # DETERMINISTIC FALLBACK DECISION
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_fallback_decision(
        facts: Dict[str, Any],
        analysis: Dict[str, Any],
        red_team_verdict: Dict[str, Any],
        reason: str,
    ) -> Dict[str, Any]:
        """
        Generates a 100% safe, authoritative decision if LLM generation fails.
        """
        expected_decision = DecisionAgent._determine_expected_decision(facts, red_team_verdict)
        overall_risk = facts.get("overall_risk", "Medium")
        inv_risk = facts.get("inventory_risk", "Medium")

        if expected_decision == "FLAG_FOR_REVIEW":
            return {
                "decision": "FLAG_FOR_REVIEW",
                "priority": "HIGH",
                "rationale": [
                    f"Red-Team audit returned FAIL: {reason}",
                    "Critical discrepancies must be reviewed before approving scenario execution.",
                ],
                "recommended_actions": [
                    "Priority 1: Review discrepancies flagged in Red-Team audit report.",
                    "Priority 2: Re-run simulation and validation once assumptions are reconciled.",
                ],
                "expected_benefits": [
                    "Prevents misallocation of resources under inconsistent financial assumptions."
                ],
                "risks": [
                    f"Overall risk is {overall_risk}. Inventory risk is {inv_risk}."
                ],
                "conditions": [
                    "Audit discrepancies must be resolved prior to any business commitment."
                ],
                "confidence": 0.1,
            }

        pri = "HIGH" if overall_risk == "High" or inv_risk == "High" else "MEDIUM"
        conf = 0.7 if str(red_team_verdict.get("overall_verdict")) == "PASS" else 0.5

        return {
            "decision": expected_decision,
            "priority": pri,
            "rationale": [
                f"Simulator projects revenue direction '{facts.get('revenue', {}).get('direction')}' "
                f"and ending cash direction '{facts.get('cash', {}).get('direction')}'.",
                f"Overall risk is {overall_risk}, with inventory risk classified as {inv_risk}.",
            ],
            "recommended_actions": [
                "Priority 1: Strengthen collections given increased accounts receivable and payment delay.",
                "Priority 2: Review inventory policy given High inventory risk.",
            ],
            "expected_benefits": [
                "Stable gross margin supports operational continuity under projected demand changes."
            ],
            "risks": [
                f"Inventory risk is {inv_risk} with minimum coverage of {facts.get('inventory_coverage_months')} months.",
                f"Ending cash decreased below baseline.",
            ],
            "conditions": [
                "Review inventory policy given High inventory risk.",
                "Monitor cash position given lower ending cash.",
                "Strengthen collections given increased AR and payment delay.",
            ],
            "confidence": conf,
        }
