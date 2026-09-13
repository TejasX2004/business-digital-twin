"""
Red-Team Agent — agents/red_team_agent.py

Architecture:
    Simulator (source of truth)
        ↓
    Analysis Agent output
        ↓
    Python deterministic pre-checks   (numbers, risk levels, directions)
        ↓
    Qwen semantic review              (logic, unsupported claims, missing risks)
        ↓
    Python merge + verdict            (deterministic final classification)

The Red-Team Agent NEVER modifies simulator facts.
It only produces a verdict JSON that grades the Analysis Agent output.
"""

from __future__ import annotations

import json
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a Red-Team auditor for a Business Digital Twin.\n"
    "\n"
    "You receive:\n"
    "1. SIMULATOR_FACTS: authoritative, deterministic simulation output.\n"
    "2. ANALYSIS_TEXT: the Analysis Agent's interpretation.\n"
    "\n"
    "Your job is to find GENUINE errors. Not to summarise or rewrite.\n"
    "\n"
    "FIELD DEFINITIONS (STRICT):\n"
    "  critical_issues:       A claim that DIRECTLY AND CLEARLY CONTRADICTS a number or\n"
    "                          fact in SIMULATOR_FACTS (e.g. says revenue decreased when\n"
    "                          SIMULATOR_FACTS shows it increased).\n"
    "  numerical_issues:      A wrong number, wrong unit, or wrong magnitude relative to\n"
    "                          SIMULATOR_FACTS.\n"
    "  logic_issues:          Internally inconsistent reasoning WITHIN the analysis\n"
    "                          (section A says X, section B says not-X).\n"
    "  unsupported_claims:    A specific assertion in ANALYSIS_TEXT that is NOT present\n"
    "                          in SIMULATOR_FACTS AND cannot be derived from SIMULATOR_FACTS.\n"
    "                          Examples: invented industry thresholds, invented causal\n"
    "                          explanations, policy assumptions not in SIMULATOR_FACTS.\n"
    "  missing_risks:         A risk value that is clearly present in SIMULATOR_FACTS\n"
    "                          AND is NOT mentioned anywhere in ANALYSIS_TEXT.\n"
    "  recommendation_issues: A recommendation that has NO corresponding risk, metric\n"
    "                          direction, or fact in SIMULATOR_FACTS.\n"
    "  verified_claims:       Claims in ANALYSIS_TEXT you confirm are CORRECT and SUPPORTED\n"
    "                          by SIMULATOR_FACTS.\n"
    "\n"
    "MANDATORY RULES — any violation causes the entire output to be rejected:\n"
    "  R1. Simulator-provided risk labels (inventory_risk, overall_risk) are AUTHORITATIVE.\n"
    "      The analysis is ALLOWED to state these labels directly. Do NOT flag them as\n"
    "      unsupported merely because no external threshold is cited.\n"
    "  R2. cash_runway_raw = 'Not reached in projection' is a SIMULATOR OUTPUT.\n"
    "      It means the model did not reach the exhaustion point. Do NOT flag it as a\n"
    "      benchmark, threshold, or missing risk. Do NOT call it a liquidity problem.\n"
    "  R3. If a risk or metric IS mentioned in ANALYSIS_TEXT, do NOT flag it as missing_risk.\n"
    "  R4. A verified/supported claim must go in verified_claims ONLY — never in issue fields.\n"
    "  R5. If you are uncertain whether something is an issue, omit it.\n"
    "  R6. Only report an issue when you can point to the exact SIMULATOR_FACTS field that\n"
    "      contradicts the analysis.\n"
    "  R7. Simulator-derived recommendations (cash monitoring, AR monitoring, inventory\n"
    "      review) are VALID when the corresponding simulator metric moved in that direction.\n"
    "      Do NOT classify them as unsupported.\n"
    "  R8. Leave any list EMPTY if you find no genuine issues in that category.\n"
    "  R9. Scenario parameters (e.g. sales_growth) represent percentages (e.g. '70%' means 70% growth, NOT 0.7%). Analysis stating '70%' is accurate and matches the scenario.\n"
    "\n"
    "Return only the required JSON."
)

# ---------------------------------------------------------------------------
# OUTPUT SCHEMA
# ---------------------------------------------------------------------------
_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "critical_issues":        {"type": "array", "items": {"type": "string"}},
        "numerical_issues":       {"type": "array", "items": {"type": "string"}},
        "logic_issues":           {"type": "array", "items": {"type": "string"}},
        "unsupported_claims":     {"type": "array", "items": {"type": "string"}},
        "missing_risks":          {"type": "array", "items": {"type": "string"}},
        "recommendation_issues":  {"type": "array", "items": {"type": "string"}},
        "verified_claims":        {"type": "array", "items": {"type": "string"}},
        "confidence":             {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "critical_issues",
        "numerical_issues",
        "logic_issues",
        "unsupported_claims",
        "missing_risks",
        "recommendation_issues",
        "verified_claims",
        "confidence",
    ],
}


@dataclass
class RedTeamAgent:
    """
    Critically challenges Analysis Agent output using:
      1. Deterministic Python checks (numbers, risk levels, metric directions)
      2. Qwen semantic review
      3. Deterministic Python verdict
    """

    model: str = "qwen3:4b-instruct-2507-q4_K_M"
    ollama_url: str = "http://localhost:11434/api/generate"

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE
    # ------------------------------------------------------------------
    def review(
        self,
        scenario: Any,
        simulation_result: Dict[str, Any],
        analysis: Dict[str, Any],
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Run the full Red-Team pipeline.

        Parameters
        ----------
        scenario          : Scenario dataclass (or dict)
        simulation_result : Raw simulate() output — the source of truth
        analysis          : dict from AnalysisAgent.analyze()

        Returns
        -------
        Red-Team verdict dict
        """
        if not isinstance(simulation_result, dict):
            raise TypeError("simulation_result must be a dictionary.")
        if not isinstance(analysis, dict):
            raise TypeError("analysis must be a dictionary.")

        # --- Step 1: Extract authoritative facts ---
        facts = self._extract_facts(scenario, simulation_result)

        # --- Step 2: Build authoritative claims set ---
        auth_claims = self._build_authoritative_claims(facts)

        # --- Step 3: Python deterministic pre-checks ---
        python_issues = self._python_checks(facts, analysis)

        # --- Step 4: Build Qwen prompt ---
        analysis_text = self._analysis_to_text(analysis)
        base_prompt = self._build_prompt(facts, analysis_text)

        # --- Step 5: Call Qwen with retry ---
        last_error: Optional[Exception] = None
        llm_result: Dict[str, Any] = {}
        extra_prefix = ""

        for attempt in range(max_retries):
            prompt = (extra_prefix + "\n\n" + base_prompt) if extra_prefix else base_prompt
            try:
                raw = self._call_ollama(prompt)
                llm_result = self._parse_json(raw)
                break
            except (ValueError, RuntimeError) as exc:
                last_error = exc
                extra_prefix = f"CORRECTION: Previous response was invalid ({exc}). Return valid JSON only."

        if not llm_result:
            llm_result = {k: [] for k in _OUTPUT_SCHEMA["required"]}
            llm_result["confidence"] = 0.0
            llm_result["logic_issues"] = [
                f"Red-Team LLM failed after {max_retries} attempts: {last_error}"
            ]

        # --- Step 6: LLM output cleanup (Python authority) ---
        llm_result = self._validate_llm_result(llm_result)
        llm_result = self._sanitize_llm_result(llm_result)
        llm_result = self._python_validate_llm_issues(llm_result, facts, auth_claims, analysis)

        # --- Step 7: Merge Python + LLM results ---
        merged = self._merge(python_issues, llm_result)

        # --- Step 8: Deterministic verdict ---
        merged["overall_verdict"] = self._compute_verdict(merged)

        return merged

    # ------------------------------------------------------------------
    # FACT EXTRACTION (mirrors AnalysisAgent._build_analysis_facts logic)
    # ------------------------------------------------------------------
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
    def _extract_facts(
        scenario: Any,
        simulation_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        def get(data: dict, key: str) -> Any:
            val = data.get(key)
            return "NOT PROVIDED" if val is None else val

        def pct(num, denom) -> Any:
            try:
                d = float(denom)
                if d == 0:
                    return "NOT PROVIDED"
                return round(float(num) / d * 100, 4)
            except (TypeError, ValueError):
                return "NOT PROVIDED"

        def direction(b, p) -> str:
            if b == "NOT PROVIDED" or p == "NOT PROVIDED":
                return "cannot_determine"
            try:
                bv, pv = float(b), float(p)
                denom = max(abs(bv), abs(pv), 1e-9)
                rel = abs(pv - bv) / denom
                if rel < 0.0001:
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

        baseline  = simulation_result.get("baseline", {})
        projected = simulation_result.get("projected", {})
        risk      = simulation_result.get("risk", {})

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

        raw_runway = get(projected, "cash_runway_months")

        return {
            "scenario": RedTeamAgent._format_scenario(scenario),
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
    # PYTHON DETERMINISTIC CHECKS
    # ------------------------------------------------------------------
    @staticmethod
    def _python_checks(
        facts: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        """
        Pure Python checks against authoritative facts.
        Returns a dict of issue lists, same keys as the final verdict.
        """

        numerical_issues:      List[str] = []
        logic_issues:          List[str] = []
        unsupported_claims:    List[str] = []
        missing_risks:         List[str] = []
        recommendation_issues: List[str] = []
        verified_claims:       List[str] = []
        critical_issues:       List[str] = []

        # Flatten all analysis text for scanning
        all_text = json.dumps(analysis, ensure_ascii=False).lower()

        # --- 1. Currency check ---
        for bad_currency in ("$", "usd", "eur", "euros", "dollars", "£", "gbp"):
            if bad_currency in all_text:
                critical_issues.append(
                    f"Analysis uses forbidden currency symbol/word: '{bad_currency}'. "
                    "All values must be INR."
                )

        # --- 2. Risk level verification ---
        inv_risk     = facts.get("inventory_risk", "NOT PROVIDED")
        overall_risk = facts.get("overall_risk", "NOT PROVIDED")

        if inv_risk != "NOT PROVIDED":
            if inv_risk.lower() not in all_text:
                missing_risks.append(
                    f"Simulator inventory_risk = '{inv_risk}' but this risk level "
                    "does not appear to be mentioned in the analysis."
                )
            else:
                verified_claims.append(
                    f"Inventory risk level '{inv_risk}' correctly mentioned."
                )

        if overall_risk != "NOT PROVIDED":
            if overall_risk.lower() not in all_text:
                missing_risks.append(
                    f"Simulator overall_risk = '{overall_risk}' but this risk level "
                    "does not appear to be mentioned in the analysis."
                )
            else:
                verified_claims.append(
                    f"Overall risk level '{overall_risk}' correctly mentioned."
                )

        # --- 3. Metric direction verification ---
        direction_checks = [
            ("revenue",             "revenue increased",    "revenue decreased"),
            ("gross_profit",        "gross profit",         None),
            ("cash",                "cash",                 None),
            ("accounts_receivable", "accounts receivable",  None),
            ("inventory_value",     "inventory",            None),
            ("operating_expenses",  "operating expenses",   None),
            ("net_margin",          "net margin",           None),
        ]

        for metric_key, label, _alt_label in direction_checks:
            metric = facts.get(metric_key, {})
            if not isinstance(metric, dict):
                continue
            d = metric.get("direction")
            if d == "cannot_determine":
                continue

            # Check that the analysis doesn't say the opposite direction
            if d == "increased":
                opposite_phrases = [
                    f"{label} decreased",
                    f"{label} fell",
                    f"{label} dropped",
                    f"{label} declined",
                ]
                for phrase in opposite_phrases:
                    if phrase in all_text:
                        numerical_issues.append(
                            f"Analysis states '{phrase}' but simulator shows "
                            f"{metric_key} direction = '{d}'."
                        )
                        break
                else:
                    verified_claims.append(
                        f"{metric_key} direction '{d}' not contradicted in analysis."
                    )

            elif d == "decreased":
                opposite_phrases = [
                    f"{label} increased",
                    f"{label} rose",
                    f"{label} grew",
                ]
                for phrase in opposite_phrases:
                    if phrase in all_text:
                        numerical_issues.append(
                            f"Analysis states '{phrase}' but simulator shows "
                            f"{metric_key} direction = '{d}'."
                        )
                        break
                else:
                    verified_claims.append(
                        f"{metric_key} direction '{d}' not contradicted in analysis."
                    )

            elif d == "unchanged":
                # Flag if analysis says it changed
                change_phrases = [
                    f"{label} increased",
                    f"{label} decreased",
                    f"{label} rose",
                    f"{label} fell",
                    f"{label} changed",
                ]
                for phrase in change_phrases:
                    if phrase in all_text:
                        numerical_issues.append(
                            f"Analysis states '{phrase}' but simulator shows "
                            f"{metric_key} direction = 'unchanged'."
                        )
                        break

        # --- 4. Lost sales / stockout check ---
        lost_sales   = facts.get("lost_sales", 0)
        stockout_exp = facts.get("stockout_exposure", 0)

        try:
            no_actual_lost = float(lost_sales) == 0 and float(stockout_exp) == 0
        except (TypeError, ValueError):
            no_actual_lost = False

        if no_actual_lost:
            stockout_phrases = [
                "stockout occurred",
                "stockouts occurred",
                "actual stockout",
                "lost sales due to",
                "demand could not be fulfilled",
            ]
            for phrase in stockout_phrases:
                if phrase in all_text:
                    logic_issues.append(
                        f"Analysis implies actual stockouts ('{phrase}') but "
                        f"simulator shows lost_sales = {lost_sales} and "
                        f"stockout_exposure = {stockout_exp} (no actual stockouts)."
                    )
                    break
            else:
                verified_claims.append(
                    "No actual stockouts or lost sales — analysis does not contradict this."
                )

        # --- 5. Inventory coverage unit check ---
        inv_cov = facts.get("inventory_coverage_months")
        if inv_cov not in (None, "NOT PROVIDED"):
            if "inventory coverage" in all_text:
                # Check that 'days' is not used when it should be 'months'
                if (
                    f"{float(inv_cov):.1f} days" in all_text
                    or f"{float(inv_cov):.2f} days" in all_text
                    or f"{int(float(inv_cov))} days" in all_text
                ):
                    numerical_issues.append(
                        f"Analysis expresses inventory coverage ({inv_cov}) in days "
                        "but the simulator measures minimum_inventory_coverage in months."
                    )
                else:
                    verified_claims.append(
                        f"Inventory coverage ({inv_cov}) appears to use correct unit."
                    )

        # --- 6. Cash runway interpretation check ---
        raw_runway = facts.get("cash_runway_raw", "NOT PROVIDED")
        if raw_runway == "Not reached in projection":
            bad_runway_phrases = [
                "lack of runway",
                "insufficient runway",
                "no runway",
                "cash crisis",
                "cash shortfall",
                "potential shortfall",
                "liquidity failure",
            ]
            for phrase in bad_runway_phrases:
                if phrase in all_text:
                    logic_issues.append(
                        f"Analysis uses '{phrase}' but simulator cash_runway_months = "
                        f"'{raw_runway}' — this means the exhaustion threshold was "
                        "not reached, NOT that there is a liquidity crisis."
                    )
                    break
            else:
                verified_claims.append(
                    "Cash runway 'Not reached in projection' not misinterpreted as a crisis."
                )

        # --- 7. Operating expenses direction check ---
        opex = facts.get("operating_expenses", {})
        if isinstance(opex, dict) and opex.get("direction") == "unchanged":
            # Should not claim operating expense pressure
            for phrase in ("rising operating expenses", "increased operating expenses", "operating expense pressure"):
                if phrase in all_text:
                    logic_issues.append(
                        f"Analysis claims '{phrase}' but simulator shows "
                        "operating_expenses direction = 'unchanged'."
                    )
                    break

        # --- 8. Scenario driver verification ---
        scenario = facts.get("scenario", {})
        if isinstance(scenario, dict):
            inv_change = scenario.get("inventory_change", 0)
            try:
                inv_val = float(str(inv_change).replace("%", ""))
                if inv_val == 0 and "inventory policy changed" in all_text:
                    logic_issues.append(
                        "Analysis claims inventory policy changed but scenario "
                        f"inventory_change = {inv_change} (no inventory policy change)."
                    )
            except (TypeError, ValueError):
                pass

        # --- 9. Assessment verdict sanity ---
        overall_assessment = analysis.get("overall_assessment", "")
        inv_risk_level   = facts.get("inventory_risk", "Low")
        overall_risk_lvl = facts.get("overall_risk", "Low")
        cash_dir         = facts.get("cash", {})
        cash_direction   = cash_dir.get("direction", "") if isinstance(cash_dir, dict) else ""

        if (
            overall_assessment == "Positive"
            and cash_direction == "decreased"
            and inv_risk_level in ("High", "Critical")
        ):
            logic_issues.append(
                f"Analysis verdict is 'Positive' despite cash decreasing and "
                f"inventory risk being '{inv_risk_level}'. "
                "This may be overconfident."
            )

        if overall_assessment == "Negative" and overall_risk_lvl in ("Low", "Medium"):
            logic_issues.append(
                f"Analysis verdict is 'Negative' but simulator overall_risk = "
                f"'{overall_risk_lvl}'. Verdict may be overly pessimistic."
            )

        return {
            "critical_issues":        critical_issues,
            "numerical_issues":       numerical_issues,
            "logic_issues":           logic_issues,
            "unsupported_claims":     unsupported_claims,
            "missing_risks":          missing_risks,
            "recommendation_issues":  recommendation_issues,
            "verified_claims":        verified_claims,
        }

    # ------------------------------------------------------------------
    # BUILD QWEN PROMPT
    # ------------------------------------------------------------------
    @staticmethod
    def _build_prompt(
        facts: Dict[str, Any],
        analysis_text: str,
    ) -> str:
        return (
            "SIMULATOR_FACTS (authoritative — do not contradict)\n"
            + json.dumps(facts, indent=2, default=str)
            + "\n\nANALYSIS_TEXT\n"
            + analysis_text
        )

    @staticmethod
    def _analysis_to_text(analysis: Dict[str, Any]) -> str:
        """Flatten the analysis dict into readable prose for Qwen to audit."""
        lines: List[str] = []
        for key, val in analysis.items():
            if isinstance(val, list):
                lines.append(f"{key.upper().replace('_', ' ')}:")
                for item in val:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key.upper().replace('_', ' ')}: {val}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # OLLAMA
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
    # JSON PARSE
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        try:
            result = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Red-Team LLM returned invalid JSON:\n{text}") from e
        if not isinstance(result, dict):
            raise ValueError("Red-Team LLM JSON must be an object.")
        return result

    # ------------------------------------------------------------------
    # AUTHORITATIVE CLAIMS
    # ------------------------------------------------------------------
    @staticmethod
    def _build_authoritative_claims(facts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a dict of claims that are DEFINITIVELY TRUE based on simulator facts.
        Python uses this to dismiss LLM issues that challenge true statements.

        Keys are human-readable labels; values are authoritative strings/values.
        """
        auth: Dict[str, Any] = {}

        inv_risk     = facts.get("inventory_risk", "NOT PROVIDED")
        overall_risk = facts.get("overall_risk", "NOT PROVIDED")
        raw_runway   = facts.get("cash_runway_raw", "NOT PROVIDED")
        inv_cov      = facts.get("inventory_coverage_months", "NOT PROVIDED")
        lost_sales   = facts.get("lost_sales", 0)
        stockout_exp = facts.get("stockout_exposure", 0)

        # --- Risk labels (simulator-provided, always authoritative) ---
        if inv_risk != "NOT PROVIDED":
            auth["inventory_risk_label"] = str(inv_risk).lower()
        if overall_risk != "NOT PROVIDED":
            auth["overall_risk_label"] = str(overall_risk).lower()

        # --- Cash runway ---
        if raw_runway == "Not reached in projection":
            auth["cash_runway_safe"] = True  # runway NOT exhausted

        # --- Inventory coverage ---
        if inv_cov not in ("NOT PROVIDED", None):
            auth["inventory_coverage_months"] = float(inv_cov)

        # --- Lost sales / stockout ---
        try:
            auth["no_lost_sales"]   = float(lost_sales)   == 0
            auth["no_stockout_exp"] = float(stockout_exp) == 0
        except (TypeError, ValueError):
            pass

        # --- Metric directions (from already-computed facts) ---
        for metric_key in (
            "revenue", "gross_profit", "gross_margin", "net_margin",
            "operating_expenses", "cash", "accounts_receivable", "inventory_value",
        ):
            m = facts.get(metric_key)
            if isinstance(m, dict) and m.get("direction") not in (None, "cannot_determine"):
                auth[f"{metric_key}_direction"] = m["direction"]

        # --- Keywords the analysis IS ALLOWED to use without justification ---
        # (These are facts — the analysis needs no external source for them)
        auth["allowed_risk_labels"] = {
            str(inv_risk).lower(), str(overall_risk).lower()
        }

        return auth

    # ------------------------------------------------------------------
    # PYTHON VALIDATION OF LLM ISSUES  (final authority)
    # ------------------------------------------------------------------
    @staticmethod
    def _python_validate_llm_issues(
        llm_result: Dict[str, Any],
        facts: Dict[str, Any],
        auth_claims: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Python has final authority to dismiss LLM-raised issues that:
          - Challenge a simulator-provided fact (risk labels, numbers, directions)
          - Flag an already-mentioned risk as "missing"
          - Misinterpret cash_runway 'Not reached' as a problem
          - Require external thresholds for simulator-provided classifications

        Dismissed items are moved to verified_claims with a [DISMISSED] note.
        """
        analysis_text = json.dumps(analysis, ensure_ascii=False).lower()
        promoted: List[str] = []

        # ---- Patterns in LLM text that indicate false-positive flag ----
        # These patterns mean the LLM is requiring external justification
        # for simulator-provided facts — which is NEVER required.
        external_justification_markers = (
            "without referencing any threshold",
            "without referencing a threshold",
            "no threshold provided",
            "no benchmark provided",
            "no external reference",
            "without external reference",
            "without citing",
            "no citation",
            "lacks a basis",
            "lacks grounding",
            "not defined in simulator",
            "simulator does not define",
            "not provided in simulator_facts",
            "not grounded in any metric",
            "not grounded in data",
            "lacks data support",
        )

        # ---- Simulator risk labels — never need external justification ----
        allowed_labels = auth_claims.get("allowed_risk_labels", set())

        def is_false_flag_unsupported(item: str) -> bool:
            """Return True if this 'unsupported_claim' is actually a simulator fact."""
            lower = item.lower()

            # Rule R1: Risk labels don't need external thresholds
            if any(label in lower for label in allowed_labels):
                if any(marker in lower for marker in external_justification_markers):
                    return True
                # Also dismiss if it's saying the risk label itself is unsupported
                if any(phrase in lower for phrase in (
                    "classified as high without",
                    "classified as medium without",
                    "risk is high is unsupported",
                    "risk is medium is unsupported",
                    "risk classification is not supported",
                )):
                    return True

            # Rule R2: Cash runway 'Not reached' is a simulator fact
            if auth_claims.get("cash_runway_safe") and any(phrase in lower for phrase in (
                "cash runway",
                "runway",
                "not reached",
            )):
                if any(phrase in lower for phrase in (
                    "unsupported",
                    "invented",
                    "threshold",
                    "benchmark",
                    "not in simulator",
                    "not grounded",
                    "no basis",
                )):
                    return True

            # Simulator-provided metric directions don't need justification
            for key, direction in (
                ("inventory_risk_label", None),
                ("overall_risk_label",   None),
            ):
                label = auth_claims.get(key, "")
                if label and label in lower:
                    if any(m in lower for m in external_justification_markers):
                        return True

            return False

        def is_false_flag_missing_risk(item: str) -> bool:
            """Return True if the 'missing_risk' is actually mentioned in the analysis."""
            lower = item.lower()

            # Map keywords in the LLM complaint to what to search for in analysis
            risk_keyword_map = [
                (["inventory risk", "inventory_risk"],   ["inventory risk", "inventory"]),
                (["overall risk",   "overall_risk"],     ["overall risk",   "medium", "risk"]),
                (["cash runway",    "runway"],           ["runway",         "cash runway"]),
                (["cash"],                               ["cash",           "ending cash"]),
                (["accounts receivable", "receivable"],  ["receivable",     "accounts receivable"]),
                (["stockout",       "lost sales"],       ["stockout",       "lost sales", "no lost"]),
                (["net margin",     "net profit"],       ["net margin",     "net profit"]),
                (["gross margin",   "gross profit"],     ["gross margin",   "gross profit"]),
            ]

            for trigger_kws, search_kws in risk_keyword_map:
                if any(kw in lower for kw in trigger_kws):
                    if any(kw in analysis_text for kw in search_kws):
                        return True   # risk IS mentioned → dismiss

            return False

        def is_false_flag_recommendation(item: str) -> bool:
            """Return True if the recommendation issue is actually backed by simulator."""
            lower = item.lower()

            # Recommendations backed by clear simulator facts are VALID
            backed_patterns = [
                # Inventory recommendation backed by High inventory risk
                (["inventory"],        auth_claims.get("inventory_risk_label", "") in ("high", "critical")),
                # Cash monitoring backed by cash decreasing
                (["cash", "monitor"],  auth_claims.get("cash_direction") == "decreased"),
                # AR monitoring backed by AR increasing
                (["collection", "receivable", "ar"], auth_claims.get("accounts_receivable_direction") == "increased"),
            ]

            for kws, backed in backed_patterns:
                if any(kw in lower for kw in kws) and backed:
                    return True

            return False

        # ----- Apply rules -----

        issue_fields_rules = [
            ("unsupported_claims",    is_false_flag_unsupported),
            ("missing_risks",         is_false_flag_missing_risk),
            ("recommendation_issues", is_false_flag_recommendation),
        ]

        for field, is_false_positive in issue_fields_rules:
            items = llm_result.get(field, [])
            real  = []
            for item in items:
                if is_false_positive(item):
                    promoted.append(f"[DISMISSED-FALSE-POSITIVE] {item}")
                else:
                    real.append(item)
            llm_result[field] = real

        # critical_issues and logic_issues: remove items that reference
        # simulator facts as if they were problems
        for field in ("critical_issues", "logic_issues"):
            items = llm_result.get(field, [])
            real  = []
            for item in items:
                lower = item.lower()
                # Dismiss if it's flagging a simulator-authoritative risk label
                if any(label in lower for label in allowed_labels):
                    if any(m in lower for m in external_justification_markers):
                        promoted.append(f"[DISMISSED-FALSE-POSITIVE] {item}")
                        continue
                real.append(item)
            llm_result[field] = real

        llm_result["verified_claims"] = llm_result.get("verified_claims", []) + promoted
        return llm_result

    # ------------------------------------------------------------------
    # SANITIZE LLM RESULT
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_llm_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move any LLM 'issue' that uses confirmation language into verified_claims.
        Qwen sometimes puts review-summaries ("This is accurate/supported/correct")
        into issue fields instead of only actual problems.
        This prevents false-positive FAILs.
        """
        confirmation_markers = (
            "is accurate",
            "is correct",
            "is supported",
            "is valid",
            "is consistent",
            "is directly supported",
            "is fully supported",
            "correctly reported",
            "correctly interpreted",
            "correctly reflected",
            "correctly identified",
            "accurately",
            "this is a correct",
            "this claim is supported",
            "does not constitute a contradiction",
            "does not contradict",
            "this is not an error",
            "this is not a factual error",
            "no error exists",
            "no numerical error",
            "no contradiction",
            "no issue here",
            "not an unsupported claim",
            "not a missing risk",
            "no recommendation issue",
            "this is a valid",
            "this is valid",
            "matches simulator",
            "matching simulator",
            "matches simulator_facts",
        )

        issue_fields = [
            "critical_issues",
            "numerical_issues",
            "logic_issues",
            "unsupported_claims",
            "missing_risks",
            "recommendation_issues",
        ]

        promoted: List[str] = []

        for field in issue_fields:
            items     = result.get(field, [])
            real      = []
            confirmed = []
            for item in items:
                lower = item.lower()
                if any(marker in lower for marker in confirmation_markers):
                    confirmed.append(item)
                else:
                    real.append(item)
            result[field] = real
            promoted.extend(confirmed)

        result["verified_claims"] = result.get("verified_claims", []) + promoted
        return result

        issue_fields = [
            "critical_issues",
            "numerical_issues",
            "logic_issues",
            "unsupported_claims",
            "missing_risks",
            "recommendation_issues",
        ]

        promoted: List[str] = []

        for field in issue_fields:
            items     = result.get(field, [])
            real      = []
            confirmed = []
            for item in items:
                lower = item.lower()
                if any(marker in lower for marker in confirmation_markers):
                    confirmed.append(item)
                else:
                    real.append(item)
            result[field] = real
            promoted.extend(confirmed)

        result["verified_claims"] = result.get("verified_claims", []) + promoted
        return result

    # ------------------------------------------------------------------
    # VALIDATE LLM RESULT
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_llm_result(result: Dict[str, Any]) -> Dict[str, Any]:
        list_fields = [
            "critical_issues", "numerical_issues", "logic_issues",
            "unsupported_claims", "missing_risks", "recommendation_issues",
            "verified_claims",
        ]
        for field in list_fields:
            val = result.get(field)
            if val is None:
                result[field] = []
            elif not isinstance(val, list):
                result[field] = [str(val)]
            else:
                result[field] = [
                    item if isinstance(item, str) else str(item)
                    for item in val
                ]

        try:
            result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
        except (TypeError, ValueError):
            result["confidence"] = 0.5

        return result

    # ------------------------------------------------------------------
    # MERGE PYTHON + LLM
    # ------------------------------------------------------------------
    @staticmethod
    def _merge(
        python_issues: Dict[str, List[str]],
        llm_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Combine Python-detected issues with LLM-detected issues.
        Python issues take priority and are prepended with [PYTHON] tag.
        LLM issues are tagged with [LLM].
        """
        def tag(items: List[str], prefix: str) -> List[str]:
            return [f"[{prefix}] {item}" for item in items]

        list_fields = [
            "critical_issues", "numerical_issues", "logic_issues",
            "unsupported_claims", "missing_risks", "recommendation_issues",
            "verified_claims",
        ]

        merged: Dict[str, Any] = {}
        for field in list_fields:
            py_items  = tag(python_issues.get(field, []),    "PYTHON")
            llm_items = tag(llm_result.get(field, []),       "LLM")
            merged[field] = py_items + llm_items

        merged["confidence"] = llm_result.get("confidence", 0.5)
        return merged

    # ------------------------------------------------------------------
    # VERDICT (deterministic)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_verdict(merged: Dict[str, Any]) -> str:
        """
        FAIL              — any critical_issues OR 2+ numerical_issues
        PASS_WITH_WARNINGS — any logic/unsupported/missing/recommendation issues
        PASS              — only verified_claims or empty lists
        """
        critical     = len(merged.get("critical_issues", []))
        numerical    = len(merged.get("numerical_issues", []))
        logic        = len(merged.get("logic_issues", []))
        unsupported  = len(merged.get("unsupported_claims", []))
        missing      = len(merged.get("missing_risks", []))
        rec_issues   = len(merged.get("recommendation_issues", []))

        if critical > 0 or numerical >= 2:
            return "FAIL"

        if numerical >= 1 or logic >= 1 or unsupported >= 1 or missing >= 1 or rec_issues >= 1:
            return "PASS_WITH_WARNINGS"

        return "PASS"
