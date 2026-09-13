"""
agents/test_red_team.py

Full pipeline test:
    Planner → Simulator → Analysis Agent → Red-Team Agent
"""
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.planner_agent  import PlannerAgent
from agents.analysis_agent import AnalysisAgent
from agents.red_team_agent import RedTeamAgent
from backend.digital_twin.digital_twin import build_digital_twin
from backend.digital_twin.simulation   import simulate


SEP = "=" * 70


def main():

    # ------------------------------------------------------------------
    # 1. PLANNER
    # ------------------------------------------------------------------
    planner = PlannerAgent(model="qwen3:4b-instruct-2507-q4_K_M")

    question = (
        "What if sales grow by 30% and customers "
        "take 15 extra days to pay?"
    )

    print(f"\n{SEP}")
    print("QUESTION")
    print(SEP)
    print(question)

    plan     = planner.plan(question)
    scenario = planner.to_scenario(plan)

    print(f"\n{SEP}")
    print("SCENARIO")
    print(SEP)
    print(scenario)


    # ------------------------------------------------------------------
    # 2. SIMULATION
    # ------------------------------------------------------------------
    twin              = build_digital_twin()
    simulation_result = simulate(scenario, twin)

    print(f"\n{SEP}")
    print("SIMULATION COMPLETE")
    print(SEP)

    proj = simulation_result.get("projected", {})
    base = simulation_result.get("baseline", {})
    risk = simulation_result.get("risk", {})

    print(f"  Revenue:         {base.get('revenue'):>15,.0f}  →  {proj.get('revenue'):>15,.0f}")
    print(f"  Gross Profit:    {base.get('gross_profit'):>15,.0f}  →  {proj.get('gross_profit'):>15,.0f}")
    print(f"  Cash:            {base.get('current_cash'):>15,.0f}  →  {proj.get('ending_cash'):>15,.0f}")
    print(f"  Ending AR:       {base.get('accounts_receivable'):>15,.0f}  →  {proj.get('ending_accounts_receivable'):>15,.0f}")
    print(f"  Inventory:       {base.get('inventory_value'):>15,.0f}  →  {proj.get('ending_inventory'):>15,.0f}")
    print(f"  Overall Risk:    {risk.get('overall')}")
    print(f"  Inventory Risk:  {risk.get('inventory')}")
    print(f"  Lost Sales:      {proj.get('total_lost_sales', 0):.0f}")
    print(f"  Min Coverage:    {proj.get('minimum_inventory_coverage', 0):.2f} months")
    print(f"  Cash Runway:     {proj.get('cash_runway_months')}")


    # ------------------------------------------------------------------
    # 3. ANALYSIS AGENT
    # ------------------------------------------------------------------
    analyst  = AnalysisAgent(model="qwen3:4b-instruct-2507-q4_K_M")

    print(f"\n{SEP}")
    print("ANALYSIS AGENT  (running...)")
    print(SEP)

    analysis = analyst.analyze(scenario, simulation_result)

    print(f"\nEXECUTIVE SUMMARY:\n{analysis.get('executive_summary', '')}")
    print(f"\nOVERALL ASSESSMENT: {analysis.get('overall_assessment')}")
    print(f"CONFIDENCE:         {analysis.get('confidence')}")


    # ------------------------------------------------------------------
    # 4. RED-TEAM AGENT
    # ------------------------------------------------------------------
    red_team = RedTeamAgent(model="qwen3:4b-instruct-2507-q4_K_M")

    print(f"\n{SEP}")
    print("RED-TEAM AGENT  (running...)")
    print(SEP)

    verdict = red_team.review(scenario, simulation_result, analysis)


    # ------------------------------------------------------------------
    # 5. DISPLAY
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("RED-TEAM VERDICT")
    print(SEP)

    overall_verdict = verdict.get("overall_verdict", "UNKNOWN")
    verdict_emoji   = {"PASS": "✅", "PASS_WITH_WARNINGS": "⚠️", "FAIL": "❌"}.get(overall_verdict, "❓")
    print(f"\n  {verdict_emoji}  OVERALL VERDICT: {overall_verdict}")
    print(f"  CONFIDENCE: {verdict.get('confidence', 0):.0%}")

    sections = [
        ("CRITICAL ISSUES",       "critical_issues"),
        ("NUMERICAL ISSUES",      "numerical_issues"),
        ("LOGIC ISSUES",          "logic_issues"),
        ("UNSUPPORTED CLAIMS",    "unsupported_claims"),
        ("MISSING RISKS",         "missing_risks"),
        ("RECOMMENDATION ISSUES", "recommendation_issues"),
        ("VERIFIED CLAIMS",       "verified_claims"),
    ]

    for title, key in sections:
        items = verdict.get(key, [])
        if items:
            print(f"\n  {title}:")
            for item in items:
                print(f"    • {item}")

    print(f"\n{SEP}")
    print("FULL JSON")
    print(SEP)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
