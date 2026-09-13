"""
agents/test_decision_agent.py

Full end-to-end pipeline test:
    Scenario → Simulator → Analysis → Red-Team → Decision

Question:
    "What if sales grow by 30% and customers take 15 extra days to pay?"
"""

import sys
import os
import json

# Ensure unbuffered output on Windows
sys.stdout.reconfigure(line_buffering=True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.planner_agent import PlannerAgent
from agents.analysis_agent import AnalysisAgent
from agents.red_team_agent import RedTeamAgent
from agents.decision_agent import DecisionAgent
from backend.digital_twin.digital_twin import build_digital_twin
from backend.digital_twin.simulation import simulate

SEP = "=" * 70


def main():
    print(f"\n{SEP}")
    print("DIGITAL TWIN PIPELINE: PLANNER -> SIMULATOR -> ANALYSIS -> RED-TEAM -> DECISION")
    print(SEP)

    # ------------------------------------------------------------------
    # 1. PLANNER AGENT
    # ------------------------------------------------------------------
    question = "What if sales grow by 30% and customers take 15 extra days to pay?"
    print(f"\n1. QUESTION:\n   {question}")

    planner = PlannerAgent(model="qwen3:4b-instruct-2507-q4_K_M")
    print("\n   Running Planner Agent...")
    plan = planner.plan(question)
    scenario = planner.to_scenario(plan)
    print(f"   Parsed Scenario: {scenario}")

    # ------------------------------------------------------------------
    # 2. SIMULATOR
    # ------------------------------------------------------------------
    print("\n2. SIMULATOR:")
    print("   Building digital twin and running simulation...")
    twin = build_digital_twin()
    sim_result = simulate(scenario, twin)

    base = sim_result.get("baseline", {})
    proj = sim_result.get("projected", {})
    risk = sim_result.get("risk", {})

    print(f"   Revenue:      INR {base.get('revenue', 0):>12,.0f}  ->  INR {proj.get('revenue', 0):>12,.0f}")
    print(f"   Gross Profit: INR {base.get('gross_profit', 0):>12,.0f}  ->  INR {proj.get('gross_profit', 0):>12,.0f}")
    print(f"   Cash:         INR {base.get('current_cash', 0):>12,.0f}  ->  INR {proj.get('ending_cash', 0):>12,.0f}")
    print(f"   Ending AR:    INR {base.get('accounts_receivable', 0):>12,.0f}  ->  INR {proj.get('ending_accounts_receivable', 0):>12,.0f}")
    print(f"   Inventory:    INR {base.get('inventory_value', 0):>12,.0f}  ->  INR {proj.get('ending_inventory', 0):>12,.0f}")
    print(f"   Overall Risk: {risk.get('overall')} | Inventory Risk: {risk.get('inventory')}")

    # ------------------------------------------------------------------
    # 3. ANALYSIS AGENT
    # ------------------------------------------------------------------
    print("\n3. ANALYSIS AGENT:")
    print("   Running Analysis Agent...")
    analyst = AnalysisAgent(model="qwen3:4b-instruct-2507-q4_K_M")
    analysis = analyst.analyze(scenario, sim_result)
    print(f"   Assessment: {analysis.get('overall_assessment')} | Confidence: {analysis.get('confidence')}")
    print(f"   Summary: {analysis.get('executive_summary')[:120]}...")

    # ------------------------------------------------------------------
    # 4. RED-TEAM AGENT
    # ------------------------------------------------------------------
    print("\n4. RED-TEAM AGENT:")
    print("   Running Red-Team Agent...")
    red_team = RedTeamAgent(model="qwen3:4b-instruct-2507-q4_K_M")
    red_team_verdict = red_team.review(scenario, sim_result, analysis)
    verdict_str = red_team_verdict.get("overall_verdict", "UNKNOWN")
    print(f"   Verdict: {verdict_str} | Confidence: {red_team_verdict.get('confidence', 0):.0%}")
    print(f"   Critical Issues: {len(red_team_verdict.get('critical_issues', []))}")
    print(f"   Numerical Issues: {len(red_team_verdict.get('numerical_issues', []))}")

    # ------------------------------------------------------------------
    # 5. DECISION AGENT
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("5. DECISION AGENT")
    print(SEP)
    decision_agent = DecisionAgent(model="qwen3:4b-instruct-2507-q4_K_M")
    print("   Running Decision Agent...")
    decision = decision_agent.decide(scenario, sim_result, analysis, red_team_verdict)

    print(f"\n{SEP}")
    print("FINAL DECISION AGENT OUTPUT")
    print(SEP)
    print(f"Decision:   {decision.get('decision')}")
    print(f"Priority:   {decision.get('priority')}")
    print(f"Confidence: {decision.get('confidence')}")

    print("\nRationale:")
    for r in decision.get("rationale", []):
        print(f"  • {r}")

    print("\nRecommended Actions:")
    for a in decision.get("recommended_actions", []):
        print(f"  • {a}")

    print("\nExpected Benefits:")
    for b in decision.get("expected_benefits", []):
        print(f"  • {b}")

    print("\nRisks:")
    for rk in decision.get("risks", []):
        print(f"  • {rk}")

    print("\nConditions:")
    for c in decision.get("conditions", []):
        print(f"  • {c}")

    print(f"\n{SEP}")
    print("RAW DECISION JSON")
    print(SEP)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
