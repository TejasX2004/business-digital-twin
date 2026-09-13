"""
Demonstration of Autonomous Control Tower / Operations Agent
Objective: 'Find the biggest business risk right now.'
"""

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.operations_agent import OperationsAgent


def run_demo():
    print("=" * 70)
    print("AUTONOMOUS CONTROL TOWER - AGENTIC INVESTIGATION DEMONSTRATION")
    print("=" * 70)

    objective = "Find the biggest business risk right now."
    print(f"\n[OBJECTIVE]: {objective}\n")

    agent = OperationsAgent(max_iterations=4)
    result = agent.investigate(objective)

    print("-" * 70)
    print("1. TOOLS SELECTED & EXECUTED BY AGENT:")
    print("-" * 70)
    for t in result.get("tools_used", []):
        print(f"  • {t}")

    print("\n" + "-" * 70)
    print("2. INVESTIGATION TIMELINE & REASONING:")
    print("-" * 70)
    for step in result.get("timeline", []):
        print(f"  [{step.get('stage')}] Step {step.get('step')}: {step.get('description')}")
        print(f"     Tool: {step.get('tool')}")
        print(f"     Data Summary: {step.get('summary')}\n")

    print("-" * 70)
    print("3. AUTHORITATIVE FACTS OBSERVED:")
    print("-" * 70)
    for f in result.get("observed_facts", []):
        print(f"  • {f}")

    print("\n" + "-" * 70)
    print("4. MATERIAL ISSUES IDENTIFIED:")
    print("-" * 70)
    for issue in result.get("identified_issues", []):
        print(f"  ⚠️ {issue}")

    print("\n" + "-" * 70)
    print("5. SIMULATIONS PERFORMED:")
    print("-" * 70)
    for sim in result.get("simulations_run", []):
        if isinstance(sim, dict):
            args = sim.get("args", {})
            res = sim.get("result", {})
            fi = res.get("financial_impact", {})
            li = res.get("liquidity_impact", {})
            ra = res.get("risk_assessment", {})
            print(f"  • Parameters: {args}")
            print(f"    Projected Revenue: Rs. {fi.get('projected_revenue', 0):,.2f}")
            print(f"    Projected Ending Cash: Rs. {li.get('projected_ending_cash', 0):,.2f}")
            print(f"    Projected Net Profit: Rs. {fi.get('projected_net_profit', 0):,.2f}")
            print(f"    Assessed Overall Risk: {ra.get('overall_risk', 'Unknown')}")
        else:
            print(f"  • {sim}")

    print("\n" + "-" * 70)
    print("6. RETRIEVED COMPANY POLICIES (RAG):")
    print("-" * 70)
    for pol in result.get("business_knowledge_used", []):
        if isinstance(pol, dict):
            print(f"  • Source: {pol.get('source')}")
            content_preview = pol.get('content', '').replace('\n', ' ')[:140]
            print(f"    Rule Excerpt: {content_preview}...\n")
        else:
            print(f"  • {pol}")

    print("-" * 70)
    print("7. STRATEGIC RECOMMENDATION & ACTION:")
    print("-" * 70)
    print(f"  Recommendation: {result.get('recommendation')}")
    print(f"  Concrete Action: {result.get('recommended_action')}")
    print(f"  Requires User Approval: {result.get('requires_user_approval')} (Human-in-the-loop enforced)")
    print(f"  Agent Confidence: {result.get('confidence')}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_demo()
