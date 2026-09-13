import sys
import os

# Add parent directory to sys.path so it can find the 'backend' and 'agents' modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.planner_agent import PlannerAgent
from agents.analysis_agent import AnalysisAgent

from backend.digital_twin.digital_twin import build_digital_twin
from backend.digital_twin.simulation import simulate


def main():

    # -----------------------------
    # 1. PLANNER
    # -----------------------------

    planner = PlannerAgent(
        model="qwen3:4b-instruct-2507-q4_K_M"
    )

    question = (
        "What if sales grow by 30% and customers "
        "take 15 extra days to pay?"
    )

    plan = planner.plan(question)

    scenario = planner.to_scenario(plan)

    print("\nSCENARIO")
    print(scenario)


    # -----------------------------
    # 2. DIGITAL TWIN
    # -----------------------------

    twin = build_digital_twin()


    # -----------------------------
    # 3. SIMULATION
    # -----------------------------

    simulation_result = simulate(
        scenario,
        twin
    )

    print("\nSIMULATION COMPLETE")


    # -----------------------------
    # 4. ANALYSIS AGENT
    # -----------------------------

    analyst = AnalysisAgent(
        model="qwen3:4b-instruct-2507-q4_K_M"
    )

    analysis = analyst.analyze(
        scenario,
        simulation_result
    )


    # -----------------------------
    # 5. DISPLAY
    # -----------------------------

    print("\n" + "=" * 70)
    print("ANALYSIS AGENT")
    print("=" * 70)

    print(
        "\nEXECUTIVE SUMMARY:"
    )

    print(
        analysis["executive_summary"]
    )

    print(
        "\nKEY DRIVERS:"
    )

    for item in analysis["key_drivers"]:
        print("-", item)

    print(
        "\nFINANCIAL IMPACT:"
    )

    for item in analysis["financial_impact"]:
        print("-", item)

    print(
        "\nCASH FLOW IMPACT:"
    )

    for item in analysis["cash_flow_impact"]:
        print("-", item)

    print(
        "\nWORKING CAPITAL:"
    )

    for item in analysis["working_capital_impact"]:
        print("-", item)

    print(
        "\nINVENTORY:"
    )

    for item in analysis["inventory_analysis"]:
        print("-", item)

    print(
        "\nRISKS:"
    )

    for item in analysis["risk_analysis"]:
        print("-", item)

    print(
        "\nTRADE-OFFS:"
    )

    for item in analysis["tradeoffs"]:
        print("-", item)

    print(
        "\nRECOMMENDATIONS:"
    )

    for item in analysis["recommendations"]:
        print("-", item)

    print(
        "\nOVERALL ASSESSMENT:",
        analysis["overall_assessment"]
    )

    print(
        "CONFIDENCE:",
        analysis["confidence"]
    )


if __name__ == "__main__":
    main()