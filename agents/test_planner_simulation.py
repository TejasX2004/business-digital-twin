import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.planner_agent import PlannerAgent
from backend.digital_twin.digital_twin import build_digital_twin
from backend.digital_twin.simulation import simulate


def main():

    planner = PlannerAgent(
        model="qwen3:4b-instruct-2507-q4_K_M"
    )

    query = (
        "What if sales grow by 30% and customers "
        "take 15 extra days to pay?"
    )

    print("=" * 70)
    print("USER QUESTION")
    print("=" * 70)
    print(query)

    # 1. Planner
    plan = planner.plan(query)

    print("\n" + "=" * 70)
    print("PLANNER")
    print("=" * 70)
    print(plan)

    # 2. Convert to deterministic Scenario
    scenario = planner.to_scenario(plan)

    print("\n" + "=" * 70)
    print("SCENARIO")
    print("=" * 70)
    print(scenario)

    # 3. Load Digital Twin
    twin = build_digital_twin()

    # 4. Run deterministic simulation
    result = simulate(scenario, twin)

    print("\n" + "=" * 70)
    print("SIMULATION RESULT")
    print("=" * 70)

    print(result)


if __name__ == "__main__":
    main()