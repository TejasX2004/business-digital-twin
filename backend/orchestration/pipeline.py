"""
Digital Twin Orchestration Pipeline.
Single orchestration entry point connecting:
Question -> Planner -> Scenario -> Simulator -> Analysis -> Red-Team -> Decision -> PipelineResult
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional, Union

from backend.digital_twin.digital_twin import build_digital_twin
from backend.digital_twin.simulation import Scenario, simulate
from agents.planner_agent import PlannerAgent
from agents.analysis_agent import AnalysisAgent
from agents.red_team_agent import RedTeamAgent
from agents.decision_agent import DecisionAgent

from backend.orchestration.state import PipelineResult, PipelineStatus, StageStatus


class DigitalTwinOrchestrator:
    """
    Coordinates the execution of the full Business Digital Twin intelligence pipeline.
    Maintains deterministic guardrails, prevents simulator state mutation, and
    ensures graceful degradation if downstream LLM stages fail.
    """

    def __init__(
        self,
        twin: Optional[Dict[str, Any]] = None,
        planner: Optional[PlannerAgent] = None,
        analyst: Optional[AnalysisAgent] = None,
        red_team: Optional[RedTeamAgent] = None,
        decision_agent: Optional[DecisionAgent] = None,
    ):
        self.twin = twin if twin is not None else build_digital_twin()
        self.planner = planner if planner is not None else PlannerAgent()
        self.analyst = analyst if analyst is not None else AnalysisAgent()
        self.red_team = red_team if red_team is not None else RedTeamAgent()
        self.decision_agent = decision_agent if decision_agent is not None else DecisionAgent()

    def run(self, input_data: Union[str, Scenario]) -> PipelineResult:
        """
        Execute the full pipeline:
        Planner -> Simulator -> Analysis -> Red-Team -> Decision
        """
        start_time = time.time()
        result = PipelineResult()
        stage_timings: Dict[str, float] = {}

        # ----------------------------------------------------
        # STAGE 1: PLANNER
        # ----------------------------------------------------
        t0 = time.time()
        result.stage_statuses["planner"] = StageStatus.RUNNING.value

        if isinstance(input_data, str):
            result.question = input_data.strip()
            if not result.question:
                result.stage_statuses["planner"] = StageStatus.FAILED.value
                result.errors.append("Planner Error: Question cannot be empty.")
                self._mark_skipped(result, ["simulator", "analysis", "red_team", "decision"])
                result.pipeline_status = PipelineStatus.FAILED.value
                return result

            try:
                plan_dict = self.planner.plan(result.question)
                scenario_obj = self.planner.to_scenario(plan_dict)
                result.plan = plan_dict
                result.scenario = scenario_obj
                result.stage_statuses["planner"] = StageStatus.SUCCESS.value
            except Exception as e:
                result.stage_statuses["planner"] = StageStatus.FAILED.value
                result.errors.append(f"Planner Error: {e}")
                self._mark_skipped(result, ["simulator", "analysis", "red_team", "decision"])
                result.pipeline_status = PipelineStatus.FAILED.value
                result.execution_metadata["total_duration_sec"] = round(time.time() - start_time, 2)
                return result

        elif isinstance(input_data, Scenario):
            result.question = "Manual Scenario Execution"
            result.scenario = input_data
            result.stage_statuses["planner"] = StageStatus.SUCCESS.value
            result.plan = {
                "interpretation": "Direct Scenario Input",
                "sales_growth": input_data.sales_growth,
                "price_change": input_data.price_change,
                "cost_change": input_data.cost_change,
                "payment_delay_days": input_data.payment_delay_days,
                "inventory_change": input_data.inventory_change,
                "expense_change": input_data.expense_change,
                "supplier_payment_delay_days": input_data.supplier_payment_delay_days,
                "months": input_data.months,
            }
        else:
            result.stage_statuses["planner"] = StageStatus.FAILED.value
            result.errors.append(f"Planner Error: Invalid input type {type(input_data).__name__}. Expected str or Scenario.")
            self._mark_skipped(result, ["simulator", "analysis", "red_team", "decision"])
            result.pipeline_status = PipelineStatus.FAILED.value
            return result

        stage_timings["planner_sec"] = round(time.time() - t0, 2)

        # ----------------------------------------------------
        # STAGE 2: DIGITAL TWIN + SIMULATOR (CRITICAL)
        # ----------------------------------------------------
        t0 = time.time()
        result.stage_statuses["simulator"] = StageStatus.RUNNING.value

        try:
            raw_sim = simulate(result.scenario, self.twin)
            # Freeze/deepcopy simulation results so downstream agents cannot mutate authoritative numbers
            result.simulation = copy.deepcopy(raw_sim)
            result.stage_statuses["simulator"] = StageStatus.SUCCESS.value
        except Exception as e:
            result.stage_statuses["simulator"] = StageStatus.FAILED.value
            result.errors.append(f"Simulator Error: {e}")
            self._mark_skipped(result, ["analysis", "red_team", "decision"])
            result.pipeline_status = PipelineStatus.FAILED.value
            result.execution_metadata["stage_timings"] = stage_timings
            result.execution_metadata["total_duration_sec"] = round(time.time() - start_time, 2)
            return result

        stage_timings["simulator_sec"] = round(time.time() - t0, 2)

        # ----------------------------------------------------
        # STAGE 3: ANALYSIS AGENT (NON-CRITICAL LLM)
        # ----------------------------------------------------
        t0 = time.time()
        result.stage_statuses["analysis"] = StageStatus.RUNNING.value

        try:
            # Pass fresh deepcopy of simulator facts to Analysis Agent
            analysis_facts = copy.deepcopy(result.simulation)
            analysis_out = self.analyst.analyze(result.scenario, analysis_facts)
            result.analysis = analysis_out
            result.stage_statuses["analysis"] = StageStatus.SUCCESS.value
        except Exception as e:
            result.stage_statuses["analysis"] = StageStatus.FAILED.value
            result.errors.append(f"Analysis Agent Error: {e}")
            result.warnings.append("AI Analysis failed; simulation KPIs and charts are preserved.")
            self._mark_skipped(result, ["red_team", "decision"])
            result.pipeline_status = PipelineStatus.PARTIAL_SUCCESS.value
            stage_timings["analysis_sec"] = round(time.time() - t0, 2)
            result.execution_metadata["stage_timings"] = stage_timings
            result.execution_metadata["total_duration_sec"] = round(time.time() - start_time, 2)
            return result

        stage_timings["analysis_sec"] = round(time.time() - t0, 2)

        # ----------------------------------------------------
        # STAGE 4: RED-TEAM AGENT (NON-CRITICAL LLM)
        # ----------------------------------------------------
        t0 = time.time()
        result.stage_statuses["red_team"] = StageStatus.RUNNING.value

        try:
            # Pass fresh deepcopy of simulator facts and analysis to Red-Team
            rt_sim_facts = copy.deepcopy(result.simulation)
            rt_analysis = copy.deepcopy(result.analysis)
            rt_verdict = self.red_team.review(result.scenario, rt_sim_facts, rt_analysis)
            result.red_team = rt_verdict

            # Flag warning status if Red-Team returned warnings or issues
            if rt_verdict.get("overall_verdict") == "PASS_WITH_WARNINGS":
                result.stage_statuses["red_team"] = StageStatus.WARNING.value
                result.warnings.append("Red-Team audit passed with warnings.")
            elif rt_verdict.get("overall_verdict") == "FAIL":
                result.stage_statuses["red_team"] = StageStatus.WARNING.value
                result.warnings.append("Red-Team audit flagged critical issues in analysis.")
            else:
                result.stage_statuses["red_team"] = StageStatus.SUCCESS.value
        except Exception as e:
            result.stage_statuses["red_team"] = StageStatus.FAILED.value
            result.errors.append(f"Red-Team Agent Error: {e}")
            result.warnings.append("Red-Team audit failed; upstream analysis and simulation preserved.")
            self._mark_skipped(result, ["decision"])
            result.pipeline_status = PipelineStatus.PARTIAL_SUCCESS.value
            stage_timings["red_team_sec"] = round(time.time() - t0, 2)
            result.execution_metadata["stage_timings"] = stage_timings
            result.execution_metadata["total_duration_sec"] = round(time.time() - start_time, 2)
            return result

        stage_timings["red_team_sec"] = round(time.time() - t0, 2)

        # ----------------------------------------------------
        # STAGE 5: DECISION AGENT (NON-CRITICAL LLM)
        # ----------------------------------------------------
        t0 = time.time()
        result.stage_statuses["decision"] = StageStatus.RUNNING.value

        try:
            # Pass authoritative facts, analysis, and Red-Team verdict to Decision Agent
            dec_sim_facts = copy.deepcopy(result.simulation)
            dec_analysis = copy.deepcopy(result.analysis)
            dec_rt = copy.deepcopy(result.red_team)

            decision_out = self.decision_agent.decide(
                result.scenario,
                dec_sim_facts,
                dec_analysis,
                dec_rt
            )
            result.decision = decision_out
            result.stage_statuses["decision"] = StageStatus.SUCCESS.value
            result.pipeline_status = PipelineStatus.SUCCESS.value
        except Exception as e:
            result.stage_statuses["decision"] = StageStatus.FAILED.value
            result.errors.append(f"Decision Agent Error: {e}")
            result.warnings.append("Business decision generation failed; analysis and audit preserved.")
            result.pipeline_status = PipelineStatus.PARTIAL_SUCCESS.value

        stage_timings["decision_sec"] = round(time.time() - t0, 2)

        # ----------------------------------------------------
        # EXECUTION METADATA
        # ----------------------------------------------------
        result.execution_metadata["stage_timings"] = stage_timings
        result.execution_metadata["total_duration_sec"] = round(time.time() - start_time, 2)

        return result

    @staticmethod
    def _mark_skipped(result: PipelineResult, stages: list[str]):
        for stage in stages:
            result.stage_statuses[stage] = StageStatus.SKIPPED.value


def run_digital_twin_pipeline(
    question: Union[str, Scenario],
    twin: Optional[Dict[str, Any]] = None,
    planner: Optional[PlannerAgent] = None,
    analyst: Optional[AnalysisAgent] = None,
    red_team: Optional[RedTeamAgent] = None,
    decision_agent: Optional[DecisionAgent] = None,
) -> PipelineResult:
    """
    Public single entry point for orchestrating the complete Digital Twin workflow:
    Question/Scenario -> Planner -> Simulator -> Analysis -> Red-Team -> Decision -> PipelineResult
    """
    orchestrator = DigitalTwinOrchestrator(
        twin=twin,
        planner=planner,
        analyst=analyst,
        red_team=red_team,
        decision_agent=decision_agent,
    )
    return orchestrator.run(question)
