"""
Pipeline State and Result Definitions for Digital Twin Orchestrator.
Defines structured results, stage statuses, and execution metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from backend.digital_twin.simulation import Scenario


class StageStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class PipelineStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"


@dataclass
class PipelineResult:
    """
    Comprehensive container for the output of the complete Digital Twin pipeline.
    Preserves all stage artifacts and tracks granular execution status.
    """
    question: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None
    scenario: Optional[Scenario] = None
    simulation: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    red_team: Optional[Dict[str, Any]] = None
    decision: Optional[Dict[str, Any]] = None
    pipeline_status: str = PipelineStatus.FAILED.value
    stage_statuses: Dict[str, str] = field(default_factory=lambda: {
        "planner": StageStatus.PENDING.value,
        "simulator": StageStatus.PENDING.value,
        "analysis": StageStatus.PENDING.value,
        "red_team": StageStatus.PENDING.value,
        "decision": StageStatus.PENDING.value,
    })
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)

    def get_status_indicator(self) -> str:
        """
        Returns a formatted pipeline status string with appropriate stage icons:
        e.g., 'Planner ✓ → Simulator ✓ → Analysis ✓ → Red-Team ✓ → Decision ✓'
        """
        icon_map = {
            StageStatus.SUCCESS.value: "✓",
            StageStatus.WARNING.value: "⚠️",
            StageStatus.FAILED.value: "✗",
            StageStatus.SKIPPED.value: "○",
            StageStatus.PENDING.value: "⋯",
            StageStatus.RUNNING.value: "⏳",
        }

        stages = [
            ("Planner", self.stage_statuses.get("planner", StageStatus.PENDING.value)),
            ("Simulator", self.stage_statuses.get("simulator", StageStatus.PENDING.value)),
            ("Analysis", self.stage_statuses.get("analysis", StageStatus.PENDING.value)),
            ("Red-Team", self.stage_statuses.get("red_team", StageStatus.PENDING.value)),
            ("Decision", self.stage_statuses.get("decision", StageStatus.PENDING.value)),
        ]

        formatted = [f"{name} {icon_map.get(status, '⋯')}" for name, status in stages]
        return " → ".join(formatted)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result object into JSON-serializable dictionary."""
        return {
            "question": self.question,
            "plan": self.plan,
            "scenario": vars(self.scenario) if self.scenario else None,
            "simulation": self.simulation,
            "analysis": self.analysis,
            "red_team": self.red_team,
            "decision": self.decision,
            "pipeline_status": self.pipeline_status,
            "stage_statuses": self.stage_statuses,
            "errors": self.errors,
            "warnings": self.warnings,
            "execution_metadata": self.execution_metadata,
            "status_indicator": self.get_status_indicator(),
        }
