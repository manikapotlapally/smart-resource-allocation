"""
Typed Pydantic models for the Smart Resource Allocation OpenEnv environment.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ─── Sub-models ────────────────────────────────────────────────────────────────

class TaskSummary(BaseModel):
    id: str
    title: str
    location: str
    urgency: int = Field(..., ge=1, le=10)
    required_volunteers: int
    required_skills: List[str]
    duration_hours: float
    deadline_hours: float
    status: str  # "open" | "assigned" | "completed"
    beneficiaries: int
    category: str
    assigned_volunteers: List[str] = Field(default_factory=list)


class VolunteerSummary(BaseModel):
    id: str
    name: str
    location: str
    availability_hours: float
    skills: List[str]
    reliability_score: float = Field(..., ge=0.0, le=1.0)
    active: bool
    max_travel_km: int
    currently_assigned_task: Optional[str] = None


class AllocationRecord(BaseModel):
    volunteer_id: str
    task_id: str
    match_score: float = Field(..., ge=0.0, le=1.0)


# ─── Observation ───────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """
    What the agent sees at each step.
    Contains unassigned tasks, available volunteers, and current allocations.
    """
    open_tasks: List[TaskSummary]
    available_volunteers: List[VolunteerSummary]
    current_allocations: List[AllocationRecord]
    step_number: int
    total_steps: int
    episode_score: float
    alerts: List[str] = Field(default_factory=list)  # e.g. "Flood task deadline in 2h"


# ─── Action ────────────────────────────────────────────────────────────────────

class AssignAction(BaseModel):
    volunteer_id: str
    task_id: str


class UnassignAction(BaseModel):
    volunteer_id: str
    task_id: str


class PrioritizeAction(BaseModel):
    task_id: str
    priority_boost: float = Field(..., ge=0.0, le=1.0)


class Action(BaseModel):
    """
    Agent's action at each step.
    Can perform multiple assignments in one step.
    """
    assignments: List[AssignAction] = Field(default_factory=list)
    unassignments: List[UnassignAction] = Field(default_factory=list)
    prioritizations: List[PrioritizeAction] = Field(default_factory=list)
    skip: bool = False  # Agent passes/does nothing


# ─── Reward ────────────────────────────────────────────────────────────────────

class Reward(BaseModel):
    """
    Detailed reward breakdown for transparency and debugging.
    """
    total: float = Field(..., ge=-1.0, le=1.0)

    # Positive components
    skill_match_score: float = Field(default=0.0)    # Skills covered
    urgency_score: float = Field(default=0.0)         # High-urgency tasks first
    coverage_score: float = Field(default=0.0)        # Volunteer slots filled
    deadline_score: float = Field(default=0.0)        # Within deadline
    beneficiary_score: float = Field(default=0.0)     # People helped

    # Penalty components
    overassignment_penalty: float = Field(default=0.0)   # Vol assigned > availability
    skill_mismatch_penalty: float = Field(default=0.0)   # Wrong skills assigned
    idle_penalty: float = Field(default=0.0)              # High-urgency tasks ignored
    repeat_action_penalty: float = Field(default=0.0)    # Same action twice

    info: Dict[str, Any] = Field(default_factory=dict)
