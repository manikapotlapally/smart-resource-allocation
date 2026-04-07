"""
StateManager: tracks all mutable state across episodes.
"""

import json
import copy
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


class StateManager:
    """
    Owns the ground-truth state of the environment.
    All mutations go through this class to ensure consistency.
    """

    def __init__(self):
        self._raw_tasks: List[dict] = []
        self._raw_volunteers: List[dict] = []
        self.tasks: Dict[str, dict] = {}
        self.volunteers: Dict[str, dict] = {}
        self.allocations: Dict[str, str] = {}          # volunteer_id → task_id
        self.task_volunteers: Dict[str, List[str]] = {} # task_id → [vol_ids]
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self.step_count: int = 0
        self.last_actions: List[dict] = []             # for repeat-action detection
        self._load_data()

    def _load_data(self):
        with open(DATA_DIR / "tasks.json") as f:
            self._raw_tasks = json.load(f)
        with open(DATA_DIR / "volunteers.json") as f:
            self._raw_volunteers = json.load(f)

    def reset(self, task_ids: Optional[List[str]] = None, volunteer_ids: Optional[List[str]] = None):
        """Reset state to initial conditions, optionally scoped to specific IDs."""
        self.tasks = {}
        self.volunteers = {}
        self.allocations = {}
        self.task_volunteers = {}
        self.completed_tasks = set()
        self.failed_tasks = set()
        self.step_count = 0
        self.last_actions = []

        for t in self._raw_tasks:
            if task_ids is None or t["id"] in task_ids:
                task = copy.deepcopy(t)
                task["status"] = "open"
                task["assigned_volunteers"] = []
                self.tasks[task["id"]] = task
                self.task_volunteers[task["id"]] = []

        for v in self._raw_volunteers:
            if volunteer_ids is None or v["id"] in volunteer_ids:
                if v["active"]:
                    vol = copy.deepcopy(v)
                    vol["currently_assigned_task"] = None
                    vol["hours_committed"] = 0.0
                    self.volunteers[vol["id"]] = vol

    def assign(self, volunteer_id: str, task_id: str) -> Tuple[bool, str]:
        """Attempt assignment. Returns (success, reason)."""
        if volunteer_id not in self.volunteers:
            return False, f"Volunteer {volunteer_id} not found or inactive"
        if task_id not in self.tasks:
            return False, f"Task {task_id} not found"
        if self.tasks[task_id]["status"] != "open":
            return False, f"Task {task_id} is not open"
        if self.allocations.get(volunteer_id) == task_id:
            return False, f"Volunteer {volunteer_id} already on task {task_id}"

        vol = self.volunteers[volunteer_id]
        task = self.tasks[task_id]

        # Check if volunteer is already assigned elsewhere
        if vol["currently_assigned_task"] is not None:
            return False, f"Volunteer {volunteer_id} already assigned to {vol['currently_assigned_task']}"

        # Check hours
        if vol["hours_committed"] + task["duration_hours"] > vol["availability_hours"]:
            return False, f"Volunteer {volunteer_id} doesn't have enough available hours"

        # Do the assignment
        self.allocations[volunteer_id] = task_id
        self.task_volunteers[task_id].append(volunteer_id)
        self.tasks[task_id]["assigned_volunteers"].append(volunteer_id)
        vol["currently_assigned_task"] = task_id
        vol["hours_committed"] = vol.get("hours_committed", 0) + task["duration_hours"]

        # Mark task as assigned if enough volunteers
        if len(self.task_volunteers[task_id]) >= task["required_volunteers"]:
            self.tasks[task_id]["status"] = "assigned"

        return True, "ok"

    def unassign(self, volunteer_id: str, task_id: str) -> Tuple[bool, str]:
        """Remove a volunteer from a task."""
        if self.allocations.get(volunteer_id) != task_id:
            return False, f"Volunteer {volunteer_id} not assigned to task {task_id}"

        vol = self.volunteers[volunteer_id]
        task = self.tasks[task_id]

        self.allocations.pop(volunteer_id)
        if volunteer_id in self.task_volunteers[task_id]:
            self.task_volunteers[task_id].remove(volunteer_id)
        if volunteer_id in task["assigned_volunteers"]:
            task["assigned_volunteers"].remove(volunteer_id)
        vol["currently_assigned_task"] = None
        vol["hours_committed"] = max(0, vol.get("hours_committed", 0) - task["duration_hours"])

        # Revert task status if not enough volunteers
        if task["status"] == "assigned" and len(self.task_volunteers[task_id]) < task["required_volunteers"]:
            task["status"] = "open"

        return True, "ok"

    def tick_deadlines(self, hours: float = 1.0):
        """Advance time — reduces deadlines and fails overdue open tasks."""
        for tid, task in self.tasks.items():
            if task["status"] == "open":
                task["deadline_hours"] = max(0, task["deadline_hours"] - hours)
                if task["deadline_hours"] == 0:
                    self.failed_tasks.add(tid)
                    task["status"] = "failed"

    def get_open_tasks(self) -> List[dict]:
        return [t for t in self.tasks.values() if t["status"] == "open"]

    def get_available_volunteers(self) -> List[dict]:
        return [v for v in self.volunteers.values() if v["currently_assigned_task"] is None]

    def get_all_allocations(self) -> List[dict]:
        result = []
        for vid, tid in self.allocations.items():
            vol = self.volunteers.get(vid, {})
            task = self.tasks.get(tid, {})
            skill_match = self._compute_skill_match(vol.get("skills", []), task.get("required_skills", []))
            result.append({
                "volunteer_id": vid,
                "task_id": tid,
                "match_score": skill_match
            })
        return result

    def _compute_skill_match(self, vol_skills: List[str], required_skills: List[str]) -> float:
        if not required_skills:
            return 1.0
        matched = sum(1 for s in required_skills if s in vol_skills)
        return matched / len(required_skills)

    def get_snapshot(self) -> dict:
        return {
            "tasks": copy.deepcopy(self.tasks),
            "volunteers": copy.deepcopy(self.volunteers),
            "allocations": copy.deepcopy(self.allocations),
            "task_volunteers": copy.deepcopy(self.task_volunteers),
            "completed_tasks": list(self.completed_tasks),
            "failed_tasks": list(self.failed_tasks),
            "step_count": self.step_count,
        }
