"""
SmartResourceEnv — OpenEnv-compliant environment for volunteer–task allocation.
Implements: reset(), step(), state() per OpenEnv spec.
"""

import copy
from typing import List, Optional, Tuple, Dict, Any

from env.models import Observation, Action, Reward, TaskSummary, VolunteerSummary, AllocationRecord
from env.state_manager import StateManager
from env.reward import compute_reward


class SmartResourceEnv:
    """
    An OpenEnv environment that simulates NGO volunteer coordination.
    
    The agent must intelligently assign available volunteers to open community
    tasks, maximising skill fit, urgency response, and beneficiary coverage.
    
    Episode terminates when:
      - All tasks are assigned / completed / failed, OR
      - max_steps is reached.
    """

    metadata = {
        "name": "smart-resource-env",
        "version": "1.0.0",
        "description": "Volunteer-to-task allocation for NGO social impact operations",
        "action_space": "AssignAction | UnassignAction | PrioritizeAction | skip",
        "observation_space": "TaskSummary[], VolunteerSummary[], AllocationRecord[]",
        "reward_range": (-1.0, 1.0),
    }

    def __init__(
        self,
        task_ids: Optional[List[str]] = None,
        volunteer_ids: Optional[List[str]] = None,
        max_steps: int = 20,
        tick_hours: float = 1.0,
    ):
        self.task_ids = task_ids
        self.volunteer_ids = volunteer_ids
        self.max_steps = max_steps
        self.tick_hours = tick_hours

        self._state = StateManager()
        self._cumulative_reward = 0.0
        self._done = False
        self._step_count = 0

    # ─── OpenEnv API ─────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset to a clean initial state and return the first observation."""
        self._state.reset(task_ids=self.task_ids, volunteer_ids=self.volunteer_ids)
        self._cumulative_reward = 0.0
        self._done = False
        self._step_count = 0
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one agent action.
        Returns: (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        state_before = self._state.get_snapshot()
        assignment_results = []

        if not action.skip:
            # Process assignments
            for assign in action.assignments:
                success, reason = self._state.assign(assign.volunteer_id, assign.task_id)
                assignment_results.append({
                    "type": "assign",
                    "volunteer_id": assign.volunteer_id,
                    "task_id": assign.task_id,
                    "success": success,
                    "reason": reason,
                })

            # Process unassignments
            for unassign in action.unassignments:
                success, reason = self._state.unassign(unassign.volunteer_id, unassign.task_id)
                assignment_results.append({
                    "type": "unassign",
                    "volunteer_id": unassign.volunteer_id,
                    "task_id": unassign.task_id,
                    "success": success,
                    "reason": reason,
                })

        # Advance time / deadlines
        self._state.tick_deadlines(hours=self.tick_hours)
        self._step_count += 1
        self._state.step_count = self._step_count

        state_after = self._state.get_snapshot()

        # Compute reward
        reward = compute_reward(action, state_before, state_after, assignment_results)
        self._cumulative_reward += reward.total

        # Check termination
        open_count = len(self._state.get_open_tasks())
        self._done = (self._step_count >= self.max_steps) or (open_count == 0)

        obs = self._build_observation()

        info = {
            "step": self._step_count,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "assignment_results": assignment_results,
            "open_tasks_remaining": open_count,
            "failed_tasks": list(self._state.failed_tasks),
        }

        return obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return the full current state (for inspection/debugging)."""
        return self._state.get_snapshot()

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        open_tasks = [
            TaskSummary(
                id=t["id"],
                title=t["title"],
                location=t["location"],
                urgency=t["urgency"],
                required_volunteers=t["required_volunteers"],
                required_skills=t["required_skills"],
                duration_hours=t["duration_hours"],
                deadline_hours=t["deadline_hours"],
                status=t["status"],
                beneficiaries=t["beneficiaries"],
                category=t["category"],
                assigned_volunteers=t.get("assigned_volunteers", []),
            )
            for t in self._state.get_open_tasks()
        ]

        available_vols = [
            VolunteerSummary(
                id=v["id"],
                name=v["name"],
                location=v["location"],
                availability_hours=v["availability_hours"],
                skills=v["skills"],
                reliability_score=v["reliability_score"],
                active=v["active"],
                max_travel_km=v["max_travel_km"],
                currently_assigned_task=v.get("currently_assigned_task"),
            )
            for v in self._state.get_available_volunteers()
        ]

        allocations = [
            AllocationRecord(
                volunteer_id=a["volunteer_id"],
                task_id=a["task_id"],
                match_score=a["match_score"],
            )
            for a in self._state.get_all_allocations()
        ]

        # Generate alerts for near-deadline urgent tasks
        alerts = []
        for t in self._state.tasks.values():
            if t["status"] == "open" and t["deadline_hours"] <= 4 and t["urgency"] >= 7:
                alerts.append(
                    f"⚠️ URGENT: '{t['title']}' deadline in {t['deadline_hours']:.1f}h (urgency {t['urgency']})"
                )

        return Observation(
            open_tasks=open_tasks,
            available_volunteers=available_vols,
            current_allocations=allocations,
            step_number=self._step_count,
            total_steps=self.max_steps,
            episode_score=max(0.0, min(1.0, self._cumulative_reward)),
            alerts=alerts,
        )
