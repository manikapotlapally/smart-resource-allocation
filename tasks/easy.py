"""
Easy Task: Assign the single most urgent volunteer to a single task.
Target: Simple skill match for one task with 1 required volunteer.
"""

from env.environment import SmartResourceEnv
from env.models import Action, AssignAction

TASK_IDS = ["task_007"]        # Digital Literacy Training – low urgency, 1 location, 3 volunteers
VOLUNTEER_IDS = ["vol_001", "vol_011", "vol_003"]  # comm/teaching skills present


def build_env() -> SmartResourceEnv:
    return SmartResourceEnv(
        task_ids=TASK_IDS,
        volunteer_ids=VOLUNTEER_IDS,
        max_steps=5,
        tick_hours=0.5,
    )


def describe() -> dict:
    return {
        "id": "easy",
        "name": "Single Volunteer Assignment",
        "description": (
            "Assign the best-matched volunteer to a digital literacy training task. "
            "The task requires 'teaching' and 'communication' skills. "
            "One of the three available volunteers is a perfect match."
        ),
        "difficulty": "easy",
        "max_steps": 5,
        "tasks": TASK_IDS,
        "volunteers": VOLUNTEER_IDS,
        "success_criteria": "Task assigned with skill-matched volunteer within 3 steps.",
    }
