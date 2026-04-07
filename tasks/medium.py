"""
Medium Task: Allocate volunteers across 3 simultaneous open tasks with
mixed urgency levels, varying skill requirements, and limited volunteer pool.
"""

from env.environment import SmartResourceEnv

TASK_IDS = ["task_001", "task_002", "task_005"]
# Food Distribution (logistics, physical), Medical Camp (medical, comm), Tree Plantation (env, physical)
VOLUNTEER_IDS = ["vol_001", "vol_002", "vol_003", "vol_006", "vol_009"]


def build_env() -> SmartResourceEnv:
    return SmartResourceEnv(
        task_ids=TASK_IDS,
        volunteer_ids=VOLUNTEER_IDS,
        max_steps=10,
        tick_hours=1.0,
    )


def describe() -> dict:
    return {
        "id": "medium",
        "name": "Multi-Task Volunteer Distribution",
        "description": (
            "Distribute 5 available volunteers across 3 concurrent tasks with different "
            "urgency levels (9, 8, 3) and distinct skill requirements. "
            "The agent must balance: cover the high-urgency food distribution task first, "
            "then the medical camp, while using leftover volunteers for the tree plantation. "
            "Volunteers cannot be over-scheduled beyond their available hours."
        ),
        "difficulty": "medium",
        "max_steps": 10,
        "tasks": TASK_IDS,
        "volunteers": VOLUNTEER_IDS,
        "success_criteria": (
            "All 3 tasks partially staffed with appropriate skill-matched volunteers "
            "within 10 steps. High-urgency tasks prioritised first."
        ),
    }
