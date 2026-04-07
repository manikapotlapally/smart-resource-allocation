"""
Hard Task: Full disaster response coordination.
- Urgency-10 flood rescue task with imminent 4h deadline
- Multiple competing tasks across locations
- Scarce, partially-skilled volunteer pool
- Cascading deadlines and skill constraints
- Agent must triage, prioritise, and re-assign under time pressure
"""

from env.environment import SmartResourceEnv

# All tasks active — flood rescue (urgency 10), medical camp, food distribution,
# vaccination drive, school renovation, old age home, digital literacy, tree plantation
TASK_IDS = [
    "task_001", "task_002", "task_003", "task_004",
    "task_005", "task_006", "task_007", "task_008"
]

# All active volunteers — mixed skills, varying availability
VOLUNTEER_IDS = [
    "vol_001", "vol_002", "vol_003", "vol_004", "vol_005",
    "vol_006", "vol_007", "vol_008", "vol_009", "vol_010", "vol_011"
]


def build_env() -> SmartResourceEnv:
    return SmartResourceEnv(
        task_ids=TASK_IDS,
        volunteer_ids=VOLUNTEER_IDS,
        max_steps=20,
        tick_hours=1.0,
    )


def describe() -> dict:
    return {
        "id": "hard",
        "name": "Full Disaster Response Coordination",
        "description": (
            "Coordinate all 8 simultaneous community tasks with 11 volunteers. "
            "A flood rescue task (urgency 10, 4h deadline) demands 12 volunteers with "
            "rescue + logistics skills — but only ~3 volunteers have those skills. "
            "The agent must: (1) immediately triage and assign rescue-capable volunteers "
            "to the flood task, (2) reassign non-rescue volunteers to lower-urgency tasks, "
            "(3) avoid wasting availability hours on low-beneficiary tasks, "
            "(4) handle cascading deadlines as time ticks forward each step."
        ),
        "difficulty": "hard",
        "max_steps": 20,
        "tasks": TASK_IDS,
        "volunteers": VOLUNTEER_IDS,
        "success_criteria": (
            "Urgency-10 flood rescue task assigned within first 3 steps. "
            "At least 6 of 8 tasks have at least 1 assigned volunteer. "
            "No volunteer overassigned beyond availability. "
            "Cumulative reward ≥ 0.60."
        ),
    }
