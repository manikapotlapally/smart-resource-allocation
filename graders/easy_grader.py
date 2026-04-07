"""
Easy Grader: Score agent performance on the easy task.
Deterministic. Returns 0.0 – 1.0.
"""

from typing import Dict, Any


def grade(episode_history: list, final_state: Dict[str, Any]) -> float:
    """
    Scoring rubric for easy task:
      - 0.4  → any volunteer assigned to task_007
      - 0.3  → assigned volunteer has 'teaching' skill
      - 0.2  → assigned volunteer has 'communication' skill  
      - 0.1  → done within 3 steps (speed bonus)
    
    Total max: 1.0
    """
    score = 0.0

    tasks = final_state.get("tasks", {})
    volunteers = final_state.get("volunteers", {})
    allocations = final_state.get("allocations", {})  # vol_id → task_id

    task = tasks.get("task_007", {})
    assigned_vols = task.get("assigned_volunteers", [])

    if not assigned_vols:
        return 0.0  # Nothing assigned → 0

    # Base assignment credit
    score += 0.4

    # Check skill quality
    assigned_vol_id = assigned_vols[0]
    vol = volunteers.get(assigned_vol_id, {})
    vol_skills = vol.get("skills", [])

    if "teaching" in vol_skills:
        score += 0.3
    if "communication" in vol_skills:
        score += 0.2

    # Speed bonus — done in ≤3 steps
    total_steps = final_state.get("step_count", 99)
    if total_steps <= 3:
        score += 0.1

    return round(min(1.0, score), 4)
