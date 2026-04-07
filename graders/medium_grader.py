"""
Medium Grader: Score agent on multi-task allocation.
Deterministic. Returns 0.0 – 1.0.
"""

from typing import Dict, Any, List


TASK_SKILL_REQUIREMENTS = {
    "task_001": ["logistics", "physical_fitness"],   # Food distribution
    "task_002": ["medical", "communication"],         # Medical camp
    "task_005": ["physical_fitness", "environment"],  # Tree plantation
}

TASK_URGENCY = {
    "task_001": 9,
    "task_002": 8,
    "task_005": 3,
}


def grade(episode_history: list, final_state: Dict[str, Any]) -> float:
    """
    Scoring rubric for medium task:
      - 0.15 per task covered (≥1 volunteer assigned)     → max 0.45
      - 0.15 per task with ≥1 skill-matched volunteer     → max 0.45
      - 0.10 if high-urgency tasks (task_001, task_002)
               assigned before task_005                   → priority bonus
    """
    score = 0.0
    tasks = final_state.get("tasks", {})
    volunteers = final_state.get("volunteers", {})

    for task_id in ["task_001", "task_002", "task_005"]:
        task = tasks.get(task_id, {})
        assigned = task.get("assigned_volunteers", [])

        if assigned:
            score += 0.15  # coverage credit

            # Skill match bonus
            required = TASK_SKILL_REQUIREMENTS.get(task_id, [])
            for vol_id in assigned:
                vol = volunteers.get(vol_id, {})
                vol_skills = vol.get("skills", [])
                matched = sum(1 for s in required if s in vol_skills)
                if matched > 0:
                    score += 0.15
                    break  # One matched volunteer per task is enough

    # Priority ordering bonus
    # We check the episode history to see if urgency-9 and urgency-8 tasks
    # got their first assignment before the urgency-3 task
    first_assign_step: Dict[str, int] = {}
    for step_idx, step_data in enumerate(episode_history):
        for result in step_data.get("assignment_results", []):
            if result.get("success") and result["task_id"] not in first_assign_step:
                first_assign_step[result["task_id"]] = step_idx

    t1_step = first_assign_step.get("task_001", 999)
    t2_step = first_assign_step.get("task_002", 999)
    t5_step = first_assign_step.get("task_005", 999)

    if t1_step < t5_step and t2_step < t5_step:
        score += 0.10

    return round(min(1.0, score), 4)
