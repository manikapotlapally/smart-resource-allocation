"""
Hard Grader: Full disaster response scoring.
Deterministic, multi-dimensional rubric. Returns 0.0 – 1.0.
"""

from typing import Dict, Any, List


def grade(episode_history: list, final_state: Dict[str, Any]) -> float:
    """
    Scoring rubric for hard (disaster response) task:

    1. Flood rescue speed  (0.25): task_006 gets rescue-skilled vol within step 3
    2. Coverage breadth    (0.25): fraction of 8 tasks with ≥1 vol assigned
    3. Skill quality       (0.25): avg skill-match ratio across all assignments
    4. No overassignment   (0.15): no volunteer assigned beyond their hours
    5. Failure avoidance   (0.10): no high-urgency tasks (urgency ≥7) failed
    """
    score = 0.0
    tasks = final_state.get("tasks", {})
    volunteers = final_state.get("volunteers", {})
    failed_tasks = set(final_state.get("failed_tasks", []))
    step_count = final_state.get("step_count", 20)

    # ── 1. Flood rescue speed (0.25) ────────────────────────────────────────
    rescue_task = tasks.get("task_006", {})
    rescue_assigned = rescue_task.get("assigned_volunteers", [])
    rescue_required_skills = {"rescue", "communication", "physical_fitness", "logistics"}

    rescue_speed_score = 0.0
    for step_idx, step_data in enumerate(episode_history):
        for result in step_data.get("assignment_results", []):
            if result.get("success") and result.get("task_id") == "task_006":
                vol = volunteers.get(result["volunteer_id"], {})
                vol_skills = set(vol.get("skills", []))
                if vol_skills & rescue_required_skills:  # any overlap
                    if step_idx <= 2:   # steps 0,1,2 = within 3 steps
                        rescue_speed_score = 0.25
                    elif step_idx <= 5:
                        rescue_speed_score = 0.15
                    else:
                        rescue_speed_score = 0.05
                    break
        if rescue_speed_score > 0:
            break

    score += rescue_speed_score

    # ── 2. Coverage breadth (0.25) ──────────────────────────────────────────
    all_task_ids = [
        "task_001", "task_002", "task_003", "task_004",
        "task_005", "task_006", "task_007", "task_008"
    ]
    covered = sum(
        1 for tid in all_task_ids
        if tasks.get(tid, {}).get("assigned_volunteers")
    )
    coverage_score = (covered / len(all_task_ids)) * 0.25
    score += coverage_score

    # ── 3. Skill quality (0.25) ─────────────────────────────────────────────
    skill_reqs = {
        "task_001": ["logistics", "physical_fitness"],
        "task_002": ["medical", "communication"],
        "task_003": ["construction", "physical_fitness"],
        "task_004": ["medical", "communication", "data_entry"],
        "task_005": ["physical_fitness", "environment"],
        "task_006": ["rescue", "communication", "physical_fitness", "logistics"],
        "task_007": ["teaching", "communication", "data_entry"],
        "task_008": ["communication", "empathy"],
    }

    match_scores = []
    for tid, req_skills in skill_reqs.items():
        task = tasks.get(tid, {})
        assigned_vols = task.get("assigned_volunteers", [])
        if assigned_vols and req_skills:
            best_match = 0.0
            for vid in assigned_vols:
                vol = volunteers.get(vid, {})
                vol_skills = vol.get("skills", [])
                matched = sum(1 for s in req_skills if s in vol_skills)
                best_match = max(best_match, matched / len(req_skills))
            match_scores.append(best_match)

    if match_scores:
        avg_skill = sum(match_scores) / len(match_scores)
        score += avg_skill * 0.25

    # ── 4. No overassignment (0.15) ──────────────────────────────────────────
    overassigned = any(
        v.get("hours_committed", 0) > v.get("availability_hours", 0)
        for v in volunteers.values()
    )
    if not overassigned:
        score += 0.15

    # ── 5. Failure avoidance (0.10) ──────────────────────────────────────────
    high_urgency_ids = {
        tid for tid, t in tasks.items() if t.get("urgency", 0) >= 7
    }
    if not (failed_tasks & high_urgency_ids):
        score += 0.10

    return round(min(1.0, score), 4)
