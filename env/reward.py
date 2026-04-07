"""
Reward computation for the Smart Resource Allocation environment.
Provides dense, shaped rewards — not just binary win/loss.
"""

from typing import List, Dict, Any
from env.models import Reward, Action


def compute_reward(
    action: Action,
    state_before: dict,
    state_after: dict,
    assignment_results: List[Dict[str, Any]],
) -> Reward:
    """
    Compute a shaped reward signal based on the quality of allocations made.
    
    Positive signals:
      - Skill match: volunteers' skills cover task requirements
      - Urgency: high-urgency tasks are prioritised
      - Coverage: more volunteers assigned → closer to filling tasks
      - Deadline adherence: act before deadline runs out
      - Beneficiary reach: tasks serving more people
    
    Negative signals:
      - Skill mismatch: assigning wrong-skilled volunteers
      - Overassignment: volunteer hours exceed availability
      - Ignoring urgency-10 tasks
      - Repeating same no-op actions
    """

    tasks_after = state_after["tasks"]
    volunteers_after = state_after["volunteers"]
    allocations_after = state_after["allocations"]  # vol_id → task_id

    skill_match_score = 0.0
    urgency_score = 0.0
    coverage_score = 0.0
    deadline_score = 0.0
    beneficiary_score = 0.0
    overassignment_penalty = 0.0
    skill_mismatch_penalty = 0.0
    idle_penalty = 0.0
    repeat_action_penalty = 0.0

    successful_assigns = [r for r in assignment_results if r.get("success")]
    failed_assigns = [r for r in assignment_results if not r.get("success")]

    if not successful_assigns and not action.skip:
        # Penalise meaningless repeated failed actions
        repeat_action_penalty = -0.05

    for record in successful_assigns:
        task = tasks_after.get(record["task_id"], {})
        vol = volunteers_after.get(record["volunteer_id"], {})

        if not task or not vol:
            continue

        required_skills = task.get("required_skills", [])
        vol_skills = vol.get("skills", [])

        # Skill match
        if required_skills:
            matched = sum(1 for s in required_skills if s in vol_skills)
            sm = matched / len(required_skills)
            skill_match_score += sm * 0.25  # up to 0.25 per assignment

            # Skill mismatch penalty for zero overlap
            if matched == 0:
                skill_mismatch_penalty -= 0.10

        # Urgency signal: reward proportional to urgency (1-10 → 0 to 0.15)
        urgency = task.get("urgency", 1)
        urgency_score += (urgency / 10.0) * 0.15

        # Deadline signal: more reward for acting early
        deadline_hours = task.get("deadline_hours", 24)
        if deadline_hours > 0:
            # exponential decay — urgency drops as deadline approaches 0
            deadline_factor = min(1.0, deadline_hours / 24.0)
            deadline_score += deadline_factor * 0.10

        # Beneficiary reach
        beneficiaries = task.get("beneficiaries", 0)
        # Normalise: 500 people = full 0.10
        bscore = min(1.0, beneficiaries / 500.0) * 0.10
        beneficiary_score += bscore

    # Overassignment penalty
    for vol in volunteers_after.values():
        committed = vol.get("hours_committed", 0)
        available = vol.get("availability_hours", 0)
        if committed > available:
            overassignment_penalty -= 0.15

    # Coverage score: fraction of open tasks that now have >= 1 volunteer
    tasks_before = state_before["tasks"]
    newly_covered = 0
    total_open = sum(1 for t in tasks_before.values() if t["status"] == "open")
    for tid, task in tasks_after.items():
        if tasks_before.get(tid, {}).get("assigned_volunteers", []) == [] and \
                task.get("assigned_volunteers", []):
            newly_covered += 1
    if total_open > 0:
        coverage_score = (newly_covered / total_open) * 0.20

    # Idle penalty: if urgency-10 task is open and nothing was assigned to it
    for task in tasks_after.values():
        if task.get("urgency", 0) == 10 and task.get("status") == "open":
            if not task.get("assigned_volunteers"):
                idle_penalty -= 0.20

    # Normalise all components
    raw_total = (
        skill_match_score
        + urgency_score
        + coverage_score
        + deadline_score
        + beneficiary_score
        + overassignment_penalty
        + skill_mismatch_penalty
        + idle_penalty
        + repeat_action_penalty
    )
    # Clip to [-1, 1]
    total = max(-1.0, min(1.0, raw_total))

    return Reward(
        total=total,
        skill_match_score=round(skill_match_score, 4),
        urgency_score=round(urgency_score, 4),
        coverage_score=round(coverage_score, 4),
        deadline_score=round(deadline_score, 4),
        beneficiary_score=round(beneficiary_score, 4),
        overassignment_penalty=round(overassignment_penalty, 4),
        skill_mismatch_penalty=round(skill_mismatch_penalty, 4),
        idle_penalty=round(idle_penalty, 4),
        repeat_action_penalty=round(repeat_action_penalty, 4),
        info={
            "successful_assignments": len(successful_assigns),
            "failed_assignments": len(failed_assigns),
            "newly_covered_tasks": newly_covered,
        },
    )
