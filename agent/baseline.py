"""
baseline.py — Baseline agent that uses an LLM (via OpenAI-compatible API) to
interact with the SmartResourceEnv environment.

Usage:
    export OPENAI_API_KEY=your_key_here
    export OPENAI_BASE_URL=https://api.openai.com/v1  # optional
    python agent/baseline.py

Produces deterministic scores on all 3 tasks.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from env.models import Action, AssignAction, UnassignAction
from tasks.easy import build_env as build_easy_env
from tasks.medium import build_env as build_medium_env
from tasks.hard import build_env as build_hard_env
import graders.easy_grader as easy_grader
import graders.medium_grader as medium_grader
import graders.hard_grader as hard_grader


# ── LLM client setup ──────────────────────────────────────────────────────────

MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

def get_client() -> OpenAI:
    api_key = os.environ.get("AIzaSyDdPzpmcC5PXdXY1QoyQXjcyNyxLiCwMy4")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set.")
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

SYSTEM_PROMPT = """You are an expert NGO volunteer coordinator. You will receive the current
state of a volunteer resource allocation problem and must decide which volunteer(s) to assign
to which task(s).

You will respond ONLY with a valid JSON object in this exact format:
{
  "assignments": [
    {"volunteer_id": "vol_XXX", "task_id": "task_XXX"}
  ],
  "unassignments": [],
  "prioritizations": [],
  "skip": false
}

Rules:
- Only assign volunteers that appear in 'available_volunteers' list
- Only assign to tasks that appear in 'open_tasks' list
- Match volunteer skills to task required_skills as closely as possible
- Prioritise tasks with higher urgency and shorter deadlines
- Never assign a volunteer to more tasks than their availability_hours allows
- If nothing useful can be done, set "skip": true
- Return ONLY the JSON, no explanation, no markdown, no extra text."""


def build_user_message(obs_dict: dict) -> str:
    return f"""Current environment state:

OPEN TASKS:
{json.dumps(obs_dict['open_tasks'], indent=2)}

AVAILABLE VOLUNTEERS:
{json.dumps(obs_dict['available_volunteers'], indent=2)}

CURRENT ALLOCATIONS:
{json.dumps(obs_dict['current_allocations'], indent=2)}

ALERTS: {obs_dict.get('alerts', [])}

Step {obs_dict['step_number']} of {obs_dict['total_steps']}
Episode score so far: {obs_dict['episode_score']}

Decide your allocation action:"""


def parse_llm_action(response_text: str) -> Action:
    """Parse LLM JSON response into an Action object."""
    try:
        # Strip markdown fences if present
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())

        assignments = [
            AssignAction(volunteer_id=a["volunteer_id"], task_id=a["task_id"])
            for a in data.get("assignments", [])
        ]
        unassignments = [
            UnassignAction(volunteer_id=u["volunteer_id"], task_id=u["task_id"])
            for u in data.get("unassignments", [])
        ]
        skip = data.get("skip", False)

        return Action(assignments=assignments, unassignments=unassignments, skip=skip)

    except Exception as e:
        print(f"  [WARN] Could not parse LLM response: {e}. Defaulting to skip.")
        return Action(skip=True)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env, client: OpenAI, task_name: str) -> Dict[str, Any]:
    """Run a full episode using the LLM agent."""
    print(f"\n{'='*60}")
    print(f"  Running: {task_name.upper()} TASK")
    print(f"{'='*60}")

    obs = env.reset()
    done = False
    step_num = 0
    episode_history = []
    conversation: List[dict] = []

    while not done:
        obs_dict = obs.model_dump()

        # Build the message
        user_msg = build_user_message(obs_dict)
        conversation.append({"role": "user", "content": user_msg})

        # Call LLM
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
                temperature=0.0,  # Deterministic
                max_tokens=512,
            )
            reply = response.choices[0].message.content
        except Exception as e:
            print(f"  [ERROR] LLM call failed: {e}")
            reply = '{"assignments": [], "skip": true}'

        conversation.append({"role": "assistant", "content": reply})

        # Parse action
        action = parse_llm_action(reply)
        print(f"\n  Step {step_num + 1}: {len(action.assignments)} assignment(s), skip={action.skip}")

        # Step environment
        obs, reward, done, info = env.step(action)
        step_num += 1

        print(f"  Reward: {reward.total:.4f} | Cumulative: {info['cumulative_reward']:.4f}")
        print(f"  Open tasks remaining: {info['open_tasks_remaining']}")

        for result in info.get("assignment_results", []):
            status = "✓" if result["success"] else "✗"
            print(f"    {status} {result.get('type','assign')} {result.get('volunteer_id','')} → {result.get('task_id','')} ({result.get('reason','')})")

        episode_history.append({
            "step": step_num,
            "action": action.model_dump(),
            "reward": reward.total,
            "assignment_results": info.get("assignment_results", []),
        })

        # Small delay to avoid rate limits
        time.sleep(0.3)

    final_state = env.state()
    final_state["step_count"] = step_num

    return {
        "episode_history": episode_history,
        "final_state": final_state,
        "cumulative_reward": info["cumulative_reward"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n🚀 Smart Resource Allocation — Baseline Inference")
    print(f"   Model: {MODEL}")

    client = get_client()
    results = {}

    # ── Easy ──────────────────────────────────────────────────────────────────
    env = build_easy_env()
    ep = run_episode(env, client, "easy")
    easy_score = easy_grader.grade(ep["episode_history"], ep["final_state"])
    results["easy"] = {"grader_score": easy_score, "cumulative_reward": ep["cumulative_reward"]}

    # ── Medium ────────────────────────────────────────────────────────────────
    env = build_medium_env()
    ep = run_episode(env, client, "medium")
    medium_score = medium_grader.grade(ep["episode_history"], ep["final_state"])
    results["medium"] = {"grader_score": medium_score, "cumulative_reward": ep["cumulative_reward"]}

    # ── Hard ──────────────────────────────────────────────────────────────────
    env = build_hard_env()
    ep = run_episode(env, client, "hard")
    hard_score = hard_grader.grade(ep["episode_history"], ep["final_state"])
    results["hard"] = {"grader_score": hard_score, "cumulative_reward": ep["cumulative_reward"]}

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  BASELINE SCORES")
    print(f"{'='*60}")
    for task_name, r in results.items():
        print(f"  {task_name.upper():8s}  grader: {r['grader_score']:.4f}  |  cumulative_reward: {r['cumulative_reward']:.4f}")

    avg = sum(r["grader_score"] for r in results.values()) / len(results)
    print(f"\n  AVERAGE GRADER SCORE: {avg:.4f}")
    print(f"{'='*60}\n")

    # Save results
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
