"""
FINAL BASELINE AGENT (Hackathon Submission Ready)

Features:
✅ Reduced LLM calls (every 2 steps)
✅ Smart fallback (high scoring)
✅ Avoids repeated assignments
✅ Skips low-confidence actions (prevents penalties)
✅ Stable + deterministic
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

load_dotenv()

# Add root path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from env.models import Action, AssignAction
from tasks.easy import build_env as build_easy_env
from tasks.medium import build_env as build_medium_env
from tasks.hard import build_env as build_hard_env
import graders.easy_grader as easy_grader
import graders.medium_grader as medium_grader
import graders.hard_grader as hard_grader


# ───────────────────────── CONFIG ─────────────────────────
MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")


def get_client():
    return OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )


# ───────────────────────── PROMPT ─────────────────────────
SYSTEM_PROMPT = """You are an expert volunteer coordinator.

Return ONLY JSON:
{
  "assignments": [{"volunteer_id": "...", "task_id": "..."}],
  "skip": false
}
"""


def build_user_message(obs):
    return json.dumps(obs, indent=2)


# ───────────────────────── LLM CALL ─────────────────────────
def call_llm(client, messages, retries=3):
    for i in range(retries):
        try:
            res = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=80
            )
            return res.choices[0].message.content

        except Exception as e:
            print(f"[Retry {i+1}] {e}")
            time.sleep(2 ** i)

    return '{"assignments": [], "skip": true}'


# ───────────────────────── PARSER ─────────────────────────
def parse_action(text):
    try:
        data = json.loads(text)

        assignments = [
            AssignAction(
                volunteer_id=a["volunteer_id"],
                task_id=a["task_id"]
            )
            for a in data.get("assignments", [])
        ]

        return Action(assignments=assignments, skip=data.get("skip", False))

    except Exception:
        return Action(skip=True)


# ───────────────────────── FINAL SMART FALLBACK ─────────────────────────
def fallback_action(obs, used_pairs):
    best_score = -1
    best_pair = None

    # 🔥 prioritize urgent tasks first
    tasks_sorted = sorted(obs["open_tasks"], key=lambda x: -x["urgency"])

    for t in tasks_sorted:
        for v in obs["available_volunteers"]:

            pair_key = (v["id"], t["id"])
            if pair_key in used_pairs:
                continue

            required = set(t["required_skills"])
            have = set(v["skills"])

            skill_match = len(required & have)

            # ❌ skip useless matches
            if skill_match == 0:
                continue

            skill_ratio = skill_match / len(required)

            slots_needed = t["required_volunteers"] - len(t["assigned_volunteers"])

            # 🔥 final scoring formula
            score = (
                skill_ratio * 6
                + t["urgency"] * 3
                + slots_needed * 2
                + v["availability_hours"] * 0.1
            )

            if score > best_score:
                best_score = score
                best_pair = (t, v)

    # ✅ skip if no good option (prevents penalty)
    if best_pair is None or best_score < 3:
        return Action(skip=True)

    t, v = best_pair
    used_pairs.add((v["id"], t["id"]))

    return Action(assignments=[
        AssignAction(volunteer_id=v["id"], task_id=t["id"])
    ])


# ───────────────────────── EPISODE ─────────────────────────
def run_episode(env, client, name):
    print(f"\n=== {name.upper()} ===")

    obs = env.reset()
    done = False
    step = 0
    MAX_STEPS = 8

    last_reply = None
    used_pairs = set()
    history = []

    while not done and step < MAX_STEPS:
        obs_dict = obs.model_dump()

        # 🔥 reduce LLM calls
        if step % 2 == 0:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(obs_dict)}
            ]

            reply = call_llm(client, messages)
            last_reply = reply
        else:
            reply = last_reply

        action = parse_action(reply)

        # fallback if LLM fails
        if not action.assignments:
            action = fallback_action(obs_dict, used_pairs)

        obs, reward, done, info = env.step(action)

        print(f"Step {step+1} | Reward: {reward.total:.3f}")

        history.append({
            "step": step,
            "reward": reward.total,
            "assignment_results": info.get("assignment_results", [])
        })

        step += 1
        time.sleep(2)

    final_state = env.state()
    return history, final_state, final_state.get("cumulative_reward", 0)


# ───────────────────────── MAIN ─────────────────────────
def main():
    print("🚀 FINAL Optimized Baseline")

    client = get_client()
    results = {}

    # EASY
    env = build_easy_env()
    hist, state, rew = run_episode(env, client, "easy")
    results["easy"] = {
        "score": easy_grader.grade(hist, state),
        "reward": rew
    }

    time.sleep(5)

    # MEDIUM
    env = build_medium_env()
    hist, state, rew = run_episode(env, client, "medium")
    results["medium"] = {
        "score": medium_grader.grade(hist, state),
        "reward": rew
    }

    time.sleep(5)

    # HARD
    env = build_hard_env()
    hist, state, rew = run_episode(env, client, "hard")
    results["hard"] = {
        "score": hard_grader.grade(hist, state),
        "reward": rew
    }

    # OUTPUT
    print("\n=== FINAL RESULTS ===")
    for k, v in results.items():
        print(f"{k}: score={v['score']:.3f}, reward={v['reward']:.3f}")

    avg = sum(v["score"] for v in results.values()) / len(results)
    print(f"\nAverage Score: {avg:.3f}")

    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
