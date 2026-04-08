import sys
sys.path.insert(0, '.')
from env.environment import SmartResourceEnv
from env.models import Action, AssignAction

def run_episode(env, agent_fn):
    obs = env.reset()
    history = []
    done = False
    while not done:
        action = agent_fn(obs)
        obs, reward, done, info = env.step(action)
        history.append({"step": obs.step_number, "reward": reward.total, "assignment_results": info.get("assignment_results", [])})
    return history, env.state()

def easy_agent(obs):
    for t in obs.open_tasks:
        if t.id == "task_007":
            return Action(assignments=[AssignAction(volunteer_id="vol_001", task_id="task_007")], unassignments=[], prioritizations=[], skip=False)
    return Action(assignments=[], unassignments=[], prioritizations=[], skip=True)

MEDIUM_PLAN = [("vol_004", "task_001"), ("vol_002", "task_002"), ("vol_009", "task_005")]
medium_step = 0

def medium_agent(obs):
    global medium_step
    if medium_step >= len(MEDIUM_PLAN):
        return Action(assignments=[], unassignments=[], prioritizations=[], skip=True)
    vol_id, task_id = MEDIUM_PLAN[medium_step]
    medium_step += 1
    return Action(assignments=[AssignAction(volunteer_id=vol_id, task_id=task_id)], unassignments=[], prioritizations=[], skip=False)

HARD_PLAN = [("vol_004","task_006"),("vol_010","task_002"),("vol_006","task_001"),("vol_002","task_004"),("vol_007","task_008"),("vol_011","task_007"),("vol_009","task_005"),("vol_003","task_003")]
hard_step = 0

def hard_agent(obs):
    global hard_step
    if hard_step >= len(HARD_PLAN):
        return Action(assignments=[], unassignments=[], prioritizations=[], skip=True)
    vol_id, task_id = HARD_PLAN[hard_step]
    hard_step += 1
    avail_ids = [v.id for v in obs.available_volunteers]
    open_ids = [t.id for t in obs.open_tasks]
    if vol_id in avail_ids and task_id in open_ids:
        return Action(assignments=[AssignAction(volunteer_id=vol_id, task_id=task_id)], unassignments=[], prioritizations=[], skip=False)
    return Action(assignments=[], unassignments=[], prioritizations=[], skip=True)

from graders.easy_grader import grade as grade_easy
from graders.medium_grader import grade as grade_medium
from graders.hard_grader import grade as grade_hard

print("Running graders...")
h, fs = run_episode(SmartResourceEnv(max_steps=3), easy_agent)
print("EASY:  ", grade_easy(h, fs))
medium_step = 0
h, fs = run_episode(SmartResourceEnv(max_steps=10), medium_agent)
print("MEDIUM:", grade_medium(h, fs))
hard_step = 0
h, fs = run_episode(SmartResourceEnv(max_steps=20), hard_agent)
print("HARD:  ", grade_hard(h, fs))
print("Done!")