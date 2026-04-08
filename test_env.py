import sys
sys.path.insert(0, '.')

from env.environment import SmartResourceEnv
from env.models import Action, AssignAction

print("Testing environment...\n")

env = SmartResourceEnv(max_steps=5)
obs = env.reset()
print(f"Reset OK — tasks: {len(obs.open_tasks)} | volunteers: {len(obs.available_volunteers)}")

# Get first available volunteer and task IDs from the observation
first_task = obs.open_tasks[0].id
first_vol = obs.available_volunteers[0].id
print(f"First task: {first_task} | First volunteer: {first_vol}")

action = Action(
    assignments=[AssignAction(volunteer_id=first_vol, task_id=first_task)],
    unassignments=[],
    prioritizations=[],
    skip=False
)

obs, reward, done, info = env.step(action)
print(f"Step  OK — reward: {reward.total} | done: {done}")

s = env.state()
print(f"State OK — step: {obs.step_number}")

print("\nAll checks passed!")