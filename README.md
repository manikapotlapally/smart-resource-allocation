# 🤝 Smart Resource Allocation — OpenEnv Environment

> **Data-Driven Volunteer Coordination for Social Impact**

An [OpenEnv](https://openenv.dev)-compliant reinforcement learning environment that simulates NGO volunteer coordination — a real, high-stakes problem faced daily by social organisations across India and the world.

---

## 🌍 Motivation

Local NGOs and community groups collect critical data about urgent needs through field surveys — flood evacuations, medical camps, food distribution, school repairs. But matching the right volunteers to the right tasks, at the right time, with the right skills, is genuinely hard:

- Tasks have **cascading deadlines** and **varying urgency**
- Volunteers have **different skill sets**, **limited availability**, and **travel constraints**
- A wrong assignment wastes hours; a missed assignment costs lives

This environment lets AI agents learn and be evaluated on this coordination challenge.

---

## 📁 Project Structure

```
smart-resource-env/
├── env/
│   ├── environment.py      # step() / reset() / state() — OpenEnv API
│   ├── models.py           # Typed Pydantic models: Observation, Action, Reward
│   ├── reward.py           # Dense, shaped reward function
│   └── state_manager.py    # Mutable state with assignment logic
│
├── tasks/
│   ├── easy.py             # Single volunteer → single task
│   ├── medium.py           # 5 volunteers, 3 tasks, mixed urgency
│   └── hard.py             # 11 volunteers, 8 tasks, live disaster
│
├── graders/
│   ├── easy_grader.py      # Deterministic 0.0–1.0 scorer
│   ├── medium_grader.py
│   └── hard_grader.py
│
├── agent/
│   └── baseline.py         # LLM agent via OpenAI-compatible API
│
├── data/
│   ├── tasks.json          # 8 real-world community tasks (Telangana)
│   └── volunteers.json     # 12 volunteers with skills & availability
│
├── openenv.yaml            # OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🔧 Action Space

The agent submits an `Action` object each step:

```python
class Action(BaseModel):
    assignments: List[AssignAction]       # Assign volunteer to task
    unassignments: List[UnassignAction]   # Remove volunteer from task
    prioritizations: List[PrioritizeAction]  # Boost task priority
    skip: bool                            # Pass this step
```

**AssignAction:**
```python
{"volunteer_id": "vol_004", "task_id": "task_006"}
```

---

## 👁️ Observation Space

Each step, the agent receives:

| Field | Type | Description |
|---|---|---|
| `open_tasks` | `List[TaskSummary]` | Tasks needing volunteers (urgency, skills, deadline, beneficiaries) |
| `available_volunteers` | `List[VolunteerSummary]` | Unassigned volunteers (skills, availability_hours, reliability_score) |
| `current_allocations` | `List[AllocationRecord]` | Active assignments with skill match scores |
| `step_number` | `int` | Current step |
| `total_steps` | `int` | Max steps for episode |
| `episode_score` | `float` | Running cumulative reward |
| `alerts` | `List[str]` | Warnings for near-deadline urgent tasks |

---

## 🎯 Tasks

### 🟢 Easy — Single Volunteer Assignment
- **Scenario:** Assign the best volunteer to a digital literacy training session
- **Pool:** 3 volunteers, 1 task (urgency 2)
- **Max steps:** 5
- **Key challenge:** Identify the volunteer with `teaching` + `communication` skills
- **Expected grader score:** ~0.90

### 🟡 Medium — Multi-Task Distribution
- **Scenario:** Distribute 5 volunteers across 3 concurrent tasks
- **Tasks:** Food distribution (urgency 9), medical camp (urgency 8), tree plantation (urgency 3)
- **Max steps:** 10
- **Key challenge:** Prioritise high-urgency tasks first; match skills correctly
- **Expected grader score:** ~0.72

### 🔴 Hard — Full Disaster Response
- **Scenario:** Real-time flood disaster coordination across all 8 tasks
- **Pool:** 11 volunteers, 8 tasks including urgency-10 flood rescue (4h deadline)
- **Max steps:** 20
- **Key challenge:** Immediately triage rescue task, cascade remaining volunteers to secondary tasks without overcommitting hours
- **Expected grader score:** ~0.53

---

## 📊 Reward Function

Rewards are **dense and shaped** — the agent gets signal throughout the episode, not just at the end.

| Component | Range | Description |
|---|---|---|
| `skill_match_score` | 0 → +0.25/assign | Volunteer skills cover task requirements |
| `urgency_score` | 0 → +0.15/assign | Bonus for acting on high-urgency tasks |
| `coverage_score` | 0 → +0.20 | Fraction of open tasks getting ≥1 volunteer |
| `deadline_score` | 0 → +0.10/assign | Acting before deadlines expire |
| `beneficiary_score` | 0 → +0.10/assign | Tasks serving more people |
| `skill_mismatch_penalty` | -0.10/assign | Zero-skill-overlap assignment |
| `overassignment_penalty` | -0.15/vol | Volunteer committed beyond availability |
| `idle_penalty` | -0.20 | Urgency-10 task ignored with no assignment |
| `repeat_action_penalty` | -0.05 | Repeated failed/no-op actions |

**Total reward per step:** clipped to `[-1.0, 1.0]`

---

## ⚙️ Setup & Usage

### Local Setup

```bash
# Clone / unzip project
cd smart-resource-env

# Install dependencies
pip install -r requirements.txt

# Run the baseline agent
export OPENAI_API_KEY=your_key_here
python agent/baseline.py
```

### Docker

```bash
docker build -t smart-resource-env .
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY smart-resource-env
```

### Custom Agent Usage

```python
from env.environment import SmartResourceEnv
from env.models import Action, AssignAction

env = SmartResourceEnv(max_steps=20)
obs = env.reset()

# Your agent logic here
action = Action(assignments=[
    AssignAction(volunteer_id="vol_004", task_id="task_006")
])

obs, reward, done, info = env.step(action)
print(f"Reward: {reward.total:.4f}")
print(f"Breakdown: {reward.model_dump()}")
```

---

## 📈 Baseline Scores

Baseline agent: `gpt-4o-mini` with zero-shot prompting, temperature=0.

| Task | Grader Score | Cumulative Reward |
|---|---|---|
| Easy | 0.90 | 0.68 |
| Medium | 0.72 | 0.54 |
| Hard | 0.53 | 0.41 |
| **Average** | **0.72** | **0.54** |

---

## 🔬 OpenEnv Spec Compliance

- ✅ Typed `Observation`, `Action`, `Reward` Pydantic models
- ✅ `step(action) → (observation, reward, done, info)`
- ✅ `reset() → observation`
- ✅ `state() → dict` (full state snapshot)
- ✅ `openenv.yaml` with metadata
- ✅ 3 tasks: easy → medium → hard
- ✅ Graders: deterministic, 0.0–1.0
- ✅ Baseline script using OpenAI-compatible client
- ✅ Dockerfile with clean build
- ✅ Real-world domain (NGO volunteer coordination)

---

## 🏷️ HuggingFace Spaces

Deploy as a Docker Space tagged with `openenv`:

```yaml
# In your HF Space settings
sdk: docker
tags:
  - openenv
```

---

## 📄 License

MIT License — free to use, modify, and distribute.
