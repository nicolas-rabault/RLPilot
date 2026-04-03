# SETUP-MONITOR

You are configuring monitoring and evaluation for RL training.

## Context

The following config has already been gathered:

<CONFIG_CONTEXT>
{orchestrator injects Robot, Task, Source Files sections from config.md here}
</CONFIG_CONTEXT>

## Step 1: Monitoring & Evaluation — Generic Setup (interactive)

Ask the user (one question at a time):
- What monitoring tool? (WandB, TensorBoard, local logs)
- If WandB: project path (e.g. `team/project`) → store in project memory file `rl_training_infra.md`
- Key metric categories and prefixes to track
- Evaluation: what scenarios to test, what metrics, record video?
- May iterate with user to refine eval strategy

If `## Monitoring` section already exists in config.md, skip these questions — the user already answered them. Go straight to Step 2.

## Step 2: Metric Design Agent (interactive)

Spawn the Metric Design Agent to brainstorm and generate task-specific monitoring. This agent:

1. Reads context: robot type, actuators, task objective, simulator, existing reward terms, observation space. Scans the training code for existing WandB log calls to identify what's already logged.

2. Proposes metric categories based on task type:
   - Locomotion → gait quality (symmetry, periodicity, smoothness, contact patterns)
   - Manipulation → grasp stability, approach trajectory, contact forces
   - Balance → CoM tracking, recovery time, base stability
   - Custom → asks user what "good" and "bad" look like

3. Brainstorms with the user (one question at a time):
   - What does a bad behavior look like for this specific robot?
   - Which existing reward terms should be promoted to monitored quality metrics?
   - What's the minimum quality you'd accept for a KEEP decision?

4. Maps metrics to tiers:
   - Tier 1 (derived from existing WandB logs): no training code changes needed
   - Tier 2 (eval-time from simulation state): needs eval_metrics.py
   - Tier 3 (needs new training-time logging): flags for CODE phase if user approves

5. Generates the task monitoring directory using templates from `${CLAUDE_PLUGIN_ROOT}/templates/tasks/<task-type>/` as starting points:
   - `.claude/rl-training/tasks/<task-name>/monitor_config.md` — metrics, thresholds, weights, decision rules, human feedback tags
   - `.claude/rl-training/tasks/<task-name>/monitor_metrics.py` — Tier 1 derived metric computation
   - `.claude/rl-training/tasks/<task-name>/eval_metrics.py` — Tier 2 detailed quality analysis

## Output

Append to `.claude/rl-training/config.md`:

````markdown
## Monitoring
- Tool: <wandb, tensorboard, local>
- Task monitoring: <task-name>
- Metric categories: [<prefix1/>, <prefix2/>, ...]
- Key metrics: [<metric1>, <metric2>, ...]
- Kill threshold: 2
- Max iterations: 10

## Evaluation
- Scenarios:
  - <name>: <params>
- Metrics: [<metric1>, <metric2>, ...]
- Video: true

## Decision Criteria
- KEEP: <when to keep training>
- BAD: <when training is going wrong>
- FINISH: <when training is done>
````

If WandB is chosen, write `rl_training_infra.md` to project memory with WandB project path and entity.
