# Evaluation Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Elevate evaluation from an optional afterthought to the most important part of the RL training skill — with its own brainstorming session during setup, mandatory execution every monitoring tick, and a hard gate for FINISH decisions.

**Architecture:** SETUP-MONITOR splits into two sequential sub-agents (Metric Design + Eval Design). Eval Design invokes brainstorming to deeply understand the task, scans existing eval scripts for patterns, generates a task-specific eval script with custom behavioral prints, and validates it with a local dry-run. The monitoring cron runs eval every tick and uses the behavioral report as a first-class decision input.

**Tech Stack:** Claude Code skills (markdown prompts), Python (eval scripts), MuJoCo (dry-run validation)

**Spec:** `docs/superpowers/specs/2026-04-07-evaluation-redesign.md`

---

### Task 1: Update SETUP-MONITOR to split into two sub-agents

**Files:**
- Modify: `skills/rlpilot/setup/monitor.md`

The current `monitor.md` is a single agent prompt that handles both metric design and evaluation config. We split it into two sequential sub-agents, with the first being the existing metric design flow and the second being the new Eval Design agent.

- [ ] **Step 1: Read the current monitor.md**

Read `skills/rlpilot/setup/monitor.md` to understand the current structure.

- [ ] **Step 2: Restructure monitor.md into orchestrator + Metric Design sub-agent**

Replace the contents of `skills/rlpilot/setup/monitor.md` with:

````markdown
# SETUP-MONITOR

You are configuring monitoring and evaluation for RL training. This phase runs two sequential sub-agents.

## Context

The following config has already been gathered:

<CONFIG_CONTEXT>
{orchestrator injects Robot, Task, Source Files sections from config.md here}
</CONFIG_CONTEXT>

## Sub-agent 1: Metric Design

### Purpose
Design WandB monitoring metrics, thresholds, and decision rules.

### Process (interactive)

Ask the user (one question at a time):
- What monitoring tool? (WandB, TensorBoard, local logs)
- If WandB: project path (e.g. `team/project`) → store in project memory file `rl_training_infra.md`
- Key metric categories and prefixes to track

If `## Monitoring` section already exists in config.md, skip these questions — the user already answered them. Go straight to the Metric Design Agent.

### Metric Design Agent (interactive)

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

### Metric Design Output

Append to `.claude/rl-training/config.md`:

```markdown
## Monitoring
- Tool: <wandb, tensorboard, local>
- Task monitoring: <task-name>
- Metric categories: [<prefix1/>, <prefix2/>, ...]
- Key metrics: [<metric1>, <metric2>, ...]
- Kill threshold: 2
- Max iterations: 10

## Decision Criteria
- KEEP: <when to keep training>
- BAD: <when training is going wrong>
- FINISH: <when training is done>
```

If WandB is chosen, write `rl_training_infra.md` to project memory with WandB project path and entity.

### Metric Design Checkpoint

Sub-agent 1 is done when `monitor_config.md` exists in `.claude/rl-training/tasks/<task-name>/`.

---

## Sub-agent 2: Eval Design

### Purpose
Design the evaluation script through brainstorming — the most important part of the training skill. This agent deeply understands the task and creates a custom evaluation pipeline with behavioral prints that give the monitoring agent maximum information for decision-making.

### Process

1. **Template learning**: Before brainstorming, scan existing `.claude/rl-training/tasks/*/` directories for `evaluate_policy.py` and `eval_metrics.py` files. Read them to understand patterns and approaches used for other tasks. Summarize what you found to inform the brainstorm.

2. **Invoke `superpowers:brainstorming`**: Design the evaluation strategy for this specific task. Feed the brainstorming with:
   - Robot context from config.md (type, actuators, specificities)
   - Task objective and scenarios from config.md
   - Monitoring metrics from `monitor_config.md` (designed by sub-agent 1)
   - Patterns learned from existing eval scripts (step 1)
   - The constraint that eval runs every ~30 min monitoring tick, so it must complete in a few minutes

   The brainstorming should explore:
   - What behavioral information best serves the monitoring agent's decisions?
   - What does "good" vs "bad" look like in practice for this robot/task?
   - What prints would make the difference between a useful eval and a useless one?
   - What level of interpretation should the script do (raw data, detected events, layered)?
   - How to structure the output for both quick decisions and deep investigation?

3. **Generate eval files** based on the brainstorming design:
   - `.claude/rl-training/tasks/<task-name>/evaluate_policy.py` — task-specific evaluation script with custom behavioral prints
   - `.claude/rl-training/tasks/<task-name>/eval_metrics.py` — Tier 2 detailed quality analysis
   - Make both executable: `chmod +x`

   The eval script must produce these output files:
   - `eval_report.md` — behavioral report (the monitoring agent's primary eval input)
   - `eval_metrics.json` — machine-readable metrics for `decide.py`
   - `eval_raw_data.json` — per-timestep raw data for deep investigation
   - `*.mp4` — video for Discord notification (not for evaluation)

   The eval report must follow this structure:
   ```markdown
   # Behavioral Evaluation Report

   ## Summary
   <high-level verdict: what the robot is doing well, what it's doing badly>

   ## Per-Scenario Results
   ### <scenario_name> (vx=X, vy=Y, vz=Z)
   - Tracking performance: <velocity errors>
   - Behavioral observations: <task-specific prints designed during brainstorm>
   - Notable events: <stumbles, anomalies, phase transitions>
   - Quality score: <composite>

   ## Cross-Scenario Analysis
   <patterns across scenarios>

   ## Raw Metrics
   <!-- RAW_METRICS:{json}-->
   ```

4. **Dry-run validation**: Run the eval script locally with a random policy to verify the pipeline works end-to-end.
   ```bash
   uv run .claude/rl-training/tasks/<task-name>/evaluate_policy.py --dry-run --output-dir /tmp/eval-dry-run/ --config .claude/rl-training/config.md
   ```
   The `--dry-run` flag makes the script:
   - Instantiate the environment
   - Use a random policy (no checkpoint needed)
   - Run a small number of steps per scenario (e.g. 10 instead of the full count)
   - Produce all output files (eval_report.md, eval_metrics.json, eval_raw_data.json, video)

   Verify:
   - Exit code is 0
   - `eval_report.md` exists and contains all expected sections
   - `eval_metrics.json` is valid JSON
   - `eval_raw_data.json` is valid JSON
   - Behavioral prints are present and meaningful (not empty placeholders)

   If the dry-run fails, read the error, fix the script, and retry. Repeat until the dry-run passes.

5. **Write validation marker**: After dry-run passes, create `.claude/rl-training/tasks/<task-name>/.eval_validated` with the current timestamp.

### Eval Design Output

Append to `.claude/rl-training/config.md`:

```markdown
## Evaluation
- Scenarios:
  - <name>: <params>
- Video: true
```

### Eval Design Checkpoint

Sub-agent 2 is done when `.claude/rl-training/tasks/<task-name>/.eval_validated` exists.
````

- [ ] **Step 3: Commit**

```bash
git add skills/rlpilot/setup/monitor.md
git commit -m "refactor: split SETUP-MONITOR into Metric Design + Eval Design sub-agents"
```

---

### Task 2: Update SETUP-GENERATE to remove evaluate_policy.py generation

**Files:**
- Modify: `skills/rlpilot/setup/generate.md`

- [ ] **Step 1: Read the current generate.md**

Read `skills/rlpilot/setup/generate.md` to confirm current content.

- [ ] **Step 2: Remove evaluate_policy.py from the generation list**

Replace the contents of `skills/rlpilot/setup/generate.md` with:

````markdown
# SETUP-GENERATE

You are generating the shared scripts for RL training. This is fully autonomous — no user interaction needed.

## Context

<CONFIG_CONTEXT>
{orchestrator injects complete config.md here}
</CONFIG_CONTEXT>

<HOST_CONFIGS>
{orchestrator injects all hosts/<name>/host.md contents here}
</HOST_CONFIGS>

<TASK_MONITOR_CONFIG>
{orchestrator injects tasks/<task-name>/monitor_config.md here}
</TASK_MONITOR_CONFIG>

<INFRA_MEMORY>
{orchestrator injects rl_training_infra.md here}
</INFRA_MEMORY>

## Generate Scripts

Generate shared scripts in `.claude/rl-training/scripts/`, using templates from `${CLAUDE_PLUGIN_ROOT}/templates/scripts/` as a starting point and adapting to the project:

1. `init_session.sh` — session state management (branch-based)
2. `get_latest_run.py` — find active run (WandB or other, based on Monitoring tool in config)
3. `monitor.py` — fetch metrics and format markdown report
4. `learnings.py` — gather structured run data for the learning agent

**Do NOT generate:** `notify.sh` (already created by SETUP-NOTIFY), per-host scripts (already created by SETUP-HOSTS), task monitoring files (already created by SETUP-MONITOR), `evaluate_policy.py` (already created and validated by SETUP-MONITOR's Eval Design sub-agent).

## Generate Training Learnings Template

Create `docs/training-learnings.md` if it doesn't exist:

```markdown
# Training Learnings

Actionable tips and insights accumulated from training experiments.
Each entry is backed by observed evidence from training runs.

## Reward Design

## Observation Space

## Training Hyperparameters

## Physical Limits & Robot Capabilities

## Common Failure Modes

## What Doesn't Work
```

## Finalize

- Make all scripts executable: `chmod +x .claude/rl-training/scripts/*.sh .claude/rl-training/scripts/*.py`
- Present a summary of all generated files to the user for review, listing each file and its purpose
````

- [ ] **Step 3: Commit**

```bash
git add skills/rlpilot/setup/generate.md
git commit -m "refactor: remove evaluate_policy.py from SETUP-GENERATE (now in SETUP-MONITOR)"
```

---

### Task 3: Update SKILL.md checkpoint table

**Files:**
- Modify: `skills/rlpilot/SKILL.md` (lines 76-85)

- [ ] **Step 1: Update the checkpoint table**

In `skills/rlpilot/SKILL.md`, replace the checkpoint table (lines 76-85):

```markdown
**Checkpoints (skip phase if signal exists):**

| Phase | Done signal |
|-------|------------|
| SETUP-DISCOVER | `config.md` has `## Robot` and `## Task` sections |
| SETUP-MONITOR | At least one `.claude/rl-training/tasks/*/monitor_config.md` exists |
| SETUP-HOSTS | `config.md` has `## Hosts` section + matching `hosts/<name>/host.md` files exist |
| SETUP-NOTIFY | `config.md` has `## Notifications` section |
| SETUP-GENERATE | `.claude/rl-training/scripts/monitor.py` exists |
```

with:

```markdown
**Checkpoints (skip phase if signal exists):**

| Phase | Done signal |
|-------|------------|
| SETUP-DISCOVER | `config.md` has `## Robot` and `## Task` sections |
| SETUP-MONITOR | At least one `.claude/rl-training/tasks/*/monitor_config.md` exists AND `.claude/rl-training/tasks/*/.eval_validated` exists |
| SETUP-HOSTS | `config.md` has `## Hosts` section + matching `hosts/<name>/host.md` files exist |
| SETUP-NOTIFY | `config.md` has `## Notifications` section |
| SETUP-GENERATE | `.claude/rl-training/scripts/monitor.py` exists |
```

- [ ] **Step 2: Commit**

```bash
git add skills/rlpilot/SKILL.md
git commit -m "feat: update SETUP-MONITOR checkpoint to require eval validation"
```

---

### Task 4: Update monitoring cron — eval mandatory every tick

**Files:**
- Modify: `skills/rlpilot/SKILL.md` (lines 385-406, the monitoring cron steps 3-5)

- [ ] **Step 1: Replace cron steps 3, 3b, 4, and 5**

In `skills/rlpilot/SKILL.md`, replace the monitoring cron steps 3 through 5 (lines 385-405):

```
3. Evaluate policy (if config says Video: true or Evaluation section exists):
   uv run .claude/rl-training/scripts/evaluate_policy.py <wandb_run_path> --output-dir run_NNN/ --config .claude/rl-training/config.md
   Capture the "Video: <path>" line from stdout — this is the video file path.
   If eval fails, continue without video (set VIDEO_PATH to empty).

3b. Compute detailed quality metrics (if TASK_DIR exists and eval ran):
    evaluate_policy.py should call TASK_DIR/eval_metrics.py's analyze_trajectory() for each scenario.
    The task-specific quality breakdown is included in eval_metrics.md and eval_metrics.json.

4. Send pre-decision notification (if enabled and monitor_update in When list):
   Read the quality metrics markdown from step 2b stderr (appended to monitor file).
   If VIDEO_PATH is non-empty and the file exists, attach it:
     bash .claude/rl-training/scripts/notify.sh "$MSG" --branch "<BRANCH>" --file "$VIDEO_PATH"
   Otherwise:
     bash .claude/rl-training/scripts/notify.sh "$MSG" --branch "<BRANCH>"

5. DECIDE using the decision script:
   Run: uv run .claude/rl-training/scripts/decide.py run_NNN/ --monitor M --config .claude/rl-training/config.md --session-dir <session_dir> [--task-config TASK_DIR/monitor_config.md]
   Read the JSON output: {decision, should_kill, reasons, consecutive_bad, eval_requested, notification}.
   If eval_requested is true:
     uv run .claude/rl-training/scripts/session.py update <session_dir> --set eval_requested=true
```

with:

```
3. Evaluate policy (mandatory — runs every tick):
   uv run TASK_DIR/evaluate_policy.py <wandb_run_path> --output-dir run_NNN/ --config .claude/rl-training/config.md
   This produces:
   - run_NNN/eval_report.md — behavioral report (primary eval output)
   - run_NNN/eval_metrics.json — machine-readable metrics
   - run_NNN/eval_raw_data.json — per-timestep raw data
   - run_NNN/*.mp4 — video for notifications
   Capture the "Video: <path>" line from stdout — this is the video file path.
   If eval fails, notify via notify.sh --branch "<BRANCH>" "Eval error — <error>" and continue without eval data.

3b. Read the behavioral report:
    Read run_NNN/eval_report.md. This contains the task-specific behavioral analysis
    designed during SETUP-MONITOR's Eval Design brainstorming. Use it as context for
    notifications and decision-making.

4. Send pre-decision notification (if enabled and monitor_update in When list):
   Compose message from monitor metrics (step 2b) and eval summary (from eval_report.md Summary section).
   If VIDEO_PATH is non-empty and the file exists, attach it:
     bash .claude/rl-training/scripts/notify.sh "$MSG" --branch "<BRANCH>" --file "$VIDEO_PATH"
   Otherwise:
     bash .claude/rl-training/scripts/notify.sh "$MSG" --branch "<BRANCH>"

5. DECIDE using the decision script:
   Run: uv run .claude/rl-training/scripts/decide.py run_NNN/ --monitor M --config .claude/rl-training/config.md --session-dir <session_dir> [--task-config TASK_DIR/monitor_config.md] [--eval-report run_NNN/eval_report.md]
   Read the JSON output: {decision, should_kill, reasons, consecutive_bad, notification}.
   NOTE: FINISH requires eval confirmation. If decide.py outputs FINISH but eval_report.md
   shows detailed_quality_score < quality_finish_minimum, the decision downgrades to KEEP.
   decide.py handles this internally when --eval-report is provided.
```

- [ ] **Step 2: Commit**

```bash
git add skills/rlpilot/SKILL.md
git commit -m "feat: make eval mandatory every monitoring tick, add hard gate for FINISH"
```

---

### Task 5: Update SKILL.md scripts table

**Files:**
- Modify: `skills/rlpilot/SKILL.md` (lines 500-518, the Scripts section)

- [ ] **Step 1: Update the scripts table**

In `skills/rlpilot/SKILL.md`, replace the eval-related row in the scripts table:

```markdown
| `evaluate_policy.py <run_path> --output-dir <dir> [--config path]` | Headless eval with video + metrics |
```

with:

```markdown
| `tasks/<name>/evaluate_policy.py <run_path> --output-dir <dir> [--config path] [--dry-run]` | Task-specific behavioral eval with prints, metrics, and video |
```

- [ ] **Step 2: Commit**

```bash
git add skills/rlpilot/SKILL.md
git commit -m "docs: update scripts table for task-specific eval location"
```

---

### Task 6: Update SKILL.md Metric Design Agent re-invocation note

**Files:**
- Modify: `skills/rlpilot/SKILL.md` (lines 105-107)

- [ ] **Step 1: Update the re-invocation note**

In `skills/rlpilot/SKILL.md`, replace:

```markdown
### Metric Design Agent (re-invocable)

The Metric Design Agent (part of SETUP-MONITOR) can also be re-invoked on demand: user says "improve monitoring for this task," or the ITERATE phase detects 3+ iterations with human feedback tags that don't correspond to any monitored metric. To re-invoke, run just the SETUP-MONITOR phase with its checkpoint skipped.
```

with:

```markdown
### Metric Design Agent (re-invocable)

The Metric Design Agent (part of SETUP-MONITOR sub-agent 1) can also be re-invoked on demand: user says "improve monitoring for this task," or the ITERATE phase detects 3+ iterations with human feedback tags that don't correspond to any monitored metric. To re-invoke, run just the SETUP-MONITOR phase with its checkpoint skipped.

### Eval Design Agent (re-invocable)

The Eval Design Agent (part of SETUP-MONITOR sub-agent 2) can be re-invoked when the user says "improve evaluation for this task" or when behavioral reports are consistently uninformative. To re-invoke, delete `.claude/rl-training/tasks/<task-name>/.eval_validated` and re-run SETUP-MONITOR (sub-agent 1 will be skipped via its checkpoint, only sub-agent 2 runs).
```

- [ ] **Step 2: Commit**

```bash
git add skills/rlpilot/SKILL.md
git commit -m "docs: add Eval Design Agent re-invocation instructions"
```

---

### Task 7: Demote templates/scripts/evaluate_policy.py to reference

**Files:**
- Modify: `templates/scripts/evaluate_policy.py`

- [ ] **Step 1: Add reference header to the template**

In `templates/scripts/evaluate_policy.py`, replace the docstring:

```python
"""Evaluate a trained policy across a set of command scenarios.

Usage:
    uv run .claude/rl-training/scripts/evaluate_policy.py <checkpoint> --config <config_path> [--output <file>]

Output: markdown report with per-scenario metrics.
"""
```

with:

```python
"""REFERENCE EXAMPLE — not used for generation.

This file is a reference showing the general structure of an evaluation script.
The actual evaluate_policy.py for each task is created by SETUP-MONITOR's Eval Design
sub-agent through brainstorming, tailored to the specific task with custom behavioral prints.

See: skills/rlpilot/setup/monitor.md (Sub-agent 2: Eval Design)
"""
```

- [ ] **Step 2: Commit**

```bash
git add templates/scripts/evaluate_policy.py
git commit -m "docs: demote evaluate_policy.py template to reference example"
```

---

### Task 8: Bump plugin version

**Files:**
- Modify: `.claude-plugin/plugin.json`

- [ ] **Step 1: Bump minor version**

In `.claude-plugin/plugin.json`, change:

```json
"version": "0.4.0",
```

to:

```json
"version": "0.5.0",
```

This is a minor bump — new feature (dedicated eval design with brainstorming + mandatory eval per tick).

- [ ] **Step 2: Commit**

```bash
git add .claude-plugin/plugin.json
git commit -m "bump version to 0.5.0 for evaluation redesign"
```
