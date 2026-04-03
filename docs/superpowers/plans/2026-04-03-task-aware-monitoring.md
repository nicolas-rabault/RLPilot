# Task-Aware Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add task-specific quality metrics, non-blocking human feedback, and a Metric Design Agent to the Monitor system so it can detect bad behavior (not just bad numbers).

**Architecture:** Per-task monitoring directories (`tasks/<name>/`) contain config, Tier 1 derived metrics script, and Tier 2 eval metrics script. The Monitor cron runs these after fetching raw WandB data. A Metric Design Agent brainstorms metrics with the user during SETUP. Human feedback is captured as structured tags in session_state.json, never blocking ITERATE.

**Tech Stack:** Python 3 (numpy for FFT/jerk), markdown configs, bash scripts, WandB API.

---

### Task 1: Locomotion Task Template — monitor_config.md

**Files:**
- Create: `templates/tasks/locomotion/monitor_config.md`

- [ ] **Step 1: Create the locomotion monitor config template**

```markdown
# Monitoring Configuration — <task-name>

## Task Type
locomotion

## Quality Metrics — Tier 1 (Derived from WandB)

Computed every monitor tick from raw WandB metrics. No training code changes needed.

| Metric | Source | Weight | Description |
|--------|--------|--------|-------------|
| symmetry_ratio | Torque/* L vs R joints | 0.3 | 1.0 = perfectly symmetric, 0.0 = fully asymmetric |
| action_smoothness | Episode_Reward/action_rate_l2 | 0.2 | Normalized smoothness score from action rate penalty |
| survival_ratio | Episode_Termination/* | 0.2 | Fraction of episodes ending in timeout (survived) vs fell |
| reward_balance | Episode_Reward/* ratios | 0.3 | Balance between locomotion and regularization rewards |

Composite quality_score = weighted sum of above, clamped to [0, 1].

## Quality Metrics — Tier 2 (Eval-Time)

Computed during policy evaluation from full simulation trajectories.

| Metric | Description | Good Range |
|--------|-------------|------------|
| joint_jerk | Mean jerk (d³pos/dt³) across joints | < 50.0 |
| step_periodicity | FFT dominant frequency strength on foot contacts | > 0.5 |
| stance_swing_ratio | Per-foot ground/air time ratio | 0.5 — 0.7 stance |
| phase_offset | L/R foot contact timing offset | 0.4 — 0.6 |
| grf_profile_score | Vertical force pattern naturalness | > 0.5 |

Composite detailed_quality_score = mean of normalized sub-scores, clamped to [0, 1].

## Actuator Mapping

Left/right pairs for symmetry computation (customize per robot):
- left: [<left_joint_1>, <left_joint_2>, ...]
- right: [<right_joint_1>, <right_joint_2>, ...]

Locomotion reward prefixes (for reward_balance):
- locomotion: [track_linear_velocity, track_angular_velocity]
- regularization: [upright, pose, action_rate_l2, foot_slip]

## Decision Rules

- quality_score_bad_threshold: 0.4
- quality_declining_monitors: 2
- quality_finish_minimum: 0.7
- reward_vs_quality_divergence: true

When reward_vs_quality_divergence is true:
- If reward is improving (> 5% gain) but quality_score is declining (> 5% drop) for quality_declining_monitors consecutive monitors → BAD

## Human Feedback Tags

- asymmetric_gait: Left/right legs behave differently
- jerky_motion: Abrupt, non-smooth movements
- shuffling: Feet barely leave the ground
- stumbling: Frequent near-falls or recovery motions
- reward_hacking: Achieves reward through unintended behavior
- too_conservative: Overly cautious, barely moving
- unstable_at_speed: Falls or wobbles at higher commanded velocities
- unnatural_posture: Body orientation or lean looks wrong
```

- [ ] **Step 2: Commit**

```bash
git add templates/tasks/locomotion/monitor_config.md
git commit -m "feat: add locomotion monitor_config.md template"
```

---

### Task 2: Locomotion Task Template — monitor_metrics.py (Tier 1)

**Files:**
- Create: `templates/tasks/locomotion/monitor_metrics.py`

- [ ] **Step 1: Create the Tier 1 derived metrics script**

This script reads raw WandB metrics JSON and computes derived quality metrics. It is called by the Monitor cron after `monitor.py` fetches raw data.

```python
#!/usr/bin/env python3
"""Compute task-specific derived quality metrics from raw WandB data.

Usage:
    uv run .claude/rl-training/tasks/<name>/monitor_metrics.py <raw_metrics.json> [--previous <prev_derived.json>] [--config <monitor_config.md>]

Input:  JSON file of raw WandB metrics (from <!-- RAW_METRICS:... --> in monitor output)
Output: JSON to stdout with derived metrics + markdown summary to stderr
"""

import argparse
import json
import re
import sys
from pathlib import Path


def parse_monitor_config(config_path):
    """Extract actuator mapping and reward prefixes from monitor_config.md."""
    text = Path(config_path).read_text()

    left_match = re.search(r"^- left:\s*\[(.+)\]", text, re.MULTILINE)
    right_match = re.search(r"^- right:\s*\[(.+)\]", text, re.MULTILINE)
    left_joints = [j.strip() for j in left_match.group(1).split(",")] if left_match else []
    right_joints = [j.strip() for j in right_match.group(1).split(",")] if right_match else []

    loco_match = re.search(r"^- locomotion:\s*\[(.+)\]", text, re.MULTILINE)
    reg_match = re.search(r"^- regularization:\s*\[(.+)\]", text, re.MULTILINE)
    loco_prefixes = [p.strip() for p in loco_match.group(1).split(",")] if loco_match else []
    reg_prefixes = [p.strip() for p in reg_match.group(1).split(",")] if reg_match else []

    weights = {}
    for line in text.splitlines():
        m = re.match(r"\|\s*(\w+)\s*\|.*\|\s*([\d.]+)\s*\|", line)
        if m and m.group(1) not in ("Metric",):
            weights[m.group(1)] = float(m.group(2))

    thresholds = {}
    for key in ("quality_score_bad_threshold", "quality_declining_monitors", "quality_finish_minimum"):
        m = re.search(rf"^- {key}:\s*(.+)$", text, re.MULTILINE)
        if m:
            val = m.group(1).strip()
            thresholds[key] = float(val) if "." in val else int(val)

    return {
        "left_joints": left_joints,
        "right_joints": right_joints,
        "loco_prefixes": loco_prefixes,
        "reg_prefixes": reg_prefixes,
        "weights": weights,
        "thresholds": thresholds,
    }


def compute_symmetry_ratio(metrics, left_joints, right_joints):
    """Compare left vs right joint torques. Returns 0-1 (1 = symmetric)."""
    if not left_joints or not right_joints:
        return None
    left_torques, right_torques = [], []
    for lj, rj in zip(left_joints, right_joints):
        lk = f"Torque/{lj}_mean"
        rk = f"Torque/{rj}_mean"
        if lk in metrics and rk in metrics:
            left_torques.append(abs(metrics[lk]))
            right_torques.append(abs(metrics[rk]))
    if not left_torques:
        return None
    total = sum(left_torques) + sum(right_torques)
    if total < 1e-6:
        return 1.0
    diff = sum(abs(l - r) for l, r in zip(left_torques, right_torques))
    return max(0.0, 1.0 - diff / total)


def compute_action_smoothness(metrics):
    """Normalize action_rate_l2 reward into 0-1 smoothness score.

    action_rate_l2 is typically negative (penalty). More negative = less smooth.
    We normalize assuming range [-1, 0] maps to [0, 1].
    """
    key = None
    for k in metrics:
        if "action_rate" in k.lower() and "Episode_Reward" in k:
            key = k
            break
    if key is None or metrics[key] is None:
        return None
    val = metrics[key]
    return max(0.0, min(1.0, 1.0 + val))


def compute_survival_ratio(metrics):
    """Fraction of episodes ending in timeout (survived full episode)."""
    timeout = None
    total_terms = 0.0
    for k, v in metrics.items():
        if not k.startswith("Episode_Termination/") or v is None:
            continue
        if "time_out" in k:
            timeout = v
        total_terms += v
    if timeout is None or total_terms < 1e-6:
        return None
    return timeout / total_terms


def compute_reward_balance(metrics, loco_prefixes, reg_prefixes):
    """Ratio indicating balance between locomotion and regularization rewards.

    Returns 0-1 where 0.5 = balanced, <0.3 = regularization dominated,
    >0.7 = locomotion dominated (potential reward hacking).
    """
    loco_sum, reg_sum = 0.0, 0.0
    loco_count, reg_count = 0, 0
    for k, v in metrics.items():
        if not k.startswith("Episode_Reward/") or v is None:
            continue
        name = k.replace("Episode_Reward/", "")
        if any(name.startswith(p) or name == p for p in loco_prefixes):
            loco_sum += abs(v)
            loco_count += 1
        elif any(name.startswith(p) or name == p for p in reg_prefixes):
            reg_sum += abs(v)
            reg_count += 1
    if loco_count == 0 or reg_count == 0:
        return None
    total = loco_sum + reg_sum
    if total < 1e-6:
        return 0.5
    ratio = loco_sum / total
    score = 1.0 - 2.0 * abs(ratio - 0.5)
    return max(0.0, min(1.0, score))


def compute_quality_score(sub_scores, weights):
    """Weighted composite of sub-scores."""
    total_weight = 0.0
    weighted_sum = 0.0
    for name, score in sub_scores.items():
        if score is not None and name in weights:
            weighted_sum += score * weights[name]
            total_weight += weights[name]
    if total_weight < 1e-6:
        return None
    return max(0.0, min(1.0, weighted_sum / total_weight))


def format_markdown(derived, previous):
    """Format derived metrics as markdown summary."""
    lines = ["\n## Quality Metrics (Tier 1)"]
    for name, val in derived.items():
        if val is None:
            lines.append(f"- **{name}**: N/A")
            continue
        trend = ""
        if previous and name in previous and previous[name] is not None:
            diff = val - previous[name]
            if abs(diff) > 0.01:
                trend = f" ({'↑' if diff > 0 else '↓'} {abs(diff):.2f})"
            else:
                trend = " (stable)"
        lines.append(f"- **{name}**: {val:.3f}{trend}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_metrics", help="Path to raw metrics JSON file")
    parser.add_argument("--previous", help="Path to previous derived metrics JSON")
    parser.add_argument("--config", default=".claude/rl-training/tasks/locomotion/monitor_config.md")
    args = parser.parse_args()

    metrics = json.loads(Path(args.raw_metrics).read_text())
    config = parse_monitor_config(args.config)

    previous = None
    if args.previous:
        p = Path(args.previous)
        if p.exists():
            previous = json.loads(p.read_text())

    sub_scores = {
        "symmetry_ratio": compute_symmetry_ratio(metrics, config["left_joints"], config["right_joints"]),
        "action_smoothness": compute_action_smoothness(metrics),
        "survival_ratio": compute_survival_ratio(metrics),
        "reward_balance": compute_reward_balance(metrics, config["loco_prefixes"], config["reg_prefixes"]),
    }

    quality = compute_quality_score(sub_scores, config["weights"])
    derived = {**sub_scores, "quality_score": quality}

    print(json.dumps(derived))
    print(format_markdown(derived, previous), file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add templates/tasks/locomotion/monitor_metrics.py
git commit -m "feat: add locomotion Tier 1 monitor_metrics.py template"
```

---

### Task 3: Locomotion Task Template — eval_metrics.py (Tier 2)

**Files:**
- Create: `templates/tasks/locomotion/eval_metrics.py`

- [ ] **Step 1: Create the Tier 2 eval metrics script**

This script is called by `evaluate_policy.py` after each scenario rollout. It receives trajectory data and computes detailed quality analysis.

```python
#!/usr/bin/env python3
"""Compute detailed gait quality metrics from policy evaluation trajectories.

Called by evaluate_policy.py after each scenario rollout.
Not run standalone — imported and called via analyze_trajectory().
"""

import json
import sys
from pathlib import Path

import numpy as np


def compute_joint_jerk(joint_positions, dt):
    """Mean jerk (third derivative of position) across all joints.

    Lower jerk = smoother motion. Returns mean absolute jerk.
    joint_positions: np.array of shape (timesteps, num_joints)
    """
    if len(joint_positions) < 4:
        return None
    vel = np.diff(joint_positions, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return float(np.mean(np.abs(jerk)))


def compute_step_periodicity(contact_states, dt):
    """Dominant frequency strength from FFT on foot contact signals.

    contact_states: np.array of shape (timesteps, num_feet), binary (0/1)
    Returns strength of dominant frequency relative to total power (0-1).
    Higher = more periodic/rhythmic gait.
    """
    if len(contact_states) < 16:
        return None
    scores = []
    for foot_idx in range(contact_states.shape[1]):
        signal = contact_states[:, foot_idx].astype(float)
        signal = signal - signal.mean()
        fft = np.fft.rfft(signal)
        power = np.abs(fft) ** 2
        if power[1:].sum() < 1e-6:
            scores.append(0.0)
            continue
        dominant = power[1:].max()
        total = power[1:].sum()
        scores.append(float(dominant / total))
    return float(np.mean(scores))


def compute_stance_swing_ratio(contact_states):
    """Per-foot fraction of time spent in stance (on ground).

    Natural walking is ~60% stance, 40% swing.
    Returns list of ratios per foot.
    """
    if len(contact_states) < 2:
        return None
    ratios = []
    for foot_idx in range(contact_states.shape[1]):
        stance_frac = contact_states[:, foot_idx].mean()
        ratios.append(float(stance_frac))
    return ratios


def compute_phase_offset(contact_states):
    """Cross-correlation based phase offset between left and right foot.

    For bipeds: returns offset as fraction of gait cycle (0-1).
    0.5 = perfect alternation (walking), 0.0 = in-phase (hopping).
    """
    if contact_states.shape[1] < 2 or len(contact_states) < 16:
        return None
    left = contact_states[:, 0].astype(float) - contact_states[:, 0].mean()
    right = contact_states[:, 1].astype(float) - contact_states[:, 1].mean()
    corr = np.correlate(left, right, mode="full")
    mid = len(corr) // 2
    half = len(left) // 2
    corr_half = corr[mid : mid + half]
    if len(corr_half) == 0 or corr_half.max() < 1e-6:
        return None
    peak_idx = np.argmax(corr_half)
    fft_left = np.fft.rfft(contact_states[:, 0].astype(float))
    power = np.abs(fft_left) ** 2
    if power[1:].max() < 1e-6:
        return None
    dominant_freq_idx = np.argmax(power[1:]) + 1
    period_samples = len(contact_states) / dominant_freq_idx
    if period_samples < 1:
        return None
    offset = (peak_idx % period_samples) / period_samples
    return float(min(offset, 1.0 - offset) * 2)


def compute_grf_profile_score(contact_forces, contact_states):
    """Score how close vertical ground reaction forces match natural gait pattern.

    Natural walking has a double-hump M-shape in vertical GRF per stride.
    Returns 0-1 (1 = natural pattern).
    """
    if contact_forces is None or len(contact_forces) < 16:
        return None
    scores = []
    for foot_idx in range(contact_forces.shape[1]):
        force = contact_forces[:, foot_idx]
        contact = contact_states[:, foot_idx]
        stance_forces = force[contact > 0.5]
        if len(stance_forces) < 4:
            scores.append(0.0)
            continue
        mid = len(stance_forces) // 2
        first_half_max = stance_forces[:mid].max() if mid > 0 else 0
        second_half_max = stance_forces[mid:].max() if mid < len(stance_forces) else 0
        mid_val = stance_forces[mid] if mid < len(stance_forces) else 0
        peak_avg = (first_half_max + second_half_max) / 2
        if peak_avg < 1e-6:
            scores.append(0.0)
            continue
        dip_ratio = 1.0 - (mid_val / peak_avg)
        scores.append(float(max(0.0, min(1.0, dip_ratio * 2))))
    return float(np.mean(scores)) if scores else None


def normalize_score(value, good_min, good_max, metric_type="higher_better"):
    """Normalize a metric value to 0-1 based on good range."""
    if value is None:
        return None
    if metric_type == "lower_better":
        if value <= good_min:
            return 1.0
        if value >= good_max:
            return 0.0
        return 1.0 - (value - good_min) / (good_max - good_min)
    if value >= good_max:
        return 1.0
    if value <= good_min:
        return 0.0
    return (value - good_min) / (good_max - good_min)


def analyze_trajectory(joint_positions, joint_velocities, contact_states, contact_forces, dt):
    """Main entry point — called by evaluate_policy.py per scenario.

    Args:
        joint_positions: np.array (timesteps, num_joints)
        joint_velocities: np.array (timesteps, num_joints)
        contact_states: np.array (timesteps, num_feet) — binary 0/1
        contact_forces: np.array (timesteps, num_feet) — vertical force magnitude, or None
        dt: float — simulation timestep

    Returns:
        dict with raw metrics and normalized scores
    """
    jerk = compute_joint_jerk(joint_positions, dt)
    periodicity = compute_step_periodicity(contact_states, dt)
    stance_swing = compute_stance_swing_ratio(contact_states)
    phase = compute_phase_offset(contact_states)
    grf = compute_grf_profile_score(contact_forces, contact_states)

    stance_score = None
    if stance_swing is not None:
        stance_score = float(np.mean([1.0 - 2.0 * abs(r - 0.6) for r in stance_swing]))
        stance_score = max(0.0, min(1.0, stance_score))

    sub_scores = {
        "joint_jerk_score": normalize_score(jerk, 0, 100, "lower_better"),
        "periodicity_score": normalize_score(periodicity, 0, 1, "higher_better"),
        "stance_swing_score": stance_score,
        "phase_offset_score": normalize_score(phase, 0, 1, "higher_better") if phase is not None else None,
        "grf_score": grf,
    }

    valid = [s for s in sub_scores.values() if s is not None]
    detailed_quality_score = float(np.mean(valid)) if valid else None

    return {
        "joint_jerk": jerk,
        "step_periodicity": periodicity,
        "stance_swing_ratio": stance_swing,
        "phase_offset": phase,
        "grf_profile_score": grf,
        "sub_scores": sub_scores,
        "detailed_quality_score": detailed_quality_score,
    }


def format_markdown(results, scenario_name):
    """Format eval quality metrics as markdown."""
    lines = [f"\n### Gait Quality — {scenario_name}"]
    for key in ("joint_jerk", "step_periodicity", "phase_offset", "grf_profile_score"):
        val = results.get(key)
        lines.append(f"- **{key}**: {val:.4f}" if val is not None else f"- **{key}**: N/A")
    if results.get("stance_swing_ratio") is not None:
        ratios = ", ".join(f"{r:.2f}" for r in results["stance_swing_ratio"])
        lines.append(f"- **stance_swing_ratio**: [{ratios}]")
    dqs = results.get("detailed_quality_score")
    lines.append(f"- **detailed_quality_score**: {dqs:.3f}" if dqs is not None else "- **detailed_quality_score**: N/A")
    return "\n".join(lines)
```

- [ ] **Step 2: Commit**

```bash
git add templates/tasks/locomotion/eval_metrics.py
git commit -m "feat: add locomotion Tier 2 eval_metrics.py template"
```

---

### Task 4: Update config.md Template

**Files:**
- Modify: `templates/config.md:26-30` (Monitoring section)

- [ ] **Step 1: Add Task monitoring field to config template**

In `templates/config.md`, replace the Monitoring section:

```markdown
## Monitoring
- Tool: <wandb, tensorboard, local>
- Task monitoring: <task-name>
- Metric categories: [<prefix1/>, <prefix2/>, ...]
- Key metrics: [<metric1>, <metric2>, ...]
- Kill threshold: 2
- Max iterations: 10
```

The `Task monitoring` field points to `.claude/rl-training/tasks/<task-name>/` which contains the task-specific monitoring config and scripts.

- [ ] **Step 2: Commit**

```bash
git add templates/config.md
git commit -m "feat: add Task monitoring field to config template"
```

---

### Task 5: Update evaluate_policy.py Template to Call Task eval_metrics

**Files:**
- Modify: `templates/scripts/evaluate_policy.py`

- [ ] **Step 1: Read current evaluate_policy.py template**

Read `templates/scripts/evaluate_policy.py` to understand the current template structure.

- [ ] **Step 2: Add task eval_metrics integration**

The template `evaluate_policy.py` is a starting point that the SETUP phase adapts per-project. Update it to show how to integrate with `tasks/<name>/eval_metrics.py`. Add after the per-scenario metrics collection loop:

After the line that computes `metrics[cmd_name]` with rms_vel_error, falls, etc., add trajectory data collection and eval_metrics call. The template should include:

1. Import the task's eval_metrics module:
```python
import importlib.util

def load_task_eval_metrics(config_path):
    """Load the task-specific eval_metrics module."""
    text = Path(config_path).read_text()
    task_match = re.search(r"^- Task monitoring:\s*(.+)$", text, re.MULTILINE)
    if not task_match:
        return None
    task_name = task_match.group(1).strip()
    eval_path = Path(f".claude/rl-training/tasks/{task_name}/eval_metrics.py")
    if not eval_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("eval_metrics", eval_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
```

2. In the scenario loop, collect trajectory data (joint positions, contact states) alongside existing metrics, then call `eval_mod.analyze_trajectory(...)` and merge the results into the scenario metrics dict.

3. Include the quality results in the markdown output and JSON output.

The exact implementation depends on the project's framework (what data is available during rollouts), so this template shows the integration pattern. The SETUP phase generates the actual code.

- [ ] **Step 3: Commit**

```bash
git add templates/scripts/evaluate_policy.py
git commit -m "feat: add task eval_metrics integration pattern to evaluate_policy template"
```

---

### Task 6: Update SKILL.md — SETUP Phase (Metric Design Agent)

**Files:**
- Modify: `skills/rlpilot/SKILL.md:167-178` (Step 4 section)

- [ ] **Step 1: Split Step 4 into 4a, 4b, 4c**

Replace the current Step 4 content (lines 167-175) with:

```markdown
### Step 4a: Monitoring & Evaluation — Generic Setup (interactive)

- What monitoring tool? (WandB, TensorBoard, local logs)
- If WandB: project path → store in `rl_training_infra.md`
- Key metric categories and prefixes to track
- Evaluation: what scenarios to test, what metrics, record video?
- May iterate with user to refine eval strategy

### Step 4b: Metric Design Agent (interactive)

Spawn the Metric Design Agent to brainstorm and generate task-specific monitoring. This agent:

1. Reads context: robot type (from Step 2), actuators, task objective, simulator, existing reward terms, observation space. Scans the training code for existing WandB log calls to identify what's already logged.

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

6. Updates `config.md` with `Task monitoring: <task-name>`

The Metric Design Agent can also be re-invoked on demand: user says "improve monitoring for this task," or the ITERATE phase detects 3+ iterations with human feedback tags that don't correspond to any monitored metric.

### Step 4c: Monitoring Validation (sequential, one agent per host)
```

(Step 4c content is the existing Step 4b monitoring validation — keep it unchanged, just renumber.)

- [ ] **Step 2: Commit**

```bash
git add skills/rlpilot/SKILL.md
git commit -m "feat: add Metric Design Agent step to SETUP phase"
```

---

### Task 7: Update SKILL.md — MONITOR CRON PROMPT

**Files:**
- Modify: `skills/rlpilot/SKILL.md:428-513` (MONITOR CRON PROMPT section)

- [ ] **Step 1: Update STEP 0 to read task monitoring config**

In the MONITOR CRON PROMPT section, after the existing STEP 0 lines, add:

```
- Read config.md → Monitoring.Task monitoring field. If present, set TASK_DIR = .claude/rl-training/tasks/<task-name>/
- If TASK_DIR exists, read TASK_DIR/monitor_config.md → extract: quality thresholds, decision rules, human feedback tags.
```

- [ ] **Step 2: Add Tier 1 quality metrics step after metrics fetch**

After the existing step 2 (Fetch metrics), add a new sub-step:

```
2b. Compute quality metrics (if TASK_DIR exists):
    Extract raw metrics JSON from the monitor output file (the <!-- RAW_METRICS:...--> comment).
    Save raw metrics to a temp file: run_NNN/raw_metrics_MMM.json
    Previous derived: run_NNN/derived_metrics_{M-1 padded}.json (if M > 1)
    Run: uv run TASK_DIR/monitor_metrics.py run_NNN/raw_metrics_MMM.json [--previous <prev_derived>] --config TASK_DIR/monitor_config.md
    Capture stdout → save to run_NNN/derived_metrics_{M padded}.json
    Capture stderr → append to run_NNN/monitor_{M padded}.md (quality metrics markdown)
    If this fails, log warning but continue with standard metrics only.
```

- [ ] **Step 3: Update eval step to include Tier 2**

Update existing step 3 (Evaluate policy) to add:

```
3b. If TASK_DIR exists and TASK_DIR/eval_metrics.py exists:
    evaluate_policy.py should call the task's eval_metrics.analyze_trajectory() for each scenario.
    The task-specific quality breakdown is included in eval_metrics.md and eval_metrics.json.
```

- [ ] **Step 4: Update notification to include quality scores**

Update step 4 (Send notification) to include quality score:

```
   If derived metrics exist (run_NNN/derived_metrics_{M padded}.json):
     Read quality_score and sub-scores.
     Append to MSG: printf '\nQuality: %.2f' "$QUALITY_SCORE"
     If any sub-score is below quality_score_bad_threshold from monitor_config.md:
       Append warning: printf ' (%s: %.2f ⚠)' "$SUB_METRIC" "$SUB_SCORE"
```

- [ ] **Step 5: Update DECIDE step with quality rules**

Update step 5 (DECIDE) to add after the existing rules:

```
   Additional rules (if TASK_DIR and monitor_config.md exist):
   Read quality decision rules from monitor_config.md.
   - If reward_vs_quality_divergence is true:
     Check if reward is improving (> 5% gain) but quality_score is declining (> 5% drop)
     for quality_declining_monitors consecutive monitors → BAD
   - If quality_score < quality_score_bad_threshold → BAD
   - For FINISH: also require quality_score >= quality_finish_minimum
   - If quality_score dropped below quality_score_bad_threshold, trigger eval on next tick
     (set a flag in session_state.json: "eval_requested": true)
```

- [ ] **Step 6: Update ACT step for enriched result.md and human feedback**

Update step 6 (ACT), in the KILL section, update result.md writing:

```
   - Write run_NNN/result.md with: goal, kill step, metrics at death, eval results, trend, assessment,
     quality metrics trend across monitors (read derived_metrics_*.json files),
     which quality sub-scores triggered the kill (if applicable).
   - Append Human Assessment section to result.md:
     ## Human Assessment
     Tags: []
     Notes: No feedback yet.
   - Update session_state.json iterations entry:
     Add to iterations: {run: N, result: "<one-line>", human_feedback: null}
```

- [ ] **Step 7: Commit**

```bash
git add skills/rlpilot/SKILL.md
git commit -m "feat: update Monitor cron prompt with quality metrics and human feedback"
```

---

### Task 8: Update SKILL.md — ITERATE Phase (Human Feedback + Quality Diagnosis)

**Files:**
- Modify: `skills/rlpilot/SKILL.md:360-424` (Phase 4: ITERATE section)

- [ ] **Step 1: Update DIAGNOSE step to read quality data and human feedback**

In the DIAGNOSE section (Step 1), after the existing step 5 (Gather evidence), add:

```
5b. Read quality metrics: scan run_{previous_run}/derived_metrics_*.json files.
    Identify which quality sub-scores degraded and when.
    Read run_{previous_run}/result.md → Human Assessment section for any user feedback.
5c. Read human_feedback from previous iterations in session_state.json.
    Look for patterns: if 3+ iterations share the same human feedback tag
    with no corresponding metric improvement, flag this in the diagnosis:
    "Persistent issue: <tag> reported across runs N, M, P — suggest re-running Metric Design Agent."
```

- [ ] **Step 2: Add non-blocking human feedback request**

After the DIAGNOSE section and before IMPLEMENT, add:

```
### Step 1b: Request Human Feedback (non-blocking)

In the same message as the diagnosis summary, include:
"How did the gait look in the last run? If you have feedback, I'll incorporate it — otherwise I'm proceeding with the metrics-based diagnosis."

This is a single message, not a blocking wait. The agent continues to IMPLEMENT in the same turn.

If the user later provides qualitative feedback (in a follow-up message or during any subsequent interaction):
- Parse feedback into structured tags from the vocabulary in tasks/<name>/monitor_config.md → Human Feedback Tags
- Accept free-text notes that don't map to tags
- Update the relevant run's result.md → Human Assessment section in-place
- Update session_state.json → iterations[N].human_feedback with {tags: [...], notes: "..."}
- This feedback is then available to future ITERATE phases
```

- [ ] **Step 3: Commit**

```bash
git add skills/rlpilot/SKILL.md
git commit -m "feat: add quality diagnosis and human feedback to ITERATE phase"
```

---

### Task 9: Update SKILL.md — Scripts Table and Generate Step

**Files:**
- Modify: `skills/rlpilot/SKILL.md:236-248` (Step 6: Generate section)
- Modify: `skills/rlpilot/SKILL.md:565-580` (Scripts table)

- [ ] **Step 1: Update Generate step to include task directory**

In Step 6 (Generate), after the existing script generation list, add:

```
- Task monitoring directory `.claude/rl-training/tasks/<task-name>/` is generated by the Metric Design Agent in Step 4b (not in this step)
```

- [ ] **Step 2: Update Scripts table**

Add to the shared scripts table:

```markdown
| `tasks/<name>/monitor_metrics.py <raw.json> [--previous prev.json] [--config cfg.md]` | Compute Tier 1 derived quality metrics |
| `tasks/<name>/eval_metrics.py` (imported by evaluate_policy.py) | Compute Tier 2 detailed quality analysis |
| `tasks/<name>/monitor_config.md` | Task-specific metrics, thresholds, decision rules |
```

- [ ] **Step 3: Commit**

```bash
git add skills/rlpilot/SKILL.md
git commit -m "feat: update scripts table and generate step for task monitoring"
```

---

### Task 10: Add Manipulation and Balance Template Stubs

**Files:**
- Create: `templates/tasks/manipulation/monitor_config.md`
- Create: `templates/tasks/balance/monitor_config.md`

- [ ] **Step 1: Create manipulation monitor_config.md stub**

```markdown
# Monitoring Configuration — <task-name>

## Task Type
manipulation

## Quality Metrics — Tier 1 (Derived from WandB)

| Metric | Source | Weight | Description |
|--------|--------|--------|-------------|
| grasp_stability | Grasp reward terms | 0.3 | Consistency of grasp across episodes |
| approach_smoothness | Action rate metrics | 0.2 | Smoothness of approach trajectory |
| success_rate | Episode success/failure | 0.3 | Fraction of successful task completions |
| force_efficiency | Contact force metrics | 0.2 | Appropriate force usage (not too high/low) |

## Quality Metrics — Tier 2 (Eval-Time)

| Metric | Description | Good Range |
|--------|-------------|------------|
| trajectory_jerk | End-effector trajectory smoothness | < 20.0 |
| grasp_force_profile | Force application pattern during grasp | > 0.5 |
| approach_directness | Path efficiency (straight line ratio) | > 0.7 |

## Decision Rules

- quality_score_bad_threshold: 0.4
- quality_declining_monitors: 2
- quality_finish_minimum: 0.7
- reward_vs_quality_divergence: true

## Human Feedback Tags

- fumbling: Unstable or uncertain grasping
- excessive_force: Applying too much contact force
- inefficient_path: Taking unnecessary detours to reach target
- dropping: Failing to maintain grasp during transport
- collision: Hitting obstacles or unintended surfaces
- too_slow: Overly cautious movements
```

- [ ] **Step 2: Create balance monitor_config.md stub**

```markdown
# Monitoring Configuration — <task-name>

## Task Type
balance

## Quality Metrics — Tier 1 (Derived from WandB)

| Metric | Source | Weight | Description |
|--------|--------|--------|-------------|
| com_stability | CoM tracking metrics | 0.3 | Center of mass deviation from target |
| recovery_speed | Episode length after perturbation | 0.2 | How quickly robot recovers from disturbances |
| posture_score | Upright/pose reward terms | 0.3 | Quality of maintained posture |
| energy_efficiency | Torque/action metrics | 0.2 | Energy usage for balance maintenance |

## Quality Metrics — Tier 2 (Eval-Time)

| Metric | Description | Good Range |
|--------|-------------|------------|
| com_jerk | CoM trajectory smoothness | < 10.0 |
| ankle_strategy_score | Use of ankle vs hip strategy | > 0.5 |
| base_oscillation | Frequency and amplitude of body sway | < 0.3 |

## Decision Rules

- quality_score_bad_threshold: 0.4
- quality_declining_monitors: 2
- quality_finish_minimum: 0.7
- reward_vs_quality_divergence: true

## Human Feedback Tags

- wobbling: Excessive oscillation during balance
- stiff: Overly rigid posture, not natural
- overcorrecting: Large corrective movements for small disturbances
- drifting: Slowly moving away from target position
- collapsing: Gradual loss of posture over time
```

- [ ] **Step 3: Commit**

```bash
git add templates/tasks/manipulation/monitor_config.md templates/tasks/balance/monitor_config.md
git commit -m "feat: add manipulation and balance monitor_config.md template stubs"
```

---

### Task 11: Final Integration — Verify All Changes Consistent

- [ ] **Step 1: Verify file structure**

```bash
find templates/tasks -type f | sort
```

Expected:
```
templates/tasks/balance/monitor_config.md
templates/tasks/locomotion/eval_metrics.py
templates/tasks/locomotion/monitor_config.md
templates/tasks/locomotion/monitor_metrics.py
templates/tasks/manipulation/monitor_config.md
```

- [ ] **Step 2: Verify config.md template has Task monitoring field**

```bash
grep "Task monitoring" templates/config.md
```

Expected: `- Task monitoring: <task-name>`

- [ ] **Step 3: Verify SKILL.md references are consistent**

Check that all references to task monitoring files use consistent paths:

```bash
grep -n "tasks/" skills/rlpilot/SKILL.md | head -20
```

Verify: all references use `.claude/rl-training/tasks/<name>/` or `TASK_DIR` consistently.

- [ ] **Step 4: Verify monitor_metrics.py runs without errors on empty input**

```bash
echo '{}' > /tmp/test_metrics.json
python3 templates/tasks/locomotion/monitor_metrics.py /tmp/test_metrics.json --config templates/tasks/locomotion/monitor_config.md
```

Expected: JSON output with null values for all metrics (no crash).

- [ ] **Step 5: Verify eval_metrics.py imports without errors**

```bash
python3 -c "import sys; sys.path.insert(0, 'templates/tasks/locomotion'); import eval_metrics; print('OK')"
```

Expected: `OK` (requires numpy installed).

- [ ] **Step 6: Final commit if any fixes were needed**

```bash
git add -A
git status
# If changes exist:
git commit -m "fix: integration fixes for task-aware monitoring"
```
