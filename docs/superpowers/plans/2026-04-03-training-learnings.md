# Training Learnings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared learning document that accumulates actionable training insights across sessions, updated by a learning agent triggered during ITERATE/FINISH/PAUSED phases.

**Architecture:** A deterministic `learnings.py` script gathers structured run data. An LLM subagent reads that report + the current `docs/training-learnings.md` and decides what to update. The learning agent runs as background during ITERATE (non-blocking) and foreground during FINISH/PAUSED.

**Tech Stack:** Python 3, JSON, markdown, git.

---

### Task 1: Create the `learnings.py` template script

**Files:**
- Create: `templates/scripts/learnings.py`

- [ ] **Step 1: Create the template script**

```python
#!/usr/bin/env python3
"""Gather structured run data for the learning agent.

Usage:
    uv run .claude/rl-training/scripts/learnings.py <session-dir> --run <N>

Reads run data (result.md, analysis.md, metrics, session_state.json, git diff)
and outputs a structured markdown report to stdout for the learning agent.
"""

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path


def load_text(path):
    p = Path(path)
    return p.read_text() if p.exists() else None


def load_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def collect_metrics_trajectory(run_dir):
    """Collect quality scores across all derived_metrics files."""
    files = sorted(glob.glob(str(run_dir / "derived_metrics_*.json")))
    trajectory = []
    for f in files:
        data = load_json(f)
        if data and data.get("quality_score") is not None:
            entry = {"quality_score": data["quality_score"]}
            for k, v in data.items():
                if k not in ("quality_score", "coverage", "missing") and isinstance(v, (int, float)):
                    entry[k] = v
            trajectory.append(entry)
    return trajectory


def get_reward_trend(run_dir):
    """Determine reward trend from raw metrics."""
    files = sorted(glob.glob(str(run_dir / "raw_metrics_*.json")))
    rewards = []
    for f in files:
        data = load_json(f)
        if not data:
            continue
        reward_keys = [k for k in data if "Episode_Reward" in k and isinstance(data[k], (int, float))]
        if reward_keys:
            total = sum(data[k] for k in reward_keys)
            rewards.append(total)
    if len(rewards) < 2:
        return "insufficient data"
    change = (rewards[-1] - rewards[0]) / (abs(rewards[0]) + 1e-8)
    if change > 0.05:
        return "improving"
    elif change < -0.05:
        return "degrading"
    return "plateauing"


def get_git_diff_summary(branch):
    """Get a summary of code changes on the branch."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--no-merges", "-10", branch, "--not", "main"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        commits = result.stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--stat", "main..." + branch],
            capture_output=True, text=True, timeout=10
        )
        return f"Commits:\n{commits}\n\nFiles changed:\n{diff.stdout.strip()}" if diff.returncode == 0 else commits
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def find_similar_iterations(iterations, current_run, current_tags):
    """Find past iterations with similar feedback tags or diagnosis patterns."""
    similar = []
    for it in iterations:
        if it["run"] >= current_run:
            continue
        fb = it.get("human_feedback")
        if fb and fb.get("tags"):
            overlap = set(fb["tags"]) & set(current_tags)
            if overlap:
                similar.append(f"Run {it['run']}: tags {list(overlap)} — {it.get('result', 'no result')}")
    return similar


def generate_report(session_dir, run_num):
    session_dir = Path(session_dir)
    run_dir = session_dir / f"run_{run_num:03d}"
    state = load_json(session_dir / "session_state.json")

    if not state:
        print("ERROR: session_state.json not found", file=sys.stderr)
        sys.exit(1)

    lines = []

    # Run Summary
    lines.append("## Run Summary")
    lines.append(f"- Goal: {state.get('goal', 'unknown')}")
    lines.append(f"- Branch: {state.get('branch', 'unknown')}")
    lines.append(f"- Run: {run_num} of {state.get('current_run', '?')} total iterations")
    lines.append(f"- Phase: {state.get('phase', 'unknown')}")

    result_text = load_text(run_dir / "result.md")
    if result_text:
        for line in result_text.splitlines():
            if line.startswith("**Reason:**"):
                lines.append(f"- Reason: {line.replace('**Reason:**', '').strip()}")
                break
            if line.startswith("**Killed at:**"):
                lines.append(f"- {line.replace('**', '').strip()}")
    lines.append("")

    # Metrics Trajectory
    trajectory = collect_metrics_trajectory(run_dir)
    lines.append("## Metrics Trajectory")
    if trajectory:
        first = trajectory[0]
        last = trajectory[-1]
        lines.append(f"- Starting quality_score: {first['quality_score']:.3f} -> Final: {last['quality_score']:.3f}")
        sub_keys = [k for k in last if k != "quality_score"]
        for k in sub_keys:
            start_val = first.get(k)
            end_val = last.get(k)
            if start_val is not None and end_val is not None:
                lines.append(f"- {k}: {start_val:.3f} -> {end_val:.3f}")
    lines.append(f"- Reward trend: {get_reward_trend(run_dir)}")
    lines.append("")

    # Diagnosis & Fix
    lines.append("## Diagnosis & Fix")
    analysis = load_text(run_dir / "analysis.md")
    if analysis:
        lines.append(f"- Hypothesis: (from analysis.md)")
        for aline in analysis.splitlines():
            if "hypothesis" in aline.lower() or "root cause" in aline.lower() or "because" in aline.lower():
                lines.append(f"  {aline.strip()}")
    else:
        lines.append("- No analysis.md found (training may have finished successfully)")

    diff_summary = get_git_diff_summary(state.get("branch", ""))
    if diff_summary:
        lines.append(f"- Code changes:\n{diff_summary}")
    else:
        lines.append("- Code changes: not available")

    # Did the fix help?
    if trajectory and len(trajectory) >= 2:
        delta = trajectory[-1]["quality_score"] - trajectory[0]["quality_score"]
        if delta > 0.05:
            lines.append("- Fix assessment: quality improved")
        elif delta < -0.05:
            lines.append("- Fix assessment: quality degraded")
        else:
            lines.append("- Fix assessment: inconclusive (quality unchanged)")
    else:
        lines.append("- Fix assessment: insufficient metrics data")
    lines.append("")

    # Human Feedback
    lines.append("## Human Feedback")
    current_tags = []
    iterations = state.get("iterations", [])
    for it in iterations:
        if it["run"] == run_num:
            fb = it.get("human_feedback")
            if fb and (fb.get("tags") or fb.get("notes")):
                current_tags = fb.get("tags", [])
                lines.append(f"- Tags: {current_tags}")
                lines.append(f"- Notes: {fb.get('notes', '')}")
            else:
                lines.append("- No human feedback for this run")
            break
    else:
        lines.append("- No iteration entry found for this run")
    lines.append("")

    # Iteration History Context
    lines.append("## Iteration History Context")
    if iterations:
        lines.append(f"- Total iterations so far: {len(iterations)}")
        for it in iterations:
            fb = it.get("human_feedback")
            fb_str = f" [tags: {fb['tags']}]" if fb and fb.get("tags") else ""
            lines.append(f"  - Run {it['run']}: {it.get('result', 'no result')}{fb_str}")
        similar = find_similar_iterations(iterations, run_num, current_tags)
        if similar:
            lines.append("- Similar past iterations:")
            for s in similar:
                lines.append(f"  - {s}")
    else:
        lines.append("- No previous iterations")
    lines.append("")

    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Gather run data for the learning agent")
    parser.add_argument("session_dir", help="Path to session directory")
    parser.add_argument("--run", type=int, required=True, help="Run number to analyze")
    args = parser.parse_args()

    generate_report(args.session_dir, args.run)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the file was created**

Run: `ls -la templates/scripts/learnings.py`
Expected: file exists

- [ ] **Step 3: Commit**

```bash
git add templates/scripts/learnings.py
git commit -m "feat: add learnings.py template for training knowledge extraction"
```

---

### Task 2: Add `learnings.py` to SETUP phase generation

**Files:**
- Modify: `skills/rlpilot/SKILL.md:266-279` (Phase 0, Step 6: Generate)

- [ ] **Step 1: Add learnings.py to the script generation list in SKILL.md**

In the SETUP Phase 0, Step 6 (Generate), add `learnings.py` to the list of shared scripts:

Find this block in `skills/rlpilot/SKILL.md` (around line 270):
```
- Generate shared scripts in `.claude/rl-training/scripts/`, using templates from `${CLAUDE_PLUGIN_ROOT}/templates/scripts/` as a starting point and adapting to the project:
  - `init_session.sh` — session state management (branch-based)
  - `get_latest_run.py` — find active run (WandB or other)
  - `monitor.py` — fetch metrics and format markdown report
  - `evaluate_policy.py` — headless eval with video and metrics (framework-specific, generated from scratch based on the project's RL framework)
  - `notify.sh` — notification delivery with `--branch` support
```

Add after `notify.sh`:
```
  - `learnings.py` — gather structured run data for the learning agent
```

- [ ] **Step 2: Add learnings.py to the Scripts table**

Find the Scripts table in SKILL.md (around line 630) and add a row:

```
| `learnings.py <session-dir> --run <N>` | Gather structured run data for learning agent |
```

Add it after the `generate_result.py` row.

- [ ] **Step 3: Add the initial training-learnings.md creation to SETUP**

In the same Step 6 (Generate) section, after the script generation bullet points, add:

```
- Create initial `docs/training-learnings.md` if it doesn't exist:
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
```

- [ ] **Step 4: Commit**

```bash
git add skills/rlpilot/SKILL.md
git commit -m "feat: add learnings.py generation and initial doc creation to SETUP phase"
```

---

### Task 3: Add learnings doc reads to CODE and ITERATE phases

**Files:**
- Modify: `skills/rlpilot/SKILL.md:293-298` (Phase 2 CODE, Step 1: INVESTIGATE)
- Modify: `skills/rlpilot/SKILL.md:396-424` (Phase 4 ITERATE, Step 1: DIAGNOSE)

- [ ] **Step 1: Add learnings read to CODE phase INVESTIGATE step**

Find the Phase 2 CODE, Step 1: INVESTIGATE section (around line 293):
```
Read the relevant code before changing anything. Get source file paths from `config.md` → Source Files section.
- Read task config, rewards, curriculum, observations
- Previous run results if available (`logs/sessions/`)
- Write findings in `logs/sessions/<branch-sanitized>/run_NNN/analysis.md`
```

Add a bullet after "Previous run results":
```
- Read `docs/training-learnings.md` for known pitfalls and proven techniques from past experiments
```

- [ ] **Step 2: Add learnings read to ITERATE phase DIAGNOSE step**

Find the Phase 4 ITERATE, Step 1: DIAGNOSE section. In the "Root Cause Investigation" list (around line 400), after step 5c (read human_feedback), add:

```
5d. Read `docs/training-learnings.md` — check if the current failure matches a known pattern.
    If a learning entry directly addresses the observed failure, reference it in the diagnosis.
```

- [ ] **Step 3: Commit**

```bash
git add skills/rlpilot/SKILL.md
git commit -m "feat: add training learnings doc reads to CODE and ITERATE phases"
```

---

### Task 4: Add learning agent dispatch to ITERATE phase

**Files:**
- Modify: `skills/rlpilot/SKILL.md:470-478` (Phase 4 ITERATE, between Step 4 CODE REVIEW and Step 5 RELAUNCH)

- [ ] **Step 1: Add learning agent step to ITERATE phase**

Find the section between Step 4 (CODE REVIEW) and Step 5 (RELAUNCH) in the ITERATE phase. After the push and context.md write (lines 472-474):

```
16. Commit: `cd ../<project>-wt-<branch-sanitized> && git add <files> && git commit -m "<what and why>"`
17. Push: `cd ../<project>-wt-<branch-sanitized> && git push origin HEAD`
18. Write `logs/sessions/<branch-sanitized>/run_{current_run}/context.md`
```

Add a new step before RELAUNCH:

```
### Step 4b: UPDATE LEARNINGS (background, non-blocking)

19. Run the data extraction script:
    `uv run .claude/rl-training/scripts/learnings.py <session_dir> --run <previous_run>`
    Capture the output.

20. Spawn a background Agent with this prompt:
    ```
    You are the training learning agent. Your job is to update docs/training-learnings.md with actionable insights from this training run.

    Here is the structured report from this run:
    <paste learnings.py output>

    Read docs/training-learnings.md. Based on the report:
    - Add new insights that are actionable and backed by the evidence in the report
    - Update existing entries if new evidence refines or contradicts them
    - Remove entries that are proven wrong by this run's data
    - Keep entries concise — one bullet per insight
    - Use existing categories or add new ones if needed
    - Task-qualify entries when they only apply to specific task types (e.g., "For locomotion: ...")
    - Do nothing if there's no new insight worth capturing

    If you make changes, commit with a message like: "learnings: <what was learned>"
    ```

    This runs in the background — proceed to RELAUNCH immediately without waiting.
```

- [ ] **Step 2: Renumber RELAUNCH step**

Update the existing Step 5 header to reflect the new step numbering:

Change:
```
### Step 5: RELAUNCH
```
To:
```
### Step 5: RELAUNCH
```

(The header stays the same — it's still Step 5. The internal numbering of sub-steps continues from 21.)

- [ ] **Step 3: Commit**

```bash
git add skills/rlpilot/SKILL.md
git commit -m "feat: add learning agent dispatch to ITERATE phase (background)"
```

---

### Task 5: Add learning agent dispatch to MONITOR CRON (FINISH and PAUSED)

**Files:**
- Modify: `skills/rlpilot/SKILL.md:566-571` (MONITOR CRON PROMPT, step 6 FINISH and KILL/PAUSED)

- [ ] **Step 1: Add learning agent to FINISH decision in cron prompt**

Find the FINISH handling in the MONITOR CRON PROMPT (around line 566):
```
   If decision = FINISH:
   - uv run .claude/rl-training/scripts/session.py update <session_dir> --set phase=FINISHED
   - Notify: bash .claude/rl-training/scripts/notify.sh "<notification from decide.py>" --branch "<BRANCH>"
   - Delete this cron: CronDelete with ID <CRON_ID>. Exit.
```

Replace with:
```
   If decision = FINISH:
   - uv run .claude/rl-training/scripts/session.py update <session_dir> --set phase=FINISHED
   - Run learning agent (foreground, before finishing):
     Run: uv run .claude/rl-training/scripts/learnings.py <session_dir> --run <current_run>
     Read docs/training-learnings.md.
     Based on the report, update docs/training-learnings.md with new insights:
     - Add actionable insights backed by evidence from this successful run
     - Update or remove entries contradicted by new evidence
     - Keep entries concise, task-qualified when needed
     - Commit changes: "learnings: <what was learned>"
   - Notify: bash .claude/rl-training/scripts/notify.sh "<notification from decide.py>" --branch "<BRANCH>"
   - Delete this cron: CronDelete with ID <CRON_ID>. Exit.
```

- [ ] **Step 2: Add learning agent to PAUSED path in KILL handling**

Find the KILL handling (around line 558), specifically the PAUSED check:
```
   - If current_run > max_iterations from config: use --set phase=PAUSED instead of ITERATE.
```

After that line, add:
```
     When setting phase=PAUSED, also run the learning agent (foreground):
     Run: uv run .claude/rl-training/scripts/learnings.py <session_dir> --run <N>
     Read docs/training-learnings.md.
     Based on the report, update docs/training-learnings.md with new insights.
     Focus especially on "What Doesn't Work" — this training hit max iterations without success.
     Commit changes if any: "learnings: <what was learned>"
```

- [ ] **Step 3: Commit**

```bash
git add skills/rlpilot/SKILL.md
git commit -m "feat: add learning agent to MONITOR CRON for FINISH and PAUSED decisions"
```

---

### Task 6: Final verification

**Files:**
- Read: `skills/rlpilot/SKILL.md` (verify all changes are consistent)
- Read: `templates/scripts/learnings.py` (verify script exists)

- [ ] **Step 1: Verify learnings.py template exists and is valid Python**

Run: `python3 -c "import ast; ast.parse(open('templates/scripts/learnings.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 2: Verify SKILL.md mentions learnings.py in all expected locations**

Run: `grep -n "learnings" skills/rlpilot/SKILL.md`
Expected: matches in SETUP (Step 6), CODE (Step 1), ITERATE (Step 4b), MONITOR CRON (FINISH + PAUSED), and Scripts table.

- [ ] **Step 3: Verify docs/training-learnings.md reference is consistent**

Run: `grep -n "training-learnings.md" skills/rlpilot/SKILL.md`
Expected: matches in CODE (INVESTIGATE), ITERATE (DIAGNOSE), ITERATE (Step 4b agent prompt), MONITOR CRON (FINISH + PAUSED).

- [ ] **Step 4: Commit any fixes if needed**

Only if verification found issues. Otherwise, done.
