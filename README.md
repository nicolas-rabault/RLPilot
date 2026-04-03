# RLPilot — Claude Code Plugin

Autonomous RL training management for Claude Code. Launch training on remote GPU servers, monitor metrics via WandB, auto-evaluate policies, and iterate on failures — all driven by Claude.

## What it does

When you tell Claude "train the robot to run faster" or "the training looks bad", this plugin:

1. **SETUP** — Scans your project, asks about your robot/task, configures hosts, generates scripts
2. **CLARIFY** — Understands what behavior you want to change
3. **CODE** — Brainstorms approach, implements reward/observation/curriculum changes
4. **LAUNCH** — Pushes code, launches training on the best available GPU host
5. **MONITOR** — Autonomous cron checks metrics every ~30 min, evaluates policy, decides keep/kill
6. **ITERATE** — On failure: diagnoses root cause, fixes code, relaunches automatically

Multiple training sessions can run in parallel on different branches and hosts.

## Requirements

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI

## Install

```bash
# Add the marketplace
/plugin marketplace add nicolas-rabault/RLPilot

# Install the plugin
/plugin install rlpilot@nicolas-rabault/RLPilot
```

Or for local development:

```bash
claude --plugin-dir /path/to/rlpilot
```

## First-time setup

Just tell Claude to train something. If no config exists, it runs the interactive SETUP phase:

1. Scans your codebase for simulator, RL framework, task definitions
2. Asks about your robot and training objective
3. Configures SSH hosts with validation
4. Sets up monitoring (WandB) and notifications (Discord/Slack)
5. Generates all scripts into your project's `.claude/rl-training/`

## Project structure after setup

```
your-project/
├── .claude/rl-training/
│   ├── config.md              # Project-specific training config
│   ├── scripts/               # Generated scripts
│   │   ├── init_session.sh
│   │   ├── get_latest_run.py
│   │   ├── monitor.py
│   │   ├── evaluate_policy.py
│   │   └── notify.sh
│   └── hosts/
│       └── <host-name>/
│           ├── host.md
│           ├── launch.sh
│           └── kill.sh
└── logs/sessions/             # Training session logs (gitignored)
```

## Supported setups

- **Simulators**: MuJoCo, Isaac Sim, PyBullet, or any simulator with a CLI training command
- **RL frameworks**: rsl_rl, Stable Baselines3, CleanRL, or any framework
- **Monitoring**: WandB (primary), TensorBoard (partial)
- **Hosts**: Direct SSH, SLURM clusters, or local
- **Notifications**: Discord, Slack, or custom webhook

## How it works

### Session management

Each training session is tied to a git branch. Sessions track:
- Current phase (LAUNCH → MONITOR → ITERATE or FINISHED)
- Run history with metrics snapshots
- Iteration log (what was tried and why it failed)

### Monitoring decisions

The cron agent compares key metrics against configurable thresholds:
- **KEEP**: Metrics improving or stable
- **BAD**: Reward dropped >10% from peak, or errors increasing
- **FINISH**: Reward plateaued and eval metrics are acceptable

After consecutive bad readings (default: 2), training is killed and the ITERATE phase begins.

### Iteration loop

On failure, Claude:
1. Reads the metric history and previous iteration attempts
2. Diagnoses the root cause using systematic debugging
3. Makes a targeted fix (one hypothesis, one change)
4. Gets the fix code-reviewed
5. Relaunches training

After too many failed iterations, it pauses and asks for human guidance.

## Contributing

PRs welcome. The plugin has two layers:

- `skills/rl-training/SKILL.md` — The orchestration logic (phases, decisions, prompts)
- `templates/` — Example scripts that the SETUP phase adapts per-project

## License

MIT
