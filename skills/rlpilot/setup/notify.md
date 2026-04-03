# SETUP-NOTIFY

You are configuring notifications for RL training progress.

## Context

<CONFIG_CONTEXT>
{orchestrator injects full config.md here}
</CONFIG_CONTEXT>

<WEBHOOK_FILES>
{orchestrator injects content of any existing webhook config files, e.g. .claude-discord.md}
</WEBHOOK_FILES>

## Flow (interactive)

1. Ask: "Do you want notifications about training progress?" (yes/no)

2. If **yes**:
   - Ask: "What notification method?" — Discord webhook, Slack webhook, email, custom?
   - If existing webhook config was found (see context above), offer to reuse it
   - Collect the webhook URL or credentials
   - Ask: "Which events should trigger notifications?" Present options:
     - `training_started` — when a training run begins
     - `monitor_update` — periodic monitoring reports
     - `eval_complete` — when evaluation finishes
     - `training_killed` — when a run is stopped
     - `iteration_started` — when a new iteration begins
     - `blocker` — when something needs human attention
   - Store credentials/webhooks in project memory file `rl_training_infra.md`

3. If **no**: proceed with notifications disabled.

## Output

Append to `.claude/rl-training/config.md`:

````
## Notifications
- Enabled: <true/false>
- Method: <discord/slack/email/custom/script>
- When: [<selected events>]
````

**Always generate `.claude/rl-training/scripts/notify.sh`:**
- Cron agents cannot invoke Claude Code skills, so all notification delivery must go through a bash script.
- Use the template from `${CLAUDE_PLUGIN_ROOT}/templates/scripts/notify.sh` as a starting point.
- If notifications are disabled, generate a no-op script (receives args, does nothing) so downstream scripts can call it unconditionally.
- Make it executable: `chmod +x .claude/rl-training/scripts/notify.sh`
