# SETUP-HOSTS

You are configuring and validating training hosts for RL training.

## Context

<CONFIG_CONTEXT>
{orchestrator injects full config.md here}
</CONFIG_CONTEXT>

<MONITOR_CONFIG>
{orchestrator injects monitor_config.md here, if it exists}
</MONITOR_CONFIG>

<INFRA_MEMORY>
{orchestrator injects rl_training_infra.md here, if it exists}
</INFRA_MEMORY>

## Recovery Check

If `## Hosts` section already exists in config.md with host entries, and matching `hosts/<name>/host.md` files exist:
- Show the existing hosts to the user
- Ask: "These hosts are already configured: [list]. Do you want to add more, re-validate existing ones, or continue?"
- Act accordingly

## Step 1: Collect Hosts (interactive)

Ask user to add hosts one by one. For each host:
- Name (identifier, e.g. `lerobot`, `cluster1`, `local`)
- Type: direct SSH, SLURM cluster, or local
- Connection details (SSH alias, remote dir, tunnel command)
- GPU check command and threshold
- PATH setup if needed
- Cluster-specific params if applicable (partition, GPU type, allocation command)
- Dependencies command for this host

Generate per-host directory `.claude/rl-training/hosts/<name>/`:
- `host.md` — host config
- `launch.sh` — availability check + worktree setup + training launch
- `kill.sh` — stop training for a session

Refer to template examples in `${CLAUDE_PLUGIN_ROOT}/templates/hosts/example/` for the expected format and structure of host scripts. Adapt them to the specific host type and configuration.

Ask "Add another host?" until done.

Write `## Hosts` section in config.md with ordered host list:
```
## Hosts
Order: [<host1>, <host2>, ...]
```

## Step 2: Host Validation (sequential, one agent per host)

Validate each host **sequentially** — one at a time, using a **separate Agent per host** (clean context per machine). Fix issues with the user before moving to the next host.

For each host in order, spawn a dedicated Agent (foreground) with this prompt:

```
You are validating training host "<HOST_NAME>" for the RL training setup.
Host config: <paste host.md content>
Monitoring tool: <tool from config.md Monitoring section>
Monitoring config: <WandB project path or other relevant config>

Run these checks in order. For each check, report PASS or FAIL with details.

1. SSH CONNECTIVITY
   - Run: ssh -o ConnectTimeout=10 -o BatchMode=yes <ssh-alias> "echo ok"
   - If fail: report the exact error. Common fixes:
     * SSH key not added: suggest `ssh-add` or `ssh-copy-id`
     * Host not in known_hosts: suggest `ssh-keyscan`
     * Wrong alias: ask user to check ~/.ssh/config
     * Connection refused: check if host is up, port is correct
   - If SSH fails, STOP here — remaining checks require SSH access.

2. REMOTE DIRECTORY
   - Run: ssh <ssh-alias> "test -d <remote-dir> && echo exists || echo missing"
   - If missing: offer to create it: ssh <ssh-alias> "mkdir -p <remote-dir>"

3. GPU AVAILABILITY
   - Run: ssh <ssh-alias> "<gpu-check-command>"
   - Report GPU count, type, memory if available
   - If no GPU found: warn but don't fail (might be a CPU-only host)

4. DEPENDENCIES
   - Run: ssh <ssh-alias> "which python3 && python3 --version"
   - Run: ssh <ssh-alias> "which uv && uv --version" (if uv is used)
   - If PATH setup is configured, test it: ssh <ssh-alias> "source <path-setup> && which python3"
   - Report missing dependencies

5. GIT ACCESS
   - Run: ssh <ssh-alias> "cd <remote-dir> && git status --short" (if repo exists)
   - Or: ssh <ssh-alias> "cd <remote-dir> && git clone --depth 1 <repo-url> ." (if empty dir)
   - Report git availability and repo state

6. MONITORING TOOL INSTALLED
   - For WandB: ssh <ssh-alias> "<path-setup-if-any> && python3 -c 'import wandb; print(wandb.__version__)'"
   - For TensorBoard: ssh <ssh-alias> "<path-setup-if-any> && python3 -c 'import tensorboard; print(tensorboard.__version__)'"
   - If not installed: report and suggest install command

7. MONITORING TOOL AUTHENTICATED
   - For WandB: ssh <ssh-alias> "<path-setup-if-any> && python3 -c 'import wandb; api = wandb.Api(); print(api.viewer.entity)'"
   - If not authenticated: report FAIL. Suggest: ssh <ssh-alias> "wandb login"
   - If authenticated: report the entity/username

8. MONITORING PROJECT ACCESS (if applicable)
   - For WandB: ssh <ssh-alias> "<path-setup-if-any> && python3 -c 'import wandb; api = wandb.Api(); p = api.project(\"<project-name>\", entity=\"<entity>\"); print(p.name)'"
   - If project doesn't exist: warn (it may be created on first run, which is fine)
   - If access denied: report FAIL with details

Return a structured report:
HOST: <name>
STATUS: READY | NEEDS_SETUP | UNREACHABLE
CHECKS:
  - SSH: PASS/FAIL — <details>
  - Directory: PASS/FAIL — <details>
  - GPU: PASS/FAIL — <details>
  - Dependencies: PASS/FAIL — <details>
  - Git: PASS/FAIL — <details>
  - Monitoring installed: PASS/FAIL — <details>
  - Monitoring authenticated: PASS/FAIL — <entity or error>
  - Monitoring project access: PASS/FAIL/SKIP — <details>
ISSUES: <list of things that need fixing, or "none">
SUGGESTED_FIXES: <actionable commands the user can run to fix issues>
```

After each validation agent returns:
- If READY: confirm to user, move to the next host.
- If NEEDS_SETUP or UNREACHABLE:
  - Present the issues and suggested fixes to the user
  - Walk through each fix interactively (run commands, verify they work)
  - After fixes, re-run the validation agent for this host to confirm
  - If the user wants to skip the host instead: remove it from the host list and update config.md
- Only proceed to the next host after the current one is fully validated or skipped.
