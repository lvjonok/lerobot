# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Workstations

This project runs across three machines. Each has a `workstation` config key set via `bd config set workstation <value>`:

| Machine | Config value | Purpose |
|---|---|---|
| Personal PC | `personal-pc` | General development |
| Inference PC | `inference-pc` | Robot teleoperation, dataset collection, inference |
| DL Server | `dl-server` | GPU training |

Every issue MUST have at least one workstation label:
- `ws:any` — development tasks, executable on any machine
- `ws:inference-pc` — requires robot/sensors (inference PC only)
- `ws:dl-server` — requires GPU training (DL server only)

A task can have multiple `ws:` labels if it can be done on several machines.

## Area Labels

Every issue MUST also have at least one area label indicating which part of the codebase it relates to:

- `area:teleoperation` — robot plugins, teleop devices, dataset collection, processor pipelines
- `area:training` — policy training, dataset format, training loop
- `area:inference` — policy deployment, evaluation, real-robot rollouts, async inference

A task can have multiple `area:` labels if it spans areas.

### Creating issues with labels

Always add both `ws:` and `area:` labels when creating an issue:

```bash
bd create --title="Refactor teleoperator config" --type=task --priority=2
bd label add <id> ws:any area:teleoperation

bd create --title="Collect pick-and-place dataset" --type=task --priority=1
bd label add <id> ws:inference-pc area:teleoperation

bd create --title="Train RDP on new dataset" --type=task --priority=1
bd label add <id> ws:dl-server area:training
```

## Quick Reference

### Finding work (workstation-aware)

At session start, check which workstation you're on and filter accordingly:

```bash
# 1. Check current workstation
bd config get workstation

# 2. Find ready work for this machine
# On personal-pc:
bd ready --label-any ws:any --label-any ws:personal-pc

# On inference-pc:
bd ready --label-any ws:any --label-any ws:inference-pc

# On dl-server:
bd ready --label-any ws:any --label-any ws:dl-server
```

### General workflow

```bash
bd ready              # Find available work (add --label-any filters per above)
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Project Context

This is a fork of HuggingFace LeRobot with custom plugins for Franka/Flexiv robot teleoperation, dataset recording, and Reactive Diffusion Policy (RDP) training/inference. Key custom components:

- **Robot plugins:** `crisp_fastapi` (HTTP REST client for crisp_py servers), `slim_crisp` (ZMQ bridge)
- **Teleoperator plugins:** `spacemouse`, `haply` (Inverse3 + VerseGrip), `meta_quest` (VR hand tracking)
- **Processors:** Delta-to-absolute conversion, clutch logic, axis filtering
- **Policies:** `rdp_tokenizer` (Asymmetric Tokenizer VAE), `rdp_latent_diffusion` (Latent Diffusion Policy)

The robot server (`franka_server_crisp.py`) lives in the parent `reactive_diffusion_policy/real_world/robot/` repository — it is NOT part of this repo.

See [CLAUDE.md](CLAUDE.md) for full architecture documentation.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues (with `ws:` labels!) for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
