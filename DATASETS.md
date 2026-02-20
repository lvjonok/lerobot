# HuggingFace Dataset Storage

This document covers authentication, uploading, and downloading LeRobot datasets from HuggingFace Hub.

## Authentication

```bash
# Login (creates ~/.cache/huggingface/token)
huggingface-cli login
# or
hf auth login

# Verify
hf auth whoami
```

You need a HuggingFace account and a **write** access token (create one at https://huggingface.co/settings/tokens). The token is stored locally and reused by all HuggingFace tools.

## Dataset Location

LeRobot datasets are cached locally at:

```
~/.cache/huggingface/lerobot/<user>/<dataset_name>/
```

Structure:

```
data/
  chunk-000/
    file-000.parquet          # Observation/action columns
meta/
  info.json                   # fps, features, shapes
  stats.json                  # Normalization statistics (mean, std, min, max)
  tasks.parquet               # Task descriptions
  episodes/
    chunk-000/
      file-000.parquet        # Episode metadata
videos/
  observation.images.front/
    chunk-000/
      file-000.mp4            # Video-encoded camera frames
  observation.images.wrist/
    chunk-000/
      file-000.mp4
```

## Pushing a Dataset

After recording or reprocessing a dataset locally, push it to HuggingFace:

```bash
cd ~/.cache/huggingface/lerobot/<user>/<dataset_name>

huggingface-cli upload <user>/<dataset_name> . . \
    --repo-type dataset \
    --commit-message "description of changes"
```

- The first `.` is the local directory (current dir).
- The second `.` is the remote path (repo root).
- `--repo-type dataset` is required (default is `model`).
- The repo is created automatically on first push if it doesn't exist.
- Only changed files are uploaded; unchanged files are skipped.

### Creating a new dataset repo explicitly

```bash
huggingface-cli repo create <dataset_name> --type dataset
```

## Pulling a Dataset

### Automatic (via LeRobotDataset)

When you pass a `repo_id` to training or load a `LeRobotDataset`, it downloads automatically:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("domrachev03/franka_timing_belt_haply")
```

```bash
# Training auto-downloads the dataset
python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=domrachev03/franka_timing_belt_haply \
    ...
```

The dataset is cached at `~/.cache/huggingface/lerobot/domrachev03/franka_timing_belt_haply/`. Subsequent loads use the cache.

### Manual download

```bash
huggingface-cli download <user>/<dataset_name> \
    --repo-type dataset \
    --local-dir ~/.cache/huggingface/lerobot/<user>/<dataset_name>
```

### Forcing re-download

Delete the local cache and load again:

```bash
rm -rf ~/.cache/huggingface/lerobot/<user>/<dataset_name>
```

The next `LeRobotDataset()` call or training run will re-download from Hub.

**Warning**: If you have local modifications (e.g. reprocessed parquet files), re-downloading will overwrite them. Push local changes to Hub first.

## Current Datasets

| Dataset | Episodes | Description |
|---|---|---|
| `domrachev03/franka_timing_belt_haply` | 80 | Timing belt task, Haply teleop |
| `domrachev03/franka_timing_belt_haply_static` | 30 | Timing belt task, static grasp |

### Observation columns

| Column | Dims | Contents |
|---|---|---|
| `observation.state` | 8 | `tcp.pos`(3) + `tcp.quat`(4) + `gripper.pos`(1) |
| `observation.effort` | 6 | `ft_sensor.force`(3) + `ft_sensor.torque`(3) |
| `observation.joints` | 7 | `joint.pos`(7) |
| `observation.joint_vel` | 7 | `joint.vel`(7) |

### Inspecting a dataset

```bash
cd ~/.cache/huggingface/lerobot/domrachev03/franka_timing_belt_haply

# View metadata
cat meta/info.json | python -m json.tool

# View parquet columns
python -c "
import pyarrow.parquet as pq
t = pq.read_table('data/chunk-000/file-000.parquet')
print('Columns:', t.column_names)
print('Rows:', len(t))
for c in t.column_names:
    col = t[c]
    sample = col[0].as_py()
    if isinstance(sample, list):
        print(f'  {c}: dim={len(sample)}')
    else:
        print(f'  {c}: {type(sample).__name__}')
"
```
