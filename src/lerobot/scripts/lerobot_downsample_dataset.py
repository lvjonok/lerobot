"""Downsample a LeRobot dataset to a lower FPS.

Actions that are velocities (twists) are summed over the downsampled window so that
displacement is preserved. Absolute action dimensions (e.g. gripper.pos) take the
value from the last frame in the window. Observations and images are taken from the
first frame of each window.

Example usage:
    python -m lerobot.scripts.lerobot_downsample_dataset \
        --repo-id domrachev03/franka_timing_belt_haply_static_v2 \
        --target-fps 10 \
        --action-absolute-subfeatures gripper.pos
"""

import argparse
from copy import deepcopy
from pathlib import Path

# Workaround: datasets lib may try to import AudioDecoder from torchcodec which
# doesn't exist in older torchcodec versions. Patch it before importing datasets.
try:
    import torchcodec.decoders as _tc_decoders

    if not hasattr(_tc_decoders, "AudioDecoder"):

        class _DummyAudioDecoder:
            pass

        _tc_decoders.AudioDecoder = _DummyAudioDecoder
except ImportError:
    pass

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.datasets.video_utils import decode_video_frames


def build_action_aggregation_mask(features: dict, absolute_subfeatures: list[str]) -> np.ndarray:
    """Build a boolean mask over action dimensions: True = sum (velocity), False = take last (absolute).

    Uses ``feature_sizes`` from the action feature spec to determine which dimensions
    correspond to which subfeatures.
    """
    action_spec = features["action"]
    names = action_spec["names"]
    feature_sizes = action_spec.get("feature_sizes", {n: 1 for n in names})

    mask = []
    for name in names:
        size = feature_sizes[name]
        is_velocity = name not in absolute_subfeatures
        mask.extend([is_velocity] * size)
    return np.array(mask, dtype=bool)


def aggregate_actions(actions: np.ndarray, sum_mask: np.ndarray) -> np.ndarray:
    """Aggregate a window of actions into a single action.

    Args:
        actions: (N, D) array of actions in the window.
        sum_mask: (D,) boolean — True dims are summed, False dims take last value.

    Returns:
        (D,) aggregated action.
    """
    result = actions[-1].copy()
    result[sum_mask] = actions[:, sum_mask].sum(axis=0)
    return result


def downsample_dataset(
    repo_id: str,
    target_fps: int,
    action_absolute_subfeatures: list[str],
    output_repo_id: str | None = None,
    root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> None:
    # Load source dataset
    print(f"Loading source dataset: {repo_id}")
    src_ds = LeRobotDataset(repo_id=repo_id, root=root)
    src_fps = src_ds.fps

    if target_fps >= src_fps:
        raise ValueError(f"target_fps ({target_fps}) must be less than source fps ({src_fps})")
    if src_fps % target_fps != 0:
        raise ValueError(f"source fps ({src_fps}) must be evenly divisible by target_fps ({target_fps})")

    step = src_fps // target_fps
    print(f"Downsampling {src_fps}Hz -> {target_fps}Hz (step={step})")

    if output_repo_id is None:
        output_repo_id = f"{repo_id}_{target_fps}hz"

    # Build features dict for the new dataset (exclude DEFAULT_FEATURES — create() adds them)
    src_features = deepcopy(src_ds.meta.info["features"])
    out_features = {k: v for k, v in src_features.items() if k not in DEFAULT_FEATURES}

    print(f"Output dataset: {output_repo_id}")
    out_ds = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=target_fps,
        root=output_root,
        robot_type=src_ds.meta.robot_type,
        features=out_features,
        use_videos=len(src_ds.meta.video_keys) > 0,
    )

    # Build action aggregation mask
    sum_mask = build_action_aggregation_mask(src_ds.meta.info["features"], action_absolute_subfeatures)
    print(f"Action dim={len(sum_mask)}, velocity dims (summed): {sum_mask.sum()}, "
          f"absolute dims (last): {(~sum_mask).sum()}")

    # Identify non-video, non-default feature keys
    tabular_keys = [
        k for k in src_ds.features
        if k not in DEFAULT_FEATURES
        and src_ds.features[k]["dtype"] not in ("image", "video")
    ]
    obs_keys = [k for k in tabular_keys if k != "action"]
    video_keys = list(src_ds.meta.video_keys)

    total_episodes = src_ds.meta.total_episodes
    total_out_frames = 0

    for ep_idx in tqdm(range(total_episodes), desc="Episodes"):
        ep = src_ds.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        ep_len = ep_end - ep_start

        # Determine task string for this episode
        tasks = ep["tasks"]
        task_str = tasks[0] if isinstance(tasks, list) else tasks

        # Number of full windows
        n_windows = ep_len // step

        if n_windows == 0:
            print(f"  Episode {ep_idx}: too short ({ep_len} frames), skipping")
            continue

        # Batch-load all tabular data for this episode at once
        ep_indices = list(range(ep_start, ep_start + n_windows * step))
        ep_data = {}
        for key in tabular_keys:
            col_data = src_ds.hf_dataset[key][ep_indices]
            if isinstance(col_data, list):
                ep_data[key] = torch.stack(col_data).numpy()
            else:
                ep_data[key] = col_data.numpy()

        # Decode video frames for kept indices
        video_frames = {}
        if video_keys:
            kept_timestamps = [i / src_fps for i in range(0, n_windows * step, step)]
            for vid_key in video_keys:
                from_ts = ep[f"videos/{vid_key}/from_timestamp"]
                shifted_ts = [from_ts + ts for ts in kept_timestamps]
                video_path = src_ds.root / src_ds.meta.get_video_file_path(ep_idx, vid_key)
                frames = decode_video_frames(
                    video_path, shifted_ts, src_ds.tolerance_s, src_ds.video_backend
                )
                video_frames[vid_key] = frames  # (n_windows, C, H, W)

        # Build and add downsampled frames
        for win_idx in range(n_windows):
            frame = {}

            # Observations: take value at first frame of window
            local_first = win_idx * step
            for key in obs_keys:
                frame[key] = ep_data[key][local_first]

            # Actions: aggregate over the window
            local_start = win_idx * step
            local_end = local_start + step
            window_actions = ep_data["action"][local_start:local_end]
            frame["action"] = aggregate_actions(window_actions, sum_mask)

            # Video frames: decode_video_frames returns CHW, add_frame expects HWC
            for vid_key in video_keys:
                frame[vid_key] = video_frames[vid_key][win_idx].permute(1, 2, 0)  # CHW -> HWC

            frame["task"] = task_str
            out_ds.add_frame(frame)

        out_ds.save_episode()
        total_out_frames += n_windows

    out_ds.finalize()
    print(f"\nDone! Output: {output_repo_id}")
    print(f"  Episodes: {out_ds.meta.total_episodes}")
    print(f"  Total frames: {out_ds.meta.total_frames}")
    print(f"  FPS: {out_ds.meta.fps}")


def main():
    parser = argparse.ArgumentParser(description="Downsample a LeRobot dataset to a lower FPS.")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Source dataset repository ID (e.g. domrachev03/franka_timing_belt_haply_static_v2)",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        required=True,
        help="Target FPS (must evenly divide source FPS)",
    )
    parser.add_argument(
        "--action-absolute-subfeatures",
        type=str,
        nargs="*",
        default=[],
        help="Action subfeature names that are absolute (take last value instead of summing). "
             "E.g. gripper.pos",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        default=None,
        help="Output dataset repository ID. Defaults to {repo_id}_{target_fps}hz",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory for the source dataset",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory for the output dataset",
    )
    args = parser.parse_args()

    downsample_dataset(
        repo_id=args.repo_id,
        target_fps=args.target_fps,
        action_absolute_subfeatures=args.action_absolute_subfeatures,
        output_repo_id=args.output_repo_id,
        root=args.root,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
