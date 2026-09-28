# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Episodic rollout strategy: mirrors the behavior of ``lerobot-record``.

- Policy drives the robot during each recording episode.
- An optional teleoperator can drive the robot during reset phases so the
  operator can bring the environment back to its starting configuration.
  If no teleop is connected the robot stays in its current position.
- Keyboard controls:

      Right arrow  — end the current episode or reset phase early
      Left arrow   — discard the current episode and re-record it
      Escape       — stop the recording session
      s / f        — mark the episode a success / failure: ends it while it runs, and can be
                     pressed during the reset that follows too (the last press wins)
      Space        — with ``intervention``: hold the arm, or hand it back to the policy

- With ``intervention=true``, a teleoperator's ``TeleopEvents.IS_INTERVENTION`` takes the
  arm mid-episode (a clutch, say): its frames go into the same episode with
  ``intervention=True``. On release the arm holds until Space. An evaluation that records
  the policy's own driving and every rescue of it, in one dataset.
- ``HardwareContext.episode_start``, when given, puts the robot where each episode begins.
- Each saved episode gets a row in ``meta/episode_outcomes.json``: its outcome, how many
  times and frames the teleop took over, and what ``episode_start`` did.

Dataset naming follows the rollout convention: repo names must start with ``rollout_``.
"""

from __future__ import annotations

import contextlib
import enum
import json
import logging
import time
from pathlib import Path

import numpy as np

from lerobot.common.control_utils import (
    follower_smooth_move_to,
    teleop_smooth_move_to,
    teleop_supports_feedback,
)
from lerobot.datasets import VideoEncodingManager
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.keyboard_input import apply_recording_control, create_key_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import log_visualization_data

from ..configs import EpisodicStrategyConfig
from ..context import RolloutContext
from .core import RolloutStrategy, safe_push_to_hub, send_next_action

logger = logging.getLogger(__name__)

OUTCOMES = "episode_outcomes.json"
"""Per saved episode, in ``meta/``: outcome, interventions, and the start it was given."""


class _Phase(enum.Enum):
    POLICY = "policy"
    CORRECTING = "correcting"  # the teleop drives, recorded with intervention=True
    HELD = "held"  # nobody drives; the last command is repeated, nothing recorded


def _init_keyboard(events: dict):
    """Right/Left/Esc (n/r/q) as ``lerobot-record`` has them, plus s/f and Space."""

    def on_key(name: str) -> None:
        key = name.lower()
        if key in ("right", "n"):
            apply_recording_control("right", events)
        elif key in ("left", "r"):
            apply_recording_control("left", events)
        elif key in ("esc", "q"):
            apply_recording_control("esc", events)
        elif key in ("s", "f"):
            events["outcome"] = "success" if key == "s" else "failure"
            logger.info("Outcome: %s", events["outcome"])
        elif key == "space":
            events["toggle_policy"] = True

    return create_key_listener(
        on_key,
        controls_help="Right/Left/Esc (n/r/q), s=success, f=failure, Space=hold/resume policy",
    )


class EpisodicStrategy(RolloutStrategy):
    """Policy-driven multi-episode recording, mirrors the behavior of ``lerobot-record``.

    Each recording episode runs the policy for maximum ``dataset.episode_time_s``
    seconds, recording every frame.  A reset phase of ``dataset.reset_time_s``
    follows every episode (except the last) so the operator can manually
    reset the environment.  During the reset phase, an optional teleoperator
    drives the robot; if none is present the robot returns to its initial joint positions captured at startup.

    The policy state (hidden state, RTC queue, interpolator) is reset at
    the start of each recording episode.

    Keyboard events:
        right arrow  → end current episode or reset phase early
        left arrow   → discard & re-record current episode
        ESC          → stop the session
    """

    config: EpisodicStrategyConfig

    def __init__(self, config: EpisodicStrategyConfig) -> None:
        super().__init__(config)
        self._listener = None
        self._events: dict | None = None
        self._outcomes: list[dict] = []
        self._outcomes_path: Path | None = None
        self._session_start = 0

    def setup(self, ctx: RolloutContext) -> None:
        """Start the inference engine and attach the keyboard listener."""
        if self.config.intervention and not hasattr(ctx.hardware.teleop, "get_teleop_events"):
            raise ValueError(
                "Episodic intervention needs a teleop with get_teleop_events(); "
                f"{type(ctx.hardware.teleop).__name__} has none"
            )
        self._init_engine(ctx)
        self._events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "outcome": None,
            "toggle_policy": False,
        }
        self._listener = _init_keyboard(self._events)
        self._outcomes_path = Path(ctx.data.dataset.root) / "meta" / OUTCOMES
        if self._outcomes_path.exists():
            self._outcomes = json.loads(self._outcomes_path.read_text())
        self._session_start = len(self._outcomes)
        logger.info("Episodic strategy ready")

    def run(self, ctx: RolloutContext) -> None:
        """Main multi-episode recording loop."""
        cfg = ctx.runtime.cfg
        dataset_cfg = cfg.dataset
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        dataset = ctx.data.dataset
        events = self._events
        features = ctx.data.dataset_features

        fps = cfg.fps
        episode_time_s = dataset_cfg.episode_time_s
        reset_time_s = dataset_cfg.reset_time_s
        num_episodes = dataset_cfg.num_episodes
        single_task = dataset_cfg.single_task or cfg.task
        play_sounds = cfg.play_sounds

        display_compressed = (
            True
            if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
            else cfg.display_compressed_images
        )

        with VideoEncodingManager(dataset):
            try:
                recorded_episodes = 0
                while recorded_episodes < num_episodes and not events["stop_recording"]:
                    if ctx.runtime.shutdown_event.is_set():
                        break

                    start = None
                    if ctx.hardware.episode_start is not None:
                        start = ctx.hardware.episode_start(robot)
                        # The robot moved under something other than the teleop.
                        ctx.processors.teleop_action_processor.reset()
                    events["outcome"] = None
                    events["toggle_policy"] = False

                    # Reset policy state at episode start (discard leftover hidden state / queue)
                    self._engine.reset()
                    self._interpolator.reset()
                    self._engine.resume()

                    log_say(f"Recording episode {dataset.num_episodes}", play_sounds)
                    counts = self._policy_loop(
                        ctx=ctx,
                        robot=robot,
                        events=events,
                        features=features,
                        fps=fps,
                        control_time_s=episode_time_s,
                        dataset=dataset,
                        single_task=single_task,
                    )

                    # Reset phase, skip after the last episode (but run when re-recording)
                    if not events["stop_recording"] and (
                        recorded_episodes < num_episodes - 1 or events["rerecord_episode"]
                    ):
                        log_say("Reset the environment", play_sounds)

                        if teleop:
                            # Smooth handover so the transition to teleop control is jerk-free.
                            # For actuated teleops: drive the leader arm to the follower's current
                            # position so the operator takes over without fighting the arm.
                            # For non-actuated teleops: slide the follower to the teleop's current
                            # pose instead, since the leader cannot be driven.
                            # Disabled entirely with --strategy.smooth_handover=false (useful for
                            # clutch-style teleops that re-reference at the current robot pose on
                            # engage).
                            if self.config.smooth_handover:
                                obs = robot.get_observation()
                                current_pos = {k: v for k, v in obs.items() if k.endswith(".pos")}
                                if (
                                    teleop_supports_feedback(teleop)
                                    and self.config.smooth_leader_to_follower_handover
                                ):
                                    logger.info("Smooth handover: moving leader arm to follower position")
                                    teleop_smooth_move_to(teleop, current_pos, duration_s=2)
                                    teleop.disable_torque()
                                else:
                                    logger.info("Smooth handover: sliding follower to teleop position")
                                    teleop_action = teleop.get_action()
                                    processed = ctx.processors.teleop_action_processor((teleop_action, obs))
                                    target = ctx.processors.robot_action_processor((processed, obs))
                                    follower_smooth_move_to(robot, current_pos, target, duration_s=1)

                        elif self.config.reset_to_initial_position:
                            # No teleop: return the robot to its startup position.
                            self._return_to_initial_position(hw=ctx.hardware, duration_s=1)

                        self._reset_loop(
                            ctx=ctx,
                            robot=robot,
                            teleop=teleop,
                            events=events,
                            fps=fps,
                            control_time_s=reset_time_s,
                            display_data=cfg.display_data,
                            display_mode=cfg.display_mode,
                            display_compressed=display_compressed,
                        )

                    if events["rerecord_episode"]:
                        log_say("Re-record episode", play_sounds)
                        events["rerecord_episode"] = False
                        events["exit_early"] = False
                        events["outcome"] = None
                        dataset.clear_episode_buffer()

                        # returns to its initial joint positions captured at startup
                        if not teleop and self.config.reset_to_initial_position:
                            self._return_to_initial_position(hw=ctx.hardware, duration_s=1)

                        continue

                    # Ended before a frame was recorded (a key at the start, or held
                    # throughout): save_episode() raises on an empty buffer.
                    if not dataset.has_pending_frames():
                        continue
                    index = dataset.num_episodes
                    dataset.save_episode()
                    recorded_episodes += 1
                    self._file_outcome(
                        {"episode_index": index, "outcome": events["outcome"], **counts, "start": start}
                    )
            finally:
                # Save any frames buffered in the current episode so an unexpected
                # exception or KeyboardInterrupt does not silently drop recorded data.
                # suppress: save_episode raises if the buffer is empty (nothing to lose).
                logger.info("Episodic control loop ended — saving any in-progress episode")
                with contextlib.suppress(Exception):
                    dataset.save_episode()

    def _policy_loop(
        self,
        ctx: RolloutContext,
        robot,
        events: dict,
        features: dict,
        fps: float,
        control_time_s: float,
        dataset,
        single_task: str,
    ) -> dict[str, int]:
        """Policy-driven recording loop for a single episode; with ``intervention``, the teleop's too.

        Returns:
            ``frames``, ``intervention_frames`` and ``interventions`` (takeovers) recorded.
        """
        interpolator = self._interpolator
        control_interval = interpolator.get_control_interval(fps)
        processors = ctx.processors
        teleop = ctx.hardware.teleop
        intervene = self.config.intervention

        phase = _Phase.POLICY
        last_sent: dict | None = None
        counts = {"frames": 0, "intervention_frames": 0, "interventions": 0}

        def add(obs_processed: dict, action: dict, intervention: bool) -> None:
            frame = {
                **build_dataset_frame(features, obs_processed, prefix=OBS_STR),
                **build_dataset_frame(features, action, prefix=ACTION),
                "task": single_task,
            }
            if intervene:
                frame["intervention"] = np.array([intervention], dtype=bool)
            dataset.add_frame(frame)
            counts["frames"] += 1
            counts["intervention_frames"] += intervention

        timestamp = 0.0
        start_t = time.perf_counter()

        while timestamp < control_time_s:
            loop_start = time.perf_counter()

            if events["exit_early"]:
                events["exit_early"] = False
                break
            if events["outcome"] is not None:
                break

            if ctx.runtime.shutdown_event.is_set():
                break

            obs = robot.get_observation()

            teleop_action = None
            if intervene:
                active = bool(teleop.get_teleop_events().get(TeleopEvents.IS_INTERVENTION, False))
                if active and phase != _Phase.CORRECTING:
                    # Whatever the teleop pipeline last targeted, the policy has moved the arm
                    # since: start it over from where the arm is.
                    processors.teleop_action_processor.reset()
                    self._engine.pause()
                    phase = _Phase.CORRECTING
                    counts["interventions"] += 1
                    logger.info("Intervention %d: the teleop has the arm", counts["interventions"])
                elif not active and phase == _Phase.CORRECTING:
                    phase = _Phase.HELD
                    logger.info("Intervention released: holding; Space hands the arm to the policy")
                if events["toggle_policy"]:
                    events["toggle_policy"] = False
                    if phase == _Phase.HELD:
                        # Predict afresh from where the arm is, not from before the takeover.
                        self._engine.reset()
                        interpolator.reset()
                        self._engine.resume()
                        phase = _Phase.POLICY
                        logger.info("Policy resumed")
                    elif phase == _Phase.POLICY:
                        self._engine.pause()
                        phase = _Phase.HELD
                        logger.info("Policy held; Space resumes it")
                # Every tick, so the channels the teleop adds to the action have a value on the
                # policy's frames too (a clutch that is not engaged, say).
                teleop_action = processors.teleop_action_processor((teleop.get_action(), obs))

            if phase == _Phase.CORRECTING:
                obs_processed = processors.robot_observation_processor(obs)
                last_sent = processors.robot_action_processor((teleop_action, obs))
                robot.send_action(last_sent)
                add(obs_processed, teleop_action, True)
                self._log_telemetry(obs_processed, teleop_action, ctx.runtime)
            elif phase == _Phase.HELD:
                if last_sent is not None:
                    robot.send_action(last_sent)
            else:
                obs_processed = self._process_observation_and_notify(processors, obs)

                if self._handle_warmup(ctx.runtime.cfg.use_torch_compile, loop_start, control_interval):
                    continue

                action_dict = send_next_action(obs_processed, obs, ctx, interpolator)

                if action_dict is not None:
                    last_sent = processors.robot_action_processor((action_dict, obs))
                    recorded = action_dict if teleop_action is None else {**teleop_action, **action_dict}
                    add(obs_processed, recorded, False)
                    self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt = time.perf_counter() - loop_start
            sleep_t = control_interval - dt
            if sleep_t < 0:
                logger.warning(
                    f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({fps} Hz). "
                    "Dataset frames might be dropped and robot control might be unstable. "
                    "Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long "
                    "3) CPU starvation"
                )
            precise_sleep(max(sleep_t, 0.0))
            timestamp = time.perf_counter() - start_t

        self._engine.pause()
        return counts

    def _reset_loop(
        self,
        ctx: RolloutContext,
        robot,
        teleop,
        events: dict,
        fps: float,
        control_time_s: float,
        display_data: bool,
        display_mode: str,
        display_compressed: bool,
    ) -> None:
        """Reset-phase loop: teleop drives the robot if available, no recording."""
        processors = ctx.processors
        control_interval = 1.0 / fps
        # The policy moved the robot since the teleop pipeline last ran.
        processors.teleop_action_processor.reset()

        timestamp = 0.0
        start_t = time.perf_counter()

        while timestamp < control_time_s:
            loop_start = time.perf_counter()

            if events["exit_early"]:
                events["exit_early"] = False
                break

            if ctx.runtime.shutdown_event.is_set():
                break

            obs = robot.get_observation()

            if teleop is not None:
                act = teleop.get_action()
                act_teleop = processors.teleop_action_processor((act, obs))
                robot_action = processors.robot_action_processor((act_teleop, obs))
                robot.send_action(robot_action)

                if display_data:
                    obs_processed = processors.robot_observation_processor(obs)
                    log_visualization_data(
                        display_mode,
                        observation=obs_processed,
                        action=act_teleop,
                        compress_images=display_compressed,
                    )

            dt = time.perf_counter() - loop_start
            sleep_t = control_interval - dt
            precise_sleep(max(sleep_t, 0.0))
            timestamp = time.perf_counter() - start_t

    def _file_outcome(self, row: dict) -> None:
        """Append one saved episode's row to ``meta/episode_outcomes.json``."""
        self._outcomes.append(row)
        self._outcomes_path.write_text(json.dumps(self._outcomes, indent=2))
        logger.info(
            "Episode %d saved: %s, %d frames, %d intervention(s)",
            row["episode_index"],
            row["outcome"] or "no outcome marked",
            row["frames"],
            row["interventions"],
        )

    def teardown(self, ctx: RolloutContext) -> None:
        """Finalise dataset, stop listener, push to hub, and disconnect hardware."""
        cfg = ctx.runtime.cfg
        play_sounds = cfg.play_sounds

        log_say("Stop recording", play_sounds, blocking=True)

        session = self._outcomes[self._session_start :]
        if session:
            marked = [r for r in session if r["outcome"] is not None]
            wins = sum(r["outcome"] == "success" for r in marked)
            logger.info(
                "This session: %d episode(s), %d/%d marked a success, %d with an intervention",
                len(session),
                wins,
                len(marked),
                sum(r["interventions"] > 0 for r in session),
            )

        if self._listener is not None:
            self._listener.stop()

        if ctx.data.dataset is not None:
            logger.info("Finalizing dataset...")
            ctx.data.dataset.finalize()

        if (
            cfg.dataset is not None
            and cfg.dataset.push_to_hub
            and ctx.data.dataset is not None
            and safe_push_to_hub(
                ctx.data.dataset,
                tags=cfg.dataset.tags,
                private=cfg.dataset.private,
            )
        ):
            logger.info("Dataset uploaded to hub")
            log_say("Dataset uploaded to hub", play_sounds)

        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=cfg.return_to_initial_position,
        )
        log_say("Exiting", play_sounds)
        logger.info("Episodic strategy teardown complete")
