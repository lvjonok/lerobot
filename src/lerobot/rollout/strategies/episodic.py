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
      Space        — with ``intervention``: take the arm from the policy, or hand it back
      t            — the next attempt's task: the next of the policy's training tasks

- With ``intervention=true``, the teleoperator takes the arm mid-episode — its
  ``TeleopEvents.IS_INTERVENTION`` (a clutch press, say), or Space — and keeps it until Space
  hands it back: in between it drives exactly as when recording demonstrations (a clutch that
  is released holds the arm, the teleop's other controls — a gripper button — still act), and
  every frame goes into the same episode with ``intervention=True``. An evaluation that
  records the policy's own driving and every rescue of it, in one dataset.
- ``HardwareContext.episode_start``, when given, puts the robot where each episode begins.
- Each saved episode gets a row in ``meta/episode_outcomes.json``: its outcome, its task,
  how many times and frames the teleop took over, and what ``episode_start`` did.
- A policy trained on several tasks (``config.tasks``) is asked one per attempt, chosen
  between attempts; frames are recorded under the task the attempt was asked.
- ``control_port``: the same commands, and the state, over :mod:`lerobot.rollout.control`
  for ``lerobot-rollout-tui`` instead of the keyboard, and a "ready" phase before the first
  attempt, so the session starts when the operator says.

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
from ..control import ControlServer
from .core import RolloutStrategy, safe_push_to_hub, send_next_action

logger = logging.getLogger(__name__)

OUTCOMES = "episode_outcomes.json"
"""Per saved episode, in ``meta/``: outcome, interventions, and the start it was given."""


class _Phase(enum.Enum):
    POLICY = "policy"
    HUMAN = "human"  # the teleop drives, clutch engaged or not; recorded with intervention=True


_KEYS = {
    "right": "next",
    "n": "next",
    "left": "discard",
    "r": "discard",
    "esc": "quit",
    "q": "quit",
    "s": "success",
    "f": "failure",
    "space": "takeover",
    "t": "task",
}
"""Keyboard to command: Right/Left/Esc (n/r/q) as ``lerobot-record`` has them, and the rest."""


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
        self._control: ControlServer | None = None
        self._tasks: list[str] = []
        self._task: str | None = None
        self._next_task: str | None = None
        self._phase = "starting"
        self._attempt_t0: float | None = None
        self._counts = {"frames": 0, "intervention_frames": 0, "interventions": 0}
        self._episode_time_s = 0.0
        self._index = 0

    def setup(self, ctx: RolloutContext) -> None:
        """Start the inference engine and attach the keyboard listener."""
        if self.config.intervention and not hasattr(ctx.hardware.teleop, "get_teleop_events"):
            raise ValueError(
                "Episodic intervention needs a teleop with get_teleop_events(); "
                f"{type(ctx.hardware.teleop).__name__} has none"
            )
        cfg = ctx.runtime.cfg
        asked = cfg.dataset.single_task or cfg.task
        # A policy trained on several tasks lists them; any other is asked what it was given.
        self._tasks = list(getattr(ctx.policy.policy.config, "tasks", None) or [asked])
        if asked not in self._tasks:
            raise ValueError(f"the policy was trained on {self._tasks}; the task given is {asked!r}")
        self._task = self._next_task = asked
        self._episode_time_s = cfg.dataset.episode_time_s
        self._init_engine(ctx)
        self._events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "outcome": None,
            "toggle_policy": False,
        }
        if self.config.control_port is not None:
            self._control = ControlServer(self.config.control_port, self._command, self._state)
            self._control.start()
        else:
            self._listener = create_key_listener(
                lambda name: self._key(name.lower()),
                controls_help="Right/Left/Esc (n/r/q), s=success, f=failure, Space=take/hand back, t=next task",
            )
        self._outcomes_path = Path(ctx.data.dataset.root) / "meta" / OUTCOMES
        if self._outcomes_path.exists():
            self._outcomes = json.loads(self._outcomes_path.read_text())
        self._session_start = len(self._outcomes)
        self._index = ctx.data.dataset.num_episodes
        logger.info("Episodic strategy ready; tasks %s", self._tasks)

    def _key(self, name: str) -> None:
        if name in _KEYS:
            self._command(_KEYS[name], None)

    def _command(self, cmd: str, arg) -> None:
        """Apply one operator command (a key, or the control channel's).

        Raises:
            ValueError: A task the policy was not trained on.
        """
        events = self._events
        if cmd in ("next", "discard", "quit"):
            apply_recording_control({"next": "right", "discard": "left", "quit": "esc"}[cmd], events)
        elif cmd in ("success", "failure"):
            events["outcome"] = cmd
            logger.info("Outcome: %s", cmd)
        elif cmd == "takeover":
            events["toggle_policy"] = True
        elif cmd == "task":
            if arg is None:
                arg = self._tasks[(self._tasks.index(self._next_task) + 1) % len(self._tasks)]
            if arg not in self._tasks:
                raise ValueError(f"{arg!r} is not one of the policy's tasks {self._tasks}")
            self._next_task = arg
            logger.info("Next attempt's task: %s", arg)

    def _state(self) -> dict:
        """What the control channel reports."""
        session = self._outcomes[self._session_start :]
        return {
            "phase": self._phase,
            "episode": self._index,
            "elapsed_s": None if self._attempt_t0 is None else time.perf_counter() - self._attempt_t0,
            "episode_time_s": self._episode_time_s,
            "task": self._task,
            "next_task": self._next_task,
            "tasks": self._tasks,
            "outcome": self._events["outcome"] if self._events else None,
            "counts": dict(self._counts),
            "session": {
                "episodes": len(session),
                "success": sum(r["outcome"] == "success" for r in session),
                "failure": sum(r["outcome"] == "failure" for r in session),
                "by_task": {
                    t: [
                        sum(r["outcome"] == o for r in session if r.get("task") == t)
                        for o in ("success", "failure", None)
                    ]
                    for t in self._tasks
                },
            },
            "intervention": self.config.intervention,
        }

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
        play_sounds = cfg.play_sounds

        display_compressed = (
            True
            if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
            else cfg.display_compressed_images
        )

        with VideoEncodingManager(dataset):
            try:
                if self._control is not None:
                    # The session starts when the operator says: the reset loop, nothing recorded.
                    self._phase = "ready"
                    self._reset_loop(
                        ctx=ctx,
                        robot=robot,
                        teleop=teleop,
                        events=events,
                        fps=fps,
                        control_time_s=float("inf"),
                        display_data=cfg.display_data,
                        display_mode=cfg.display_mode,
                        display_compressed=display_compressed,
                    )
                recorded_episodes = 0
                while recorded_episodes < num_episodes and not events["stop_recording"]:
                    if ctx.runtime.shutdown_event.is_set():
                        break

                    self._task = self._next_task
                    self._engine.task = self._task
                    self._index = dataset.num_episodes
                    self._counts = {"frames": 0, "intervention_frames": 0, "interventions": 0}
                    start = None
                    if ctx.hardware.episode_start is not None:
                        self._phase = "walking"
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
                    logger.info("Attempt %d: %s", dataset.num_episodes, self._task)
                    self._attempt_t0 = time.perf_counter()
                    self._policy_loop(
                        ctx=ctx,
                        robot=robot,
                        events=events,
                        features=features,
                        fps=fps,
                        control_time_s=episode_time_s,
                        dataset=dataset,
                    )
                    self._attempt_t0 = None
                    counts = dict(self._counts)

                    # Reset phase, skip after the last episode (but run when re-recording)
                    if not events["stop_recording"] and (
                        recorded_episodes < num_episodes - 1 or events["rerecord_episode"]
                    ):
                        log_say("Reset the environment", play_sounds)
                        self._phase = "reset"

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

                    # Ended before a frame was recorded (a key at the start): save_episode()
                    # raises on an empty buffer.
                    if not dataset.has_pending_frames():
                        continue
                    self._phase = "saving"
                    index = dataset.num_episodes
                    dataset.save_episode()
                    recorded_episodes += 1
                    self._file_outcome(
                        {
                            "episode_index": index,
                            "outcome": events["outcome"],
                            "task": self._task,
                            **counts,
                            "start": start,
                        }
                    )
                    self._index = dataset.num_episodes
            finally:
                self._phase = "done"
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
    ) -> None:
        """Policy-driven recording loop for a single episode; with ``intervention``, the teleop's too.

        Counts ``frames``, ``intervention_frames`` and ``interventions`` (takeovers) into
        ``self._counts``, and keeps ``self._phase`` current.
        """
        interpolator = self._interpolator
        control_interval = interpolator.get_control_interval(fps)
        processors = ctx.processors
        teleop = ctx.hardware.teleop
        intervene = self.config.intervention

        phase = _Phase.POLICY
        self._phase = phase.value
        counts = self._counts

        def add(obs_processed: dict, action: dict, intervention: bool) -> None:
            frame = {
                **build_dataset_frame(features, obs_processed, prefix=OBS_STR),
                **build_dataset_frame(features, action, prefix=ACTION),
                "task": self._task,
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
                clutched = bool(teleop.get_teleop_events().get(TeleopEvents.IS_INTERVENTION, False))
                toggled = events["toggle_policy"]
                events["toggle_policy"] = False
                if phase == _Phase.POLICY and (clutched or toggled):
                    # Whatever the teleop pipeline last targeted, the policy has moved the arm
                    # since: start it over from where the arm is.
                    processors.teleop_action_processor.reset()
                    self._engine.pause()
                    phase = _Phase.HUMAN
                    counts["interventions"] += 1
                    logger.info(
                        "Intervention %d: the teleop has the arm (%s); Space hands it back",
                        counts["interventions"],
                        "clutch" if clutched else "Space",
                    )
                elif phase == _Phase.HUMAN and toggled:
                    if clutched:
                        # Handing back under a held clutch would take the arm again next tick.
                        logger.info("Release the clutch before handing the arm to the policy")
                    else:
                        # Predict afresh from where the arm is, not from before the takeover.
                        self._engine.reset()
                        interpolator.reset()
                        self._engine.resume()
                        phase = _Phase.POLICY
                        logger.info("Policy resumed")
                # Every tick, so the channels the teleop adds to the action have a value on the
                # policy's frames too (a clutch that is not engaged, say).
                teleop_action = processors.teleop_action_processor((teleop.get_action(), obs))
            self._phase = phase.value

            if phase == _Phase.HUMAN:
                # As a demonstration is recorded: every tick, engaged or not. The pipeline holds
                # the arm while the clutch is released and passes the teleop's other controls.
                obs_processed = processors.robot_observation_processor(obs)
                robot.send_action(processors.robot_action_processor((teleop_action, obs)))
                add(obs_processed, teleop_action, True)
                self._log_telemetry(obs_processed, teleop_action, ctx.runtime)
            else:
                obs_processed = self._process_observation_and_notify(processors, obs)

                if self._handle_warmup(ctx.runtime.cfg.use_torch_compile, loop_start, control_interval):
                    continue

                action_dict = send_next_action(obs_processed, obs, ctx, interpolator)

                if action_dict is not None:
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
        if self._control is not None:
            self._control.stop()

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
