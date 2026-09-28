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

"""Minimal tests for the rollout module's public API."""

from __future__ import annotations

import contextlib
import dataclasses
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


def test_rollout_top_level_imports():
    import lerobot.rollout

    for name in lerobot.rollout.__all__:
        assert hasattr(lerobot.rollout, name), f"Missing export: {name}"


def test_inference_submodule_imports():
    import lerobot.rollout.inference

    for name in lerobot.rollout.inference.__all__:
        assert hasattr(lerobot.rollout.inference, name), f"Missing export: {name}"


def test_strategies_submodule_imports():
    import lerobot.rollout.strategies

    for name in lerobot.rollout.strategies.__all__:
        assert hasattr(lerobot.rollout.strategies, name), f"Missing export: {name}"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_strategy_config_types():
    from lerobot.rollout import (
        BaseStrategyConfig,
        DAggerStrategyConfig,
        EpisodicStrategyConfig,
        HighlightStrategyConfig,
        SentryStrategyConfig,
    )

    assert BaseStrategyConfig().type == "base"
    assert SentryStrategyConfig().type == "sentry"
    assert HighlightStrategyConfig().type == "highlight"
    assert DAggerStrategyConfig().type == "dagger"
    assert EpisodicStrategyConfig().type == "episodic"


def test_dagger_config_invalid_input_device():
    from lerobot.rollout import DAggerStrategyConfig

    with pytest.raises(ValueError, match="input_device must be 'keyboard', 'pedal' or 'teleop'"):
        DAggerStrategyConfig(input_device="joystick")


def test_dagger_config_defaults():
    from lerobot.rollout import DAggerStrategyConfig

    cfg = DAggerStrategyConfig()
    assert cfg.num_episodes is None
    assert cfg.record_autonomous is False
    assert cfg.input_device == "keyboard"


def test_inference_config_types():
    from lerobot.rollout import RTCInferenceConfig, SyncInferenceConfig

    assert SyncInferenceConfig().type == "sync"

    rtc = RTCInferenceConfig()
    assert rtc.type == "rtc"
    assert rtc.queue_threshold == 30
    assert rtc.rtc is not None


def test_sentry_config_defaults():
    from lerobot.rollout import SentryStrategyConfig

    cfg = SentryStrategyConfig()
    assert cfg.upload_every_n_episodes == 5
    assert cfg.target_video_file_size_mb is None


def test_rollout_config_passes_policy_pretrained_revision(monkeypatch):
    from lerobot.configs import PreTrainedConfig, parser
    from lerobot.rollout import RolloutConfig
    from tests.mocks.mock_robot import MockRobotConfig

    captured = {}

    def fake_from_pretrained(cls, pretrained_name_or_path, **kwargs):
        captured["pretrained_name_or_path"] = pretrained_name_or_path
        captured.update(kwargs)
        return SimpleNamespace(device="cpu", pretrained_revision=kwargs["revision"])

    monkeypatch.setattr(parser, "get_yaml_overrides", lambda _: ["--pretrained_revision=yaml-sha"])
    monkeypatch.setattr(
        sys,
        "argv",
        ["lerobot-rollout", "--policy.path=user/policy", "--policy.pretrained_revision=cli-sha"],
    )
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", classmethod(fake_from_pretrained))

    cfg = RolloutConfig(robot=MockRobotConfig())

    assert captured["pretrained_name_or_path"] == "user/policy"
    assert captured["revision"] == "cli-sha"
    assert captured["cli_overrides"] == [
        "--pretrained_revision=yaml-sha",
        "--pretrained_revision=cli-sha",
    ]
    assert cfg.policy.pretrained_path == "user/policy"
    assert cfg.policy.pretrained_revision == "cli-sha"


def test_load_pretrained_policy_passes_revision(monkeypatch):
    import lerobot.rollout.context as rollout_context

    policy_config = SimpleNamespace(
        type="mock",
        use_peft=False,
        pretrained_path="user/policy",
        pretrained_revision="policy-sha",
    )
    policy_class = MagicMock()
    loaded_policy = MagicMock()
    policy_class.from_pretrained.return_value = loaded_policy
    monkeypatch.setattr(rollout_context, "get_policy_class", lambda _: policy_class)

    policy = rollout_context._load_pretrained_policy(policy_config)

    assert policy is loaded_policy
    policy_class.from_pretrained.assert_called_once_with(
        "user/policy",
        config=policy_config,
        revision="policy-sha",
    )


def test_load_pretrained_peft_policy_keeps_adapter_and_base_revisions_separate(monkeypatch):
    import lerobot.rollout.context as rollout_context

    policy_config = SimpleNamespace(
        type="mock",
        use_peft=True,
        pretrained_path="user/adapter",
        pretrained_revision="adapter-sha",
    )
    policy_class = MagicMock()
    base_policy = MagicMock()
    policy_class.from_pretrained.return_value = base_policy
    monkeypatch.setattr(rollout_context, "get_policy_class", lambda _: policy_class)

    peft_config = SimpleNamespace(
        base_model_name_or_path="user/base-policy",
        revision="base-sha",
    )
    peft_config_from_pretrained = MagicMock(return_value=peft_config)
    adapted_policy = MagicMock()
    peft_model_from_pretrained = MagicMock(return_value=adapted_policy)
    require_package = MagicMock()
    monkeypatch.setattr(rollout_context, "require_package", require_package)
    monkeypatch.setattr(
        rollout_context,
        "PeftConfig",
        SimpleNamespace(from_pretrained=peft_config_from_pretrained),
        raising=False,
    )
    monkeypatch.setattr(
        rollout_context,
        "PeftModel",
        SimpleNamespace(from_pretrained=peft_model_from_pretrained),
        raising=False,
    )

    policy = rollout_context._load_pretrained_policy(policy_config)

    assert policy is adapted_policy
    require_package.assert_called_once_with("peft", extra="peft")
    peft_config_from_pretrained.assert_called_once_with("user/adapter", revision="adapter-sha")
    policy_class.from_pretrained.assert_called_once_with(
        pretrained_name_or_path="user/base-policy",
        config=policy_config,
        revision="base-sha",
    )
    peft_model_from_pretrained.assert_called_once_with(
        base_policy,
        "user/adapter",
        config=peft_config,
        revision="adapter-sha",
    )


# ---------------------------------------------------------------------------
# RolloutRingBuffer
# ---------------------------------------------------------------------------


def test_ring_buffer_append_and_eviction():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=0.5, max_memory_mb=100.0, fps=10.0)
    # max_frames = 5
    for i in range(8):
        buf.append({"val": i})
    assert len(buf) == 5


def test_ring_buffer_drain():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    for i in range(3):
        buf.append({"val": i})
    frames = buf.drain()
    assert len(frames) == 3
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_clear():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    buf.append({"val": 1})
    buf.clear()
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_tensor_bytes():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    t = torch.zeros(100, dtype=torch.float32)  # 400 bytes
    buf.append({"tensor": t})
    assert buf.estimated_bytes >= 400


# ---------------------------------------------------------------------------
# ThreadSafeRobot
# ---------------------------------------------------------------------------


def test_thread_safe_robot_delegates():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    obs = wrapper.get_observation()
    assert "motor_1.pos" in obs
    assert "motor_2.pos" in obs
    assert "motor_3.pos" in obs

    action = {"motor_1.pos": 0.0, "motor_2.pos": 1.0, "motor_3.pos": 2.0}
    result = wrapper.send_action(action)
    assert result == action

    robot.disconnect()


def test_thread_safe_robot_properties():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    assert wrapper.name == "mock_robot"
    assert "motor_1.pos" in wrapper.observation_features
    assert "motor_1.pos" in wrapper.action_features
    assert wrapper.is_connected is True
    assert wrapper.inner is robot

    robot.disconnect()


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def test_create_strategy_dispatches():
    from lerobot.rollout import (
        BaseStrategy,
        BaseStrategyConfig,
        DAggerStrategy,
        DAggerStrategyConfig,
        EpisodicStrategy,
        EpisodicStrategyConfig,
        SentryStrategy,
        SentryStrategyConfig,
        create_strategy,
    )

    assert isinstance(create_strategy(BaseStrategyConfig()), BaseStrategy)
    assert isinstance(create_strategy(SentryStrategyConfig()), SentryStrategy)
    assert isinstance(create_strategy(DAggerStrategyConfig()), DAggerStrategy)
    assert isinstance(create_strategy(EpisodicStrategyConfig()), EpisodicStrategy)


def test_create_strategy_unknown_raises():
    from lerobot.rollout import create_strategy

    cfg = MagicMock()
    cfg.type = "bogus"
    with pytest.raises(ValueError, match="Unknown strategy type"):
        create_strategy(cfg)


# ---------------------------------------------------------------------------
# Inference factory
# ---------------------------------------------------------------------------


def test_create_inference_engine_sync():
    from lerobot.rollout import SyncInferenceConfig, SyncInferenceEngine, create_inference_engine

    engine = create_inference_engine(
        SyncInferenceConfig(),
        policy=MagicMock(),
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=MagicMock(robot_type="mock"),
        hw_features={},
        dataset_features={"action": {"names": ["k"]}},
        ordered_action_keys=["k"],
        task="test",
        fps=30.0,
        device="cpu",
    )
    assert isinstance(engine, SyncInferenceEngine)


class _ChunkPolicy:
    """Predicts ``±[1, 2, 3, ...] * 0.1`` as a chunk of relative steps, counting its calls."""

    def __init__(self, chunk_size=5, n_action_steps=3, **config):
        self.config = SimpleNamespace(
            n_action_steps=n_action_steps, use_amp=False, temporal_ensemble_coeff=None, **config
        )
        self.chunk_size = chunk_size
        self.calls = 0

    def reset(self):
        pass

    def predict_action_chunk(self, batch):
        self.calls += 1
        steps = torch.arange(1, self.chunk_size + 1, dtype=torch.float32) * 0.1
        return torch.stack([steps, -steps], dim=-1).unsqueeze(0)


class _AnchorPre:
    """Stands in for a preprocessor holding an enabled relative step: caches the state."""

    def __init__(self):
        from lerobot.processor import RelativeActionsProcessorStep

        self.steps = [RelativeActionsProcessorStep(enabled=True)]
        self.anchor = None

    def __call__(self, observation):
        self.anchor = observation["observation.state"].clone()
        return observation

    def reset(self):
        pass


class _AnchorPost:
    """Composes a relative chunk onto the anchor the preprocessor cached."""

    def __init__(self, pre):
        self.pre = pre

    def __call__(self, action):
        return action + self.pre.anchor.view(1, 1, -1)

    def reset(self):
        pass


def _relative_engine(policy, action_names=("x", "y"), ordered_action_keys=("x", "y")):
    from lerobot.rollout import SyncInferenceEngine

    pre = _AnchorPre()
    return SyncInferenceEngine(
        policy=policy,
        preprocessor=pre,
        postprocessor=_AnchorPost(pre),
        dataset_features={"action": {"names": list(action_names)}},
        ordered_action_keys=list(ordered_action_keys),
        task="test",
        device="cpu",
        robot_type="mock",
    )


def test_sync_relative_policy_runs_whole_chunks_on_one_anchor():
    """A chunk's actions are composed onto the state it was predicted from, not a later one."""
    import numpy as np

    policy = _ChunkPolicy(chunk_size=5, n_action_steps=3)
    engine = _relative_engine(policy)
    served = []
    for tick in range(6):
        # The arm moves between ticks; only the observation a chunk starts from anchors it.
        state = np.array([10.0 * tick, 0.0], dtype=np.float32)
        served.append(engine.get_action({"observation.state": state}).tolist())

    assert policy.calls == 2, "one prediction per n_action_steps"
    expected = [[0.1, -0.1], [0.2, -0.2], [0.3, -0.3], [30.1, -0.1], [30.2, -0.2], [30.3, -0.3]]
    assert served == [pytest.approx(row) for row in expected]

    engine.reset()
    engine.get_action({"observation.state": np.array([50.0, 0.0], dtype=np.float32)})
    assert policy.calls == 3, "reset drops the rest of the chunk"


def test_sync_engine_names_only_the_robot_actions_a_teleop_pipeline_adds_to():
    """A teleop pipeline's own action features (a clutch state) are not the policy's output."""
    import numpy as np

    engine = _relative_engine(
        _ChunkPolicy(),
        action_names=("x", "y", "teleop.engaged", "teleop.engage_id"),
        ordered_action_keys=("y", "x"),
    )
    action = engine.get_action({"observation.state": np.zeros(2, dtype=np.float32)})
    assert action.tolist() == pytest.approx([-0.1, 0.1]), "named in dataset order, then reordered"


def test_sync_relative_policy_refuses_temporal_ensembling_and_history_queues():
    ensembling = _ChunkPolicy()
    ensembling.config.temporal_ensemble_coeff = 0.01
    with pytest.raises(ValueError, match="temporal ensembling"):
        _relative_engine(ensembling)

    queued = _ChunkPolicy()
    queued._queues = {}
    with pytest.raises(NotImplementedError, match="observation-history"):
        _relative_engine(queued)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def test_estimate_max_episode_seconds_no_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    assert estimate_max_episode_seconds({}, fps=30.0) == 300.0


def test_estimate_max_episode_seconds_with_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    features = {"cam": {"dtype": "video", "shape": (480, 640, 3)}}
    result = estimate_max_episode_seconds(features, fps=30.0)
    assert result > 0
    # With a real camera, duration should differ from the fallback
    assert result != 300.0


def test_safe_push_to_hub():
    from lerobot.rollout.strategies import safe_push_to_hub

    ds = MagicMock()
    ds.num_episodes = 0
    assert safe_push_to_hub(ds) is False
    ds.push_to_hub.assert_not_called()

    ds.num_episodes = 5
    assert safe_push_to_hub(ds, tags=["test"]) is True
    ds.push_to_hub.assert_called_once_with(tags=["test"], private=False)


# ---------------------------------------------------------------------------
# DAgger state machine
# ---------------------------------------------------------------------------


def test_dagger_full_transition_cycle():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    assert events.phase == DAggerPhase.AUTONOMOUS

    # AUTONOMOUS -> PAUSED
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED)

    # PAUSED -> CORRECTING
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.CORRECTING)

    # CORRECTING -> PAUSED
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.CORRECTING, DAggerPhase.PAUSED)

    # PAUSED -> AUTONOMOUS
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.AUTONOMOUS)


def test_dagger_teleop_intervention_takes_over_from_any_phase_and_release_holds():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    auto, paused, corr = DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED, DAggerPhase.CORRECTING
    events = DAggerEvents()
    assert events.follow_intervention(False) == []
    # Pressed while the policy drives: through PAUSED, so both steps' side effects run.
    assert events.follow_intervention(True) == [(auto, paused), (paused, corr)]
    assert events.follow_intervention(True) == [], "held: stays correcting"
    # Released: the correction ends and the robot holds; the policy does not resume.
    assert events.follow_intervention(False) == [(corr, paused)]
    assert events.phase == paused
    assert events.follow_intervention(False) == []
    # Pressed again from the hold.
    assert events.follow_intervention(True) == [(paused, corr)]


class _Scripted:
    """A robot whose position is ``x``, and the keys, teleop and ticks that script an episodic run."""

    def __init__(self, events_at, engaged_at):
        self.x, self.tick = 0.0, -1
        self.events_at, self.engaged_at = events_at, engaged_at
        self.strategy = None

    # robot
    def get_observation(self):
        self.tick += 1
        for key, value in self.events_at.get(self.tick, {}).items():
            self.strategy._events[key] = value
        return {"x": self.x}

    def send_action(self, action):
        self.x = float(action["x"])

    # teleop: drives to x=50 while engaged
    def get_teleop_events(self):
        from lerobot.teleoperators.utils import TeleopEvents

        return {TeleopEvents.IS_INTERVENTION: self.tick in self.engaged_at}

    def get_action(self):
        return {"x": 50.0, "teleop.engaged": float(self.tick in self.engaged_at)}


class _Pipeline:
    def __init__(self):
        self.resets = 0

    def __call__(self, transition):
        return dict(transition[0])

    def reset(self):
        self.resets += 1


class _Engine:
    ready = True

    def get_action(self, obs_frame):
        return torch.tensor([float(obs_frame["observation.state"][0]) + 1.0])

    def start(self):
        pass

    def reset(self):
        pass

    def pause(self):
        pass

    def resume(self):
        pass

    def notify_observation(self, obs):
        pass


class _Dataset:
    def __init__(self, root):
        self.root, self.frames, self.episodes = root, [], []

    @property
    def num_episodes(self):
        return len(self.episodes)

    def add_frame(self, frame):
        self.frames.append(frame)

    def has_pending_frames(self):
        return bool(self.frames)

    def save_episode(self):
        if not self.frames:
            raise RuntimeError("save_episode() on an empty buffer")
        self.episodes.append(self.frames)
        self.frames = []

    def clear_episode_buffer(self):
        self.frames = []


def _episodic_ctx(tmp_path, robot, episode_start, *, num_episodes, tasks=None, single_task="t"):
    """A rollout context around a :class:`_Scripted` robot, and its teleop pipeline."""
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": ["x", "teleop.engaged"]},
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["x"]},
    }
    teleop_pipe = _Pipeline()
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            episode_time_s=30, reset_time_s=30, num_episodes=num_episodes, single_task=single_task
        ),
        fps=200,
        task=None,
        play_sounds=False,
        display_data=False,
        display_ip=None,
        display_port=None,
        display_compressed_images=False,
        display_mode=None,
        use_torch_compile=False,
        interpolation_multiplier=1,
    )
    (tmp_path / "meta").mkdir()
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(cfg=cfg, shutdown_event=SimpleNamespace(is_set=lambda: False)),
        hardware=SimpleNamespace(robot_wrapper=robot, teleop=robot, episode_start=episode_start),
        policy=SimpleNamespace(
            inference=_Engine(), policy=SimpleNamespace(config=SimpleNamespace(tasks=tasks))
        ),
        processors=SimpleNamespace(
            teleop_action_processor=teleop_pipe,
            robot_action_processor=lambda t: dict(t[0]),
            robot_observation_processor=lambda o: o,
        ),
        data=SimpleNamespace(
            dataset=_Dataset(tmp_path), dataset_features=features, ordered_action_keys=["x"]
        ),
    )
    return ctx, teleop_pipe


def test_episodic_intervention_records_policy_and_takeover_with_outcomes(tmp_path, monkeypatch):
    import json

    from lerobot.rollout import EpisodicStrategyConfig
    from lerobot.rollout.strategies import EpisodicStrategy, episodic

    monkeypatch.setattr(episodic, "create_key_listener", lambda *a, **k: None)
    monkeypatch.setattr(episodic, "VideoEncodingManager", lambda dataset: contextlib.nullcontext())

    # Episode 0: the policy (t0-2), a takeover (t3-4), held after release (t5), Space (t6)
    # hands back, `s` (t7) ends it; the reset (t8) ends on the next-episode key. Episode 1:
    # `f` on its first tick. The last episode has no reset.
    robot = _Scripted(
        events_at={
            6: {"toggle_policy": True},
            7: {"outcome": "success"},
            8: {"exit_early": True},
            9: {"outcome": "failure"},
        },
        engaged_at={3, 4},
    )
    starts = []

    def episode_start(r):
        r.x = 100.0
        starts.append(r.tick)
        return {"at": 100.0}

    ctx, teleop_pipe = _episodic_ctx(tmp_path, robot, episode_start, num_episodes=2)
    dataset = ctx.data.dataset
    strategy = EpisodicStrategy(EpisodicStrategyConfig(intervention=True, smooth_handover=False))
    robot.strategy = strategy
    strategy.setup(ctx)
    strategy.run(ctx)

    first, second = dataset.episodes
    assert [f["action"].tolist() for f in first] == [
        [101, 0],
        [102, 0],
        [103, 0],  # the policy
        [50, 1],
        [50, 1],  # the takeover, as the teleop pipeline made it
        [51, 0],
        [52, 0],  # after Space: predicted from where the teleop left the arm
    ]
    assert [bool(f["intervention"][0]) for f in first] == [False] * 3 + [True] * 2 + [False] * 2
    assert [f["action"].tolist() for f in second] == [[101, 0]]
    assert starts == [-1, 8], "a start before each episode, after the reset"
    # Episode start, the takeover, the reset, the second start: each after something else
    # moved the robot.
    assert teleop_pipe.resets == 4
    rows = json.loads((tmp_path / "meta" / episodic.OUTCOMES).read_text())
    common = {"task": "t", "start": {"at": 100.0}}
    assert rows == [
        {
            "episode_index": 0,
            "outcome": "success",
            "frames": 7,
            "intervention_frames": 2,
            "interventions": 1,
            **common,
        },
        {
            "episode_index": 1,
            "outcome": "failure",
            "frames": 1,
            "intervention_frames": 0,
            "interventions": 0,
            **common,
        },
    ]


def test_episodic_control_channel_starts_the_session_and_picks_the_task(tmp_path, monkeypatch):
    from lerobot.rollout import EpisodicStrategyConfig
    from lerobot.rollout.control import request
    from lerobot.rollout.strategies import EpisodicStrategy, episodic

    monkeypatch.setattr(episodic, "VideoEncodingManager", lambda dataset: contextlib.nullcontext())
    replies = {}

    class Operator(_Scripted):
        """Commands over the socket, as lerobot-rollout-tui sends them, at given ticks."""

        def get_observation(self):
            obs = super().get_observation()
            port = self.strategy._control._server.server_address[1]
            for cmd, arg in self.commands.get(self.tick, []):
                replies[(self.tick, cmd)] = request(port, cmd, arg)
            return obs

    # t0: ready, nothing recorded; the task is chosen, a typo refused, then "next" starts.
    # t1-3: the attempt, marked a success over the channel at t3.
    robot = Operator(events_at={}, engaged_at=set())
    robot.commands = {
        0: [("task", "remove"), ("task", "remvoe"), ("next", None)],
        3: [("success", None)],
    }
    ctx, _ = _episodic_ctx(
        tmp_path, robot, None, num_episodes=1, tasks=["insert", "remove"], single_task="insert"
    )
    strategy = EpisodicStrategy(
        EpisodicStrategyConfig(intervention=True, smooth_handover=False, control_port=0)
    )
    robot.strategy = strategy
    strategy.setup(ctx)
    assert strategy._state()["phase"] == "starting"
    strategy.run(ctx)
    strategy._control.stop()

    assert replies[(0, "task")]["ok"] is False, "the last task reply is the refused typo"
    assert "not one of the policy's tasks" in replies[(0, "task")]["error"]
    assert replies[(0, "next")]["state"]["phase"] == "ready"
    assert replies[(0, "next")]["state"]["next_task"] == "remove"
    (episode,) = ctx.data.dataset.episodes
    assert {f["task"] for f in episode} == {"remove"}, "recorded under the task it was asked"
    assert ctx.policy.inference.task == "remove", "and the policy was asked it"
    # The ready phase handed the robot to the teleop, which put it at 50.
    assert [f["action"].tolist() for f in episode] == [[51, 0], [52, 0], [53, 0]]
    state = strategy._state()
    assert state["session"]["by_task"] == {"insert": [0, 0, 0], "remove": [1, 0, 0]}


def test_episodic_refuses_a_task_the_policy_was_not_trained_on(tmp_path):
    from lerobot.rollout import EpisodicStrategyConfig
    from lerobot.rollout.strategies import EpisodicStrategy

    ctx, _ = _episodic_ctx(
        tmp_path, _Scripted({}, set()), None, num_episodes=1, tasks=["insert"], single_task="pick"
    )
    with pytest.raises(ValueError, match="trained on"):
        EpisodicStrategy(EpisodicStrategyConfig()).setup(ctx)


def test_dagger_takeover_restarts_the_teleop_pipeline_where_the_robot_is():
    from lerobot.rollout import DAggerStrategyConfig
    from lerobot.rollout.strategies import DAggerPhase, DAggerStrategy

    pipe = _Pipeline()
    ctx = SimpleNamespace(
        hardware=SimpleNamespace(teleop=SimpleNamespace(feedback_features={}), robot_wrapper=None),
        processors=SimpleNamespace(teleop_action_processor=pipe),
    )
    strategy = DAggerStrategy(DAggerStrategyConfig(input_device="teleop", smooth_handover=False))
    strategy._apply_transition(DAggerPhase.PAUSED, DAggerPhase.CORRECTING, None, None, ctx, None)
    assert pipe.resets == 1


def test_episodic_intervention_requires_a_teleop():
    from lerobot.rollout import EpisodicStrategyConfig, RolloutConfig

    with pytest.raises(ValueError, match="intervention=true requires --teleop"):
        RolloutConfig.__post_init__(
            SimpleNamespace(strategy=EpisodicStrategyConfig(intervention=True), teleop=None)
        )


def test_dagger_config_accepts_teleop_input():
    from lerobot.rollout import DAggerStrategyConfig

    assert DAggerStrategyConfig(input_device="teleop").input_device == "teleop"


def test_dagger_invalid_transition_ignored():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("correction")  # Not valid from AUTONOMOUS
    assert events.consume_transition() is None
    assert events.phase == DAggerPhase.AUTONOMOUS


def test_dagger_events_reset():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("pause_resume")
    events.consume_transition()  # -> PAUSED
    events.upload_requested.set()
    events.reset()
    assert events.phase == DAggerPhase.AUTONOMOUS
    assert not events.upload_requested.is_set()


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


def test_rollout_context_fields():
    from lerobot.rollout import RolloutContext

    field_names = {f.name for f in dataclasses.fields(RolloutContext)}
    assert field_names == {"runtime", "hardware", "policy", "processors", "data"}
