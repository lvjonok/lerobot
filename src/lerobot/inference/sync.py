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

"""Synchronous inference engine: inline policy call per control tick."""

from __future__ import annotations

import logging
import time
from collections import deque
from contextlib import nullcontext
from copy import copy

import torch

from lerobot.lerobot_types import EnvTransition
from lerobot.policies import PreTrainedPolicy, prepare_observation_for_inference
from lerobot.processor import (
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    create_transition,
    policy_output_to_transition,
    transition_to_policy_action,
    transition_to_prediction,
)
from lerobot.utils import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

from .base import InferenceEngine, PolicyQuery

logger = logging.getLogger(__name__)


class SyncInferenceEngine(InferenceEngine):
    """Inline synchronous inference: compute one action per call.

    ``get_action`` runs the full policy pipeline (pre/post-processor +
    ``select_action``) on the given observation frame and returns a CPU action tensor
    in the policy's own dimension order.

    A relative-action policy that conditions on one observation runs whole chunks instead
    (:meth:`_predict_chunk`): one observation is preprocessed, ``predict_action_chunk`` gives
    the chunk, the postprocessor makes its first ``n_action_steps`` absolute at once against
    that observation's anchor, and they are served from a FIFO. The anchor then never has to
    outlive a call, so a relative step that re-caches on every observation is right too, and a
    chunk is one call for a caller that plans chunks itself. A policy with an observation
    history or ACT's temporal ensembling (both live in ``select_action``) keeps the per-tick
    path, where ``bind_relative_anchor`` holds the anchor while the policy's queue drains.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
    ) -> None:
        """Own the local policy and processors used for synchronous calls."""
        super().__init__(task=task)
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        self._relative = (
            any(
                isinstance(step, RelativeActionsProcessorStep) and step.enabled
                for step in getattr(preprocessor, "steps", ())
            )
            and getattr(policy.config, "n_obs_steps", 1) <= 1
            and getattr(policy.config, "temporal_ensemble_coeff", None) is None
        )
        self._chunk: deque[torch.Tensor] = deque()
        logger.info(
            "SyncInferenceEngine initialized (device=%s, action_keys=%d, whole chunks=%s)",
            self._device,
            len(ordered_action_keys),
            self._relative,
        )

    def start(self) -> None:
        """No background resources to start."""
        logger.info("SyncInferenceEngine started (inline mode — no background thread)")

    def stop(self) -> None:
        """No background resources to stop."""
        logger.info("SyncInferenceEngine stopped")

    def reset(self) -> None:
        """Reset the policy and pre/post-processors."""
        logger.info("Resetting sync inference state (policy + processors)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._chunk.clear()
        # The policy was just reset, so a pending task change has nothing stale to flush.
        self._discard_task_change()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | EnvTransition | None:
        """Run the full inference pipeline on ``obs_frame`` and return an action tensor.

        When the policy returned a prediction besides the action, returns it with it as a transition.
        """
        if obs_frame is None:
            return None
        if self._relative:
            task, task_changed = self._take_task()
            if task_changed:
                logger.info("Task changed to '%s' — dropping the rest of the chunk", task)
                self._chunk.clear()
                self._postprocessor.reset()
            if not self._chunk:
                started = time.monotonic()
                self._chunk.extend(
                    self._predict_chunk(obs_frame, getattr(self._policy.config, "n_action_steps", None))
                )
                if self.latency.interval_s > 0:
                    completed = time.monotonic()
                    self.latency.record(completed - started, completed)
                self._set_dispatched_task(task)
            return self._chunk.popleft()
        # Shallow copy is intentional: the caller (`send_next_action`) builds
        # ``obs_frame`` fresh per tick via ``build_dataset_frame``, so the
        # tensor/array values are not shared with any other reader.
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        task, task_changed = self._take_task()
        with torch.inference_mode(), autocast_ctx:
            if task_changed:
                # Chunking policies queue actions computed under the previous instruction,
                # so drop them and let the new one take effect on this tick.  Narrower
                # than ``policy.reset``: observation history and other episode state stay.
                logger.info("Task changed to '%s' — dropping precomputed actions", task)
                self._policy.drop_queued_actions()
                self._postprocessor.reset()
            measure = self.latency.interval_s > 0 and self._policy.count_queued_actions() == 0
            started = time.monotonic() if measure else 0.0
            observation = prepare_observation_for_inference(observation, self._device, task, self._robot_type)
            observation = self._preprocessor(observation)
            transition = policy_output_to_transition(self._policy.select_action(observation))
            # Only the action goes through the postprocessor; the prediction travels beside it.
            action = self._postprocessor(transition_to_policy_action(transition))
        action_tensor = action.squeeze(0).cpu()
        if measure:
            completed = time.monotonic()
            self.latency.record(completed - started, completed)

        # ``task`` is the pre-inference snapshot: a /subtask landing mid-inference must
        # not relabel this action.
        self._set_dispatched_task(task)
        # Policy's own dimension order — the order ``send_next_action`` labels it with.
        if not (prediction := transition_to_prediction(transition)):
            return action_tensor
        return create_transition(action=action_tensor, prediction=prediction)

    def _predict_chunk(self, obs_frame: dict, steps: int | None) -> list[torch.Tensor]:
        """Predict one chunk from ``obs_frame`` and make it absolute against its anchor.

        Returns:
            The first ``steps`` actions (all, for ``None``), each ``(action_dim,)`` on the CPU.
        """
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(
                copy(obs_frame), self._device, self.task, self._robot_type
            )
            # The relative step caches this observation's state; the postprocessor below composes
            # the chunk onto it before any later observation can replace it.
            observation = self._preprocessor(observation)
            chunk = transition_to_policy_action(
                policy_output_to_transition(self._policy.predict_action_chunk(observation))
            )
            chunk = self._postprocessor(chunk[:, : steps or chunk.shape[1]])
        return list(chunk.squeeze(0).float().cpu())

    def dispatch_allowed(self) -> bool:
        """Permit dispatch unless an external planner has yet to issue its first instruction.

        Inline inference has no worker to pause, so the planner hold is applied here: the
        control loop clears interpolation and holds the robot until the first reply.
        """
        return not self.hold_for_planner()

    # ------------------------------------------------------------------
    # Text queries
    # ------------------------------------------------------------------

    @property
    def supports_text_queries(self) -> bool:
        """True when an external text backend is attached or the policy has a text head."""
        return super().supports_text_queries or self._policy.supports_text_generation()

    @property
    def control_thread_owns_policy(self) -> bool:
        """Inference runs inline on the control thread, so queries are served there too."""
        return True

    def _generate_text(self, obs_processed: dict, query: PolicyQuery) -> str:
        """Run the policy's text head on the current observation."""
        obs_frame = build_dataset_frame(self._dataset_features, obs_processed, prefix=OBS_STR)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        # Live task, read without consuming the task-changed edge (the action path needs it).
        task = self.task
        # Generation blocks the control thread, so any precomputed chunk is stale by its
        # full duration afterwards. Drop it and resume from fresh inference, matching the
        # hold-and-invalidate rule of the asynchronous backends.
        self._policy.drop_queued_actions()
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(obs_frame, self._device, task, self._robot_type)
            observation = self._mark_query(observation, query)
            # Reusing the action path's preprocessor is safe only while its steps are
            # stateless per call; the one that is not (an enabled RelativeActionsProcessorStep)
            # is rejected for this backend at context-build time.
            observation = self._preprocessor(observation)
            # No str() coercion: _service_query validates the return value.
            return self._policy.generate_text(observation)
