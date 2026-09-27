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
from collections import deque
from contextlib import nullcontext
from copy import copy

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline, RelativeActionsProcessorStep
from lerobot.utils.constants import ACTION

from .base import InferenceEngine

logger = logging.getLogger(__name__)


# Relative-action policies run whole chunks.  The per-tick flow refreshes
# ``RelativeActionsProcessorStep._last_state`` every call, so chunk actions
# queued inside ``select_action`` and popped on later ticks would be re-anchored
# to the *current* robot state and the absolute targets would drift through the
# chunk.  Instead the engine preprocesses one observation, predicts the whole
# chunk with ``predict_action_chunk``, postprocesses it at once against that
# observation's anchor, and serves the first ``n_action_steps`` from a local
# FIFO.  This bypasses ``select_action``, so the two policies that need it are
# refused: ACT with temporal ensembling (the ensembler lives in
# ``select_action``, and it would average chunks anchored at different states),
# and policies with observation-history queues (the Diffusion family fills
# ``_queues`` as a side effect of ``select_action``).


class SyncInferenceEngine(InferenceEngine):
    """Inline synchronous inference: compute one action per call.

    ``get_action`` runs the full policy pipeline (pre/post-processor +
    ``select_action``) on the given observation frame and returns a
    CPU action tensor reordered to match the dataset action keys.
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
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        # The policy's output, in the dataset's action order, is the robot's action keys only:
        # a teleop pipeline may add action features of its own (a clutch state, a counter),
        # which the policy never emits.
        self._policy_action_names = [
            name for name in dataset_features[ACTION]["names"] if name in ordered_action_keys
        ]
        self._relative = any(
            isinstance(step, RelativeActionsProcessorStep) and step.enabled
            for step in getattr(preprocessor, "steps", ())
        )
        self._chunk: deque[torch.Tensor] = deque()
        if self._relative:
            if getattr(policy.config, "temporal_ensemble_coeff", None) is not None:
                raise ValueError(
                    "Relative-action policies run whole chunks; temporal ensembling would average "
                    "chunks anchored at different states. Set temporal_ensemble_coeff to None."
                )
            if hasattr(policy, "_queues"):
                raise NotImplementedError(
                    f"SyncInferenceEngine does not support relative-action policies with "
                    f"observation-history queues ({type(policy).__name__}) yet."
                )
        logger.info(
            "SyncInferenceEngine initialized (device=%s, action_keys=%d, relative=%s)",
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

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Run the full inference pipeline on ``obs_frame`` and return an action tensor.

        For a relative-action policy, the next action of the current chunk; a new chunk is
        predicted from ``obs_frame`` only when the last one is used up.
        """
        if obs_frame is None:
            return None
        if self._relative:
            if not self._chunk:
                self._chunk.extend(self._predict_chunk(obs_frame))
            return self._reorder(self._chunk.popleft())
        # Shallow copy is intentional: the caller (`send_next_action`) builds
        # ``obs_frame`` fresh per tick via ``build_dataset_frame``, so the
        # tensor/array values are not shared with any other reader.
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )
            observation = self._preprocessor(observation)
            action = self._policy.select_action(observation)
            action = self._postprocessor(action)
        return self._reorder(action.squeeze(0).cpu())

    def _predict_chunk(self, obs_frame: dict) -> list[torch.Tensor]:
        """Predict one chunk and make it absolute against this observation's anchor.

        Returns:
            The first ``n_action_steps`` actions, each ``(action_dim,)`` on the CPU.
        """
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )
            # The relative step caches this observation's state; the postprocessor below composes
            # the chunk onto it before any later observation can replace it.
            observation = self._preprocessor(observation)
            chunk = self._policy.predict_action_chunk(observation)
            steps = getattr(self._policy.config, "n_action_steps", None) or chunk.shape[1]
            chunk = self._postprocessor(chunk[:, :steps])
        return list(chunk.squeeze(0).float().cpu())

    def _reorder(self, action_tensor: torch.Tensor) -> torch.Tensor:
        """Name the policy's output, then order it by ``ordered_action_keys``."""
        action_dict = dict(zip(self._policy_action_names, action_tensor.tolist(), strict=True))
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])
