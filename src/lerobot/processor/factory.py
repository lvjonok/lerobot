#!/usr/bin/env python

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

from .converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from .core import RobotAction, RobotObservation
from .crisp_fastapi_processors import FTSensorBiasSubtractionProcessor
from .pipeline import IdentityProcessorStep, RobotProcessorPipeline


def make_default_teleop_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return teleop_action_processor


def make_default_robot_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return robot_action_processor


def make_default_robot_observation_processor() -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[IdentityProcessorStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return robot_observation_processor


def make_crisp_robot_observation_processor() -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    """Observation processor for crisp_fastapi / crisp_ws robots.

    Includes per-episode FT sensor bias subtraction (gravity compensation).
    """
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[FTSensorBiasSubtractionProcessor()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return robot_observation_processor


def make_default_processors():
    teleop_action_processor = make_default_teleop_action_processor()
    robot_action_processor = make_default_robot_action_processor()
    robot_observation_processor = make_default_robot_observation_processor()
    return (teleop_action_processor, robot_action_processor, robot_observation_processor)


def make_processors_for(
    robot_type: str,
    teleop_config=None,
) -> tuple[
    RobotProcessorPipeline,
    RobotProcessorPipeline,
    RobotProcessorPipeline,
]:
    """Create processor pipelines for a given robot-teleop combination.

    Returns:
        (teleop_action_processor, robot_action_processor, robot_observation_processor)
    """
    teleop_type = teleop_config.type if teleop_config is not None else None

    crisp_robots = {"crisp_fastapi", "crisp_ws"}

    if teleop_type == "spacemouse" and robot_type in crisp_robots:
        from .crisp_fastapi_processors import (
            AbsoluteToTwistProcessor,
            SpaceMouseDeltaToAbsoluteProcessor,
            TwistToAbsolutePoseProcessor,
        )

        teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
            steps=[SpaceMouseDeltaToAbsoluteProcessor(), AbsoluteToTwistProcessor()],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )
        robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
            steps=[TwistToAbsolutePoseProcessor()],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )
        robot_observation_processor = make_crisp_robot_observation_processor()
        return teleop_action_processor, robot_action_processor, robot_observation_processor

    elif teleop_type == "haply" and robot_type in crisp_robots:
        from .crisp_fastapi_processors import (
            AbsoluteToTwistProcessor,
            HaplyToCrispClutchProcessor,
            TwistToAbsolutePoseProcessor,
        )

        teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
            steps=[
                HaplyToCrispClutchProcessor(
                    teleop_mode=getattr(teleop_config, "teleop_mode", "left_arm_6DOF"),
                    translation_scale=getattr(teleop_config, "translation_scale", 1.0),
                    rotation_scale=getattr(teleop_config, "rotation_scale", 1.0),
                    max_gripper_width=getattr(teleop_config, "max_gripper_width", 0.08),
                ),
                AbsoluteToTwistProcessor(),
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )
        robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
            steps=[TwistToAbsolutePoseProcessor()],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )
        robot_observation_processor = make_crisp_robot_observation_processor()
        return teleop_action_processor, robot_action_processor, robot_observation_processor

    elif teleop_type == "meta_quest" and robot_type in crisp_robots:
        from .crisp_fastapi_processors import (
            AbsoluteToTwistProcessor,
            DeltaPoseToAbsoluteProcessor,
            TwistToAbsolutePoseProcessor,
        )

        teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
            steps=[DeltaPoseToAbsoluteProcessor(), AbsoluteToTwistProcessor()],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )
        robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
            steps=[TwistToAbsolutePoseProcessor()],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )
        robot_observation_processor = make_crisp_robot_observation_processor()
        return teleop_action_processor, robot_action_processor, robot_observation_processor

    elif teleop_type == "haply" and robot_type == "slim_crisp":
        from scipy.spatial.transform import Rotation
        import numpy as np
        from .haply_clutch_processor import HaplyToSlimCrispClutchProcessor

        frame_transform = Rotation.from_matrix(
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        )

        teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
            steps=[HaplyToSlimCrispClutchProcessor(
                axis_scales=[-1.0, -1.0, 1.0],
                enable_orientation=True,
                rotation_deadband=0.01,
                orientation_frame_transform=frame_transform.as_quat().tolist(),
            )],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )
        _, robot_action_processor, robot_observation_processor = make_default_processors()
        return teleop_action_processor, robot_action_processor, robot_observation_processor

    elif robot_type in crisp_robots:
        teleop_action_processor = make_default_teleop_action_processor()
        robot_action_processor = make_default_robot_action_processor()
        robot_observation_processor = make_crisp_robot_observation_processor()
        return teleop_action_processor, robot_action_processor, robot_observation_processor

    else:
        return make_default_processors()
