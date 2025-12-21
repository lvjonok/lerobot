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

"""
Example script for recording demonstrations with slim-crisp robot and Haply teleoperator.

This example shows how to:
1. Configure and connect to a remote robot via slim-crisp-zmq bridge
2. Use Haply Inverse3 device for teleoperation
3. Record demonstrations to a LeRobot dataset

Prerequisites:
- slim-crisp-zmq server running on robot machine
- Haply Inverse Service running (ws://localhost:10001)
- slim-crisp-zmq installed in lerobot environment

Usage:
    python examples/slim_crisp/record_haply.py

Or use the CLI:
    lerobot-record \\
        --robot.type=slim_crisp \\
        --robot.server_ip=127.0.0.1 \\
        --robot.id=my_robot \\
        --teleop.type=haply \\
        --teleop.use_gripper=true \\
        --dataset.repo_id=<username>/<dataset_name> \\
        --dataset.num_episodes=5 \\
        --dataset.single_task="Pick and place object" \\
        --display_data=true
"""

from dataclasses import dataclass
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import (
    make_default_processors,
    RobotActionProcessorStep,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.slim_crisp import SlimCrispConfig, SlimCrispRobot
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.haply import HaplyTeleop, HaplyTeleopConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# Configuration
NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 30
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "Pick and place the object"
HF_REPO_ID = "robot-lev/slim_crisp_haply_demo"  # Replace with your HuggingFace username and dataset name


class HaplyDeltaToAbsolutePositionStep(RobotActionProcessorStep):
    """Convert Haply deltas to absolute positions with intervention tracking.
    
    Haply outputs deltas from when button 'b' is pressed: {'x': dx, 'y': dy, 'z': dz}
    Robot needs to track: remembered_position_at_button_press + current_delta
    
    This processor:
    1. Detects when intervention starts (deltas become non-zero after being zero)
    2. Remembers robot position at that moment
    3. Outputs: remembered_position + haply_delta
    """
    
    def __init__(self):
        super().__init__()
        # State tracking
        self.robot_position_at_intervention_start: dict[str, float] | None = None
        self.was_intervening: bool = False
    
    def action(self, action: RobotAction) -> RobotAction:
        """Convert delta action to absolute position using remembered intervention start position."""
        # Get observation from the current transition
        from lerobot.processor.pipeline import TransitionKey
        observation = self._current_transition.get(TransitionKey.OBSERVATION, {})
        
        # Get deltas from Haply
        delta_x = action.get('x', 0.0)
        delta_y = action.get('y', 0.0)
        delta_z = action.get('z', 0.0)
        
        # Check if we're currently controlling (any non-zero delta indicates intervention)
        is_intervening = abs(delta_x) > 1e-6 or abs(delta_y) > 1e-6 or abs(delta_z) > 1e-6
        
        # Detect intervention start: started getting deltas
        if is_intervening and not self.was_intervening:
            # Capture robot position at intervention start
            self.robot_position_at_intervention_start = {
                'x': observation.get('ee_pos_x', 0.0),
                'y': observation.get('ee_pos_y', 0.0),
                'z': observation.get('ee_pos_z', 0.0),
            }
            print(f"🎮 Intervention started! Robot position captured: x={self.robot_position_at_intervention_start['x']:.3f}, "
                  f"y={self.robot_position_at_intervention_start['y']:.3f}, z={self.robot_position_at_intervention_start['z']:.3f}")
        
        # Detect intervention end: stopped getting deltas
        elif not is_intervening and self.was_intervening:
            print("✋ Intervention ended! Robot will maintain last position until next intervention.")
            # DON'T clear the remembered position - keep it for smooth transitions
        
        self.was_intervening = is_intervening
        
        # Compute target position
        if is_intervening and self.robot_position_at_intervention_start is not None:
            # During intervention: remembered position + current delta
            target_x = self.robot_position_at_intervention_start['x'] + delta_x
            target_y = self.robot_position_at_intervention_start['y'] + delta_y
            target_z = self.robot_position_at_intervention_start['z'] + delta_z
        else:
            # Not intervening: maintain current position
            target_x = observation.get('ee_pos_x', 0.0)
            target_y = observation.get('ee_pos_y', 0.0)
            target_z = observation.get('ee_pos_z', 0.0)
        
        # Build action with SlimCrisp key names
        mapped_action = {
            'ee_pos_x': target_x,
            'ee_pos_y': target_y,
            'ee_pos_z': target_z,
        }
        
        # Pass through gripper state
        if 'gripper' in action:
            mapped_action['gripper.pos'] = action['gripper']
        
        return mapped_action
    
    def get_config(self) -> dict[str, Any]:
        return {}
    
    def transform_features(self, features):
        """Transform feature names from Haply format to SlimCrisp format."""
        # Create new features dict with transformed names
        transformed = {}
        for key, value in features.items():
            if isinstance(value, dict) and 'names' in value:
                # Transform the names mapping
                new_names = {}
                for name, idx in value['names'].items():
                    # Map Haply names to SlimCrisp names
                    if name == 'x':
                        new_names['ee_pos_x'] = idx
                    elif name == 'y':
                        new_names['ee_pos_y'] = idx
                    elif name == 'z':
                        new_names['ee_pos_z'] = idx
                    elif name == 'gripper':
                        new_names['gripper.pos'] = idx
                    else:
                        new_names[name] = idx
                
                # Create new feature with transformed names
                transformed[key] = {**value, 'names': new_names}
            else:
                transformed[key] = value
        
        return transformed


def main():
    """Record demonstrations with slim-crisp robot and Haply teleoperator."""
    
    # Create robot configuration
    robot_config = SlimCrispConfig(
        server_ip="127.0.0.1",  # Update with your robot server IP
        state_pub_port=5556,
        cmd_rep_port=5557,
        id="slim_crisp_robot",
        use_gripper=True,
    )
    
    # Create teleoperator configuration
    teleop_config = HaplyTeleopConfig(
        use_gripper=True,
    )
    
    # Initialize robot and teleoperator
    robot = SlimCrispRobot(robot_config)
    teleop = HaplyTeleop(teleop_config)
    
    # Create processors
    default_teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    
    # Add custom processor to convert Haply deltas to absolute positions
    # This processor:
    # 1. Takes delta actions from Haply (x, y, z deltas)
    # 2. Adds them to current robot position from observation
    # 3. Converts key names from Haply format to SlimCrisp format
    # This ensures smooth tracking from robot's current position
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[HaplyDeltaToAbsolutePositionStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    
    # Configure dataset features
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}
    
    # Create dataset
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=False,  # No cameras for now
    )
    
    # Connect robot and teleoperator
    print("Connecting to robot...")
    robot.connect()
    
    print("Connecting to Haply device...")
    print("Make sure Haply Inverse Service is running on ws://localhost:10001")
    teleop.connect()
    
    # Initialize keyboard listener and visualization
    listener, events = init_keyboard_listener()
    init_rerun(session_name="slim_crisp_record")
    
    if not robot.is_connected or not teleop.is_connected:
        raise ValueError("Robot or teleop is not connected!")
    
    print("\n" + "="*60)
    print("Ready to record!")
    print("="*60)
    print("\nHaply Controls:")
    print("  - Button 'b': Toggle intervention (start/stop controlling)")
    print("  - Button 'a': Toggle gripper (open/close)")
    print("  - Button 'c': Mark episode as successful")
    print("  - Keyboard 'R': Re-record current episode")
    print("\n" + "="*60)
    print()
    
    # Record episodes
    recorded_episodes = 0
    while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Recording episode {recorded_episodes + 1}/{NUM_EPISODES}")
        
        # Main record loop
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            dataset=dataset,
            teleop=teleop,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )
        
        # recorded_episodes += 1
        
        # # Reset between episodes
        # if recorded_episodes < NUM_EPISODES:
        #     log_say(f"Reset the environment. Press 'C' to continue when ready.")
        #     import time
        #     time.sleep(RESET_TIME_SEC)
    
    # Cleanup
    print("\nRecording complete!")
    robot.disconnect()
    teleop.disconnect()
    listener.stop()
    
    print(f"\nDataset saved to: {dataset.repo_id}")
    print(f"Total episodes: {len(dataset)}")


if __name__ == "__main__":
    main()
