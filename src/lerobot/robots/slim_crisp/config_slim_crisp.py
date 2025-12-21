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

from dataclasses import dataclass

from ..config import RobotConfig


@RobotConfig.register_subclass("slim_crisp")
@dataclass
class SlimCrispConfig(RobotConfig):
    """Configuration for slim-crisp-zmq remote robot control.
    
    This configuration enables control of a remote robot via ZMQ protocol
    using the slim-crisp-zmq bridge. The robot supports Cartesian space
    control for end-effector positioning.
    """
    
    # ZMQ Server connection settings
    server_ip: str = "127.0.0.1"
    state_pub_port: int = 5556
    cmd_rep_port: int = 5557
    
    # ZMQ client settings
    command_timeout: float = 5.0  # seconds
    max_state_delay: float = 1.0  # seconds
    
    # Controller settings
    default_controller: str = "cartesian_impedance_controller"
    
    # TODO(lvjonok): Implement gripper support
    # Currently gripper actions are placeholders - implementation needed
    use_gripper: bool = True
    
    # Safety settings (optional)
    max_relative_target: float | None = None
