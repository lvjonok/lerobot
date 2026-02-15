# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 1920, "height": 1080, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 1920, "height": 1080, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 1920, "height": 1080, "fps": 30}
  }' \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/tty.usbmodem5A460828611 \
  --teleop.right_arm_port=/dev/tty.usbmodem5A460826981 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import multiprocessing
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Any

import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_processors_for,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    crisp_fastapi,
    crisp_ws,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so100_follower,
    so101_follower,
    slim_crisp,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    haply,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    meta_quest,
    omx_leader,
    so100_leader,
    so101_leader,
    spacemouse,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


class RerunLoggerProcess:
    """Offloads Rerun logging to a separate process, fully isolating it from the control loop's GIL."""

    def __init__(self):
        self._queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=2)
        self._process = multiprocessing.Process(
            target=self._run, args=(self._queue,), daemon=True, name="rerun_logger"
        )
        self._process.start()

    def log(self, observation: dict[str, Any], action: dict[str, Any]) -> None:
        """Submit data for logging (non-blocking, drops oldest frame if the logger falls behind)."""
        try:
            self._queue.put_nowait((observation, action))
        except multiprocessing.queues.Full:
            try:
                self._queue.get_nowait()  # discard oldest
            except multiprocessing.queues.Empty:
                pass
            try:
                self._queue.put_nowait((observation, action))
            except multiprocessing.queues.Full:
                pass  # skip frame

    @staticmethod
    def _run(queue: multiprocessing.Queue) -> None:
        # All Rerun state lives in this child process — no GIL sharing with control loop
        init_rerun(session_name="teleoperation")
        while True:
            try:
                # Drain queue and only log the latest frame
                data = queue.get(timeout=1.0)
                while not queue.empty():
                    try:
                        data = queue.get_nowait()
                    except multiprocessing.queues.Empty:
                        break
                log_rerun_data(observation=data[0], action=data[1])
            except multiprocessing.queues.Empty:
                continue
            except (EOFError, KeyboardInterrupt):
                break
        rr.rerun_shutdown()

    def stop(self) -> None:
        self._queue.close()
        self._process.join(timeout=3.0)
        if self._process.is_alive():
            self._process.terminate()


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display action values in the terminal
    display_data: bool = False
    # Visualize observations and actions in Rerun viewer
    visualize: bool = False


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    rerun_logger: RerunLoggerProcess | None = None,
    duration: float | None = None,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, displays action values in the terminal.
        rerun_logger: Background Rerun logger thread. If provided, observations and actions are logged
            asynchronously without blocking the control loop.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        obs = robot.get_observation()

        # Get teleop action
        raw_action = teleop.get_action()

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Send processed action to robot (robot_action_processor.to_output should return dict[str, Any])
        _ = robot.send_action(robot_action_to_send)

        if rerun_logger is not None:
            # Process observation and hand off to background process (non-blocking)
            obs_transition = robot_observation_processor(obs)
            rerun_logger.log(observation=obs_transition, action=teleop_action)

        if display_data:
            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    rerun_logger = None
    if cfg.visualize:
        rerun_logger = RerunLoggerProcess()

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    # Select processor pipeline based on robot-teleop combination
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_processors_for(
        robot_type=cfg.robot.type,
        teleop_config=cfg.teleop,
    )

    teleop.connect()
    robot.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            rerun_logger=rerun_logger,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if rerun_logger is not None:
            rerun_logger.stop()
        teleop.disconnect()

        # possibly move robot home
        if robot.name == "slim_crisp":
            logging.info("Moving robot to home position")
            robot._robot.home()
        elif robot.name in ("crisp_fastapi", "crisp_ws"):
            logging.info("Moving robot to home position")
            robot.go_home()

        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
