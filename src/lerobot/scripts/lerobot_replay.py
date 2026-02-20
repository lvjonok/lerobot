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
Replays the actions of an episode from a dataset on a robot.

Examples:

```shell
lerobot-replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=aliberts/record-test \
    --dataset.episode=0
```

Example replay with bimanual so100:
```shell
lerobot-replay \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --dataset.repo_id=${HF_USER}/bimanual-so100-handover-cube \
  --dataset.episode=0
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import (
    make_processors_for,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    crisp_ws,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so100_follower,
    so101_follower,
)
from lerobot.utils.action_interpolator import divide_twist
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Number of interpolation sub-steps per action (0 = disabled).
    # E.g. 3 sends 3 sub-actions per dataset frame, tripling the robot command rate.
    action_interpolation_steps: int = 0


@parser.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    _, robot_action_processor, _ = make_processors_for(cfg.robot.type)

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])

    # Filter dataset to only include frames from the specified episode since episodes are chunked in dataset V3.0
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.dataset.episode)
    actions = episode_frames.select_columns(ACTION)

    n_substeps = cfg.action_interpolation_steps

    robot.connect()

    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    for idx in range(len(episode_frames)):
        start_episode_t = time.perf_counter()

        action_array = actions[idx][ACTION]
        action_spec = dataset.features[ACTION]
        names = action_spec["names"]
        feature_sizes = action_spec.get("feature_sizes", {n: 1 for n in names})
        action = {}
        offset = 0
        for name in names:
            size = feature_sizes[name]
            if size == 1:
                action[name] = action_array[offset]
            else:
                action[name] = action_array[offset:offset + size]
            offset += size

        if n_substeps > 0:
            divided = divide_twist(action, n_substeps)
            substep_dt = 1.0 / (dataset.fps * n_substeps)
            for _ in range(n_substeps):
                sub_start = time.perf_counter()
                robot_obs = robot.get_observation()
                absolute = robot_action_processor((divided, robot_obs))
                robot.send_action(absolute)
                elapsed = time.perf_counter() - sub_start
                precise_sleep(substep_dt - elapsed)
        else:
            robot_obs = robot.get_observation()
            processed_action = robot_action_processor((action, robot_obs))
            robot.send_action(processed_action)
            dt_s = time.perf_counter() - start_episode_t
            precise_sleep(1 / dataset.fps - dt_s)

    robot.disconnect()


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
