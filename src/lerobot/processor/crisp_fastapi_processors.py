"""Processor steps for converting teleoperator deltas to absolute TCP commands.

Processor steps for crisp_fastapi robots (any robot behind a crisp_py FastAPI server):

1. SpaceMouseDeltaToAbsoluteProcessor — accumulates per-frame deltas from
   SpaceMouse (delta_pos, delta_rot in euler) into absolute TCP targets.

2. DeltaPoseToAbsoluteProcessor — applies delta position/quaternion from
   initial hand pose to initial robot pose. Used with Haply and
   Meta Quest teleoperators that output pose deltas.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.spatial.transform as st

from lerobot.configs.types import PipelineFeatureType

from .core import RobotAction, TransitionKey
from .pipeline import ProcessorStepRegistry, RobotActionProcessorStep


@ProcessorStepRegistry.register("spacemouse_delta_to_absolute")
@dataclass
class SpaceMouseDeltaToAbsoluteProcessor(RobotActionProcessorStep):
    """Accumulate SpaceMouse per-frame deltas into absolute TCP targets.

    The SpaceMouse teleoperator outputs:
        - delta_pos: (3,) position delta per frame
        - delta_rot: (3,) euler angle delta per frame (xyz convention)
        - gripper.pos: float gripper width

    This processor:
    1. On first call, reads the robot's current TCP pose from observation
    2. Each frame: accumulated_pos += delta_pos, accumulated_rpy += delta_rot
    3. Converts accumulated RPY to quaternion
    4. Outputs absolute tcp.pos, tcp.quat, gripper.pos
    """

    # Input keys from SpaceMouse teleoperator
    teleop_delta_pos_key: str = "delta_pos"
    teleop_delta_rot_key: str = "delta_rot"
    teleop_gripper_key: str = "gripper.pos"

    # Output keys for crisp_fastapi robot
    robot_pos_key: str = "tcp.pos"
    robot_quat_key: str = "tcp.quat"
    robot_gripper_key: str = "gripper.pos"

    # Observation keys for reading initial robot pose
    obs_pos_key: str = "tcp.pos"
    obs_quat_key: str = "tcp.quat"

    # Internal state
    _commanded_pos: Any = field(default=None, init=False, repr=False)
    _commanded_rpy: Any = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def _initialize_from_observation(self, obs: dict) -> None:
        """Capture robot pose from observation as initial commanded pose."""
        pos = np.array(obs[self.obs_pos_key], dtype=np.float32)
        quat_wxyz = np.array(obs[self.obs_quat_key], dtype=np.float32)
        rot = st.Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        self._commanded_pos = pos.copy()
        self._commanded_rpy = rot.as_euler("xyz").astype(np.float32)
        self._initialized = True

    def reset_reference(self, obs: dict) -> None:
        """Reset the commanded pose to the current robot pose (right button)."""
        self._initialize_from_observation(obs)

    def action(self, action: RobotAction) -> RobotAction:
        obs = self.transition.get(TransitionKey.OBSERVATION, {})

        if not self._initialized:
            self._initialize_from_observation(obs)

        delta_pos = np.array(action[self.teleop_delta_pos_key], dtype=np.float32)
        delta_rot = np.array(action[self.teleop_delta_rot_key], dtype=np.float32)
        gripper = action[self.teleop_gripper_key]

        self._commanded_pos = self._commanded_pos + delta_pos
        self._commanded_rpy = self._commanded_rpy + delta_rot

        rot = st.Rotation.from_euler("xyz", self._commanded_rpy)
        quat_xyzw = rot.as_quat().astype(np.float32)
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32
        )

        return {
            self.robot_pos_key: self._commanded_pos.copy(),
            self.robot_quat_key: quat_wxyz,
            self.robot_gripper_key: gripper,
        }

    def transform_features(
        self, features: dict, in_key: str | None = None, out_key: str | None = None
    ) -> dict:
        features[PipelineFeatureType.ACTION] = {
            self.robot_pos_key: (3,),
            self.robot_quat_key: (4,),
            self.robot_gripper_key: float,
        }
        return features

    def get_config(self) -> dict:
        return {
            "teleop_delta_pos_key": self.teleop_delta_pos_key,
            "teleop_delta_rot_key": self.teleop_delta_rot_key,
            "teleop_gripper_key": self.teleop_gripper_key,
            "robot_pos_key": self.robot_pos_key,
            "robot_quat_key": self.robot_quat_key,
            "robot_gripper_key": self.robot_gripper_key,
        }

    def reset(self) -> None:
        self._commanded_pos = None
        self._commanded_rpy = None
        self._initialized = False


@ProcessorStepRegistry.register("delta_pose_to_absolute")
@dataclass
class DeltaPoseToAbsoluteProcessor(RobotActionProcessorStep):
    """Apply delta position/quaternion to initial robot pose.

    Used with teleoperators that output deltas from an initial hand pose
    (Haply Inverse3, Meta Quest). The delta represents the offset from the
    teleoperator's initial position/orientation, which is applied to the
    robot's initial TCP pose to produce absolute targets.

    The teleoperator outputs:
        - tcp.pos: (3,) position delta from initial hand position
        - tcp.quat: (4,) quaternion delta (wxyz) from initial hand orientation
        - gripper.pos: float gripper width

    This processor:
    1. On first call, reads the robot's current TCP pose from observation
    2. Each frame: target_pos = initial_pos + delta_pos
    3. target_rot = delta_rot * initial_rot (quaternion composition)
    4. Outputs absolute tcp.pos, tcp.quat, gripper.pos
    """

    # Input/output keys (same keys, but values change from delta to absolute)
    pos_key: str = "tcp.pos"
    quat_key: str = "tcp.quat"
    gripper_key: str = "gripper.pos"

    # Observation keys for reading initial robot pose
    obs_pos_key: str = "tcp.pos"
    obs_quat_key: str = "tcp.quat"

    # Internal state
    _initial_pos: Any = field(default=None, init=False, repr=False)
    _initial_quat_wxyz: Any = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def _initialize_from_observation(self, obs: dict) -> None:
        """Capture robot pose from observation as initial reference."""
        self._initial_pos = np.array(obs[self.obs_pos_key], dtype=np.float32).copy()
        self._initial_quat_wxyz = np.array(obs[self.obs_quat_key], dtype=np.float32).copy()
        self._initialized = True

    def action(self, action: RobotAction) -> RobotAction:
        obs = self.transition.get(TransitionKey.OBSERVATION, {})

        if not self._initialized:
            self._initialize_from_observation(obs)

        delta_pos = np.array(action[self.pos_key], dtype=np.float32)
        delta_quat_wxyz = np.array(action[self.quat_key], dtype=np.float32)
        gripper = action[self.gripper_key]

        target_pos = self._initial_pos + delta_pos

        delta_rot = st.Rotation.from_quat([
            delta_quat_wxyz[1], delta_quat_wxyz[2],
            delta_quat_wxyz[3], delta_quat_wxyz[0],
        ])
        initial_rot = st.Rotation.from_quat([
            self._initial_quat_wxyz[1], self._initial_quat_wxyz[2],
            self._initial_quat_wxyz[3], self._initial_quat_wxyz[0],
        ])
        target_rot = delta_rot * initial_rot
        target_quat_xyzw = target_rot.as_quat().astype(np.float32)
        target_quat_wxyz = np.array(
            [target_quat_xyzw[3], target_quat_xyzw[0], target_quat_xyzw[1], target_quat_xyzw[2]],
            dtype=np.float32,
        )

        return {
            self.pos_key: target_pos,
            self.quat_key: target_quat_wxyz,
            self.gripper_key: gripper,
        }

    def transform_features(
        self, features: dict, in_key: str | None = None, out_key: str | None = None
    ) -> dict:
        features[PipelineFeatureType.ACTION] = {
            self.pos_key: (3,),
            self.quat_key: (4,),
            self.gripper_key: float,
        }
        return features

    def get_config(self) -> dict:
        return {
            "pos_key": self.pos_key,
            "quat_key": self.quat_key,
            "gripper_key": self.gripper_key,
        }

    def reset(self) -> None:
        self._initial_pos = None
        self._initial_quat_wxyz = None
        self._initialized = False


@ProcessorStepRegistry.register("haply_to_crisp_clutch")
@dataclass
class HaplyToCrispClutchProcessor(RobotActionProcessorStep):
    """Convert raw Haply positions/orientations to absolute robot pose with cumulative clutching.

    Takes raw Haply output (x, y, z, qw, qx, qy, qz, is_controlling, gripper)
    and converts to absolute tcp.pos, tcp.quat, gripper.pos for a crisp_fastapi robot.

    Control scheme:
    - is_controlling (button 'b'): Clutch — hold to control, release to reposition
    - gripper: CLOSE=0, STAY=1, OPEN=2 → converted to width (0 or max_gripper_width)
    - Deltas accumulate across clutch sessions

    Position: delta = (current_haply - haply_at_clutch_start) * translation_scale
    Orientation: pure quaternion delta with optional teleop_mode axis filtering
    """

    # Teleop mode — which axes are actuated
    teleop_mode: str = "left_arm_6DOF"

    # Control scaling
    translation_scale: float = 1.0
    rotation_scale: float = 1.0

    # Gripper
    max_gripper_width: float = 0.08  # meters

    # Output keys for crisp_fastapi robot
    robot_pos_key: str = "tcp.pos"
    robot_quat_key: str = "tcp.quat"
    robot_gripper_key: str = "gripper.pos"

    # Observation keys for reading initial robot pose
    obs_pos_key: str = "tcp.pos"
    obs_quat_key: str = "tcp.quat"

    # Internal state
    _clutch_engaged: bool = field(default=False, init=False, repr=False)
    _initial_hand_pos: Any = field(default=None, init=False, repr=False)
    _initial_hand_quat_xyzw: Any = field(default=None, init=False, repr=False)
    _offset_pos: Any = field(default=None, init=False, repr=False)
    _offset_rot: Any = field(default=None, init=False, repr=False)
    _last_delta_pos: Any = field(default=None, init=False, repr=False)
    _last_delta_quat_wxyz: Any = field(default=None, init=False, repr=False)
    _initial_robot_pos: Any = field(default=None, init=False, repr=False)
    _initial_robot_quat_wxyz: Any = field(default=None, init=False, repr=False)
    _gripper_open: bool = field(default=True, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def _initialize_from_observation(self, obs: dict) -> None:
        """Capture robot pose from observation as initial reference."""
        self._initial_robot_pos = np.array(obs[self.obs_pos_key], dtype=np.float32).copy()
        self._initial_robot_quat_wxyz = np.array(obs[self.obs_quat_key], dtype=np.float32).copy()
        self._offset_pos = np.zeros(3, dtype=np.float32)
        self._offset_rot = st.Rotation.identity()
        self._last_delta_pos = np.zeros(3, dtype=np.float32)
        self._last_delta_quat_wxyz = np.array([1, 0, 0, 0], dtype=np.float32)
        self._initialized = True

    def action(self, action: RobotAction) -> RobotAction:
        obs = self.transition.get(TransitionKey.OBSERVATION, {})

        if not self._initialized:
            self._initialize_from_observation(obs)

        # Read raw Haply state
        haply_pos = np.array(
            [action.get("x", 0.0), action.get("y", 0.0), action.get("z", 0.0)],
            dtype=np.float32,
        )
        haply_quat_xyzw = np.array(
            [action.get("qx", 0.0), action.get("qy", 0.0),
             action.get("qz", 0.0), action.get("qw", 1.0)],
            dtype=np.float32,
        )
        clutch_active = action.get("is_controlling", False)

        # Handle gripper (discrete: CLOSE=0, STAY=1, OPEN=2)
        gripper_action = action.get("gripper", 1)  # STAY by default
        if gripper_action == 0:  # CLOSE
            self._gripper_open = False
        elif gripper_action == 2:  # OPEN
            self._gripper_open = True
        # STAY (1) keeps current state

        gripper_width = self.max_gripper_width if self._gripper_open else 0.0

        # Clutch state machine
        if clutch_active:
            if not self._clutch_engaged:
                # Clutch just engaged — capture references
                self._initial_hand_pos = haply_pos.copy()
                self._initial_hand_quat_xyzw = haply_quat_xyzw.copy()
                self._clutch_engaged = True

            # Compute session delta position
            session_delta_pos = (haply_pos - self._initial_hand_pos) * self.translation_scale

            # Compute session delta rotation
            initial_rot = st.Rotation.from_quat(self._initial_hand_quat_xyzw)
            current_rot = st.Rotation.from_quat(haply_quat_xyzw)
            session_delta_rot = current_rot * initial_rot.inv()

            # Filter rotation axes based on teleop_mode
            if self.teleop_mode != "left_arm_6DOF":
                delta_rpy = session_delta_rot.as_euler("xyz")
                if self.teleop_mode == "left_arm_3D_translation":
                    delta_rpy[:] = 0.0
                elif self.teleop_mode == "left_arm_3D_translation_Y_rotation":
                    delta_rpy[0] = 0.0
                    delta_rpy[2] = 0.0
                elif self.teleop_mode == "left_arm_3D_translation_Z_rotation":
                    delta_rpy[0] = 0.0
                    delta_rpy[1] = 0.0
                session_delta_rot = st.Rotation.from_euler("xyz", delta_rpy)

            # Apply rotation scaling
            if self.rotation_scale != 1.0:
                delta_rotvec = session_delta_rot.as_rotvec()
                session_delta_rot = st.Rotation.from_rotvec(delta_rotvec * self.rotation_scale)

            # Accumulate with offset from previous clutch sessions
            total_delta_pos = self._offset_pos + session_delta_pos
            total_delta_rot = session_delta_rot * self._offset_rot

            total_quat_xyzw = total_delta_rot.as_quat().astype(np.float32)
            total_quat_wxyz = np.array(
                [total_quat_xyzw[3], total_quat_xyzw[0], total_quat_xyzw[1], total_quat_xyzw[2]],
                dtype=np.float32,
            )

            self._last_delta_pos = total_delta_pos.astype(np.float32)
            self._last_delta_quat_wxyz = total_quat_wxyz

        else:
            if self._clutch_engaged:
                # Clutch just released — save accumulated offset
                self._clutch_engaged = False
                self._offset_pos = self._last_delta_pos.copy()
                q = self._last_delta_quat_wxyz
                self._offset_rot = st.Rotation.from_quat([q[1], q[2], q[3], q[0]])
                self._initial_hand_pos = None
                self._initial_hand_quat_xyzw = None

        # Apply accumulated delta to initial robot pose
        target_pos = self._initial_robot_pos + self._last_delta_pos

        delta_rot = st.Rotation.from_quat([
            self._last_delta_quat_wxyz[1], self._last_delta_quat_wxyz[2],
            self._last_delta_quat_wxyz[3], self._last_delta_quat_wxyz[0],
        ])
        initial_rot = st.Rotation.from_quat([
            self._initial_robot_quat_wxyz[1], self._initial_robot_quat_wxyz[2],
            self._initial_robot_quat_wxyz[3], self._initial_robot_quat_wxyz[0],
        ])
        target_rot = delta_rot * initial_rot
        target_quat_xyzw = target_rot.as_quat().astype(np.float32)
        target_quat_wxyz = np.array(
            [target_quat_xyzw[3], target_quat_xyzw[0], target_quat_xyzw[1], target_quat_xyzw[2]],
            dtype=np.float32,
        )

        return {
            self.robot_pos_key: target_pos,
            self.robot_quat_key: target_quat_wxyz,
            self.robot_gripper_key: gripper_width,
        }

    def transform_features(
        self, features: dict, in_key: str | None = None, out_key: str | None = None
    ) -> dict:
        features[PipelineFeatureType.ACTION] = {
            self.robot_pos_key: (3,),
            self.robot_quat_key: (4,),
            self.robot_gripper_key: float,
        }
        return features

    def get_config(self) -> dict:
        return {
            "teleop_mode": self.teleop_mode,
            "translation_scale": self.translation_scale,
            "rotation_scale": self.rotation_scale,
            "max_gripper_width": self.max_gripper_width,
            "robot_pos_key": self.robot_pos_key,
            "robot_quat_key": self.robot_quat_key,
            "robot_gripper_key": self.robot_gripper_key,
        }

    def reset(self) -> None:
        self._clutch_engaged = False
        self._initial_hand_pos = None
        self._initial_hand_quat_xyzw = None
        self._offset_pos = None
        self._offset_rot = None
        self._last_delta_pos = None
        self._last_delta_quat_wxyz = None
        self._initial_robot_pos = None
        self._initial_robot_quat_wxyz = None
        self._gripper_open = True
        self._initialized = False


@ProcessorStepRegistry.register("absolute_to_twist")
@dataclass
class AbsoluteToTwistProcessor(RobotActionProcessorStep):
    """Convert absolute TCP target to twist (linear_vel + angular_vel).

    Takes absolute target pose from a preceding processor step and the
    current robot pose from observation, computes the twist:
        linear_vel = target_pos - current_pos
        angular_vel = (target_rot * current_rot^-1).as_rotvec()

    Input (from preceding processor):
        - tcp.pos: (3,) absolute target position
        - tcp.quat: (4,) absolute target quaternion (wxyz)
        - gripper.pos: float gripper width

    Output:
        - linear_vel: (3,) position displacement
        - angular_vel: (3,) rotation vector displacement
        - gripper.pos: float gripper width
    """

    # Input keys (from preceding absolute-pose processor)
    input_pos_key: str = "tcp.pos"
    input_quat_key: str = "tcp.quat"
    input_gripper_key: str = "gripper.pos"

    # Output keys (twist format)
    output_linear_vel_key: str = "linear_vel"
    output_angular_vel_key: str = "angular_vel"
    output_gripper_key: str = "gripper.pos"

    # Observation keys for current robot pose
    obs_pos_key: str = "tcp.pos"
    obs_quat_key: str = "tcp.quat"

    def action(self, action: RobotAction) -> RobotAction:
        obs = self.transition.get(TransitionKey.OBSERVATION, {})

        target_pos = np.array(action[self.input_pos_key], dtype=np.float32)
        target_quat_wxyz = np.array(action[self.input_quat_key], dtype=np.float32)
        gripper = action[self.input_gripper_key]

        current_pos = np.array(obs[self.obs_pos_key], dtype=np.float32)
        current_quat_wxyz = np.array(obs[self.obs_quat_key], dtype=np.float32)

        # Linear velocity = position displacement
        linear_vel = target_pos - current_pos

        # Angular velocity = rotation displacement as rotation vector
        target_rot = st.Rotation.from_quat([
            target_quat_wxyz[1], target_quat_wxyz[2],
            target_quat_wxyz[3], target_quat_wxyz[0],
        ])
        current_rot = st.Rotation.from_quat([
            current_quat_wxyz[1], current_quat_wxyz[2],
            current_quat_wxyz[3], current_quat_wxyz[0],
        ])
        delta_rot = target_rot * current_rot.inv()
        angular_vel = delta_rot.as_rotvec().astype(np.float32)

        return {
            self.output_linear_vel_key: linear_vel,
            self.output_angular_vel_key: angular_vel,
            self.output_gripper_key: gripper,
        }

    def transform_features(
        self, features: dict, in_key: str | None = None, out_key: str | None = None
    ) -> dict:
        features[PipelineFeatureType.ACTION] = {
            self.output_linear_vel_key: (3,),
            self.output_angular_vel_key: (3,),
            self.output_gripper_key: float,
        }
        return features

    def get_config(self) -> dict:
        return {
            "input_pos_key": self.input_pos_key,
            "input_quat_key": self.input_quat_key,
            "input_gripper_key": self.input_gripper_key,
            "output_linear_vel_key": self.output_linear_vel_key,
            "output_angular_vel_key": self.output_angular_vel_key,
            "output_gripper_key": self.output_gripper_key,
        }

    def reset(self) -> None:
        pass


@ProcessorStepRegistry.register("twist_to_absolute_pose")
@dataclass
class TwistToAbsolutePoseProcessor(RobotActionProcessorStep):
    """Convert twist (linear_vel + angular_vel) to absolute TCP target.

    Takes twist from a preceding processor step and the current robot
    pose from observation, computes the absolute target:
        target_pos = current_pos + linear_vel
        target_rot = Rotation.from_rotvec(angular_vel) * current_rot

    Input:
        - linear_vel: (3,) position displacement
        - angular_vel: (3,) rotation vector displacement
        - gripper.pos: float gripper width

    Output:
        - tcp.pos: (3,) absolute target position
        - tcp.quat: (4,) absolute target quaternion (wxyz)
        - gripper.pos: float gripper width
    """

    # Input keys (twist format)
    input_linear_vel_key: str = "linear_vel"
    input_angular_vel_key: str = "angular_vel"
    input_gripper_key: str = "gripper.pos"

    # Output keys (absolute pose)
    output_pos_key: str = "tcp.pos"
    output_quat_key: str = "tcp.quat"
    output_gripper_key: str = "gripper.pos"

    # Observation keys for current robot pose
    obs_pos_key: str = "tcp.pos"
    obs_quat_key: str = "tcp.quat"

    def action(self, action: RobotAction) -> RobotAction:
        obs = self.transition.get(TransitionKey.OBSERVATION, {})

        linear_vel = np.array(action[self.input_linear_vel_key], dtype=np.float32)
        angular_vel = np.array(action[self.input_angular_vel_key], dtype=np.float32)
        gripper = action[self.input_gripper_key]

        current_pos = np.array(obs[self.obs_pos_key], dtype=np.float32)
        current_quat_wxyz = np.array(obs[self.obs_quat_key], dtype=np.float32)

        target_pos = current_pos + linear_vel

        current_rot = st.Rotation.from_quat([
            current_quat_wxyz[1], current_quat_wxyz[2],
            current_quat_wxyz[3], current_quat_wxyz[0],
        ])
        delta_rot = st.Rotation.from_rotvec(angular_vel)
        target_rot = delta_rot * current_rot
        target_quat_xyzw = target_rot.as_quat().astype(np.float32)
        target_quat_wxyz = np.array(
            [target_quat_xyzw[3], target_quat_xyzw[0], target_quat_xyzw[1], target_quat_xyzw[2]],
            dtype=np.float32,
        )

        return {
            self.output_pos_key: target_pos,
            self.output_quat_key: target_quat_wxyz,
            self.output_gripper_key: gripper,
        }

    def transform_features(
        self, features: dict, in_key: str | None = None, out_key: str | None = None
    ) -> dict:
        features[PipelineFeatureType.ACTION] = {
            self.output_pos_key: (3,),
            self.output_quat_key: (4,),
            self.output_gripper_key: float,
        }
        return features

    def get_config(self) -> dict:
        return {
            "input_linear_vel_key": self.input_linear_vel_key,
            "input_angular_vel_key": self.input_angular_vel_key,
            "input_gripper_key": self.input_gripper_key,
            "output_pos_key": self.output_pos_key,
            "output_quat_key": self.output_quat_key,
            "output_gripper_key": self.output_gripper_key,
        }

    def reset(self) -> None:
        pass
