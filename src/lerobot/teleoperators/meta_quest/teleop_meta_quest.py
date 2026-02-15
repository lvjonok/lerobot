"""Meta Quest teleoperator implementation for LeRobot.

Provides 6DOF teleoperation input from a Meta Quest headset via
the meta-teleop UDP multicast client. Uses hand tracking for wrist pose
and finger joints for pinch-based gripper control.
"""

import logging
import threading
import time
from functools import cached_property
from typing import Any, Optional

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_meta_quest import MetaQuestTeleopConfig

logger = logging.getLogger(__name__)


class MetaQuestTeleop(Teleoperator):
    """LeRobot-compatible Meta Quest teleoperator.

    Receives hand tracking data from a Meta Quest headset via UDP multicast
    and converts it to robot actions using relative control.
    """

    config_class = MetaQuestTeleopConfig
    name = "meta_quest"

    def __init__(self, config: MetaQuestTeleopConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._teleop_mode = config.teleop_mode

        # Meta teleop client
        self._client = None

        # State
        self._lock = threading.Lock()
        self._latest_hand_pose = None
        self._latest_finger_joints = None
        self._gripper_is_grasping = False
        self._gripper_state = 0.08

        # Initial pose for relative control
        self._initial_hand_pos: Optional[np.ndarray] = None
        self._initial_hand_quat: Optional[np.ndarray] = None
        self._tracking_active = False

        # Reader thread
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def __str__(self) -> str:
        return f"MetaQuestTeleop({self.config.id}, hand={self.config.control_hand})"

    @cached_property
    def action_features(self) -> dict:
        return {
            "tcp.pos": (3,),
            "tcp.quat": (4,),
            "gripper.pos": float,
        }

    @cached_property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the Meta Quest hand tracking stream."""
        if self._connected:
            logger.warning(f"{self} is already connected.")
            return

        try:
            from meta_teleop import MetaTeleopClient
        except ImportError as e:
            raise ImportError(
                "MetaQuestTeleop requires meta-teleop. "
                "Install with: pip install meta-teleop-client"
            ) from e

        logger.info(f"Connecting to Meta Quest on multicast {self.config.multicast_group}...")

        self._client = MetaTeleopClient(
            multicast_group=self.config.multicast_group,
            auto_start_channels=True,
        )
        self._client.connect()

        if not self._client.wait_for_channels(timeout=self.config.channel_timeout):
            self._client.disconnect()
            self._client = None
            raise ConnectionError(
                f"Timeout waiting for Quest channels after {self.config.channel_timeout}s. "
                "Make sure the Quest app is running and on the same network."
            )

        device_name = self._client.device_name or "Unknown"
        logger.info(f"Connected to Quest device: {device_name}")
        logger.info(f"Available channels: {self._client.channel_names}")

        self._stop_event.clear()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

        self._connected = True
        logger.info(f"{self} connected")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _read_loop(self):
        """Background thread to poll hand tracking data."""
        while not self._stop_event.is_set():
            try:
                channels = self._client.channels if self._client else {}

                hand_pose_ch = channels.get("hand_pose")
                if hand_pose_ch is not None:
                    data = hand_pose_ch.last_packet
                    if data is not None:
                        with self._lock:
                            self._latest_hand_pose = data

                finger_joints_ch = channels.get("finger_joints")
                if finger_joints_ch is not None:
                    data = finger_joints_ch.last_packet
                    if data is not None:
                        with self._lock:
                            self._latest_finger_joints = data

            except Exception as e:
                logger.warning(f"Quest read error: {e}")

            time.sleep(0.005)

    def _unity_to_flu(self, pos_unity: np.ndarray, quat_unity_xyzw: np.ndarray):
        """Convert Unity coordinates to FLU (X-forward, Y-left, Z-up, right-handed)."""
        pos_flu = np.array([pos_unity[2], -pos_unity[0], pos_unity[1]], dtype=np.float32)
        quat_flu = np.array(
            [quat_unity_xyzw[2], -quat_unity_xyzw[0], quat_unity_xyzw[1], quat_unity_xyzw[3]],
            dtype=np.float32,
        )
        return pos_flu, quat_flu

    def _get_hand_pose_flu(self):
        """Get the control hand's wrist pose in FLU coordinates."""
        hand_pose = self._latest_hand_pose
        if hand_pose is None:
            return None

        hand = hand_pose.right if self.config.control_hand == "right" else hand_pose.left
        if not hand.is_tracked:
            return None

        pos_unity = np.array([hand.pos_x, hand.pos_y, hand.pos_z], dtype=np.float32)
        quat_unity = np.array([hand.rot_x, hand.rot_y, hand.rot_z, hand.rot_w], dtype=np.float32)
        return self._unity_to_flu(pos_unity, quat_unity)

    def _get_palm_pose_flu(self):
        """Get control hand's palm joint pose from finger_joints in FLU."""
        from meta_teleop import JOINT_PALM

        finger_joints = self._latest_finger_joints
        if finger_joints is None:
            return None

        hand = finger_joints.right if self.config.control_hand == "right" else finger_joints.left
        if not hand.is_tracked or len(hand.joints) <= JOINT_PALM:
            return None

        palm = hand.joints[JOINT_PALM]
        pos_unity = np.array([palm.pos_x, palm.pos_y, palm.pos_z], dtype=np.float32)
        quat_unity = np.array([palm.rot_x, palm.rot_y, palm.rot_z, palm.rot_w], dtype=np.float32)
        return self._unity_to_flu(pos_unity, quat_unity)

    def _get_pinch_distance(self):
        """Get thumb-to-index tip distance for pinch detection."""
        from meta_teleop import JOINT_INDEX_TIP, JOINT_THUMB_TIP

        finger_joints = self._latest_finger_joints
        if finger_joints is None:
            return None

        hand = finger_joints.right if self.config.control_hand == "right" else finger_joints.left
        if not hand.is_tracked or len(hand.joints) < max(JOINT_THUMB_TIP, JOINT_INDEX_TIP) + 1:
            return None

        thumb_tip = hand.joints[JOINT_THUMB_TIP]
        index_tip = hand.joints[JOINT_INDEX_TIP]

        distance = np.sqrt(
            (thumb_tip.pos_x - index_tip.pos_x) ** 2
            + (thumb_tip.pos_y - index_tip.pos_y) ** 2
            + (thumb_tip.pos_z - index_tip.pos_z) ** 2
        )
        return float(distance)

    def _is_engaged(self) -> bool:
        """Check if the engagement condition is met."""
        if not self.config.require_engagement_pinch:
            return True

        from meta_teleop import JOINT_INDEX_TIP, JOINT_THUMB_TIP

        finger_joints = self._latest_finger_joints
        if finger_joints is None:
            return False

        hand = finger_joints.right if self.config.engagement_hand == "right" else finger_joints.left
        if not hand.is_tracked or len(hand.joints) < max(JOINT_THUMB_TIP, JOINT_INDEX_TIP) + 1:
            return False

        thumb_tip = hand.joints[JOINT_THUMB_TIP]
        index_tip = hand.joints[JOINT_INDEX_TIP]
        distance = np.sqrt(
            (thumb_tip.pos_x - index_tip.pos_x) ** 2
            + (thumb_tip.pos_y - index_tip.pos_y) ** 2
            + (thumb_tip.pos_z - index_tip.pos_z) ** 2
        )
        return distance < self.config.engagement_pinch_threshold

    def set_initial_pose(self, robot_pos: np.ndarray, robot_quat_wxyz: np.ndarray) -> None:
        """Set the initial robot pose for relative control."""
        with self._lock:
            self._initial_robot_pos = np.array(robot_pos, dtype=np.float32)
            self._initial_robot_quat_wxyz = np.array(robot_quat_wxyz, dtype=np.float32)
            self._initial_hand_pos = None
            self._initial_hand_quat = None
            self._tracking_active = False

    def get_action(self) -> dict[str, Any]:
        """Get the current action from Meta Quest hand tracking."""
        if not self._connected:
            raise RuntimeError(f"{self} is not connected. Call connect() first.")

        with self._lock:
            hand_result = self._get_palm_pose_flu() if self.config.use_palm_joint else self._get_hand_pose_flu()

            # Update gripper state based on pinch with hysteresis
            pinch_dist = self._get_pinch_distance()
            if pinch_dist is not None:
                if not self._gripper_is_grasping and pinch_dist < self.config.pinch_close_threshold:
                    self._gripper_is_grasping = True
                    self._gripper_state = 0.0
                    logger.debug(f"Gripper grasped (pinch={pinch_dist:.3f}m)")
                elif self._gripper_is_grasping and pinch_dist > self.config.pinch_open_threshold:
                    self._gripper_is_grasping = False
                    self._gripper_state = self.config.max_gripper_width
                    logger.debug(f"Gripper released (pinch={pinch_dist:.3f}m)")

            engaged = self._is_engaged()

            if hand_result is None or not engaged:
                if self._tracking_active:
                    logger.info("Lost hand tracking or engagement")
                    self._tracking_active = False
                return {
                    "tcp.pos": np.zeros(3, dtype=np.float32),
                    "tcp.quat": np.array([1, 0, 0, 0], dtype=np.float32),
                    "gripper.pos": float(self._gripper_state),
                }

            current_pos, current_quat_xyzw = hand_result

            if self._initial_hand_pos is None:
                self._initial_hand_pos = current_pos.copy()
                self._initial_hand_quat = current_quat_xyzw.copy()
                self._tracking_active = True
                logger.info(f"Initial hand pose recorded: pos={current_pos}")

            import scipy.spatial.transform as st

            delta_pos = (current_pos - self._initial_hand_pos) * self.config.translation_scale

            initial_rot = st.Rotation.from_quat(self._initial_hand_quat)
            current_rot = st.Rotation.from_quat(current_quat_xyzw)
            delta_rot = current_rot * initial_rot.inv()

            if self._teleop_mode != "left_arm_6DOF":
                delta_rpy = delta_rot.as_euler("xyz")
                if self._teleop_mode == "left_arm_3D_translation":
                    delta_rpy[:] = 0.0
                elif self._teleop_mode == "left_arm_3D_translation_Y_rotation":
                    delta_rpy[0] = 0.0
                    delta_rpy[2] = 0.0
                elif self._teleop_mode == "left_arm_3D_translation_Z_rotation":
                    delta_rpy[0] = 0.0
                    delta_rpy[1] = 0.0
                delta_rot = st.Rotation.from_euler("xyz", delta_rpy)

            delta_quat_xyzw = delta_rot.as_quat().astype(np.float32)
            delta_quat_wxyz = np.array(
                [delta_quat_xyzw[3], delta_quat_xyzw[0], delta_quat_xyzw[1], delta_quat_xyzw[2]],
                dtype=np.float32,
            )

        return {
            "tcp.pos": delta_pos.astype(np.float32),
            "tcp.quat": delta_quat_wxyz,
            "gripper.pos": float(self._gripper_state),
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        """Disconnect from the Meta Quest."""
        if not self._connected:
            logger.warning(f"{self} is not connected.")
            return

        logger.info(f"Disconnecting {self}")

        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None

        if self._client is not None:
            self._client.disconnect()
            self._client = None

        self._connected = False
        self._tracking_active = False
        self._initial_hand_pos = None
        self._initial_hand_quat = None
        logger.info(f"{self} disconnected")
