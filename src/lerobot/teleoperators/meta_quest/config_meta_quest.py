"""Configuration for Meta Quest teleoperator."""

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("meta_quest")
@dataclass
class MetaQuestTeleopConfig(TeleoperatorConfig):
    """Configuration for Meta Quest hand-tracking teleoperator.

    Receives hand tracking data from a Meta Quest headset via UDP multicast
    using the meta-teleop client library.
    """

    # Multicast settings
    multicast_group: str = "224.1.1.1"

    # Which hand controls the robot
    control_hand: str = "right"

    # Control scaling
    translation_scale: float = 1.0
    rotation_scale: float = 1.0

    # Teleop mode — which axes are actuated
    teleop_mode: str = "left_arm_6DOF"

    # Gripper control via pinch gesture
    pinch_close_threshold: float = 0.03  # meters
    pinch_open_threshold: float = 0.05  # meters
    max_gripper_width: float = 0.08  # meters
    gripper_velocity: float = 0.1  # m/s
    gripper_force: float = 40.0  # N

    # Use force-based grasping vs position-based
    use_force_grasp: bool = True

    # Tracking engagement: require pinch of the non-control hand
    require_engagement_pinch: bool = False
    engagement_hand: str = "left"
    engagement_pinch_threshold: float = 0.03  # meters

    # Use palm joint from finger_joints channel instead of hand_pose
    use_palm_joint: bool = True

    # Channel timeout
    channel_timeout: float = 10.0  # seconds
