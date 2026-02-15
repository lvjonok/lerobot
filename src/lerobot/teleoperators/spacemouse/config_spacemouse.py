"""Configuration for SpaceMouse teleoperator."""

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpaceMouseTeleopConfig(TeleoperatorConfig):
    """Configuration for 3Dconnexion SpaceMouse teleoperator.

    Provides 6DOF delta-based teleoperation with configurable axis filtering
    via teleop_mode.
    """

    # Control scaling
    translation_scale: float = 0.001
    rotation_scale: float = 0.05

    # Teleop mode — which axes are actuated
    # "left_arm_6DOF", "left_arm_3D_translation",
    # "left_arm_3D_translation_Y_rotation", "left_arm_3D_translation_Z_rotation"
    teleop_mode: str = "left_arm_3D_translation"

    # Dead zones
    translation_deadzone: float = 50.0
    rotation_deadzone: float = 50.0

    # Control mode
    use_delta_control: bool = True
