uv run python -m lerobot.scripts.lerobot_teleoperate \
    --teleop.type=omega6 \
    --teleop.device_index=0 \
    --teleop.recenter_on_enable=true \
    --robot.type=ur_rtde \
    --robot.cameras='{
        "dev6":  {"type": "opencv", "index_or_path": 6,  "width": 1280, "height": 720, "fps": 30},
        "dev12": {"type": "opencv", "index_or_path": 12, "width": 1280, "height": 720, "fps": 30}
        }' \
    --display_data=true --teleop.enable_keyboard_gripper=true --teleop.keyboard_toggle_key=g