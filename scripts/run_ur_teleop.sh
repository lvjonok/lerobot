#!/usr/bin/env bash
#
# Quick helper to launch UR teleoperation with an Omega device.
# Fill in the TODO fields or export the environment variables before running.

set -euo pipefail

: "${UR_ROBOT_IP:=192.168.1.10}"          # TODO: set to the UR controller IP
: "${UR_TELEOP_TYPE:=omega6}"             # omega3/omega6/etc.
: "${OMEGA_DEVICE_INDEX:=0}"              # TODO: pick the correct device index or use serial_number

lerobot-teleoperate \
  --robot.type=ur_rtde \
  --robot.robot_ip="${UR_ROBOT_IP}" \
  --teleop.type="${UR_TELEOP_TYPE}" \
  --teleop.device_index="${OMEGA_DEVICE_INDEX}" \
  --teleop.enable_button_index=0 \
  --teleop.gripper_open_button_index=1 \
  --teleop.gripper_close_button_index=2 \
  --fps=120 \
  --display_data=false \
  "$@"
