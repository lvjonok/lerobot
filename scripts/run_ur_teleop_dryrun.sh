#!/usr/bin/env bash
#
# Same as run_ur_teleop.sh but keeps robot actions disabled so you can verify
# both devices initialise correctly before moving hardware.

set -euo pipefail

: "${UR_ROBOT_IP:=192.168.1.10}"
: "${UR_TELEOP_TYPE:=omega6}"
: "${OMEGA_DEVICE_INDEX:=0}"

lerobot-teleoperate \
  --robot.type=ur_rtde \
  --robot.robot_ip="${UR_ROBOT_IP}" \
  --teleop.type="${UR_TELEOP_TYPE}" \
  --teleop.device_index="${OMEGA_DEVICE_INDEX}" \
  --disable_robot_actions=true \
  --display_data=true \
  "$@"
