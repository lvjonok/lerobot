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

"""
Simple standalone test for HaplyController.
Tests the raw device interface without the teleoperator wrapper.
"""

import sys
import time

# Add parent directory to path for imports
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.teleoperators.haply.haply_utils import HaplyController


def main():
    """Test the HaplyController directly."""
    print("=" * 60)
    print("Haply Controller Raw Test")
    print("=" * 60)
    print()
    print("This test reads raw values from the Haply device.")
    print()
    print("Instructions:")
    print("  1. Ensure Haply Inverse Service is running")
    print("  2. Move the Inverse3 device and press buttons")
    print("  3. Press Ctrl+C to exit")
    print()
    print("=" * 60)
    print()

    # Create and start controller
    try:
        controller = HaplyController()
        controller.start()
        print("✓ Connected to Haply device")
        time.sleep(1)  # Give it time to initialize
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        print("\nMake sure:")
        print("  - Haply Inverse Service is running on ws://localhost:10001")
        print("  - Dependencies installed: pip install websockets orjson")
        return

    print()
    print("Reading device state... (updates every 0.5s)")
    print()

    try:
        prev_button_state = {}

        while controller.running:
            # Get complete state
            state = controller.get_state()

            # Display position
            xyz = state["xyz"]
            print(f"\rPosition: x={xyz['x']:7.4f} y={xyz['y']:7.4f} z={xyz['z']:7.4f} ", end="")

            # Display velocity
            vel = state["velocity"]
            print(f"| Velocity: vx={vel['x']:6.3f} vy={vel['y']:6.3f} vz={vel['z']:6.3f} ", end="")

            # Display orientation (quaternion)
            quat = state["quat"]
            print(
                f"| Quat: w={quat['w']:5.2f} x={quat['x']:5.2f} y={quat['y']:5.2f} z={quat['z']:5.2f}", end=""
            )

            # Display buttons
            buttons = state["buttons"]
            if buttons:
                button_str = " | Buttons: "
                for btn_id, btn_state in buttons.items():
                    if btn_state:
                        button_str += f"{btn_id}:ON "
                    else:
                        button_str += f"{btn_id}:OFF "
                print(button_str, end="")

                # Detect button presses (edge detection)
                for btn_id, btn_state in buttons.items():
                    if btn_state and not prev_button_state.get(btn_id, False):
                        print(f"\n>>> Button {btn_id} pressed! <<<", end="")

                prev_button_state = buttons.copy()

            sys.stdout.flush()
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n")
        print("=" * 60)
        print("Shutting down...")

    finally:
        controller.stop()
        print("✓ Disconnected")


if __name__ == "__main__":
    main()
