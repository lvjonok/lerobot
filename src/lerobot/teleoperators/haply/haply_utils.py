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

import asyncio
import logging
import threading
from typing import Dict, Optional, Tuple

try:
    import orjson
    import websockets
except ImportError:
    logging.warning("Haply dependencies not installed. Install with: pip install websockets orjson")


class HaplyController:
    """Interface to Haply Inverse3 device with Verse Grip for teleoperation."""

    def __init__(self, uri: str = "ws://localhost:10001"):
        """
        Initialize the Haply controller.

        Args:
            uri: WebSocket URI for Inverse Service 3.1
        """
        self.uri = uri

        # Device state
        self.inverse3_device_id: Optional[str] = None
        self.verse_grip_device_id: Optional[str] = None
        self.handedness: Optional[str] = None
        self.running = False

        # Inverse3 state (position and velocity)
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.velocity = {"x": 0.0, "y": 0.0, "z": 0.0}

        # Verse Grip state (orientation and buttons)
        self.orientation = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}  # Quaternion
        self.buttons = {}

        # Forces to send (can be customized for haptic feedback)
        self.force = {"x": 0.0, "y": 0.0, "z": 0.0}

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._first_message = True

    def start(self):
        """Start the Haply controller and initialize WebSocket connection."""
        self.running = True
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()

        logging.info("Haply controller started")
        logging.info("  Move the Inverse3 device for X-Y-Z position")
        logging.info("  Verse Grip provides orientation (quaternion) and button states")

    def stop(self):
        """Stop the Haply controller and close WebSocket connection."""
        self.running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2.0)
        logging.info("Haply controller stopped")

    def _run_async_loop(self):
        """Run the asyncio event loop in a separate thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._haptic_loop())
        except Exception as e:
            logging.error(f"Error in Haply haptic loop: {e}")
        finally:
            self._loop.close()

    async def _haptic_loop(self):
        """Main haptic loop for communicating with the Haply device."""
        try:
            async with websockets.connect(self.uri) as ws:
                logging.info(f"Connected to Haply Inverse Service at {self.uri}")

                while self.running:
                    try:
                        # Receive data from the device
                        response = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        data = orjson.loads(response)

                        # Get devices list from the data
                        inverse3_devices = data.get("inverse3", [])
                        verse_grip_devices = data.get("wireless_verse_grip", [])

                        # Get the first device from the list
                        inverse3_data = inverse3_devices[0] if inverse3_devices else {}
                        verse_grip_data = verse_grip_devices[0] if verse_grip_devices else {}

                        # Handle the first message to get device IDs and extra information
                        if self._first_message:
                            self._first_message = False

                            if not inverse3_data:
                                logging.error("No Inverse3 device found.")
                                self.running = False
                                break
                            if not verse_grip_data:
                                logging.warning("No Wireless Verse Grip device found.")

                            # Store device ID for sending forces
                            self.inverse3_device_id = inverse3_data.get("device_id")

                            # Get handedness from Inverse3 device config data
                            self.handedness = inverse3_data.get("config", {}).get("handedness")

                            logging.info(
                                f"Inverse3 device ID: {self.inverse3_device_id}, "
                                f"Handedness: {self.handedness}"
                            )

                            if verse_grip_data:
                                logging.info(
                                    f"Wireless Verse Grip device ID: {verse_grip_data.get('device_id')}"
                                )

                                # Extracself.verse_grip_device_id = verse_grip_data.get("device_id")
                                logging.info(f"Wireless Verse Grip device ID: {self.verse_grip_device_id}")

                        # Extract position and velocity from Inverse3 device state
                        state = inverse3_data.get("state", {})
                        self.position = state.get("cursor_position", {"x": 0.0, "y": 0.0, "z": 0.0})
                        self.velocity = state.get("cursor_velocity", {"x": 0.0, "y": 0.0, "z": 0.0})

                        # Extract buttons and orientation from Wireless Verse Grip device state
                        verse_state = verse_grip_data.get("state", {})
                        self.buttons = verse_state.get("buttons", {})
                        # Orientation is a quaternion with keys: w, x, y, z
                        self.orientation = verse_state.get(
                            "orientation", {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
                        )

                        # Prepare the force command message to send
                        # Must send forces to receive state updates (even if forces are 0)
                        request_msg = {
                            "inverse3": [
                                {
                                    "device_id": self.inverse3_device_id,
                                    "commands": {"set_cursor_force": {"values": self.force}},
                                }
                            ]
                        }

                        # Send the force command message to the server
                        await ws.send(orjson.dumps(request_msg))

                    except asyncio.TimeoutError:
                        # Continue if no data received within timeout
                        continue
                    except Exception as e:
                        logging.error(f"Error in haptic loop iteration: {e}")
                        break

        except Exception as e:
            logging.error(f"Failed to connect to Haply device at {self.uri}: {e}")
            self.running = False

    def get_state(self) -> Dict:
        """
        Get the complete current state of the Haply devices.

        Returns:
            Dictionary containing:
                - xyz: Position dict with keys x, y, z (meters)
                - quat: Orientation quaternion dict with keys w, x, y, z
                - buttons: Button states dict
                - velocity: Velocity dict with keys x, y, z (meters/second)
        """
        return {
            "xyz": self.position.copy(),
            "quat": self.orientation.copy(),
            "buttons": self.buttons.copy(),
            "velocity": self.velocity.copy(),
        }

    def get_position(self) -> Tuple[float, float, float]:
        """
        Get the current position of the Inverse3 device.

        Returns:
            Tuple of (x, y, z) in meters
        """
        return (self.position["x"], self.position["y"], self.position["z"])

    def get_orientation(self) -> Tuple[float, float, float, float]:
        """
        Get the current orientation from the Verse Grip device as a quaternion.

        Returns:
            Tuple of (w, x, y, z) quaternion components
        """
        return (self.orientation["w"], self.orientation["x"], self.orientation["y"], self.orientation["z"])

    def get_buttons(self) -> Dict:
        """
        Get the current button states from the Verse Grip device.

        Returns:
            Dictionary of button states
        """
        return self.buttons.copy()

    def get_velocity(self) -> Tuple[float, float, float]:
        """
        Get the current velocity of the Inverse3 device.

        Returns:
            Tuple of (vx, vy, vz) in meters/second
        """
        return (self.velocity["x"], self.velocity["y"], self.velocity["z"])

    def set_force(self, fx: float = 0.0, fy: float = 0.0, fz: float = 0.0):
        """
        Set the force feedback to send to the Haply device.

        Args:
            fx: Force in X direction (Newtons)
            fy: Force in Y direction (Newtons)
            fz: Force in Z direction (Newtons)
        """
        self.force = {"x": fx, "y": fy, "z": fz}
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released when exiting 'with' block."""
        self.stop()
