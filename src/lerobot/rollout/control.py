# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""A control channel for a running rollout: its state out, the operator's commands in.

The episodic strategy's keys are single presses into whichever terminal the rollout logs to,
with nothing on screen saying which phase it is in, which task the next attempt will be
asked, or what was marked. With ``--strategy.control_port=N`` the strategy serves this
channel instead, and ``lerobot-rollout-tui --port N`` (another terminal) shows the state and
sends the commands.

Protocol: one TCP connection per request on 127.0.0.1 (it moves a robot: never another
interface; reach it over ssh). The client sends one JSON line, ``{"cmd": NAME}`` or
``{"cmd": "task", "arg": TASK}``, and reads one JSON line back, the state after the command.
``{"cmd": "state"}`` only reads.
"""

from __future__ import annotations

import json
import logging
import socket
import socketserver
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

COMMANDS = ("state", "next", "discard", "success", "failure", "takeover", "task", "quit")
"""What a client may ask: read; end the attempt or the reset; discard the attempt; mark it;
take the arm from the policy or hand it back; choose the next attempt's task; end the session."""


class ControlServer:
    """Serves :data:`COMMANDS` on 127.0.0.1 from a background thread.

    Args:
        port: The TCP port.
        handle: Applies one command, ``(cmd, arg)``; raises ``ValueError`` to refuse it.
        state: The state to answer with, JSON-serialisable.
    """

    def __init__(
        self,
        port: int,
        handle: Callable[[str, Any], None],
        state: Callable[[], dict[str, Any]],
    ) -> None:
        outer = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                try:
                    request = json.loads(self.rfile.readline())
                    cmd = request.get("cmd")
                    if cmd not in COMMANDS:
                        raise ValueError(f"unknown command {cmd!r}; one of {COMMANDS}")
                    if cmd != "state":
                        outer._handle(cmd, request.get("arg"))
                    reply = {"ok": True, "state": outer._state()}
                except (ValueError, json.JSONDecodeError) as error:
                    reply = {"ok": False, "error": str(error), "state": outer._state()}
                self.wfile.write((json.dumps(reply) + "\n").encode())

        self._handle, self._state = handle, state
        socketserver.TCPServer.allow_reuse_address = True
        self._server = socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        """Serve until :meth:`stop`."""
        self._thread.start()
        logger.info(
            "Rollout control on 127.0.0.1:%d: lerobot-rollout-tui --port %d",
            self._server.server_address[1],
            self._server.server_address[1],
        )

    def stop(self) -> None:
        """Stop serving."""
        self._server.shutdown()
        self._server.server_close()


def request(port: int, cmd: str, arg: Any = None, timeout_s: float = 1.0) -> dict[str, Any]:
    """Send one command, return the reply ``{"ok", "state"[, "error"]}``.

    Raises:
        OSError: Nothing is serving on ``port``.
    """
    with socket.create_connection(("127.0.0.1", port), timeout=timeout_s) as sock:
        sock.sendall((json.dumps({"cmd": cmd, "arg": arg}) + "\n").encode())
        return json.loads(sock.makefile().readline())
