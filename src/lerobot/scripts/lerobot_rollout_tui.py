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

"""Operate an episodic rollout from a second terminal.

    lerobot-rollout --strategy.type=episodic --strategy.control_port=8099 ...   # terminal 1
    lerobot-rollout-tui --port 8099                                              # terminal 2

Shows the phase, the attempt and its task, the takeovers, the session's tally per task, and
sends the commands (:mod:`lerobot.rollout.control`). Destructive keys -- discard, quit -- ask
for a second press. Standard library only (curses).
"""

from __future__ import annotations

import argparse
import contextlib
import curses
import time

from lerobot.rollout.control import request

HELP = [
    ("n / →", "start the next attempt / end this one or the reset"),
    ("s / f", "success / failure (ends a running attempt; re-marks during the reset)"),
    ("space", "take the arm from the policy / hand it back"),
    ("t, 1-9", "the next attempt's task: cycle / choose"),
    ("r / ←", "discard this attempt (twice)"),
    ("q", "end the session (twice)"),
]
PHASE_COLOR = {
    "ready": 3,
    "walking": 3,
    "policy": 2,
    "human": 4,
    "reset": 5,
    "saving": 5,
    "done": 1,
}


def _draw(screen, port: int, state: dict | None, message: str, armed: str | None) -> None:
    screen.erase()
    height, width = screen.getmaxyx()

    def put(y: int, x: int, text: str, attr: int = 0) -> None:
        if 0 <= y < height and x < width:
            screen.addnstr(y, x, text, max(width - x - 1, 0), attr)

    put(0, 0, f"lerobot-rollout control  127.0.0.1:{port}", curses.A_BOLD)
    if state is None:
        put(2, 0, "not connected: is lerobot-rollout running with --strategy.control_port?")
        put(height - 1, 0, message)
        screen.refresh()
        return

    phase = state["phase"]
    put(2, 0, "phase  ")
    put(2, 7, f" {phase.upper()} ", curses.color_pair(PHASE_COLOR.get(phase, 1)) | curses.A_BOLD)
    elapsed = state["elapsed_s"]
    clock = f"{elapsed:5.1f} / {state['episode_time_s']:.0f} s" if elapsed is not None else ""
    put(2, 24, f"attempt {state['episode']}   {clock}")
    put(3, 0, f"task   {state['task']}")
    if state["next_task"] != state["task"]:
        put(4, 0, f"next   {state['next_task']}", curses.A_BOLD)
    counts = state["counts"]
    outcome = state["outcome"] or "unmarked"
    put(5, 0, f"marked {outcome}")
    if state["intervention"]:
        put(
            6,
            0,
            f"takeovers {counts['interventions']}  ({counts['intervention_frames']} of {counts['frames']} frames)",
        )

    session = state["session"]
    put(
        8,
        0,
        f"session  {session['episodes']} saved: {session['success']} success, {session['failure']} failure",
        curses.A_BOLD,
    )
    for i, task in enumerate(state["tasks"]):
        s, f, u = session["by_task"][task]
        marker = ">" if task == state["next_task"] else " "
        put(9 + i, 0, f" {marker}{i + 1}  {s:3d} ok {f:3d} fail {u:3d} unmarked   {task}")

    row = 10 + len(state["tasks"])
    for key, text in HELP:
        put(row, 0, f"{key:>8}  {text}")
        row += 1
    if armed:
        put(row + 1, 0, f"press {armed} again to confirm", curses.A_REVERSE)
    put(height - 1, 0, message)
    screen.refresh()


def _loop(screen, port: int) -> None:
    curses.curs_set(0)
    curses.use_default_colors()
    for pair, color in enumerate(
        (curses.COLOR_WHITE, curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED, curses.COLOR_CYAN),
        start=1,
    ):
        curses.init_pair(pair, curses.COLOR_BLACK, color)
    screen.timeout(200)
    state, message = None, ""
    armed: str | None = None
    armed_at = 0.0
    keys = {
        ord("n"): ("next", None),
        curses.KEY_RIGHT: ("next", None),
        ord("s"): ("success", None),
        ord("f"): ("failure", None),
        ord(" "): ("takeover", None),
        ord("t"): ("task", None),
    }
    confirm = {ord("r"): "discard", curses.KEY_LEFT: "discard", ord("q"): "quit"}

    while True:
        key = screen.getch()
        cmd, arg = "state", None
        if key in keys:
            cmd, arg = keys[key]
        elif key in confirm:
            name = confirm[key]
            if armed == name and time.monotonic() - armed_at < 3.0:
                cmd, armed = name, None
            else:
                armed, armed_at = name, time.monotonic()
        elif ord("1") <= key <= ord("9") and state is not None:
            i = key - ord("1")
            if i < len(state["tasks"]):
                cmd, arg = "task", state["tasks"][i]
        if armed and time.monotonic() - armed_at >= 3.0:
            armed = None
        try:
            reply = request(port, cmd, arg)
            state = reply["state"]
            if cmd != "state":
                message = f"sent {cmd}" + (f" {arg}" if arg else "") if reply["ok"] else reply["error"]
            if cmd == "quit" and reply["ok"]:
                message = "session ending; the rollout saves and exits"
        except OSError:
            state = None
            if state is None and cmd != "state":
                message = f"not sent: {cmd}"
        _draw(screen, port, state, message, "the key" if armed else None)


def main() -> None:
    """``lerobot-rollout-tui``."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        curses.wrapper(_loop, args.port)


if __name__ == "__main__":
    main()
