from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Sequence

from dotenv import load_dotenv

AGENTS: Sequence[tuple[str, int]] = (
    ("reader", 8001),
    ("analyst", 8002),
    ("visualizer", 8003),
)

PROCESSES: list[subprocess.Popen[bytes]] = []


def start_agent(agent: str, port: int) -> subprocess.Popen[bytes]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.start_agent",
        "--agent",
        agent,
        "--port",
        str(port),
    ]
    print(f"[bootstrap] launching {agent} on port {port}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=os.environ.copy())


def shutdown(*_: int) -> None:
    print("\nStopping agents…")
    for proc in PROCESSES:
        if proc.poll() is None:
            proc.terminate()
    for proc in PROCESSES:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    sys.exit(0)


def main() -> None:
    load_dotenv()
    for agent, port in AGENTS:
        proc = start_agent(agent, port)
        PROCESSES.append(proc)

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)  # type: ignore[arg-type]

    print("Agents running (reader@8001, analyst@8002, visualizer@8003). Press Ctrl+C to stop.")

    try:
        for proc in PROCESSES:
            proc.wait()
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
