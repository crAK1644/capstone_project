"""Convenience launcher: one server process + N client processes locally.

Spawns `python main.py --mode server` first, waits a short warmup, then
spawns `python main.py --mode client --client_id i` for i in 0..N-1. Each
subprocess's stdout+stderr is redirected to its own file under --log_dir
so debugging one client doesn't require parsing a mixed log.

Usage:
    python launch.py --num_clients 27 --num_rounds 150 --device cpu
"""
import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from typing import List

import config

logger = logging.getLogger(__name__)


def _launch(cmd: List[str], log_path: str) -> subprocess.Popen:
    parent: str = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    log_f = open(log_path, "w")  # intentionally not closed; child owns it until exit
    return subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch SSFL server + clients as separate local processes"
    )
    parser.add_argument("--num_clients", type=int, default=config.NUM_CLIENTS)
    parser.add_argument("--num_rounds", type=int, default=config.NUM_ROUNDS)
    parser.add_argument(
        "--server_address", type=str, default=config.DEFAULT_SERVER_ADDRESS
    )
    parser.add_argument("--partition_dir", type=str, default=config.PARTITION_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument(
        "--warmup_sec",
        type=float,
        default=3.0,
        help="Seconds to wait after starting the server before launching clients.",
    )
    args = parser.parse_args()

    python_exe: str = sys.executable
    main_script: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "main.py"
    )
    os.makedirs(args.log_dir, exist_ok=True)

    procs: List[subprocess.Popen] = []

    # ---- Server ----
    server_cmd: List[str] = [
        python_exe, main_script,
        "--mode", "server",
        "--num_clients", str(args.num_clients),
        "--num_rounds", str(args.num_rounds),
        "--server_address", args.server_address,
        "--partition_dir", args.partition_dir,
        "--device", args.device,
    ]
    server_log: str = os.path.join(args.log_dir, "server.log")
    print(f"[launch] server -> {server_log}")
    procs.append(_launch(server_cmd, server_log))

    # gRPC listener warmup
    time.sleep(args.warmup_sec)

    # ---- Clients ----
    for i in range(args.num_clients):
        client_cmd: List[str] = [
            python_exe, main_script,
            "--mode", "client",
            "--client_id", str(i),
            "--server_address", args.server_address,
            "--partition_dir", args.partition_dir,
            "--device", args.device,
        ]
        client_log: str = os.path.join(args.log_dir, f"client_{i:02d}.log")
        print(f"[launch] client {i:02d} -> {client_log}")
        procs.append(_launch(client_cmd, client_log))

    def _shutdown(*_):
        print("\n[launch] shutting down all processes...")
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Block on the server process; when it exits, shut down clients.
    server_proc = procs[0]
    ret = server_proc.wait()
    print(f"[launch] server exited with code {ret}")
    for p in procs[1:]:
        try:
            p.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
