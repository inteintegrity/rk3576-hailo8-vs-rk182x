"""Small paramiko helper for driving the RK3576 board that hosts the accelerators.

The SSH password is never passed on the command line and never written to a file
by this script: it is read from an environment variable (default RK_SSH_PASSWORD)
so it stays out of shell history and process arguments.

Examples:
    python remote_ops.py run --command "hailortcli scan"
    python remote_ops.py put --local model/yolo11n_hailo8_int8.hef --remote /home/seeed/x.hef
    python remote_ops.py get --remote /home/seeed/out.jpg --local results/out.jpg
"""

from __future__ import annotations

import argparse
import os
import posixpath
import sys
from pathlib import Path

import paramiko

DEFAULT_HOST = os.environ.get("RK_SSH_HOST", "")   # set RK_SSH_HOST or pass --host
DEFAULT_USER = "seeed"
PASSWORD_ENV = "RK_SSH_PASSWORD"


def connect(host: str, user: str, password_env: str, timeout: int = 20) -> paramiko.SSHClient:
    password = os.environ.get(password_env)
    if not password:
        raise SystemExit(
            f"environment variable {password_env} is not set; source the existing "
            "RK3576 stress-test ssh-env.sh or export it before running"
        )
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host,
        username=user,
        password=password,
        look_for_keys=False,
        allow_agent=False,
        timeout=timeout,
        auth_timeout=timeout,
        banner_timeout=timeout,
    )
    return client


def run(client: paramiko.SSHClient, command: str, timeout: int) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout, get_pty=False)
    code = stdout.channel.recv_exit_status()
    return code, stdout.read().decode(errors="replace"), stderr.read().decode(errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password-env", default=PASSWORD_ENV)
    parser.add_argument("--timeout", type=int, default=600)
    subparsers = parser.add_subparsers(dest="action", required=True)

    run_parser = subparsers.add_parser("run", help="run a shell command on the board")
    run_parser.add_argument("--command", required=True)
    run_parser.add_argument("--log", type=Path, help="also write combined output to this file")

    put_parser = subparsers.add_parser("put", help="upload a file")
    put_parser.add_argument("--local", type=Path, required=True)
    put_parser.add_argument("--remote", required=True)

    get_parser = subparsers.add_parser("get", help="download a file")
    get_parser.add_argument("--remote", required=True)
    get_parser.add_argument("--local", type=Path, required=True)

    args = parser.parse_args()
    client = connect(args.host, args.user, args.password_env)
    try:
        if args.action == "run":
            code, out, err = run(client, args.command, args.timeout)
            text = out + err
            sys.stdout.write(text)
            if args.log:
                args.log.parent.mkdir(parents=True, exist_ok=True)
                args.log.write_text(f"command={args.command}\nexit_code={code}\n{text}", encoding="utf-8")
            raise SystemExit(code)
        if args.action == "put":
            if not args.local.is_file():
                raise SystemExit(f"local file not found: {args.local}")
            remote_dir = posixpath.dirname(args.remote)
            if remote_dir:
                run(client, f"mkdir -p '{remote_dir}'", 60)
            with client.open_sftp() as sftp:
                sftp.put(str(args.local.resolve()), args.remote)
            print(f"uploaded {args.local} -> {args.remote}")
            return
        if args.action == "get":
            args.local.parent.mkdir(parents=True, exist_ok=True)
            with client.open_sftp() as sftp:
                sftp.get(args.remote, str(args.local.resolve()))
            print(f"downloaded {args.remote} -> {args.local}")
            return
    finally:
        client.close()


if __name__ == "__main__":
    main()