#!/usr/bin/env python3
"""Run staged P3O-CBF curriculum training by restarting between stages.

This supervisor keeps the user-facing workflow automatic while preserving the
important implementation detail that terrain obstacles are created at
environment startup. Each stage starts a fresh IsaacLab process from the latest
checkpoint produced by the previous stage.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path("/home/ubuntu/P3O-CBF")
DEFAULT_PYTHON = Path("/home/ubuntu/miniconda3/envs/CBF/bin/python")
DEFAULT_TRAIN_SCRIPT = REPO_ROOT / "unitree_rl_lab/scripts/rsl_rl/paper_2508_07611/train_g1_p3o_cbf_realG1_paper_curriculum.py"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs"
DEFAULT_EXPERIMENT = "End2EndP3OCurriculumRebuild"

DEFAULT_STAGE_PLAN = (
    (2, 21000),
    (3, 23000),
    (4, 25000),
    (5, 27000),
    (6, 30000),
)


def parse_stage_plan(raw: str) -> list[tuple[int, int]]:
    plan: list[tuple[int, int]] = []
    if not raw:
        return list(DEFAULT_STAGE_PLAN)
    for item in raw.split(","):
        stage_s, max_iter_s = item.split(":", 1)
        plan.append((int(stage_s), int(max_iter_s)))
    return plan


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pid(pid: int, poll_seconds: float) -> None:
    print(f"[supervisor] Waiting for current training PID {pid} to finish.", flush=True)
    while pid_is_running(pid):
        time.sleep(poll_seconds)
    print(f"[supervisor] PID {pid} finished.", flush=True)


def checkpoint_iteration(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1))
    if path.name == "model_final.pt":
        siblings = [checkpoint_iteration(p) for p in path.parent.glob("model_*.pt") if p.name != "model_final.pt"]
        return max(siblings, default=0)
    return 0


def find_latest_checkpoint(experiment_dir: Path) -> Path:
    candidates = list(experiment_dir.glob("*/model_*.pt")) + list(experiment_dir.glob("*/model_final.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {experiment_dir}")
    return max(candidates, key=lambda p: (checkpoint_iteration(p), p.stat().st_mtime))


def run_stage(args: argparse.Namespace, stage: int, max_iterations: int, resume: Path) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stage_log = args.log_root / f"curriculum_supervisor_stage{stage}_{timestamp}.log"
    cmd = [
        str(args.python),
        str(args.train_script),
        "--headless",
        "--device",
        args.device,
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(max_iterations),
        "--num_steps_per_env",
        str(args.num_steps_per_env),
        "--num_mini_batches",
        str(args.num_mini_batches),
        "--num_learning_epochs",
        str(args.num_learning_epochs),
        "--save_interval",
        str(args.save_interval),
        "--experiment_name",
        args.experiment_name,
        "--resume",
        str(resume),
        "--force_stage",
        str(stage),
        "--reset_optimizers",
        "--learning_rate",
        str(args.learning_rate),
        "--cost_critic_learning_rate",
        str(args.cost_critic_learning_rate),
    ]
    print(
        f"[supervisor] Starting Stage {stage}: max_iterations={max_iterations}, resume={resume}",
        flush=True,
    )
    print(f"[supervisor] Stage {stage} log: {stage_log}", flush=True)
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":1")
    env.setdefault("XAUTHORITY", "/home/ubuntu/.Xauthority")
    env["LD_LIBRARY_PATH"] = f"/home/ubuntu/miniconda3/envs/CBF/lib:{env.get('LD_LIBRARY_PATH', '')}"
    with stage_log.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while True:
            return_code = process.poll()
            if return_code is not None:
                break
            time.sleep(args.poll_seconds)
    if return_code != 0:
        raise RuntimeError(f"Stage {stage} failed with exit code {return_code}. See {stage_log}")
    print(f"[supervisor] Stage {stage} finished successfully.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, default=None)
    parser.add_argument("--stage-plan", type=str, default=",".join(f"{s}:{m}" for s, m in DEFAULT_STAGE_PLAN))
    parser.add_argument("--experiment-name", type=str, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--num-steps-per-env", type=int, default=24)
    parser.add_argument("--num-mini-batches", type=int, default=24)
    parser.add_argument("--num-learning-epochs", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--cost-critic-learning-rate", type=float, default=5e-5)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    args.log_root.mkdir(parents=True, exist_ok=True)
    experiment_dir = args.log_root / args.experiment_name
    stage_plan = parse_stage_plan(args.stage_plan)
    print(f"[supervisor] Stage plan: {stage_plan}", flush=True)

    if args.wait_pid is not None:
        wait_for_pid(args.wait_pid, args.poll_seconds)

    stop_requested = False

    def handle_signal(signum: int, _frame) -> None:
        nonlocal stop_requested
        print(f"[supervisor] Received signal {signum}; stopping after current wait/stage.", flush=True)
        stop_requested = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    for stage, max_iterations in stage_plan:
        if stop_requested:
            break
        resume = find_latest_checkpoint(experiment_dir)
        run_stage(args, stage, max_iterations, resume)

    print("[supervisor] Curriculum supervisor finished.", flush=True)


if __name__ == "__main__":
    main()
