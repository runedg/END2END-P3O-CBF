#!/usr/bin/env python3

"""Dedicated entrypoint for the paper-closer Mid360 point-cloud P3O-CBF training line."""

from __future__ import annotations

import os
import runpy
import sys

from cuda_reduce_workaround import apply_cuda_reduce_workaround

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_SCRIPT = os.path.join(SCRIPT_DIR, "train_g1_p3o_cbf_realG1_paper_omni.py")
DEFAULT_EXPERIMENT_NAME = "End2EndP3OPaper"


def _has_flag(flag: str) -> bool:
    return flag in sys.argv[1:]


def _append_default_flag(flag: str, value: str) -> None:
    if not _has_flag(flag):
        sys.argv.extend([flag, value])


def main() -> None:
    apply_cuda_reduce_workaround()
    _append_default_flag("--experiment_name", DEFAULT_EXPERIMENT_NAME)
    _append_default_flag("--max_iterations", "10000")
    _append_default_flag("--save_interval", "1000")
    _append_default_flag("--safety_margin", "0.8")
    _append_default_flag("--cost_limit", "0.22")
    _append_default_flag("--num_obstacles", "12")
    _append_default_flag("--min_forward_speed", "0.10")
    _append_default_flag("--max_forward_speed", "0.45")
    _append_default_flag("--max_lateral_speed", "0.08")
    _append_default_flag("--max_yaw_rate", "0.30")
    _append_default_flag("--num_fps_points", "128")
    sys.argv[0] = os.path.abspath(__file__)
    runpy.run_path(TARGET_SCRIPT, run_name="__main__")


if __name__ == "__main__":
    main()
