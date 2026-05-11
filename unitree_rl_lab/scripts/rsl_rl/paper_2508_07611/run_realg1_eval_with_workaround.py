#!/usr/bin/env python3

"""Run realG1 evaluation with the local CUDA reduction workaround enabled."""

from __future__ import annotations

import os
import runpy
import sys

from cuda_reduce_workaround import apply_cuda_reduce_workaround


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_SCRIPT = os.path.join(SCRIPT_DIR, "..", "eval_g1_p3o_cbf_realG1.py")


def main() -> None:
    apply_cuda_reduce_workaround()
    sys.argv[0] = os.path.abspath(__file__)
    runpy.run_path(TARGET_SCRIPT, run_name="__main__")


if __name__ == "__main__":
    main()
