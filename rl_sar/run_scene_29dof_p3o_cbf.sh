#!/bin/bash
# P3O-CBF G1 29dof sim2sim (MuJoCo)
# 按 5 切换到 p3o_end2end 模式

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority

"${SCRIPT_DIR}/cmake_build/bin/rl_sim_mujoco" g1 scene_29dof_obstacle
