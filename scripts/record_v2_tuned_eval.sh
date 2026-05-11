#!/bin/bash
export CARB_TMPDIR=/home/ubuntu/P3O-CBF/.tmp_shm
set -euo pipefail

PROJECT_DIR="/home/ubuntu/P3O-CBF"
CONDA_ENV="CBF"
CHECKPOINT="${PROJECT_DIR}/logs/P3O_V2_Tuned/2026-05-09_01-24-50/model_26000.pt"

export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority
export LD_LIBRARY_PATH=/home/ubuntu/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${PROJECT_DIR}/unitree_rl_lab/scripts/rsl_rl/paper_2508_07611:${PROJECT_DIR}/unitree_rl_lab/source/unitree_rl_lab:${PROJECT_DIR}/rsl_rl:${PYTHONPATH:-}

export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

cd "${PROJECT_DIR}"

echo "======================================"
echo " P3O_V2_Tuned 评估视频录制"
echo " Checkpoint: ${CHECKPOINT}"
echo "======================================"

if [ ! -f "${CHECKPOINT}" ]; then
  echo "[ERROR] Checkpoint 不存在: ${CHECKPOINT}"
  exit 1
fi

/home/ubuntu/miniconda3/envs/${CONDA_ENV}/bin/python \
  "${PROJECT_DIR}/record_v2_tuned_eval.py" \
  --checkpoint "${CHECKPOINT}" \
  --device cuda:0 \
  --steps 1800 \
  --num_envs 1 \
  --topdown_follow \
  --terrain_clutter \
  --terrain_obstacles 4 \
  --terrain_layout continuous_avoidance \
  --terrain_size_x 30.0 \
  --terrain_size_y 24.0 \
  --terrain_x_min -10.5 \
  --terrain_x_max 10.5 \
  --terrain_y_span 8.5 \
  --cmd_vx 0.30 \
  --cmd_vy 0.0 \
  --cmd_wz 0.0 \
  --video_fps 30 \
  --name v2_tuned_model26000_topdown.mp4

echo "[INFO] 录制完成!"
