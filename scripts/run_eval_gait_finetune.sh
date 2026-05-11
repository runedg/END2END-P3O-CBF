#!/bin/bash
# P3O-CBF 步态微调模型评估视频录制
# 模型: End2EndP3O_GaitFinetune/2026-05-07_00-20-01/model_2500.pt

set -e

PROJECT_DIR="/home/ubuntu/P3O-CBF"
CONDA_ENV="CBF"
CHECKPOINT="${PROJECT_DIR}/logs/End2EndP3O_GaitFinetune/2026-05-07_00-20-01/model_2500.pt"

cd "${PROJECT_DIR}"
export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority
export LD_LIBRARY_PATH=/home/ubuntu/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=${PROJECT_DIR}/unitree_rl_lab/scripts/rsl_rl/paper_2508_07611:${PROJECT_DIR}/unitree_rl_lab/source/unitree_rl_lab:${PROJECT_DIR}/rsl_rl

# Force Vulkan to use NVIDIA GPU only (avoid Intel GPU crash)
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

echo "======================================"
echo " P3O-CBF 评估视频录制"
echo "======================================"
echo "Checkpoint : ${CHECKPOINT}"
echo "======================================"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "[ERROR] Checkpoint 不存在: ${CHECKPOINT}"
    exit 1
fi

# 去掉 --headless，用 GUI 窗口渲染
conda run -n ${CONDA_ENV} python \
    ${PROJECT_DIR}/unitree_rl_lab/scripts/rsl_rl/paper_2508_07611/eval_g1_p3o_cbf_realG1_paper_omni.py \
    --checkpoint ${CHECKPOINT} \
    --device cuda:0 --steps 420 --num_envs 1 \
    --topdown_follow --show_lidar_points --lidar_ring_vis \
    --terrain_clutter --terrain_obstacles 4 --terrain_layout continuous_avoidance \
    --terrain_size_x 30.0 --terrain_size_y 24.0 \
    --terrain_x_min -10.5 --terrain_x_max 10.5 --terrain_y_span 8.5 \
    --safety_margin 0.8 --collision_distance 0.2 \
    --cmd_vx 0.20 --cmd_vy 0.0 --cmd_wz 0.0 \
    --name gait_finetune_4obstacles_topdown.mp4 --video_fps 30

echo ""
echo "[INFO] 录制完成!"
