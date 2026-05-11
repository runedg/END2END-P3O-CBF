#!/bin/bash
# ============================================================
# P3O_V2_Tuned 速度扩展微调
# 基于 model_26000.pt，提高速度范围到 1.0 m/s
# 保持避障能力：12 obstacles + continuous_avoidance
# ============================================================

set -e

PROJECT_DIR="/home/ubuntu/P3O-CBF"
CONDA_ENV="CBF"
CHECKPOINT="${PROJECT_DIR}/logs/P3O_V2_Tuned/2026-05-09_01-24-50/model_26000.pt"

cd "${PROJECT_DIR}"
export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority
export LD_LIBRARY_PATH=/home/ubuntu/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${PROJECT_DIR}/unitree_rl_lab/scripts/rsl_rl/paper_2508_07611:${PROJECT_DIR}/unitree_rl_lab/source/unitree_rl_lab:${PROJECT_DIR}/rsl_rl:${PYTHONPATH:-}
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

echo "======================================"
echo " P3O_V2_Tuned 速度扩展微调"
echo "======================================"
echo "Checkpoint    : ${CHECKPOINT}"
echo "Speed target  : 0.20 ~ 0.80 m/s"
echo "Obstacles     : 12 (continuous_avoidance)"
echo "Safety margin : 0.55 (frozen)"
echo "CBF gamma     : 0.35 (frozen)"
echo "======================================"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "[ERROR] Checkpoint 不存在: ${CHECKPOINT}"
    exit 1
fi

/home/ubuntu/miniconda3/envs/${CONDA_ENV}/bin/python -u \
    ${PROJECT_DIR}/unitree_rl_lab/scripts/rsl_rl/paper_2508_07611/finetune_speed_extend.py \
    --headless \
    --device cuda:0 \
    --resume ${CHECKPOINT} \
    --num_envs 4096 \
    --max_iterations 8000 \
    --save_interval 1000 \
    --num_steps_per_env 24 \
    --num_learning_epochs 5 \
    --num_mini_batches 24 \
    --learning_rate 5e-5 \
    --cost_critic_learning_rate 1e-5 \
    --entropy_coef 0.005 \
    --num_obstacles 12 \
    --obstacle_layout continuous_avoidance \
    --safety_margin 0.55 \
    --cbf_gamma 0.35 \
    --collision_distance 0.18

echo "[INFO] 训练完成!"
