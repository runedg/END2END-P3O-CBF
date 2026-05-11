#!/bin/bash
# ============================================================
# P3O-CBF v2-tuned 训练脚本
# 基于 P3O-END2END-001/2026-04-30_06-06-44/model_final.pt 做隔离改进:
#   1. 收窄 safety_margin / cbf_gamma，不再过度保守
#   2. 提高速度课程上限，让模型学会跑快
#   3. 放开关节偏差惩罚，步态更自然
#
# 训练脚本: train_g1_p3o_cbf_v2_tuned.py
# 原始脚本保持不动
# ============================================================

set -e

# ---- 路径配置 ----
PROJECT_DIR="/home/ubuntu/P3O-CBF"
CONDA_ENV="CBF"
CHECKPOINT="${PROJECT_DIR}/logs/P3O-END2END-001/2026-04-30_06-06-44/model_final.pt"

# ---- 环境变量 ----
cd "${PROJECT_DIR}"
export DISPLAY=:1
export XAUTHORITY=/home/ubuntu/.Xauthority
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/home/ubuntu/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=${PROJECT_DIR}/unitree_rl_lab/scripts/rsl_rl/paper_2508_07611:${PROJECT_DIR}/unitree_rl_lab/source:${PROJECT_DIR}/rsl_rl

# ---- 训练参数 ----
NUM_ENVS=4096
MAX_ITERATIONS=12000
SAVE_INTERVAL=1000

# ---- 启动前检查 ----
echo "======================================"
echo " P3O-CBF v2-tuned"
echo "======================================"
echo "Checkpoint : ${CHECKPOINT}"
echo "Num Envs   : ${NUM_ENVS}"
echo "Max Iter   : ${MAX_ITERATIONS}"
echo "======================================"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "[ERROR] Checkpoint 不存在: ${CHECKPOINT}"
    exit 1
fi

# ---- GPU 预检 ----
echo "[INFO] 检查 GPU ..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || {
    echo "[ERROR] nvidia-smi 失败，请检查驱动和 GPU 状态"
    exit 1
}
echo ""

# ---- 启动训练 ----
echo "[INFO] 启动 v2-tuned 训练 ..."
conda run -n ${CONDA_ENV} python \
    ${PROJECT_DIR}/unitree_rl_lab/scripts/rsl_rl/paper_2508_07611/train_g1_p3o_cbf_v2_tuned.py \
    --headless \
    --device cuda:0 \
    --resume ${CHECKPOINT} \
    --reset_optimizers \
    --num_envs ${NUM_ENVS} \
    --max_iterations ${MAX_ITERATIONS} \
    --save_interval ${SAVE_INTERVAL} \
    --num_steps_per_env 24 \
    --num_learning_epochs 5 \
    --num_mini_batches 24 \
    --learning_rate 1e-3 \
    --cost_critic_learning_rate 1e-4 \
    --entropy_coef 0.01 \
    --cost_limit 0.22 \
    --safety_margin 0.55 \
    --cbf_gamma 0.35 \
    --collision_distance 0.18 \
    --contact_cost_weight 2.5

echo "[INFO] 训练完成!"
