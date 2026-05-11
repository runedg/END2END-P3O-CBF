#!/bin/bash
# ============================================================
# P3O-CBF 步态微调训练脚本 v2
# 提升速度跟踪 + 保持避障 + 更自然步态
# 基础模型: P3O-END2END-001/2026-04-29_12-52-49/model_final.pt
# GPU: RTX 5090 (32GB)
# ============================================================

set -e

# ---- 路径配置 ----
PROJECT_DIR="/home/ubuntu/P3O-CBF"
CONDA_ENV="CBF"
CHECKPOINT="${PROJECT_DIR}/logs/P3O-END2END-001/2026-04-29_12-52-49/model_final.pt"

# ---- 环境变量 ----
cd "${PROJECT_DIR}"
export DISPLAY=:1
export XAUTHORITY=/home/ubuntu/.Xauthority
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/home/ubuntu/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=${PROJECT_DIR}/unitree_rl_lab/scripts/rsl_rl/paper_2508_07611:${PROJECT_DIR}/unitree_rl_lab/source/unitree_rl_lab:${PROJECT_DIR}/rsl_rl

# ---- 训练参数 ----
NUM_ENVS=4096
MAX_ITERATIONS=15000
SAVE_INTERVAL=2000

# ---- 启动前检查 ----
echo "======================================"
echo " P3O-CBF 步态微调"
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
echo "[INFO] 启动步态微调训练 ..."
# 使用 python -u 强制无缓冲输出，确保每轮信息实时打印
/home/ubuntu/miniconda3/envs/${CONDA_ENV}/bin/python -u \
    ${PROJECT_DIR}/unitree_rl_lab/scripts/rsl_rl/paper_2508_07611/finetune_end2end_gait_v2.py \
    --headless \
    --device cuda:0 \
    --resume ${CHECKPOINT} \
    --num_envs ${NUM_ENVS} \
    --max_iterations ${MAX_ITERATIONS} \
    --save_interval ${SAVE_INTERVAL} \
    --num_steps_per_env 24 \
    --num_learning_epochs 5 \
    --num_mini_batches 16 \
    --learning_rate 1e-4 \
    --cost_critic_learning_rate 3e-5 \
    --entropy_coef 0.005 \
    --num_obstacles 8 \
    --max_vx 0.45

echo "[INFO] 训练完成!"
