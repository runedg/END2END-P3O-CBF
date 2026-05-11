# P3O End2End G1 29dof

Unitree G1 29自由度人形机器人 **P3O（Perception-aware Policy Optimization）** 端到端避障策略的完整训练与部署代码。

---

## 项目结构

```
.
├── models/                          # 训练产物
│   └── P3O-END2END-001/
│       └── 2026-04-30_06-06-44/
│           ├── model_final.pt        # 最终训练模型（来源：P3O-END2END-001）
│           └── events.out.tfevents.* # TensorBoard 日志
├── rl_sar/                          # Sim2Sim / Sim2Real 部署框架（G1 专用精简版）
│   ├── src/rl_sar/                  # 核心源码（含 lidar、P3O FSM 修改）
│   ├── policy/g1/p3o_end2end/       # 部署策略配置 + policy.pt
│   ├── src/rl_sar_zoo/g1_description/ # G1 URDF/MJCF 描述文件
│   └── ...
├── rsl_rl/                          # 训练算法（含 P3O 修改）
│   ├── rsl_rl/algorithms/p3o_eco.py
│   ├── rsl_rl/modules/actor_critic_safe.py
│   └── configs/p3o_config.yaml
├── unitree_rl_lab/                  # IsaacSim/IsaacLab 环境（修改过的文件）
│   ├── scripts/rsl_rl/train_g1_p3o_cbf.py
│   ├── source/unitree_rl_lab/...    # 环境配置、观察、奖励函数
│   └── deploy/robots/g1_29dof/      # 实机部署配置
├── scripts/                         # 根目录自研训练/评估脚本
│   ├── run_gait_finetune.sh
│   ├── run_v2_tuned.sh
│   ├── record_v2_tuned_eval.py
│   └── ...
├── patches/                         # 原始修改 patch（参考用）
│   ├── rsl_rl_changes.patch
│   ├── unitree_rl_lab_changes.patch
│   └── rl_sar_changes.patch
└── docs/
    └── CURRICULUM_RETRAIN_HANDOFF.md
```

---

## 核心修改说明

### 1. rsl_rl（训练算法）
- `p3o_eco.py`：P3O 安全感知强化学习算法
- `actor_critic_safe.py`：带安全约束的 Actor-Critic 网络
- `rollout_storage_safe.py`：安全 rollout buffer
- `ppo.py`：原始 PPO 的修改版本

### 2. unitree_rl_lab（训练环境）
- 新增 `obstacle_avoidance_env_cfg.py`：障碍物避障环境配置
- 新增 `realg1_mid360.py`：MID360 激光雷达接入
- 修改 `observations.py`：增加 lidar_points 观察
- 修改 `rewards.py` / `obstacle_rewards.py`：避障奖励函数
- 新增 `train_g1_p3o_cbf.py`：主训练脚本

### 3. rl_sar（部署框架）
- `rl_sdk.cpp/hpp`：增加 `lidar_points` 观察支持
- `fsm_g1.hpp`：新增 `RLFSMStateRLP3OEnd2End` 状态，按键 `5` 切换
- `rl_sim.cpp` / `rl_sim_mujoco.cpp`：Mid360 伪激光雷达（raycast / FPS 下采样）
- `policy/g1/p3o_end2end/`：部署配置 + `policy.pt`

---

## 快速开始

### 训练（参考）
```bash
cd unitree_rl_lab
python scripts/rsl_rl/train_g1_p3o_cbf.py
```

### MuJoCo Sim2Sim
```bash
cd rl_sar
./build.sh -mj
./cmake_build/bin/rl_sim_mujoco g1 scene_29dof_obstacle
# 按 5 进入 P3O End2End 模式
```

### Gazebo Sim2Sim
```bash
cd rl_sar
source /opt/ros/humble/setup.bash
./build.sh rl_sar
ros2 launch rl_sar gazebo.launch.py rname:=g1
# 新终端：ros2 run rl_sar rl_sim，按 5 切换
```

### 真机部署
参考 `rl_sar/README_DEPLOY.md`（需补充 ROS2 Livox 订阅到 `rl_real_g1`）。

---

## 模型导出

如需重新导出 `policy.pt`：
```bash
python rl_sar/policy/g1/p3o_end2end/export_model_libtorch23.py
```

---

## 关键参数

| 参数 | 值 |
|------|-----|
| 机器人 | Unitree G1 29dof |
| 观察维度 | 2400（5 帧历史堆叠） |
| 激光雷达 | Livox MID360，128 点（FPS 下采样） |
| 动作空间 | 29 维关节位置偏移 |
| `action_scale` | 0.25 |
| `dt` | 0.005 × 4 decimation = 0.02s |
| 训练框架 | rsl_rl + IsaacLab |

---

**备份日期**：2026-05-11  
**原始路径**：`/home/ubuntu/P3O-CBF`
