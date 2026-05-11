# P3O-CBF Curriculum Retrain Handoff

更新时间：2026-04-29，北京时间

## 当前结论

上一轮 `2026-04-28_12-28-10/model_final.pt` 不是加载错误。验证结果：

- 无障碍：稳定行走，不摔。
- 1 个 terrain mesh 障碍：能走，不碰撞，但会贴近到约 `0.49m`，安全余量不足。
- 4 个 terrain mesh 障碍：前半段能走，约 `285 step` 摔倒。
- 28 个混合障碍：失败。

原因不是 Mid360 点云没接上，而是课程难度一开始过高。更关键的是：terrain mesh 障碍物是地形的一部分，环境创建后不能动态隐藏。因此单个训练进程里改变 `active_obstacles` 不能真正改变物理地形难度。必须按阶段重启训练，每个阶段创建对应数量的真实 terrain mesh 障碍物。

## 已修改内容

修改文件：

- `unitree_rl_lab/scripts/rsl_rl/paper_2508_07611/train_g1_p3o_cbf_realG1_paper_curriculum.py`
- `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/obstacle_env_cfg.py`
- `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/obstacle_manager.py`
- `unitree_rl_lab/scripts/rsl_rl/paper_2508_07611/eval_g1_p3o_cbf_realG1_paper_omni.py`

核心改动：

- Curriculum stages 改为 `1 -> 2 -> 4 -> 8 -> 12 -> 20 -> 28` 个障碍物。
- `--force_stage N` 现在会决定真实 terrain mesh 里生成多少障碍物，而不是只改变指标 proxy。
- stage0 到 stage4 使用 `continuous_avoidance`，先学基础绕障。
- stage5 使用 `surrounded_front_open`。
- stage6 使用 `mixed_obstacles_no_wall`，最后才做混合形状微调。
- P3O/CBF safety distance 仍然来自 `mid360_lidar.data.ray_hits_w`，不是 obstacle manager 直接给距离。

## 推荐训练策略

不要再一上来训练 28 个混合障碍。按阶段顺序跑，每阶段结束后录视频/看指标，再进入下一阶段。

建议从当前最新模型继续，因为它无障碍步态稳定：

`/home/ubuntu/P3O-CBF/logs/End2EndP3OPaperCurriculum/2026-04-28_12-28-10/model_final.pt`

如果想更保守，也可以从旧模型开始：

`/home/ubuntu/P3O-CBF/logs/End2EndP3OPaperCurriculum/2026-04-28_00-15-14/model_final.pt`

## 启动训练

所有命令在 `/home/ubuntu/P3O-CBF` 执行。

### Stage 0：1 个前方障碍

目标：先学会提前绕开，不能贴近通过。

```bash
export DISPLAY=:1 XAUTHORITY=/home/ubuntu/.Xauthority LD_LIBRARY_PATH=/home/ubuntu/miniconda3/envs/CBF/lib:$LD_LIBRARY_PATH
cd /home/ubuntu/P3O-CBF
conda run -n CBF python /home/ubuntu/P3O-CBF/unitree_rl_lab/scripts/rsl_rl/paper_2508_07611/train_g1_p3o_cbf_realG1_paper_curriculum.py \
  --headless --device cuda:0 --num_envs 4096 \
  --max_iterations 19000 --num_steps_per_env 24 --num_mini_batches 24 --num_learning_epochs 5 \
  --save_interval 250 --experiment_name End2EndP3OCurriculumRebuild \
  --resume /home/ubuntu/P3O-CBF/logs/End2EndP3OPaperCurriculum/2026-04-28_12-28-10/model_final.pt \
  --force_stage 0 --reset_optimizers \
  --learning_rate 3e-4 --cost_critic_learning_rate 5e-5
```

注意：脚本 resume 时 `--max_iterations` 是绝对目标 iteration。当前 checkpoint 是 `18000`，所以 `19000` 表示再跑 `1000` iter。

### 后续阶段

每个阶段用上一个阶段新生成的 `model_final.pt` 作为 `--resume`。

- Stage 1：`--force_stage 1`，2 个障碍，建议再跑 `1000` iter。
- Stage 2：`--force_stage 2`，4 个障碍，建议再跑 `1500` iter。
- Stage 3：`--force_stage 3`，8 个障碍，建议再跑 `1500-2000` iter。
- Stage 4：`--force_stage 4`，12 个障碍，建议再跑 `2000` iter。
- Stage 5：`--force_stage 5`，20 个障碍，建议再跑 `2000` iter。
- Stage 6：`--force_stage 6`，28 个混合障碍，最后微调 `2000-4000` iter。

每阶段通过标准：

- `contact_rate < 1%`
- `collision_rate < 3%-5%`
- 单场景视频不摔、不接触
- 1 个障碍时 `nearest_obs_min` 不要贴到 `0.5m` 以下，最好保持 `>0.65m`

## 查看训练状态

看 GPU：

```bash
nvidia-smi
```

看最新 checkpoint：

```bash
find logs/End2EndP3OCurriculumRebuild -maxdepth 2 -type f -name 'model_*.pt' -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort | tail -n 20
```

读取 TensorBoard 指标：

```bash
CONDA_PREFIX=/home/ubuntu/miniconda3/envs/CBF /home/ubuntu/miniconda3/envs/CBF/bin/python - <<'PY'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import statistics as st
logdir = sorted(Path('/home/ubuntu/P3O-CBF/logs/End2EndP3OCurriculumRebuild').glob('*'))[-1]
event = next(logdir.glob('events.out.tfevents*'))
ea = EventAccumulator(str(event), size_guidance={'scalars': 0})
ea.Reload()
for tag in ['train/cost','paper/unsafe_rate','paper/collision_rate','paper/contact_rate','paper/distance_mean','track/actual_speed']:
    vals = ea.Scalars(tag)
    tail = vals[-100:]
    print(tag, 'last_step', vals[-1].step, 'last100_avg', st.mean(v.value for v in tail), 'last', vals[-1].value)
PY
```

## 录制评估视频

Stage 0 / 单障碍评估：

```bash
export DISPLAY=:1 XAUTHORITY=/home/ubuntu/.Xauthority LD_LIBRARY_PATH=/home/ubuntu/miniconda3/envs/CBF/lib:$LD_LIBRARY_PATH
cd /home/ubuntu/P3O-CBF
conda run -n CBF python /home/ubuntu/P3O-CBF/unitree_rl_lab/scripts/rsl_rl/paper_2508_07611/eval_g1_p3o_cbf_realG1_paper_omni.py \
  --checkpoint /PATH/TO/model_final.pt \
  --headless --device cuda:0 --steps 360 --num_envs 1 \
  --topdown_follow --show_lidar_points --lidar_ring_vis \
  --terrain_clutter --terrain_obstacles 1 --terrain_layout continuous_avoidance \
  --terrain_size_x 30.0 --terrain_size_y 24.0 --terrain_x_min -10.5 --terrain_x_max 10.5 --terrain_y_span 8.5 \
  --safety_margin 0.8 --collision_distance 0.2 \
  --cmd_vx 0.20 --cmd_vy 0.0 --cmd_wz 0.0 \
  --name curriculum_stage0_single_obstacle_topdown.mp4 --video_fps 30
```

4 个障碍评估：

```bash
conda run -n CBF python /home/ubuntu/P3O-CBF/unitree_rl_lab/scripts/rsl_rl/paper_2508_07611/eval_g1_p3o_cbf_realG1_paper_omni.py \
  --checkpoint /PATH/TO/model_final.pt \
  --headless --device cuda:0 --steps 420 --num_envs 1 \
  --topdown_follow --show_lidar_points --lidar_ring_vis \
  --terrain_clutter --terrain_obstacles 4 --terrain_layout continuous_avoidance \
  --terrain_size_x 30.0 --terrain_size_y 24.0 --terrain_x_min -10.5 --terrain_x_max 10.5 --terrain_y_span 8.5 \
  --safety_margin 0.8 --collision_distance 0.2 \
  --cmd_vx 0.20 --cmd_vy 0.0 --cmd_wz 0.0 \
  --name curriculum_stage2_four_obstacles_topdown.mp4 --video_fps 30
```

## 重要注意

- 不要直接用 `--force_stage 6` 开始训练，否则又会一开始面对 28 个混合障碍。
- 不要用单个进程期待 terrain mesh 的障碍数量随 stage 动态变化；terrain mesh 创建后固定。
- 训练过程中显示的 `paper/distance_*` 来自 Mid360 点云 ray hits。
- 如果视频失败但训练指标好，优先检查单场景 metrics：`fallen`、`contact_collision`、`nearest_obs`、`root_height`。

