# P3O-END2END-001

End-to-end P3O-CBF curriculum run for G1 obstacle avoidance.

## Recommended checkpoint

Use the Stage 5 checkpoint as the current best model:

`2026-04-30_06-06-44/model_final.pt`

This checkpoint was trained with 20 obstacles and was selected after visual checks. Stage 6 was started briefly but stopped because the Stage 5 model looked better for the current sim2sim/sim2real candidate.

## Stage mapping

- `2026-04-29_10-55-45`: Stage 0, 1 obstacle, metrics only kept.
- `2026-04-29_12-52-49`: Stage 1, 2 obstacles, final checkpoint kept.
- `2026-04-29_16-30-55`: Stage 3, 8 obstacles, final checkpoint kept.
- `2026-04-29_23-17-30`: Stage 4, 12 obstacles, final checkpoint kept.
- `2026-04-30_06-06-44`: Stage 5, 20 obstacles, final checkpoint and evaluation videos kept.

## Kept evaluation videos

- `2026-04-30_06-06-44/videos/stage5_20obs_topdown_long.mp4`
- `2026-04-30_06-06-44/videos/stage5_rect_blocks_topdown_long.mp4`
- `2026-04-30_06-06-44/videos/stage5_20obs_follow_video.mp4`

## Cleanup policy

Intermediate `model_*.pt` checkpoints were removed. TensorBoard event files, final checkpoints, videos, and metrics CSV files were kept.
