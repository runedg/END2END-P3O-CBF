# arXiv:2508.07611 experiment

This directory isolates a paper-aligned P3O-CBF variant from the existing training scripts.

Key differences from the older local experiment:
- separates task reward from obstacle safety cost
- uses a discrete CBF slack `h_next - (1 - gamma) h`
- keeps an explicit unsafe-set indicator in the cost
- uses a lower rollout-level `cost_limit` so the P3O penalty activates during training
