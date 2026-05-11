#!/bin/bash
# 清除 Isaac Sim 和 NVIDIA shader 缓存
# 解决驱动更新后 RTX 渲染引擎 crash 的问题

set -e

echo "[INFO] 清除 NVIDIA GLCache ..."
sudo rm -rf /home/ubuntu/.cache/nvidia/GLCache/*

echo "[INFO] 清除 Isaac Sim shader cache ..."
sudo rm -rf /home/ubuntu/miniconda3/envs/CBF/lib/python3.11/site-packages/isaacsim/kit/cache/shadercache/*

echo "[INFO] 清除 Isaac Sim DerivedDataCache ..."
sudo rm -rf /home/ubuntu/miniconda3/envs/CBF/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache/*

echo "[INFO] 清除 Isaac Sim kit logs ..."
rm -rf /home/ubuntu/miniconda3/envs/CBF/lib/python3.11/site-packages/isaacsim/kit/logs/*

echo "[INFO] 缓存清除完毕!"
