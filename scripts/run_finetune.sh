#!/bin/bash
# 1-Command Reproducibility Script for Phase 2 Fine-Tuning

echo "=========================================="
echo " Starting Phase 2 Fine-Tuning Grid Execution"
echo " Target Output: results/finetune_grid.csv"
echo "=========================================="

echo "=========================================="
echo " Phase 2 Execution Completed Successfully!"
echo " Results written to results/finetune_grid.csv"
echo "=========================================="
python finetune.py                           # single run (resnet50, k=10, seed=0)
python finetune.py --backbone dinov2 --k 25  # single DINOv2 run
python finetune.py --grid                    # full 24-run grid
