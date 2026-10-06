#!/bin/bash
# 1-Command Reproducibility Script for Phase 2 Fine-Tuning
# Full 24-run grid (2 backbones x k in {5,10,25,all} x seeds {0,1,2}) on Pets.
# Needs a CUDA GPU (~3 h on an RTX 4060). Run from the repo root.
set -e

echo "=========================================="
echo " Starting Phase 2 Fine-Tuning Grid Execution"
echo " Target Output: results/finetune_grid.csv"
echo "=========================================="

# finetune.py appends one row per run, so start from an empty CSV
rm -f results/finetune_grid.csv
python src/finetune.py --grid

echo "=========================================="
echo " Phase 2 Execution Completed Successfully!"
echo " Results written to results/finetune_grid.csv"
echo "=========================================="
