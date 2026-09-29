#!/usr/bin/env bash
set -euo pipefail

FEATURES_DIR="${FEATURES_DIR:-features}"
SPLITS_DIR="${SPLITS_DIR:-splits}"
FROZEN_OUTPUT="${FROZEN_OUTPUT:-results/frozen_grid_v2.csv}"
MASTER_OUTPUT="${MASTER_OUTPUT:-results/master_table_v2.csv}"
MASTER_WIDE_OUTPUT="${MASTER_WIDE_OUTPUT:-results/master_table_wide_v2.csv}"
PLOT_DIR="${PLOT_DIR:-plots/eval_v2}"

if [[ ! -d "$FEATURES_DIR" ]]; then
  echo "Missing feature-cache directory: $FEATURES_DIR" >&2
  exit 1
fi
if [[ ! -d "$SPLITS_DIR" ]]; then
  echo "Missing split directory: $SPLITS_DIR" >&2
  exit 1
fi

python3 scripts/run_frozen_grid.py \
  --features-dir "$FEATURES_DIR" \
  --splits-dir "$SPLITS_DIR" \
  --output "$FROZEN_OUTPUT"

python3 scripts/build_master_table.py \
  --frozen "$FROZEN_OUTPUT" \
  --output "$MASTER_OUTPUT" \
  --output-wide "$MASTER_WIDE_OUTPUT"

python3 scripts/make_plots.py \
  --input "$MASTER_OUTPUT" \
  --output-dir "$PLOT_DIR"

echo "Frozen evaluation completed: $FROZEN_OUTPUT"
