# Label-Efficient Image Classification with Vision Foundation Models

Deep Learning final-exam project comparing a self-supervised vision foundation
model (**DINOv2 ViT-S/14**) against a conventional supervised backbone
(**ResNet-50**) under limited-label ("k-shot") budgets, on Oxford-IIIT Pets and
EuroSAT.

**Research questions**
1. Do DINOv2 frozen features beat ResNet-50 frozen features when very few
   labels are available?
2. How much does fine-tuning improve over frozen features, and at what label
   budget does it start to pay off?
3. Does the conclusion hold cross-domain (satellite imagery, EuroSAT)?

Full experiment design, timeline, and role breakdown live in the team's
planning docs (`PROJECT_PLAN.md`, `WORKFLOW.md`) — kept local-only per team
convention, not part of this repo (see `.gitignore`).

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch, torchvision, scikit-learn, pandas, matplotlib.

## Pipeline overview

```
1. Download data      → src/download_data.py, src/download_eurosat.py
2. Generate splits     → src/sampler.py            (already run; see splits/)
3. Extract features    → src/features.py            (see note below)
4. Frozen evaluation    → scripts/run_frozen.sh      (kNN + Linear Probe)
5. Fine-tuning          → scripts/run_finetune.sh    (2-stage: freeze → unfreeze)
6. Master table + plots → scripts/build_master_table.py, scripts/make_plots.py
```

### 1–2. Data and splits

```bash
python src/download_data.py       # Oxford-IIIT Pets, 80/20 stratified trainval/test
python src/download_eurosat.py    # EuroSAT
```

`splits/{dataset}_seed{seed}_k{budget}.json` are deterministic, nested k-shot
index lists (k=1 ⊂ k=2 ⊂ ... ⊂ k=all) into each dataset's trainval split.
Already generated and committed (70 files: 2 datasets × 5 seeds × 7 budgets).

### 3. Feature extraction

`src/features.py` is a placeholder — feature caches for this run were produced
out-of-band and are not reproducible from this script yet (tracked as a TODO).
The expected output is one cache per `{backbone}_{dataset}`, in `features/`
(gitignored, not pushed — too large for git). Two layouts are supported:

- **Combined**: `{backbone}_{dataset}.pt` (or `.npz`), holding all of
  `train_features`/`train_labels`/`test_features`/`test_labels` in one file.
- **Split** (what this run actually uses): `{backbone}_{dataset}_trainval.pt`
  + `{backbone}_{dataset}_test.pt` (EuroSAT uses `_train.pt` instead of
  `_trainval.pt`), each holding a generic `features`/`labels` pair. This is
  the format Thu's cached features ship as; `src/evaluate_frozen.py` detects
  and loads either layout automatically.

Backbones: `resnet50` (torchvision, ImageNet-1K, 2048-d avgpool) and
`dinov2_vits14` (`torch.hub`, self-supervised, 384-d CLS token).

### 4. Frozen evaluation (kNN + Linear Probe)

```bash
bash scripts/run_frozen.sh
```

Runs the full grid — 2 backbones × 2 datasets × 7 label budgets × 5 seeds ×
2 methods (kNN, Linear Probe) = 280 runs — and writes:

- `results/frozen_grid_v2.csv` — raw per-run results
- `results/master_table_v2.csv` — long-format summary (mean ± std per config)
- `results/master_table_wide_v2.csv` — wide, presentation-ready table
- `plots/eval_v2/frozen_curves_{pets,eurosat}.png` — accuracy vs. label budget

The v2 protocol uses cosine kNN with `k=20` and **uniform** majority voting
(`python3 scripts/run_frozen_grid.py --help` for overrides, e.g.
`--knn-weights distance`).

⚠️ **Known limitation**: `knn_k=20` is fixed across all label budgets. At
`k=1`/`k=2` (only 37/74 Pets training samples), taking 20 uniform-weighted
neighbors out of a 37-sample pool washes out the signal — kNN accuracy there
sits near the random baseline. This is a protocol limitation to disclose on
the results slide, not a bug.

#### Legacy results

`results/frozen_grid.csv` / `results/master_table.csv` /
`plots/frozen_curves_*.png` are the original 280-run grid, produced with
**distance-weighted** kNN and a since-superseded feature-loading path. Kept
for provenance (see `results/KET_QUA_DOC_BAO.md`). Its kNN numbers at small k
run much higher than the v2 protocol's (e.g. ResNet-50/Pets/k=1: 75.7% legacy
vs. 2.7% v2) — this is the weighting-scheme difference explained above, not a
feature or correctness bug. Its Linear Probe numbers match the v2 run exactly,
confirming both were computed on the same underlying features.

### 5. Fine-tuning

```bash
python src/finetune.py --grid                       # full 24-run matrix
python src/finetune.py --backbone dinov2 --k 25 --seed 0   # single run
```

Two-stage per run: freeze backbone + train head, then unfreeze all and train
end-to-end at a lower backbone LR. Hyperparameters are tuned once (at k=10/25)
and locked in `config/finetune_recipe.yaml` — deliberately *not* re-tuned per
label budget, to avoid leaking validation-set information into the low-label
regime. Output: `results/finetune_grid.csv` (24 rows), TensorBoard logs under
`runs/`.

⚠️ **Known issue**: `results/finetune_grid.csv` currently has one duplicate
key (`resnet50/pets/k=5/seed=0`, logged twice with different accuracy:
78.28% vs 78.21%) — needs a rerun or manual resolution before it can be
merged into a combined master table.

### 6. Combined results

`scripts/build_master_table.py` can merge frozen + fine-tuning results into
one table once the duplicate above is resolved:

```bash
python3 scripts/build_master_table.py \
  --frozen results/frozen_grid_v2.csv \
  --finetune results/finetune_grid.csv \
  --output results/master_table_v2.csv \
  --output-wide results/master_table_wide_v2.csv
```

## Tests

```bash
python -m pytest tests/
```

## Repository layout

```
config/     dataset paths, label budgets, seeds, locked fine-tune hyperparameters
src/        sampler, feature extraction (TODO), frozen eval, fine-tuning, metrics
scripts/    CLI entry points (frozen grid, master table, plots, shell wrappers)
splits/     deterministic nested k-shot index files (70 files)
features/   cached backbone features (gitignored, generated locally)
results/    raw CSVs, master tables, benchmark/run logs
plots/      accuracy-vs-label-budget figures (legacy + eval_v2)
runs/       TensorBoard logs from fine-tuning
tests/      pytest suite for the evaluation pipeline
```
