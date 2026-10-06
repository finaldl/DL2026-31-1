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

Experiment design, timeline, role breakdown and the running progress/audit log
live in [`docs/`](docs/):

- [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) — experiment design, timeline, cut list, status tracker
- [`docs/project_analysis.md`](docs/project_analysis.md) — per-role task explainer and acceptance criteria
- [`docs/PROGRESS_CHECK.md`](docs/PROGRESS_CHECK.md) — audit log, open issues, who-does-what-next

## Status (as of 2026-10-01)

| Phase | State |
|---|---|
| Data download + nested k-shot splits (70 files) | ✅ Done, independently verified for Pets **and** EuroSAT (5 seeds each) |
| Feature extraction (`src/features.py`, 8 cached `.pt` files) | ✅ Done |
| Phase 1 — frozen grid (kNN + Linear Probe, 280 runs, v2 protocol) | ✅ Done |
| Phase 2 — fine-tuning grid (24 runs, Pets) | ✅ Done (clean rerun, no duplicate keys) |
| Combined master table (frozen + fine-tune, 64 summary rows) | ✅ Done |
| Fine-tune vs. frozen plot | ⏳ Not yet — `plots/eval_v2/` predates the fine-tune merge |
| Open decisions before slides | ⏳ kNN `knn_k` at tiny budgets; ResNet-50 weights mismatch (see [Known limitations](#known-limitations)) |
| Slides, reproducibility check, rehearsals | ⏳ Week 3 (experiment freeze: Tue 2026-10-06) |

## Headline results (Oxford-IIIT Pets, Top-1 %, mean ± std over seeds)

| Method | Backbone | k=1 | k=5 | k=10 | k=25 | k=all |
|---|---|---|---|---|---|---|
| Linear Probe | ResNet-50 | **76.3 ± 5.2** | 87.7 ± 0.6 | 89.8 ± 0.8 | 91.5 ± 0.2 | 93.4 ± 0.0 |
| Linear Probe | DINOv2 | 72.6 ± 4.0 | **88.3 ± 0.9** | **91.5 ± 0.2** | **93.9 ± 0.5** | **96.2 ± 0.0** |
| kNN (v2) | ResNet-50 | 2.7 ± 0.0 | 54.8 ± 2.8 | 85.4 ± 1.2 | 89.6 ± 0.2 | 92.4 ± 0.0 |
| kNN (v2) | DINOv2 | 3.2 ± 0.6 | 58.1 ± 2.9 | 82.7 ± 0.6 | 90.1 ± 0.6 | 94.4 ± 0.0 |
| Fine-tune | ResNet-50 | — | 79.4 ± 2.8 | 81.6 ± 0.7 | 86.8 ± 1.0 | 92.3 ± 0.6 |
| Fine-tune | DINOv2 | — | 81.9 ± 1.3 | 88.4 ± 0.7 | 91.3 ± 0.2 | 95.2 ± 0.4 |

Frozen: 5 seeds. Fine-tune: 3 seeds. Full tables (incl. k=2/50 and EuroSAT) in
`results/master_table_wide_v2.csv`.

Preliminary reading (still under audit — see `docs/PROGRESS_CHECK.md`):
- **RQ1**: ResNet-50 leads at k ≤ 2; DINOv2 overtakes from k = 5 and the gap
  widens with more labels (+2.8 pts at k=all, Linear Probe).
- **RQ2**: in this grid fine-tuning never beats a Linear Probe on the same
  backbone; the gap shrinks from ~6–8 pts at k=5 to ~1 pt at k=all. DINOv2
  frozen + LP beats fine-tuned ResNet-50 at every budget.
- **RQ3**: on EuroSAT the two backbones are within ~4 pts everywhere
  (ResNet-50 slightly ahead at k ≤ 10, tied from k=25; both ~95% at k=all).

## Setup

```bash
pip install -r requirements.txt
pip install pytest tensorboard   # for tests / fine-tuning logs (not in requirements.txt)
```

Requires Python 3.10+, PyTorch, torchvision, HuggingFace `datasets`,
scikit-learn, pandas, matplotlib. Fine-tuning requires a CUDA GPU.

## Pipeline overview

```
1. Download data       → src/download_data.py, src/download_eurosat.py
2. Generate splits     → src/sampler.py            (already run; see splits/)
3. Extract features    → src/features.py
4. Frozen evaluation   → scripts/run_frozen.sh      (kNN + Linear Probe)
5. Fine-tuning         → src/finetune.py --grid     (2-stage: freeze → unfreeze)
6. Master table + plots→ scripts/build_master_table.py, scripts/make_plots.py
```

### 1–2. Data and splits

```bash
python src/download_data.py       # Oxford-IIIT Pets (HF pcuenq/oxford-pets) → data/pets_{trainval,test}
python src/download_eurosat.py    # EuroSAT (HF tanganke/eurosat)          → data/eurosat_{train,test}
```

- **Pets**: the HF `train` split is re-split 80/20, stratified, `seed=42` →
  5,912 trainval / 1,478 test images (~160 / ~40 per class). Note this is
  *not* the official Pets test split.
- **EuroSAT**: the HF `train`/`test` split as-is → 21,600 / 2,700 images.

`splits/{dataset}_seed{seed}_k{budget}.json` are deterministic, nested k-shot
index lists (k=1 ⊂ k=2 ⊂ ... ⊂ k=all) into each dataset's train(val) split.
Already generated and committed (70 files: 2 datasets × 5 seeds × 7 budgets).

### 3. Feature extraction

```bash
python src/features.py
```

Reads the four `data/*` folders above and writes eight caches to `features/`
(gitignored — too large for git):

```
features/{resnet50,dinov2}_{pets_trainval,pets_test,eurosat_train,eurosat_test}.pt
```

each a dict `{"features": Tensor[N, D], "labels": Tensor[N]}`. Preprocessing:
`Resize(256) → CenterCrop(224) → Normalize(ImageNet)`, no augmentation.

Backbones: `resnet50` (torchvision **`IMAGENET1K_V2`**, 2048-d avgpool) and
`dinov2_vits14` (`torch.hub`, self-supervised, 384-d CLS token).

`src/evaluate_frozen.py` accepts both this split layout and a combined
`{backbone}_{dataset}.pt`/`.npz` file holding
`train_features`/`train_labels`/`test_features`/`test_labels`.

### 4. Frozen evaluation (kNN + Linear Probe)

```bash
bash scripts/run_frozen.sh
```

Runs the full grid — 2 backbones × 2 datasets × 7 label budgets × 5 seeds ×
2 methods (kNN, Linear Probe) = 280 runs, ~2 min on CPU — and writes:

- `results/frozen_grid_v2.csv` — raw per-run results
- `results/master_table_v2.csv` — long-format summary (mean ± std per config)
- `results/master_table_wide_v2.csv` — wide, presentation-ready table
- `plots/eval_v2/frozen_curves_{pets,eurosat}.png` — accuracy vs. label budget

Note: `run_frozen.sh` builds the master table from frozen results only. To
include fine-tuning, re-run step 6 afterwards.

The v2 protocol uses cosine kNN with `k=20` and **uniform** majority voting;
Linear Probe is `LogisticRegression(C=1.0, lbfgs, max_iter=1000)` on
L2-normalised features (`python3 scripts/run_frozen_grid.py --help` for
overrides, e.g. `--knn-k`, `--knn-weights distance`).

#### Legacy results

`results/frozen_grid.csv` / `results/master_table.csv` /
`results/master_table_long.csv` / `plots/frozen_curves_*.png` /
`results/KET_QUA_DOC_BAO.md` are the original 280-run grid (2026-09-26),
produced with **distance-weighted** kNN. Kept for provenance. Its kNN numbers
at small k are much higher than v2's (e.g. ResNet-50/Pets/k=1: 75.7% legacy vs.
2.7% v2) — this is the weighting scheme, not a feature or correctness bug. Its
Linear Probe numbers match v2 exactly, confirming both used the same features.

### 5. Fine-tuning

```bash
python src/finetune.py --grid                             # full 24-run matrix
python src/finetune.py --backbone dinov2 --k 25 --seed 0  # single run
```

Run from the repo root (paths `data/`, `config/`, `results/`, `runs/` are
relative). Pets only; 2 backbones × k ∈ {5, 10, 25, all} × seeds {0, 1, 2}.

Two-stage per run: freeze backbone + train a new linear head (10 epochs), then
unfreeze all and train end-to-end (30 epochs) with a lower backbone LR, AdamW,
AMP. Train-time augmentation: `RandomResizedCrop(224) → HFlip →
ColorJitter(0.2)`. Hyperparameters are locked in
`config/finetune_recipe.yaml` (identical across label budgets, to avoid
leaking validation information into the low-label regime):

| | lr_head | lr_backbone | weight_decay | batch | epochs |
|---|---|---|---|---|---|
| ResNet-50 | 1e-3 | 1e-4 | 0.01 | 32 | 10 + 30 |
| DINOv2 ViT-S/14 | 1e-3 | 1e-5 | 0.05 | 32 | 10 + 30 |

Output: one row appended per run to `results/finetune_grid.csv` (delete it
first for a clean rerun), TensorBoard loss curves under `runs/`. The k-shot
indices are regenerated at runtime with `src/sampler.make_kshot_splits` rather
than read from `splits/`. Memory benchmark (RTX 4060, AMP):
`results/finetune_benchmark.log`.

⚠️ `scripts/run_finetune.sh` is currently broken (prints "completed" before
running, and calls `finetune.py` from the wrong directory). Use the commands
above until it is fixed.

### 6. Combined master table + plots

```bash
python3 scripts/build_master_table.py \
  --frozen results/frozen_grid_v2.csv \
  --finetune results/finetune_grid.csv \
  --output results/master_table_v2.csv \
  --output-wide results/master_table_wide_v2.csv
python3 scripts/make_plots.py   # re-draws plots/eval_v2/ from master_table_v2.csv
```

The committed `master_table_v2.csv` already contains frozen (56) + fine-tune (8)
= 64 summary rows. The committed plots were drawn before the fine-tune merge;
re-running `make_plots.py` adds the fine-tune curves to the Pets plot.

## Known limitations

- **kNN at k=1/k=2**: `knn_k=20` is fixed across budgets. With 37/74 Pets (or
  10/20 EuroSAT) training samples, a 20-neighbour uniform vote washes out the
  signal — kNN sits near chance (EuroSAT k=1/2 is a constant 11.74%, because
  `n_train ≤ 20` means every test image votes over the whole pool). Pending
  decision: scale `knn_k` with `n_train`, or disclose on the results slide.
- **ResNet-50 weights differ between phases**: frozen features use
  `IMAGENET1K_V2`, fine-tuning uses `IMAGENET1K_V1` (the plan specifies V1).
  The ResNet frozen-vs-fine-tune comparison (RQ2) is confounded until aligned.
- **Fine-tune recipe** equals the plan's initial candidates; there is no
  validation split in `finetune.py`, so no tuning at k=10/25 is recorded.
- **Pets test set** is a random stratified 20% of the HF train split, not the
  official test split; Pets breeds also overlap with ImageNet-1K classes,
  which favours the supervised ResNet-50 at very low k.

## Tests

```bash
python -m pytest tests/
```

10 tests covering the frozen-evaluation pipeline and master-table builder.

## Repository layout

```
config/     dataset paths, label budgets, seeds, locked fine-tune recipe
docs/       project plan, task explainer, progress/audit log
src/        data download, sampler, feature extraction, frozen eval, fine-tuning, metrics
scripts/    CLI entry points (frozen grid, master table, plots, shell wrappers)
splits/     deterministic nested k-shot index files (70 files)
features/   cached backbone features (gitignored, generated locally)
results/    raw CSVs, master tables, benchmark/run logs
plots/      accuracy-vs-label-budget figures (legacy + eval_v2)
runs/       TensorBoard logs from fine-tuning (24 runs)
tests/      pytest suite for the evaluation pipeline
```
