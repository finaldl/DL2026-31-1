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

## Status (as of 2026-10-06 — experiment freeze)

| Phase | State |
|---|---|
| Data download + nested k-shot splits (70 files) | ✅ Done, independently verified for Pets **and** EuroSAT (5 seeds each) |
| Feature extraction (`src/features.py`, 8 cached `.pt` files) | ✅ Done — ResNet-50 re-extracted with `IMAGENET1K_V1` on 2026-10-06 to match fine-tuning |
| Phase 1 — frozen grid (kNN + Linear Probe, 280 runs, v2 protocol) | ✅ Done (re-run 2026-10-06 with V1 ResNet features; kNN re-run with `knn_k=5` on 2026-10-07) |
| Phase 2 — fine-tuning grid (24 runs, Pets) | ✅ Done (clean rerun, no duplicate keys) |
| Combined master table (frozen + fine-tune, 64 summary rows) | ✅ Done |
| Plots | ✅ `plots/eval_v2_knn5/frozen_curves_{pets,eurosat}.png` (all methods) + `plots/finetune_vs_frozen.png` (Linear Probe vs. fine-tune, Pets, slide-ready) |
| Fine-tuning audit | ✅ Loss curves; found a Stage-2 head bug (official numbers kept, read as lower bound); `run_finetune.sh` fixed. Bug-fixed grid (`src/finetune_fixed.py`) running 2026-10-07 as a post-freeze supplementary result |
| Experiment freeze (2026-10-06) | 🔒 Closed. ResNet-50 aligned to V1; kNN uses `knn_k=5` (`knn_k=20` kept as a reference run). No new runs |
| Slides, reproducibility check, rehearsals | ⏳ 2026-10-08 → 2026-10-11 |

## Headline results (Oxford-IIIT Pets, Top-1 %, mean ± std over seeds)

| Method | Backbone | k=1 | k=5 | k=10 | k=25 | k=all |
|---|---|---|---|---|---|---|
| Linear Probe | ResNet-50 | 71.2 ± 3.1 | 85.6 ± 0.6 | 89.2 ± 1.0 | 92.4 ± 0.3 | 94.3 ± 0.0 |
| Linear Probe | DINOv2 | **72.6 ± 4.0** | **88.3 ± 0.9** | **91.5 ± 0.2** | **93.9 ± 0.5** | **96.2 ± 0.0** |
| kNN (`knn_k=5`) | ResNet-50 | **14.9 ± 1.3** | **83.9 ± 1.0** | 87.2 ± 1.3 | 90.0 ± 0.5 | 93.1 ± 0.0 |
| kNN (`knn_k=5`) | DINOv2 | 13.8 ± 1.0 | 83.7 ± 0.9 | **88.1 ± 0.8** | **92.0 ± 0.7** | **94.7 ± 0.0** |
| Fine-tune | ResNet-50 | — | 79.4 ± 2.8 | 81.6 ± 0.7 | 86.8 ± 1.0 | 92.3 ± 0.6 |
| Fine-tune | DINOv2 | — | 81.9 ± 1.3 | 88.4 ± 0.7 | 91.3 ± 0.2 | 95.2 ± 0.4 |

Frozen: 5 seeds. Fine-tune: 3 seeds. Both phases use ResNet-50 `IMAGENET1K_V1`.
**Source of truth for slides: `results/master_table_wide_v2_knn5.csv`** (all
methods, incl. k=2/50 and EuroSAT). Bold = better backbone within a method.

Reading of the results:
- **RQ1**: DINOv2 frozen features beat ResNet-50 at **every** label budget on
  Pets (Linear Probe, +1.4 to +2.7 pts).
- **RQ2**: with the fixed recipe, fine-tuning never beats a Linear Probe on the
  same backbone. The gap shrinks as labels grow: ResNet-50 −6.1 / −7.5 / −5.6 /
  −1.9 pts, DINOv2 −6.4 / −3.1 / −2.6 / −1.0 pts at k = 5 / 10 / 25 / all.
  Because of the Stage-2 head bug (see limitations) these fine-tune numbers
  are a lower bound.
  DINOv2 frozen + LP beats fine-tuned ResNet-50 at every budget.
- **RQ3**: on EuroSAT the curves cross. ResNet-50 leads at k ≤ 10 (e.g. 70.2 vs
  64.5 at k=2); DINOv2 leads from k=25 on (+1.6 to +1.9 pts; 95.2 vs 93.3 at k=all).
- **kNN** tells a weaker version of RQ1: the backbones are tied at k ≤ 5 and
  DINOv2 leads from k=10 (Pets). Linear Probe is the primary frozen method.

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
Committed as fixed artifacts (70 files: 2 datasets × 5 seeds × 7 budgets);
verify them with `python3 scripts/verify_splits.py`.

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

Backbones: `resnet50` (torchvision **`IMAGENET1K_V1`**, same weights as fine-tuning, 2048-d avgpool) and
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

These defaults use `knn_k=20` (reference run). The official kNN numbers use
`knn_k=5`; reproduce them with:

```bash
rm -f results/frozen_grid_v2_knn5.csv     # the runner skips run_ids already in the output file
python3 scripts/run_frozen_grid.py --knn-k 5 --output results/frozen_grid_v2_knn5.csv
python3 scripts/build_master_table.py --frozen results/frozen_grid_v2_knn5.csv \
  --finetune results/finetune_grid.csv \
  --output results/master_table_v2_knn5.csv --output-wide results/master_table_wide_v2_knn5.csv
python3 scripts/make_plots_knn5.py   # official plots/eval_v2_knn5/ (defaults: master_table_v2_knn5.csv)
```

⚠️ `run_frozen_grid.py` appends to `--output` and skips any `run_id` already in
it; `run_id` hashes the configuration, not the features. Delete the output file
before re-running on new features, otherwise nothing is recomputed (`new=0`).

Note: `run_frozen.sh` builds the master table from frozen results only. To
include fine-tuning, re-run step 6 afterwards.

The v2 protocol uses cosine kNN with **uniform** majority voting (`knn_k=5` official, `knn_k=20` reference);
Linear Probe is `LogisticRegression(C=1.0, lbfgs, max_iter=1000)` on
L2-normalised features (`python3 scripts/run_frozen_grid.py --help` for
overrides, e.g. `--knn-k`, `--knn-weights distance`).

#### Legacy results

`results/frozen_grid.csv` / `results/master_table.csv` /
`results/master_table_long.csv` / `plots/frozen_curves_*.png` /
`results/KET_QUA_DOC_BAO.md` are the original 280-run grid (2026-09-26),
produced with **distance-weighted** kNN and ResNet-50 **`IMAGENET1K_V2`**
features. Kept for provenance only — do not quote it on slides. Its kNN numbers
at small k are much higher than v2's (e.g. DINOv2/Pets/k=1: 72.6% legacy vs.
3.2% v2) — this is the weighting scheme, not a correctness bug. DINOv2 Linear
Probe numbers match v2 (to within one test image), confirming the same DINOv2
features; ResNet-50 numbers differ because v2 now uses V1 weights.

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
than read from `splits/` — see [Known limitations](#known-limitations). Memory
benchmark (RTX 4060, AMP): `results/finetune_benchmark.log`. One-command
reproduction (deletes the old CSV first, ~3 h on an RTX 4060):

```bash
bash scripts/run_finetune.sh
```

Fine-tuning audit (Zam, 2026-10-07): training-loss figure
`plots/finetune_loss_curves.png` (`python scripts/plot_finetune_losses.py`,
needs `tensorboard`); findings are summarised in `docs/PROGRESS_CHECK.md`. The audit found
a bug in `src/finetune.py` (see limitations). `src/finetune.py` is kept unchanged
because it produced the official numbers; `src/finetune_fixed.py` is the same
pipeline with the bugs fixed (head trained in Stage 2, reads `splits/*.json`,
seeds everything, BatchNorm frozen in Stage 1). It writes to
`results/finetune_grid_fixed.csv` / `runs_fixed/`. The Managers approved one
post-freeze run of it (2026-10-07) as a **supplementary** result for a backup
slide; the official fine-tune numbers stay those of `src/finetune.py`.

### 6. Combined master table + plots

```bash
python3 scripts/build_master_table.py \
  --frozen results/frozen_grid_v2.csv \
  --finetune results/finetune_grid.csv \
  --output results/master_table_v2.csv \
  --output-wide results/master_table_wide_v2.csv
python3 scripts/make_plots.py   # re-draws plots/eval_v2/ from master_table_v2.csv
python3 scripts/plot_finetune_vs_frozen.py   # plots/finetune_vs_frozen.png (Pets, LP vs. fine-tune)
python3 scripts/plot_finetune_losses.py       # plots/finetune_loss_curves.png (from runs/ TensorBoard logs)
```

The committed `master_table_v2.csv` already contains frozen (56) + fine-tune (8)
= 64 summary rows, and the committed Pets plots already include the fine-tune
curves. For slides use `plots/finetune_vs_frozen.png`: Linear Probe (solid) vs.
fine-tune (dashed), one colour per backbone, k ∈ {5, 10, 25, all}. Its y-axis
starts at 75%.

## Known limitations

- **kNN at very low budgets**: with `knn_k=20` and only 37/74 Pets (10/20
  EuroSAT) training images, the uniform vote collapses to near chance (3% on
  Pets k=1; a constant 11.74% on EuroSAT k=1/2, where every test image votes
  over the whole pool). `knn_k=5` was chosen to fit the smallest training pools
  — not tuned on test accuracy — and fixes k=2/k=5 (e.g. Pets ResNet-50 k=2:
  9.8% → 59.1%). At k=1 (one image per class) a 5-neighbour vote is still
  noisy (~14% on Pets). The `knn_k=20` run stays in `results/frozen_grid_v2.csv`
  for comparison.
- **Linear Probe is not bit-exact across machines**: re-running on different
  hardware can move a result by one test image (≤ 0.07 pts on Pets). Compare
  reproductions with a ~0.1-pt tolerance.
- **Fine-tuning bug — the head is not trained in Stage 2**: `build_model()`
  returns the head parameters as a generator; the Stage-1 optimizer consumes
  it, so the Stage-2 head parameter group is empty (PyTorch does not raise).
  During Stage 2 only the backbone trains, toward a head frozen at its
  10-epoch Stage-1 values. Found in the post-freeze audit; the official
  fine-tune numbers are kept and should be read as a **lower bound**. Fixed in
  `src/finetune_fixed.py`; its results are reported separately as a
  post-freeze, bug-fixed supplement.
- **Fine-tune and frozen use different k-shot images at k ≤ 25**:
  `finetune.py` re-samples at runtime with `src/sampler.py`, while the frozen
  grid reads `splits/*.json`; the two overlap only ~3% at k=5 (chance level).
  Both are valid stratified draws from trainval with no test leakage, so
  compare means, not paired seeds. k=all is identical.
- **`splits/*.json` are fixed artifacts**: they were committed with the
  evaluation pipeline (`87ed900`) and cannot be regenerated from
  `src/sampler.py`. Their validity is checked instead:
  `python3 scripts/verify_splits.py` (nested, k images per class, no duplicates,
  k=all = range(N) so no test indices; add `--features-dir features` to also
  check per-class counts against the cached labels).
- **Fine-tune recipe** equals the plan's initial candidates; there is no
  validation split in `finetune.py`, so no tuning at k=10/25 is recorded.
- **Pets test set** is a random stratified 20% of the HF train split, not the
  official test split; Pets breeds also overlap with ImageNet-1K classes,
  which may favour the supervised ResNet-50.

## Tests

```bash
python -m pytest tests/
```

10 tests covering the frozen-evaluation pipeline and master-table builder.

## Repository layout

```
config/     dataset paths, label budgets, seeds, locked fine-tune recipe
docs/       progress/audit log (plan and task explainer are local-only)
src/        data download, sampler, feature extraction, frozen eval, fine-tuning, metrics
scripts/    CLI entry points (frozen grid, master table, plots, shell wrappers)
splits/     deterministic nested k-shot index files (70 files)
features/   cached backbone features (gitignored, generated locally)
results/    raw CSVs, master tables, benchmark/run logs
plots/      eval_v2_knn5/ = official, eval_v2/ = knn_k=20 reference, finetune_vs_frozen.png, top level = legacy
runs/       TensorBoard logs from fine-tuning (24 runs)
tests/      pytest suite for the evaluation pipeline
```
