# Frozen-feature evaluation

The evaluation code lives in the repository root rather than in a nested project:

- `src/evaluate_frozen.py` and `src/metrics.py` contain reusable evaluation logic.
- `scripts/run_frozen_grid.py` runs kNN and linear-probe experiments.
- `scripts/build_master_table.py` normalizes raw result schemas and aggregates seeds.
- `scripts/make_plots.py` produces accuracy-versus-label-budget figures.
- `splits/` contains the deterministic split manifests used by the legacy run.

## Run

Place the four feature caches in `features/` using the pattern
`{backbone}_{dataset}.pt` or `.npz`, then run:

```bash
bash scripts/run_frozen.sh
```

The script writes new artifacts to `results/*_v2.csv` and `plots/eval_v2/`; it does
not overwrite the committed legacy results. The v2 protocol uses cosine kNN with
`k=20` and uniform majority voting. Use `python3 scripts/run_frozen_grid.py --help`
for overrides.

## Legacy results

`results/frozen_grid.csv` contains the original 280-run grid produced with
distance-weighted kNN. It is retained for provenance and is not interchangeable
with the v2 majority-vote protocol. See `results/KET_QUA_DOC_BAO.md` for the audit
status and known limitations.
