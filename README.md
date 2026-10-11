# MBMF Q1 Final Package — Version 8

Version 8 is **infrastructure-only**. The scientific file `mbmf_q1_final_v7.py` is copied byte-for-byte from the user-supplied V7 bundle and remains the frozen definition of MBMF-F/C/N and Methods 1–7.

Frozen scientific-core SHA-256:

`6da726cb155ab6ad291f5f608febc37f42d1c25ac3924910f56917908c1f124c`

## What V8 adds

- First-class loaders for Synthetic, Jester, Book-Crossing, ML-100K, ML-1M and ML-10M.
- Dense/sparse execution infrastructure; Methods 1–4 have numerically equivalent sparse-input wrappers, while Methods 5–7 already use sparse histories/edges.
- Numerical regression additionally verifies one Adam/update step equivalence for Methods 1–6; Method 7 uses the frozen V7 trainer unchanged.
- Strict validation-only configuration selection and locked test evaluation.
- Train, validation and test metrics for selected checkpoints.
- Persistent selected-model save/load with data-signature and scientific-core SHA checks.
- AutoRec and AEMC-NE reimplementations; official/external prediction adapters for BMC-ADMM, Deep-MMMF and DRPRA.
- Extended accuracy, tail-error, correlation, OOB, individual-bound, ranking, uncertainty, runtime, inference-time and memory metrics.
- Prediction stability across independent seeds and validation prediction stability across K.
- Wilcoxon-Holm and Friedman/Nemenyi/critical-difference analysis.
- PNG + vector PDF figures.
- 38 unique mechanism-level ablations. The two duplicated V7 ablations were removed.
- Partial timestamps are disabled rather than silently zero-filled.
- Default main K grid is `(10, 20, 50)`.
- Jupyter notebooks use explicit unique cell IDs and top-level run switches.

## Files

- `mbmf_q1_final_v7.py` — **frozen scientific core; do not edit**.
- `mbmf_q1_infrastructure_v8.py` — V8 infrastructure.
- `mbmf_q1_final_v8.ipynb` — guided experiment notebook.
- `download_datasets_q1_v8.ipynb` — raw-dataset downloader only.
- `run_v8.py` — CLI runner.
- `SCIENTIFIC_CORE_MANIFEST_V8.json` — core/file + symbol hashes.
- `docs/V8_AUDIT_REPORT.md` — detailed audit and test report.
- `docs/V8_EXPERIMENT_PROTOCOL.md` — publication protocol.

## Quick start

```bash
pip install -r requirements.txt
python run_v8.py --self-test-only
```

Download raw data by running `download_datasets_q1_v8.ipynb`.

Quick pilot:

```bash
python run_v8.py --dataset ml-1m --quick --output ./results_q1_v8/ml1m_quick
```

Full ML-1M:

```bash
python run_v8.py \
  --dataset ml-1m \
  --data-root ./datasets \
  --output ./results_q1_v8/ml-1m \
  --split-seeds 2026,2027,2028,2029,2030 \
  --model-seeds 11,22,33,44,55 \
  --k 10,20,50
```

All six datasets:

```bash
python run_v8.py --dataset all --data-root ./datasets --output ./results_q1_v8/all
```

Ablations:

```bash
python run_v8.py --dataset ml-1m --data-root ./datasets --output ./results_q1_v8/ml1m_ablation --ablation
```

## Publication rule

Do not select K, initialization or any hyperparameter using test metrics. V8 candidate tables contain train + validation metrics but no test columns. Test metrics are computed only after validation selection. Selected checkpoints are then permanently copied to `selected_models/`.

## Infrastructure bugfix audit (V8.0.1)

This redistributed V8 package retains the **byte-identical** V7 scientific core,
original V8 38-ablation plan, original datasets/method registry and hyperparameter
scan grid. Fixes are limited to evaluation, benchmark origin adaptation,
preprocessing/export integrity, statistical reporting, memory safety, and strict
validation-only hyperparameter search. See
`docs/V8_0_1_INFRASTRUCTURE_FIX_AUDIT_FA.md` and the machine-generated
`PROTOCOL_IMMUTABILITY_REPORT_8.0.1.json`.

**Important:** for a signed rating scale (e.g. Jester), MBMF-C/N run an explicitly
labelled *fixed-origin adapter*, preserving the core Eq.42 calculation in a
nonnegative coordinate system; predictions are returned to original rating units.
The raw MBMF-C/N formulation cannot directly consume negative observations.

`--hyper-scan` is now **validation-only**, with
`hyperparameter_scan_validation_runs.csv` as the aggregate output. It never
computes test metrics or changes the parameter grid. After freezing your final
selection, use `run_v8.py` to conduct the independent locked-test study.

To validate final result artifact completeness:

```bash
python audit_results_v8.py --study ./results_q1_v8/ml-1m \
  --report ./results_q1_v8/ml-1m/Q1_COMPLETENESS_AUDIT.json
```

External author code and real-dataset Q1 study results are **not included**;
this ZIP is tested infrastructure, not an already-completed publication result.
