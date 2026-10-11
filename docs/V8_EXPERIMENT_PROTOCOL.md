# V8 Q1 Experimental Protocol

## Primary study

- Common per-user split: 80% train, 10% validation, 10% locked test unless a paper-specific reproduction is explicitly labelled.
- Multiple outer split seeds and multiple model seeds.
- Main latent dimensions: `K = {10,20,50}`.
- MBMF-F/C/N: three angle initializations (`legacy_uniform_2pi`, `canonical_uniform`, `isotropic_gaussian`).
- Validation selects K/init/hyperparameters. Test is never used for selection.
- Selected checkpoints are permanently archived.

## Dataset matrix

- Synthetic
- Jester
- Book-Crossing explicit ratings
- MovieLens 100K
- MovieLens 1M
- MovieLens 10M

The downloader only obtains raw data. Parsing, filtering, ID mapping and splitting live in the V8 infrastructure module.

## Internal methods

- BiasMF / MF
- MBMF-F / MBMF-C / MBMF-N
- AutoRec (reimplementation)
- AEMC-NE (reimplementation)
- Proposed Methods 1–7 from the frozen V7 scientific core

## External/author implementations

- BMC-ADMM
- Deep-MMMF
- DRPRA

Export the exact V8 split, execute author code, and return `split,userId,movieId,prediction`. V8 recomputes all common metrics. Literature-reported numbers must not be mixed into the same statistical test as common-split predictions.

## Metrics

### Accuracy/tail behavior
MSE, RMSE, MAE, MedianAE, P90AE, P95AE, P99AE, range-normalized MAE/RMSE, ordinal-uniform NMAE when applicable, Pearson, Spearman, mean signed error, prediction mean/std.

### Bounds
Global OOB rate, mean OOB violation, maximum OOB violation, individual-bound violation rate, mean/max individual-bound violation and mean bound width when an analytical structural bound is available.

### Ranking
Held-out-explicit NDCG, Precision, Recall, F1, MAP and MRR at configured K values. The primary F1 uses a train-derived per-user relevance threshold. Literal MBMF threshold reproduction is labelled reproduction-only.

### Uncertainty
For uncertainty-enabled Method 7: Gaussian NLL, PICP at 50/80/90/95%, coverage gap, MPIW and normalized MPIW.

### Efficiency/stability
Training wall time, inference time, peak process RSS, peak GPU allocated/reserved memory, parameter count, prediction std across independent seeds, validation prediction std across K.

## Statistics

- Paired Wilcoxon signed-rank with Holm correction for matched common-protocol blocks.
- Friedman across matched dataset/split blocks.
- Nemenyi critical-difference average ranks.
- V7 paired-user and hierarchical bootstrap outputs can be retained as complementary uncertainty analyses.

## Ablations

V8 contains 38 unique mechanism-level jobs. The duplicated V7 `m1_raw` and `m2_derived_consistency` jobs are removed. Key replacements include Method-1 no-AE/no-MBMF and Method-2 user-view-only/item-view-only.

Run ablations under the same split and validation-selection policy. Do not select an ablation based on test performance.
