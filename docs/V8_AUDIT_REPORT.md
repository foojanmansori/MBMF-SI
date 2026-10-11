# V8 Infrastructure-Only Audit Report

## Scientific immutability

The user-supplied V7 scientific core was copied **byte-for-byte** into V8.

- Original V7 SHA-256: `6da726cb155ab6ad291f5f608febc37f42d1c25ac3924910f56917908c1f124c`
- V8 bundled scientific-core SHA-256: `6da726cb155ab6ad291f5f608febc37f42d1c25ac3924910f56917908c1f124c`
- Binary comparison: identical.

`SCIENTIFIC_CORE_MANIFEST_V8.json` also stores AST-source hashes for the hyperspherical map, MBMF, Methods 1–7 and the core trainers. V8 does not assign to or monkey-patch any `core.*` model/training symbol.

## Ten requested fixes

1. **Ablation duplicates removed.** `m1_raw` and `m2_derived_consistency` were duplicates of full V7 settings. V8 replaces them with mechanism-revealing `m1_no_ae`, `m1_no_mbmf`, `m2_user_view_only`, and `m2_item_view_only`. The suite now contains 38 unique configurations and raises if a future duplicate is introduced.
2. **Six datasets are first-class.** Synthetic, Jester, Book-Crossing, ML-100K, ML-1M and ML-10M have canonical loaders, explicit documented rating ranges, ID mappings and common split creation.
3. **Comparison methods restored.** AutoRec and AEMC-NE are available as explicit reimplementations. BMC-ADMM, Deep-MMMF and DRPRA use exact-split external-author prediction adapters rather than misleading approximate implementations.
4. **Train/validation/test reporting restored.** Candidate selection remains validation-only; after selection, the locked checkpoint is evaluated on train, validation and test with the same evaluator.
5. **Persistent save/load restored.** Selected models are copied to `selected_models/`. Checkpoints include family/config, frozen-core SHA, training-data signature, state dict and training metadata. Loading verifies data identity by default.
6. **Metric/stability/efficiency coverage expanded.** RMSE, MAE, MSE, MedianAE, P90/P95/P99AE, NMAE/NRMSE, Pearson/Spearman, signed bias, OOB rate/mean/max violation, individual-bound metrics, ranking, uncertainty, training time, inference time, peak CPU/GPU memory, parameter count, seed stability and validation K-stability are produced where applicable.
7. **Statistical analysis restored.** Wilcoxon-Holm and Friedman/Nemenyi average-rank/CD outputs are supported in addition to the V7 bootstrap analyses.
8. **Publication figures restored as PNG + PDF.** Accuracy, OOB/bounds, runtime/memory, NDCG, train-vs-test, K/stability, training curves, accuracy-efficiency, heatmaps and CD figures use the shared plotting layer.
9. **Main K grid restored to 10/20/50.** V8 defaults to `(10, 20, 50)`; other K values belong in explicit ablations.
10. **Notebook/timestamp UX fixed.** V8 notebooks have unique explicit cell IDs and `RUN_*` switches. Partial timestamps are never zero-filled; time features are enabled only when the complete training split has valid timestamps.

## Sparse execution note

For matrices too large for dense history tensors, V8 uses infrastructure wrappers. The frozen model equations and optimizer step semantics are preserved:

- Methods 1–4: sparse first-layer algebra equivalent to the dense value+mask input; observed-edge objectives are accumulated without changing the objective.
- Methods 5–6: the frozen sparse-history model is used; V8 edge-chunks the exact loss terms and performs one optimizer step per epoch, matching full-batch objective semantics.
- Method 7: the frozen edge-minibatch trainer is used unchanged.
- Paper MBMF: the same dynamic-step objective is evaluated in chunks.

## Actual executed tests

### Core/infrastructure self-test

```text
[v8-test] frozen v7 core SHA256 unchanged: PASS
[self-test] hyperspherical geometry: PASS
[self-test] construction + forward held-out label invariance Methods 1-7: PASS
[smoke-test-v7] PASS: Methods 1-7 finite, leakage-safe; Methods 4/6/7 range-safe
[v8-test] dense/sparse Methods 1-4 prediction equivalence: PASS
[v8-test] one-step optimizer/update equivalence Methods 1-6: PASS
[v8-test] checkpoint round-trip all internal scientific families: PASS
[v8-test] ablation configurations unique: PASS (38 unique jobs)
[v8-test] publication infrastructure: PASS
```

### Pytest

```text
pytest -q
22 passed, 2 deselected

pytest -q -m integration
2 passed, 22 deselected
```


### Additional no-scientific-change regression

V8 also executes a one-optimizer-step numerical equivalence regression. Methods 1–3 compare the frozen dense V7 training path with the sparse algebraic wrappers; Methods 4–6 compare the frozen full-objective update with the chunked infrastructure update. Maximum parameter difference after one Adam step is required to stay within numerical tolerance; the final self-test reports PASS. This is an infrastructure translation guard, not a new training algorithm.

### Strict locked-test mini study

12 model families were run on a synthetic MovieLens-like dataset with a validation-locked test. Candidate table test columns: **none**. All selected train/validation/test RMSE values were finite and every selected checkpoint existed.

Representative smoke results (not scientific results):

| Model | Train RMSE | Val RMSE | Test RMSE | Test OOB |
|---|---:|---:|---:|---:|
| BiasMF | 0.8168 | 0.8575 | 0.9394 | 0.0000 |
| MBMF-F | 0.8013 | 0.9391 | 1.2029 | 0.0000 |
| MBMF-C | 0.7697 | 0.9340 | 1.3034 | 0.0000 |
| MBMF-N | 2.2444 | 1.6923 | 1.9114 | 0.4211 |
| Method1 | 0.8363 | 0.8501 | 0.9495 | 0.0000 |
| Method2 | 0.8801 | 0.8761 | 1.0530 | 0.0000 |
| Method3 | 1.2549 | 1.3373 | 1.2869 | 0.0000 |
| Method4 | 1.2566 | 1.3393 | 1.2885 | 0.0000 |
| Method5 | 0.8182 | 0.8944 | 0.9911 | 0.0000 |
| Method6 | 0.8172 | 0.8740 | 0.9615 | 0.0000 |
| Method7 | 0.8650 | 0.9779 | 0.9964 | 0.0000 |

These values only confirm end-to-end execution; they must not be cited as research results.


### Comparison-method/adapter smoke

AutoRec and AEMC-NE were each trained for one epoch on a tiny synthetic dataset and produced finite test RMSE. The external-author adapter was fed exact synthetic ground-truth predictions in the required `split,userId,movieId,prediction` format and returned test RMSE `0.0`. These are execution checks only.

### Sparse backend smoke

Methods 1–7 were forced onto the sparse backend. All seven produced finite test RMSE. Individual-bound violations were zero in the smoke run.

### Ablation smoke

All **38 unique** V8 ablation tags were executed at least once on a tiny synthetic integration dataset with one training epoch; all 38 produced finite test RMSE.

### Figure generation

The smoke study generated **23 PNG and 23 PDF** figures. PDF output is vector-format publication output.

### Notebook execution

`mbmf_q1_final_v8.ipynb` was executed from first to last cell with `RUN_FULL=False`, `RUN_ABLATION=False`, and self-tests enabled. It completed successfully. The V7 missing-cell-ID warning is eliminated in the V8 notebook.

## Warning retained intentionally

PyTorch currently emits a warning that sparse CSR tensor support is beta. This is a library-status warning, not a failed test. The sparse/dense equivalence regression tests are retained to detect numerical drift.
