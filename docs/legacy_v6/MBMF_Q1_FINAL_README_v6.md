# MBMF / Thesis Q1 Final Package — v6

This is the final audited revision of the experiment pipeline built from:

- the uploaded MBMF reference paper;
- the uploaded thesis chapter;
- the original thesis notebook (`model2_model1`);
- the independently revised notebook (`mbmf_two_models_complete`).

## Included model families

### Paper baselines
1. `MBMF-F`
2. `MBMF-C`
3. `MBMF-N`

Each baseline can be run with three angle initializations:
- legacy Uniform(0, 2π);
- canonical angular ranges;
- isotropic Gaussian direction -> hyperspherical angles.

### Proposed / thesis models
4. `Thesis-Method-1-LatentMagnitude`
5. `Thesis-Method-2-DualViewFusion`
6. `Proposed-Method-3-PriorResidualMagnitude`

Method 3 is the revised version of Model-2 from `mbmf_two_models_complete`.

## Method 3

Let `r0_u, r0_i` be train-only historical priors. Two AE branches produce
bottlenecks `z_u, z_i`, and zero-initialized scalar heads learn log-radius
corrections:

    delta_u = T(g_u(z_u))
    delta_i = T(g_i(z_i))

    r_u = r0_u exp(delta_u)
    r_i = r0_i exp(delta_i)

Then:

    W_u = SphericalToCartesian(phi_u, r_u)
    H_i = SphericalToCartesian(theta_i, r_i)
    p_ui = W_u^T H_i

The Q1-safe default uses:
- centered raw rating space;
- rating mask supplied to encoders;
- Eq.42-style shrinkage for the prior;
- smooth bounded log-radius correction;
- separate user/item AE losses;
- MSE losses.

A separate preset reproduces the other-chat Model-2 choices as closely as possible:
- `[0,1]` model space;
- values-only input;
- local mean+std magnitude prior;
- hard clamp of log correction to [-6,6];
- averaged AE reconstruction;
- Frobenius losses;
- alpha=beta=0.5;
- lr=5e-4;
- grad clip=5.

Known invalid behavior from the other notebook is intentionally NOT restored:
held-out labels never enter encoders and predictions are never clipped for scoring.

## Important corrections in v6

- Correct K-dimensional hyperspherical mapping with K-1 angles.
- Explicit K=3 regression test catches the old `cos[k-1]` indexing error.
- Random Cartesian -> spherical -> Cartesian round-trip test.
- Held-out-label invariance test for Methods 1, 2, and 3.
- Train/validation/test separation; validation only for checkpoint/config selection.
- Deep-copied checkpoints.
- No post-hoc prediction clamp in primary evaluation.
- Raw OOB diagnostics are retained.
- `movieId` is the item key.
- Observed-mask construction is independent of rating value.
- Explicit documented rating scale is required for publication runs.
- MBMF-C uses the current TRAIN factorizing-matrix minimum in Eq.40.
- MBMF-C feasibility modification is explicit and logged.
- MBMF-N nonnegative projection is represented with a straight-through gradient treatment.
- Paper objective/stopping uses squared Frobenius SSE scale.
- Method-2's old missing-square loss bug is corrected.
- Method-3's zero-initialized residual heads exactly recover historical radii at initialization.
- Synthetic test split now *actually* guarantees user/item training support rather than relying on chance.
- Split seed and model seed are separate.
- Hyperparameter/configuration selection uses validation RMSE only.
- Publication-facing output is `selected_by_validation_summary.csv`.

## Verification performed

The final `.py` file:
- passes Python compilation;
- passes hyperspherical geometry tests;
- passes held-out-label invariance tests;
- passes finite-loss/finite-gradient smoke tests for F/C/N + Methods 1/2/3;
- passes structural magnitude-bound checks;
- passes Method-3 zero-residual prior identity;
- was compared numerically to the supplied other-chat Model-2 at initialization under
  the corrected legacy-reproduction configuration:
  priors, AE reconstructions and MBMF predictions agree to floating-point tolerance.

Synthetic executions are engineering tests only and are NOT scientific results.

## Publication warning

MBMF-C/N in the common comparison use TRAIN observations as the historical source.
This is leakage-safe and equation-faithful, but it is a common-protocol adaptation,
not an exact reproduction of the paper's separate historical/present experiment design.
If the paper submission makes an "exact paper reproduction" claim, add a separate
historical/present protocol experiment.

The F1 metric in the common pipeline is explicitly named
`F1_trainThreshold_*`: the per-user threshold is derived from TRAIN ratings only.
This is deliberately leakage-safe; it should not be described as an exact reproduction
of the paper's somewhat underspecified F1 threshold protocol.
