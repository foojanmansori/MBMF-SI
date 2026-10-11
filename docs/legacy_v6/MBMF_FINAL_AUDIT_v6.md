# Final audit notes — v6

## Additional issues found and fixed in the last pass

1. The synthetic smoke-test comment claimed train support was guaranteed, but a shuffled
   80-pair prefix only guaranteed it probabilistically. The synthetic split is now
   constructed deterministically so every user and item has train support.

2. MBMF-C Eq.40 now uses `min(V)` from the current TRAIN factorizing matrix, rather than
   silently substituting the externally documented rating minimum.

3. The paper-style convergence check now reports/stops on the full squared Frobenius
   objective; `0.5*objective` is used only for gradient scaling.

4. The bounded scalar counterpart-radius initialization was moved away from saturation.

5. Temporal splitting no longer silently degrades into a random split when timestamps
   are unavailable.

6. Output metadata no longer hard-codes the target space.

7. Corrected-reproduction presets are separated:
   - original thesis Methods 1/2;
   - other-chat Model-2 -> Method 3.

## Method 3 provenance

Method 3 is not claimed to be in the uploaded thesis chapter. It is a new model derived
from the supplied `mbmf_two_models_complete.ipynb`.

The unsafe Model-1 in that notebook (`C2S(h) @ C2S(W)^T`) remains excluded because an
ordinary dot product of spherical-coordinate tuples is not the Cartesian Euclidean inner
product and therefore does not inherit the MBMF magnitude-bound argument.
