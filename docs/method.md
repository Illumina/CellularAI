# Method

CellularAI is a context-specific prediction method for single-cell perturbation response. It starts from **precomputed training summaries**, not raw training measurements. A new task supplies only control cells, the desired readout genes, and target perturbations. The output is a synthetic count matrix with a fixed number of cells per target per context: 400 by default, set by the task's `cells_per_perturbation`.

This guide describes the numerical workflow without requiring access to any customer data.

## 1. Align and center summary evidence

For each selected summary row, CellularAI aligns effects and two-sided probabilities to the task gene/target axes. Effects represent log2 fold changes. Probabilities are converted to signed normal scores using the effect sign:

```text
signed score = sign(effect) × magnitude(normal quantile of probability)
```

Per-gene effect and signed-score centers from the offset archives are subtracted where finite estimates exist. Missing measurements remain distinguishable from measured zero effects. Control-expression sums and response-variance estimates supply expression and uncertainty features.

## 2. Pool effects with random effects and context similarity

Pooling combines summary rows rather than treating them as identically precise:

1. Estimate between-row variance per gene with a DerSimonian–Laird random-effects calculation.
2. Combine it with each row's response variance to form inverse-variance weights, with fallbacks for missing estimates and a relative variance floor.
3. Multiply row/gene weights by configured target and whole-row factors. These include uncertainty, estimated target knockdown, optional scale information, related-record down-weighting, and auxiliary-row weighting.
4. Weight summary rows by similarity between their control-expression profiles and the new context's controls. Pooling uses rank correlation over highly variable control genes; later planning uses log-expression profile correlation. Similarity scores are standardized and bounded, planning weights are capped, and each stage checks concentration and effective row count.

Centered log2 effects are combined by weighted means. Signed evidence is combined with a weighted Stouffer/Liptak-style calculation: squared weights control the denominator, and only measured signed scores contribute. Numerical guards record clipping, variance floors, concentration, and effective-number-of-rows diagnostics.

The result is a gene-by-target effect matrix and accompanying significance matrix for the context.

## 3. Plan response genes and effect sizes

Control cells establish each gene's CPM distribution, expression eligibility, and median sequencing depth. For every perturbation, the method:

- excludes the perturbation's own target gene from response calls;
- ranks eligible genes using pooled effect magnitude, significance, and expected detection power;
- chooses the number of planned calls from training-summary response breadth, similarity-weighted breadth, and the task's `median_degs` calibration;
- assigns effect directions and priorities, with smaller nudges outside the planned set.

Additional placement rules convert evidence into target mean-expression shifts. Detection limits, expression depth, significance bands, and planned-call priorities influence these targets. `call_depth` is separately enforced as a final assembly limit on the median realized significant-gene count.

## 4. Apply control-conditioned residual correction

Pooled effects can miss systematic context-dependent structure. CellularAI therefore fits a residual correction entirely from the supplied summaries:

1. Form an inverse-variance baseline across the residual summary records.
2. Derive a low-rank response basis with randomized SVD, replacing own-target entries by off-target summaries during fitting.
3. Standardize control logCPM features and add a small set of global control-expression components.
4. For each sufficiently measured gene, ridge-regress residual low-rank coordinates on those control features.
5. Predict a correction from the new context's summed controls and map it back through the response basis.

The correction is shrunk, ratio-bounded, and prevented from changing an effect's sign. Own-target entries and unsupported genes remain protected. Agreement between the original and corrected effects controls their final blend. The released rank, ridge penalties, support thresholds, and bounds are in `configs/example.yaml`.

## 5. Design integer count totals

The method converts log2 effects into perturbation-specific gene fractions relative to the control pool. It shapes these profiles using significance, the residual-adjusted effects, and control-expression principal components.

A population-level scale is not chosen arbitrarily. Seeded control resamples estimate a delete-one jackknife variation “credit.” A numerical solve adjusts profile spread toward expected credit, synthesizes cells, recomputes their realized variation, and repeats to a fixed point (at most `totals.fixed_steps` passes, converged when the target moves less than `totals.fixed_rtol`). This keeps between-perturbation expression variation consistent with the control-derived target.

For each perturbation, normalized gene fractions are converted to integer gene totals by largest-remainder apportionment. Their sum matches the designed total count depth.

## 6. Synthesize individual cells

For each target, exactly `cells_per_perturbation` cells (default 400) are generated. The same count sets the size of the control resamples used for the variation credit in step 5:

- Target library sizes follow smoothed control-library quantiles at the designed total depth.
- For eligible genes, stratified control-CPM quantiles initialize counts. Bisection adjusts shifts toward target Wilcoxon differential-expression scores using the same control order statistics as evaluation.
- A Gaussian copula links gene ranks to library-size ranks; a bounded correlation search moves mean per-cell CPM toward its placement target while preserving score tolerances.
- Overshooting or directionally inconsistent unplanned genes receive repair or neutral treatment.
- Genes outside the eligible set receive their integer totals in small bursts.
- Target-specific deterministic random seeds make the synthesis reproducible for a fixed numerical environment. An optional task `seed` is combined with each target's seed to draw independent replicates.

The final sparse matrix stores exact non-negative integer counts as float32 values below `2^24`.

## 7. Gate and merge

Assembly independently rechecks the context, gene/target axes, per-target cell counts, integer counts, realized call depth, unmeasured targets, readout coverage, convergence, credit interval, provenance, and numerical-guard rates. Any failed check stops the build before merging.

Accepted context matrices are merged into `prediction.h5ad`. A separate conversion step with the VCC utilities is optional and is not part of model fitting or cell synthesis.

For implementation details, see the public modules `cellularai.pooling`, `cellularai.prediction`, and `cellularai.assemble`. For file contracts, see [Input files](input-files.md) and [Outputs](outputs.md).
