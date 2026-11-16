# Outputs and quality gates

For `--out run/output`, a complete run has this layout:

```text
run/output/
├── ctx_<context>/
│   ├── pooled_effects_<context>.npz
│   └── predicted_cells_<context>.h5ad
├── prediction.h5ad
├── gates.json
├── jobs/                         # Slurm backend only
│   ├── ctx_<context>.sh
│   ├── assemble.sh
│   ├── launch.sh
│   └── *_%j.log                  # created by Slurm after submission
└── score/
    └── scores.json               # only when --score is requested
```

There is one `ctx_<context>` directory per task context.

## Per-context files

### `pooled_effects_<context>.npz`

Contains pooled evidence aligned to the task's ordered genes and sorted perturbations:

- `l2`, `z`: pooled log2-fold-change and significance evidence used by the design;
- `l2_tot`, `z_tot`: the corresponding totals-design evidence;
- `genes`, `perts`: axis labels;
- `guard_names`, `guard_counts`: numerical-guard tally entries;
- `unmeasured`: targets without measured summary rows.

Prediction validates the saved axes before using this archive.

### `predicted_cells_<context>.h5ad`

An AnnData object with a sparse float32 `X` matrix of non-negative integer-valued counts. Rows are cells and columns exactly follow the task gene list. It contains:

- `obs['context']`: the single context label;
- `obs['target_gene']`: exactly `cells_per_perturbation` rows (400 by default) for each task perturbation;
- observation names of the form `<context>_<target>_<index>`;
- `uns['build']`: convergence, credit, unmeasured-target, readout-coverage, guard, and provenance reports.

Float32 is a storage type; values are checked to be exact integers below `2^24`.

## Merged outputs

Assembly reads every per-context prediction and writes `gates.json` before deciding whether to continue. If any gate fails, it stops before merging. Use a fresh output directory: assembly refuses pre-existing `prediction.h5ad` rather than leaving stale final artifacts. Candidate files are published only after all gates succeed.

- `prediction.h5ad`: accepted per-context predictions concatenated along cells, retaining the common gene axis and build metadata.
- `gates.json`: per-context quality-gate values and the list of failed checks (see below).

JSON build reports and optional evaluation scores are separate outputs; they do not change the prediction matrix.

File checksums are for transfer/integrity checking, not proof that two independently written HDF5 files are scientifically equivalent.

## Quality gates

Assembly requires every boolean check for every context to pass. `gates.json` contains a top-level `failed` list plus detailed values and human-readable `failures` for diagnosis.

### Prediction contract

- Gene names and order match the task.
- The context label matches exactly.
- The perturbation set matches the task exactly.
- Every perturbation has exactly the task's `cells_per_perturbation` cells (400 by default).
- `X` is CSR sparse float32, and all stored counts are finite, non-negative integers below `2^24`.
- The median number of predicted significant genes, excluding each perturbation's own target, does not exceed `call_depth`.

### Build and numerical validity

- The credit-margin search reached its target and the fixed-point iteration converged.
- Realized credit share lies within the configured lower band and upper kink limit.
- Unmeasured perturbation count does not exceed `max_unmeasured` (zero in the released configuration).
- Missing-readout fraction does not exceed `max_missing_readout_frac`; missing genes and name-only near matches are reported.
- Every registered numerical guard is present, and no stale guard-limit key exists.
- Guard hit counts stay within configured rates. Guards cover effect/score bounds, variance floors, clipping, search caps, similarity-weight concentration, effective sample size, and related numerical safeguards.

Similarity-share guards discourage one summary source from dominating a weighted pool; effective-count guards discourage excessively concentrated weights. Many guards also clip or floor values during computation so downstream operations remain finite, but assembly still rejects excessive guard activity.

### Provenance

The configuration and task fingerprints stored during prediction must equal those recomputed at assembly. This prevents mixing per-context files built from different model/catalog content or different task inputs. Source package version and a path-independent source hash are also recorded in each build report.

## Slurm job files

With `--scheduler slurm`, the pipeline only generates scripts. Context scripts contain pooling followed by prediction; `assemble.sh` contains assembly and optional scoring. `launch.sh` submits independent context jobs and makes assembly dependent on all of them succeeding. The `%j` in log names is expanded by Slurm to the job ID after explicit submission.

## Evaluation output

When requested through `cellularai run --score`, scoring occurs after successful assembly and merging. `score/scores.json` contains raw metrics, anchor-scaled metrics, baseline/replicate anchors, and expression-error components. See [Evaluation](evaluation.md).
