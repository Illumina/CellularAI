# Evaluation

CellularAI ships one public evaluation harness, **`vcc2025`**, built from the Virtual Cell Challenge 2025 H1 hESC CRISPRi data. Its reference bundle is published as a release asset, so anyone can score predictions from CellularAI or any other method against it.

Evaluation has two independent operations:

- **Stage** converts the bundle's controls and target list into an ordinary prediction task.
- **Score** compares an existing `prediction.h5ad` with the bundle's cached reference summaries.

Neither operation rebuilds raw ground truth. Staging consumes the controls, provision, and perturbation list; scoring additionally reads the anchors and `anchors/reference_side.npz` directly.

## Get the reference bundle

Download `vcc2025.tar` from the [`vcc2025-harness-v1` release](https://github.com/Illumina/CellularAI/releases/tag/vcc2025-harness-v1) and unpack it next to the repository's evaluation configuration:

```bash
mkdir -p evaluation
curl -L -o vcc2025.tar https://github.com/Illumina/CellularAI/releases/download/vcc2025-harness-v1/vcc2025.tar
sha256sum vcc2025.tar          # compare with the value on the release page
tar -xf vcc2025.tar -C evaluation
```

This creates `evaluation/vcc2025/` with the five files described in [Input files](input-files.md#3-reference-evaluation-bundle). [`configs/evaluation.yaml`](../configs/evaluation.yaml) points at that directory and pins the MD5 of every file, so a damaged or modified download is refused before use. Its `root` is relative to the current working directory; run the commands below from the directory that contains `evaluation/`, or edit `root` to an absolute path.

### How the harness was built

The reference data come from the Arc Institute's [Virtual Cell Challenge 2025 dataset](https://console.cloud.google.com/marketplace/product/bigquery-public-data/arc-institute?project=gcp-public-data-arc-institute) (H1 hESC CRISPRi Perturb-seq). The training, validation, and test splits were pooled and processed as follows:

| Step | Rule |
| --- | --- |
| Perturbations | Every target with **at least 400 cells**: 255 targets. |
| Reference cells | Exactly **400 cells per target**, sampled without replacement with a fixed per-target seed. |
| Controls | All non-targeting cells, pooled across splits: 114,528 cells. |
| Genes | The dataset's 18,080-gene readout axis, unchanged. |
| Depth | Integer counts at a median of about 20,000 UMI per cell, obtained by binomial thinning of the raw counts, applied identically to controls and reference cells. |
| Control provision | A fixed seeded subset of **18,400 control cells**. These are the only controls a method sees (`stage` writes them), and scoring uses the same cells. |
| Baseline anchor | The mean-response baseline: an equal-weight average of the per-target mean count vectors, assigned to every target and scored as a submission. |
| Replicate anchor | The reference scored against itself: cells (perturbed and control) split into two disjoint halves, one half scored against the other, averaged over five seeded splits. |

Predictions are therefore expected to contain **400 cells per target**, the same cell count as the reference. Use of the data is subject to the Arc Institute's terms for the source dataset.

## Validate evaluation inputs

To check a bundle together with the normal summary and task inputs:

```bash
cellularai preflight \
  --config run/config.yaml \
  --task run/task.yaml \
  --evaluation-config configs/evaluation.yaml \
  --dataset vcc2025
```

Preflight verifies every configured MD5/SHA-256 entry and loads the control pool, provision, and perturbation list. Scoring separately validates the anchor formula version and cached reference side.

## Stage an evaluation task

```bash
cellularai evaluate stage \
  --dataset vcc2025 \
  --config configs/evaluation.yaml \
  --like run/task.yaml \
  --out "$PWD/run/staged/vcc2025"
```

An absolute staging output path is recommended. Staging writes:

```text
controls.h5ad
gene_names.csv
pert_counts.csv
task.yaml
```

The generated task has one context named after the public dataset, no catalog exclusions, and `call_depth`, `median_degs`, `control_label`, and the optional `cells_per_perturbation`, `seed`, and `extra` copied from `--like`. The output directory is recorded as its `data_root`.

Staging rejects control values that are not finite non-negative integers.

Run prediction with the generated task and the ordinary model configuration:

```bash
cellularai preflight \
  --config run/config.yaml \
  --task run/staged/vcc2025/task.yaml

cellularai run \
  --config run/config.yaml \
  --task run/staged/vcc2025/task.yaml \
  --out run/output-vcc2025
```

## Score a prediction

```bash
cellularai evaluate score \
  --dataset vcc2025 \
  --config configs/evaluation.yaml \
  --prediction run/output-vcc2025/prediction.h5ad \
  --name cellularai \
  --out run/output-vcc2025/score \
  --n-jobs 4
```

`--name` defaults to `prediction`; `--n-jobs` defaults to a CPU-derived worker count. Use `--n-jobs 1` for serial scoring. Parallel scoring requires Linux `fork` support.

The prediction must have:

- the evaluation gene axis in exactly the supplied order;
- exactly the supplied perturbation set in `obs['target_gene']`;
- the same number of cells (at least two) for every perturbation. The reference anchors were computed for 400-cell predictions; other counts are scored with a warning, and their scaled values are not directly comparable to 400-cell results;
- a sparse count matrix readable as float32.

## Score predictions from another method

The scorer does not depend on how a prediction was produced. To evaluate another method:

1. Stage the harness to get the inputs a method may use. `--like` only supplies task settings; a minimal file is enough:

   ```yaml
   # like.yaml
   data_root: .
   contexts: {}
   readout_genes: gene_names.csv
   perturbations: pert_counts.csv
   call_depth: 30
   median_degs: 8
   exclude_cell_lines: []
   control_label: non-targeting
   ```

   Then run `cellularai evaluate stage --dataset vcc2025 --config configs/evaluation.yaml --like like.yaml --out "$PWD/vcc2025_inputs"`. The method should use only `controls.h5ad` (the 18,400 provided control cells), `gene_names.csv`, and `pert_counts.csv`.

2. Write the method's output as an AnnData `.h5ad` with the gene axis from `gene_names.csv` in the same order, `obs['target_gene']` naming each cell's target, exactly 400 cells for every target in `pert_counts.csv`, and non-negative counts on the same scale as the provided controls. No control cells are needed in the prediction.

3. Score it with `cellularai evaluate score` as above, giving the method a label with `--name`.

Scoring writes `scores.json` and prints the scaled overall value. The JSON contains `method`, `dataset`, `raw`, `scaled`, `baseline`, `replicate`, and `mse_parts`.

Scoring can also be appended to a prediction pipeline with `cellularai run --score ... --evaluation-config ...`; see [Configuration](configuration.md#propagating-a-custom-evaluation-configuration).

## Metric meanings

Predicted cells are compared with the provided controls using normalized expression profiles and Wilcoxon differential-expression statistics. A gene's own perturbation target is excluded where applicable.

| Metric | High-level meaning | Raw direction |
| --- | --- | --- |
| `pds` | Profile discrimination: whether a perturbation's predicted response profile is closer to its own reference profile than to other reference profiles. | Higher is better |
| `expr_mse` | Noise-aware normalized squared error between predicted and reference expression profiles, including finite-sample variation corrections. | Lower is better |
| `fid` | Agreement of significant response directions with adjudicable reference responses. | Higher is better |
| `reach` | Fraction of the reference response budget recovered by a prediction ranking while maintaining at least 0.9 directional purity. | Higher is better |
| `jac` | Jaccard overlap between predicted and reference significant-gene sets. | Higher is better |
| `nmae` | Absolute log-fold-change error normalized by reference effect magnitude; reported only with sufficient reference support. | Lower is better |

Undefined per-perturbation values are omitted from their metric means. `raw.n_fid` and `raw.n_nmae` record support counts for the corresponding metrics. `mse_parts` exposes the components of `expr_mse` for diagnostics.

Scaled scores use the bundle's baseline and replicate anchors:

```text
scaled = (raw - baseline) / (replicate - baseline)
```

The scaled `expr_mse` is clamped to `[0, 1]`, scaled `nmae` is floored at `-6`, and other scaled metrics are unclamped. `scaled.overall` is the mean of finite scaled metric values. The anchor JSON's formula version must match the implementation.

## Compare score reports

Create aligned raw and scaled tables from one or more completed `scores.json` files:

```bash
cellularai evaluate report \
  cellularai=run/output-vcc2025/score/scores.json
```

Additional arms use space-separated `NAME=path/to/scores.json` arguments. Arm names are labels for the table; they do not alter scoring.

## Integrity behavior

Evaluation YAML may specify MD5 values, SHA-256 values, or both for any of the five logical files. Every declared digest must match before staging, preflight, or scoring proceeds. Input loading rejects empty or out-of-bounds control provisions; scoring also rejects missing or unreadable anchor/reference files and unsupported anchor versions. These checks establish file integrity and input validity; they do not change the metric calculation or recreate reference measurements.
