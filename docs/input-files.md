# Input files

CellularAI consumes three logical input groups: a precomputed **training summary bundle**, an ordinary **prediction task**, and, optionally, the public **reference evaluation bundle**. Customer-specific catalogs enumerate the actual summary rows and access-controlled identifiers; this guide describes only the public file contracts.

Raw training-data preprocessing and summary generation are not included. Reproduction starts from the supplied summary bundle, not from raw training measurements.

## 1. Training summary bundle

The bundle root contains:

```text
catalog.json
SHA256SUMS
summaries/
offsets/
variance/
```

`training_data.catalog` in the run configuration points to `catalog.json`. Paths in the catalog are required to be relative to its directory; absolute paths and `..` traversal are rejected. The logical layout below is conventional; the catalog's `rows`, `files`, and `offset` values are authoritative for the supplied bundle.

### Catalog

`catalog.json` has this structure:

```json
{
  "primary_view": "<view>",
  "variance": "variance/primary.npz",
  "orders": {
    "effect_files": ["<catalog identifier>", "..."],
    "control_files": ["<catalog identifier>", "..."],
    "residual": ["<catalog identifier>", "..."]
  },
  "rows": [
    {
      "cell_line": "<catalog identifier>",
      "view": "<view>",
      "role": "primary",
      "files": {
        "effect": "summaries/<view>/<cell_line>/effect.parquet",
        "probability": "summaries/<view>/<cell_line>/probability.parquet",
        "scale": "summaries/<view>/<cell_line>/scale.parquet",
        "controls": "summaries/<view>/<cell_line>/controls.parquet"
      },
      "offset": "offsets/<view>/<cell_line>.npz",
      "pair_group": null,
      "pair_alternate": false
    }
  ]
}
```

A row's `role` is `primary` or `auxiliary`. `pair_group` may be `null` or a catalog-specific grouping label used to down-weight related records. `pair_alternate` marks an alternate measurement whose weight is reduced even if its base record is absent. The three `orders` arrays fix accumulation and residual input order; preserve them exactly. Configured primary views are selected in `model.pooling.views`, while auxiliary rows are selected by role.

### Summary Parquet files

The per-row files are:

| Logical path | Required content and use |
| --- | --- |
| `summaries/<view>/<cell_line>/effect.parquet` | Effect estimates, interpreted as log2 fold changes. Required for selected pooling rows and primary generation inputs. |
| `summaries/<view>/<cell_line>/probability.parquet` | Two-sided probabilities paired with effects. Required wherever its effect table is used for signed scores. |
| `summaries/<view>/<cell_line>/scale.parquet` | Positive per-target scale values. Required only when the configured `sef` factor uses that row; missing or absent scale information falls back to neutral weights where the implementation permits it. |
| `summaries/<view>/<cell_line>/controls.parquet` | Control-expression sums. Required for primary rows used by `orders.control_files` and by residual summaries; auxiliary controls are needed only if referenced by a workflow. |

Effect, probability, and scale tables use neutral columns:

- `feature_id`: feature/gene identifier.
- `target__<target>`: one column per summarized perturbation target.

They contain no cell-level observations. Effect-like tables align by `feature_id`; duplicate features are collapsed by the implementation's maximum-value rule. Unmeasured entries may be absent or non-finite.

Control tables instead contain:

- `feature_id`: feature/gene identifier.
- `control_sum`: non-negative summed control expression for that feature.

### Offset and variance NPZ files

Each selected summary row has an offset archive, conventionally:

```text
offsets/<view>/<cell_line>.npz
```

It contains the arrays `cell_line`, `view`, `genes`, `effect_center`, `effect_count`, `signed_score_center`, and `response_variance`. These arrays provide per-gene centering and uncertainty information; they are not raw observations.

The catalog-level variance archive is conventionally:

```text
variance/primary.npz
```

It contains `cell_lines` and, for each primary identifier, `genes__<identifier>` and `variance__<identifier>`. It supplies response-variance estimates for pooling weights and must cover the primary lines used by that calculation.

### SHA256SUMS

`SHA256SUMS` is a plain-text manifest in standard form:

```text
<64-character-sha256>  <path-relative-to-bundle-root>
```

Every catalog-referenced payload must be listed. `cellularai preflight` hashes every listed file, rejects missing files, modified content, and paths escaping the bundle, and checks that no referenced payload is omitted from the manifest. Keep the manifest and catalog together when moving a bundle.

## 2. Prediction task inputs

A task YAML references these files beneath `data_root`:

| File | Required columns/content |
| --- | --- |
| `gene_names.csv` | A `gene_name` column. File order defines the exact output feature axis; names must be non-empty and unique. |
| `pert_counts.csv` | A `target_gene` column. The exact `control_label` row is removed; remaining targets must be unique and at least three are required. Targets are sorted for prediction. |
| One control `.h5ad` per context | Cells by genes in `X`, with `var` index names exactly matching `gene_names.csv` in the same order. |

Control `X` may be dense, CSR, or CSC. Values must be finite, non-negative integers below `2^24`. Each context needs at least two control cells, and no control cell may have an all-zero count vector. These are context control cells, not raw training data.

See [Configuration](configuration.md#task-yaml) for the complete task YAML, including `call_depth`, `median_degs`, and exclusions.

## 3. Reference evaluation bundle

The `vcc2025` evaluation bundle, published as a release asset (see [Evaluation](evaluation.md#get-the-reference-bundle)), has exactly five logical input files:

```text
controls_pooled.npz
perturbation_list.json
anchors/control_provision.npy
anchors/anchor_scores.json
anchors/reference_side.npz
```

- `controls_pooled.npz` contains `counts` and `genes`.
- `perturbation_list.json` is an ordered JSON list of target names.
- `anchors/control_provision.npy` contains row indices selecting the provided controls from the pooled control matrix.
- `anchors/anchor_scores.json` contains the supported formula version and baseline/replicate metric anchors.
- `anchors/reference_side.npz` contains cached reference profiles, differential-expression tables, and jackknife moments used for scoring.

The evaluation YAML maps the logical keys `controls`, `perturbations`, `provision`, `anchor_scores`, and `reference_side` to these files; filenames are not hard-coded. Staging consumes the first three; scoring consumes all five. Optional MD5 and SHA-256 entries authenticate the configured release. Scoring uses the cached reference side directly and does not rebuild raw ground truth.

See [Evaluation](evaluation.md) and [Configuration](configuration.md#reference-evaluation-yaml) for commands and schema.
