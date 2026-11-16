# Configuration

CellularAI uses explicit YAML files for model/data configuration, prediction tasks, and reference evaluations. There is no hidden environment-variable configuration layer. Output locations and optional execution overrides are CLI arguments.

## Run configuration YAML

Start by copying [`configs/example.yaml`](../configs/example.yaml) into a writable run directory. Its `model` mapping contains the complete released numerical configuration; every model key is required and unknown keys are rejected. Normally, change only `training_data.catalog` and, if necessary, `compute` settings.

```yaml
training_data:
  catalog: customer_bundle/catalog.json
compute:
  backend: local
  workers: 8
  # python: python                 # optional; defaults to the invoking interpreter
  # slurm:                         # required when backend is slurm
  #   build: "--partition=cpu --cpus-per-task=8"
  #   assemble: "--partition=cpu --cpus-per-task=2"
model:
  # Copy every value from configs/example.yaml.
  pooling: {}
  calls: {}
  plan: {}
  totals: {}
  placement: {}
  synthesis: {}
  guards: {}
```

The empty mappings above are illustrative only; they are **not** a runnable configuration. Use the full example file.

### Training data and path resolution

- `training_data.catalog` (required): path to the supplied bundle's `catalog.json`.
- A relative catalog path is resolved relative to the **run configuration file**, not the current working directory.
- Catalog payload paths must be relative to the catalog directory; absolute paths and `..` traversal are rejected.
- Moving a run directory and bundle together therefore does not change the configuration or task fingerprints.

### Compute settings

- `backend`: `local` or `slurm`; defaults to `local` if the entire `compute` mapping is omitted by lower-level tooling.
- `workers`: worker count for prediction; the CLI `--workers` value takes precedence, and `cellularai predict --n-jobs` overrides both for a single stage. If no value is given anywhere, prediction runs serially. Use a positive integer suited to available memory.
- `tmp`: optional temporary-file directory; defaults to `OUTPUT/tmp`. Choose a short writable path on systems with Unix-socket path-length limits.
- `python`: optional interpreter used for pipeline stage commands; defaults to the interpreter running `cellularai run`.
- `slurm.build`: options inserted into each context's `sbatch` command.
- `slurm.assemble`: options inserted into the assembly job's `sbatch` command, including any scoring stage requested for that pipeline.

No external packaging tool is needed by any stage; merging writes only h5ad predictions and JSON reports.

### Complete model schema

The following is the exhaustive key hierarchy. Values and lists should be copied from `configs/example.yaml`.

| Section | Required keys |
| --- | --- |
| `pooling` | `l2_row`, `l2_col`, `l2_mat`, `z_row`, `z_col`, `z_mat`, `noise`, `se`, `se_clip`, `kd_ref`, `kd_floor`, `sim`, `sim_hvg`, `correction`, `fill`, `one_stream`, `views`, `auxiliary_weight`, `pair_weight` |
| `calls` | `eligible_cpm`, `min_lfc`, `gamma`, `n_min`, `n_max`, `w_pow`, `phi`, `strength_z`, `strength_k`, `strength_c`, `strength_phi` |
| `plan` | `curve_slope`, `sim_tau`, `sim_ntop`, `sim_cap`, `plan_z_uniform`, `xsim_tau`, `xsim_strength`, `xsim_cap_factor`, `xsim_max_total`, `power_k`, `power_c`, `power_phi`, `power_depth`, `power_z_n2`, `power_t_pen`, `power_soft_z`, `shrink_quantile`, `shrink_factor` |
| `totals` | `credit_draws`, `credit_seed`, `amp_share`, `shape_gamma`, `shape_kz`, `mix`, `gate_z`, `pcs_cells`, `pcs_seed`, `cp_j0`, `cp_k`, `cp_lam`, `cp_top`, `pw_gamma`, `pw_k`, `pw_lam`, `pw_top`, `residual_rank`, `residual_ridge`, `residual_min_measured`, `residual_alpha`, `residual_perts`, `residual_scale_floor`, `residual_global`, `residual_proj_ridge`, `residual_clip`, `fixed_steps`, `fixed_rtol` |
| `placement` | `rule`, `frozen`, `evidence_z`, `rule_band`, `boost_band`, `boost`, `damp_band`, `damp`, `plan_floor_z`, `order_az`, `order_zsim`, `order_clip`, `lib_spread` |
| `placement.rule` and `placement.frozen` | `amp`, `h0`, `kz`, `phi`, `cap`, `cl2` |
| `synthesis` | `z_plan`, `z_nudge`, `rmax`, `n_bis`, `ztol`, `ztol_plan`, `ltol`, `ne_burst`, `planned_z`, `overshoot_z`, `repair_r`, `solve_ntop`, `solve_tol`, `solve_iters`, `shift_band`, `sparsify_step` |
| `guards` | `l2_abs_max`, `z_abs_max`, `v_floor_rel`, `sim_sd_min`, `sim_z_clip`, `nr_clip`, `log2_max`, `mt_ls_clip`, `order_sd_min`, `sim_share_max`, `sim_neff_min`, `max_unmeasured`, `max_missing_readout_frac`, `credit_kink`, `credit_band_lo`, `max_hit_rate` |
| `guards.sim_share_max` and `guards.sim_neff_min` | `plan`, `sim`, `xsim`, `pooling` |
| `guards.max_hit_rate` | `nr_clip`, `log2_max`; all other registered guards have a zero-hit limit |

`totals.fixed_steps` (released value 3) caps how many times the credit target is re-solved against freshly generated cells, and `totals.fixed_rtol` (released value 0.001) is the relative change at which that loop counts as converged. If the `converged` gate fails, for example with a non-default `cells_per_perturbation`, raising `fixed_steps` is the first thing to try; each extra step costs one more full synthesis pass.

The released pooling mode is `correction: LZ`, `fill: zf`, and `one_stream: true`; other modes are rejected. Duplicate YAML keys, missing model keys, and extra model keys fail validation.

## Task YAML

```yaml
data_root: task
contexts:
  context_A:
    controls: context_A_controls.h5ad
  context_B:
    controls: context_B_controls.h5ad
readout_genes: gene_names.csv
perturbations: pert_counts.csv
call_depth: 30
median_degs: 8
exclude_cell_lines: []
control_label: negative_control
cells_per_perturbation: 400      # optional; defaults to 400
seed: 0                          # optional; omit for the released synthesis streams
round: optional_identifier       # optional
extra: {}                        # optional metadata
```

Required keys are `data_root`, `contexts`, `readout_genes`, `perturbations`, `call_depth`, `median_degs`, `exclude_cell_lines`, and `control_label`. Only `cells_per_perturbation`, `seed`, `round`, and `extra` are optional; unknown keys are rejected.

- `data_root`: base directory for the task files. If relative, it is resolved relative to the **task YAML file**.
- `contexts`: non-empty mapping. Each value must be exactly `{controls: <path>}`. Context names may contain letters, digits, underscores, hyphens, and dots; other characters are rejected. Relative control paths are resolved from `data_root`.
- `readout_genes`: path to CSV with `gene_name`; relative to `data_root`.
- `perturbations`: path to CSV with `target_gene`; relative to `data_root`.
- `call_depth`: positive number. Assembly requires the median predicted significant-gene count per context to be no greater than this limit.
- `median_degs`: positive number used to calibrate the response-plan sizes.
- `exclude_cell_lines`: list of identifiers present in the supplied catalog, or `[]`. Unknown identifiers fail before computation.
- `control_label`: non-empty exact label removed from the target list. Names differing only by case are rejected as targets.
- `cells_per_perturbation`: optional integer, at least 2, defaulting to 400. The number of cells generated for every target in every context. When set, it is part of the task fingerprint; omitting it gives the same fingerprint as before the key existed. The released model values and the `vcc2025` anchors were calibrated at 400, so other values are supported but not validated for accuracy.
- `seed`: optional non-negative integer. Each perturbation's cells are drawn from a random stream seeded by its target name; when `seed` is set, it is combined with the name, so different seeds give independent replicate predictions from the same inputs. Omitting it reproduces the released streams exactly. (Setting `seed: 0` is a valid seed, distinct from omitting the key.) It is part of the task fingerprint when set and affects only cell synthesis, not pooling or the count design.
- `round`, `extra`: optional task metadata retained in the content fingerprint.

Gene and target names are read as literal strings. The target file may have any input order; prediction sorts targets. The output always contains exactly `cells_per_perturbation` cells (400 by default) for every retained target in every context.

## Local and Slurm execution

Local execution runs all contexts and assembly in order:

```bash
cellularai run \
  --config run/config.yaml \
  --task run/task.yaml \
  --out run/output \
  --scheduler local \
  --workers 8
```

`--scheduler` overrides `compute.backend`; `--workers` overrides `compute.workers`. Local stage processes set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to `1` to avoid nested numerical threading.

For Slurm:

```bash
cellularai run \
  --config run/config.yaml \
  --task run/task.yaml \
  --out run/output \
  --scheduler slurm
```

This **writes but does not submit** `run/output/jobs/launch.sh`, per-context scripts, and an assembly script. Review the scripts and scheduler options, then submit the generated dependency chain explicitly:

```bash
bash run/output/jobs/launch.sh
```

Each context job performs pooling and prediction. Assembly starts only after every context succeeds. No scheduler command is run automatically by `cellularai run --scheduler slurm`.

## Reference evaluation YAML

Evaluation uses a separate configuration passed as `--evaluation-config` or, for standalone evaluation commands, `--config`. The repository ships [`configs/evaluation.yaml`](../configs/evaluation.yaml) for the published `vcc2025` bundle; its schema is:

```yaml
datasets:
  vcc2025:
    root: evaluation/vcc2025
    files:
      controls: controls_pooled.npz
      perturbations: perturbation_list.json
      provision: anchors/control_provision.npy
      anchor_scores: anchors/anchor_scores.json
      reference_side: anchors/reference_side.npz
    md5:                           # optional; any subset of the five logical keys
      controls: <expected-md5>
      perturbations: <expected-md5>
      provision: <expected-md5>
      anchor_scores: <expected-md5>
      reference_side: <expected-md5>
    sha256:                        # optional; same logical keys
      controls: <expected-sha256>
```

File paths are resolved relative to the dataset's `root` unless absolute. A relative `root` is interpreted from the **current working directory**, so absolute roots are recommended.

MD5 and SHA-256 checks run whenever their entries are present, before evaluation inputs are loaded. This configuration is independent of the run/model YAML.

### Propagating a custom evaluation configuration

Validate evaluation inputs during preflight:

```bash
cellularai preflight \
  --config run/config.yaml \
  --task run/task.yaml \
  --evaluation-config run/evaluation.yaml \
  --dataset vcc2025
```

`--dataset` is required whenever `--evaluation-config` is supplied. Add optional scoring to the same pipeline:

```bash
cellularai run \
  --config run/config.yaml \
  --task run/task.yaml \
  --out run/output \
  --score vcc2025 \
  --evaluation-config run/evaluation.yaml
```

The pipeline resolves the evaluation-config path and passes it to scoring after assembly. The score is written to `run/output/score/scores.json`. Scoring is never triggered by the evaluation YAML alone.

## Individual stage commands

The equivalent manual sequence for one context is:

```bash
cellularai pool \
  --config run/config.yaml --task run/task.yaml \
  --context context_A \
  --out run/output/ctx_context_A/pooled_effects_context_A.npz

cellularai predict \
  --config run/config.yaml --task run/task.yaml \
  --context context_A \
  --pooling run/output/ctx_context_A/pooled_effects_context_A.npz \
  --out run/output/ctx_context_A/predicted_cells_context_A.h5ad \
  --n-jobs 8

cellularai assemble \
  --config run/config.yaml --task run/task.yaml \
  --builds run/output
```

Repeat pooling and prediction for every task context before assembly. Assembly expects the exact directory and filenames shown above.
