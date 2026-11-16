# Reproducibility

Reproduction uses the supplied **summary bundle**, task inputs, configuration, and public source. Raw training-data preprocessing is outside this repository and is not needed to rerun prediction.

## Content-derived, path-independent identities

Each prediction records provenance in `predicted_cells_<context>.h5ad` under `uns['build']['provenance']`.

### Configuration fingerprint

The configuration fingerprint is SHA-256 over canonical JSON containing:

- the complete parsed `catalog.json` content;
- the complete `model` mapping.

The catalog's installation location and all `compute` settings are excluded. Moving the same bundle and configuration to another directory therefore preserves the identity; changing summary references, model values, or catalog order changes it. Keep catalog identifiers unchanged for numerical reproduction: some generation inputs intentionally use lexical identifier order in addition to the explicit file orders.

### Task fingerprint

The task fingerprint hashes canonical task settings together with:

- the ordered readout gene names and sorted target names;
- SHA-256 of the actual gene and perturbation CSV bytes;
- SHA-256 of every context control `.h5ad`, keyed by context.

`data_root` and the configured input pathnames are not themselves identities. Relevant task settings, including exclusions, calibration values, control label, and optional metadata, are included.

### Code identity

Provenance also includes the package version and a SHA-256 source-tree identity computed from sorted package-relative Python filenames and their bytes. Installation directory names do not enter this hash.

Assembly recomputes the expected configuration and task fingerprints and rejects context files built with different identities.

## Input integrity

- `cellularai preflight` verifies every entry in the summary bundle's `SHA256SUMS`, confines manifest paths to the bundle, and requires every catalog-referenced payload to be listed.
- Evaluation YAML can declare MD5 and/or SHA-256 digests for the five logical evaluation inputs. Every declared digest is checked before staging or scoring.

Outputs consist only of h5ad predictions and JSON reports. Checksums detect changed bytes; they do not replace scientific review of inputs or outputs.

## Numerical reproducibility

For fixed inputs, code, dependency versions, thread settings, and numerical libraries, the method uses explicit catalog orders, stable target sorting, seeded control draws, target-derived synthesis seeds (optionally combined with the task `seed`), and deterministic worker result ordering. These choices reduce avoidable run-to-run variation.

**Bit-exact predictions are nevertheless environment- and CPU-sensitive.** BLAS/LAPACK implementations, vectorized instructions, floating-point accumulation, SciPy/NumPy/Numba versions, and worker execution choices can change low-level rounding and downstream integer decisions. Pin the full software environment and use comparable CPU/thread settings when exact numerical reproduction is required. Worker count alone is not a portability guarantee.

The local pipeline sets common BLAS/OpenMP thread counts to one for its child stages. Independent stage invocations should use equivalent settings when comparing results.

## HDF5 and archive checksums are not scientific equivalence

Two scientifically equivalent predictions need not have identical file checksums. HDF5 metadata, object layout, compression, and library versions can change bytes without changing the stored count matrix or labels. Archive formats can likewise differ in metadata or serialization.

Compare the scientific payloads when that is the goal: genes, targets, context labels, sparse matrix structure and values, build settings, and metric results. Use whole-file MD5/SHA-256 only to verify that a particular artifact was transferred unchanged.

## Multiprocessing requirements

Parallel paths are Linux-specific:

- Prediction with `--n-jobs > 1` uses the Python `forkserver` start method. Workers receive the engine once and share its large control order-statistics array through a read-only memory map. Independent stage invocations prefer `/dev/shm`; the provided pipeline instead uses `compute.tmp` (default `OUTPUT/tmp`) so runtime artifacts stay in the configured data location. Ensure sufficient temporary capacity, RAM, and a short path for worker sockets.
- Evaluation scoring with `--n-jobs > 1` uses `fork` so workers can share control statistics copy-on-write.

Run these paths through the provided `cellularai`/`python -m` entry points, which protect process startup with the normal main-module convention. Platforms without the required start methods should use `--n-jobs 1`. Nested numerical threading should remain disabled for predictable resource use.

## Reproduction checklist

1. Preserve the original summary bundle, including `catalog.json` and `SHA256SUMS`.
2. Preserve the task files and run/model YAML without editing content or row order.
3. Record source version, dependency lock/environment, OS, CPU, and worker settings.
4. Run `cellularai preflight` before prediction.
5. Confirm `gates.json` has an empty `failed` list.
6. Compare configuration/task/source identities before comparing numerical outputs.
7. Use file checksums only for archive integrity, not as the sole test of equivalent predictions.
