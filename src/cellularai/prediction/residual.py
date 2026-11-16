"""The totals' profile effects: pooled-evidence l2 corrected by record control-conditioned response residuals
(correction 'ratio', sign 'preserve'), then weighted toward the uncorrected l2
by its agreement with the totals' own l2.

Fit on the record lines only: their T x G x P effects (own targets replaced by the row's off-diagonal mean) give a
low-rank response basis around their inverse-variance mean; each gene's residual coordinates are ridge-regressed
on its records' standardized control logCPM plus 5 global control scores. Predict reads this context's summed controls.
"""

from __future__ import annotations

import numpy as np

from ..pooling.catalog import inverse_variance_weights


def ivw_baseline(stack) -> np.ndarray:
    """Zero-filled inverse-variance mean over lines, gene-median variance where a line measures nothing."""
    filled = np.where(np.isfinite(stack), stack, 0.0)
    w3 = inverse_variance_weights(stack)[:, :, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w3 * filled).sum(axis=0) / w3.sum(axis=0)


def _rms(coordinates):
    scale = np.sqrt(np.mean(coordinates**2, axis=1))
    return np.where(scale > 0, scale, 1.0)


def response_basis(matrix, rows, columns, rank, seed=0):
    """(U, intercept, centred) of a G x P matrix: own-target entries -> row off-diagonal mean, randomized SVD."""
    training = np.asarray(matrix, dtype=np.float64).copy()
    training[rows, columns] = 0.0
    count = np.full(training.shape[0], training.shape[1], dtype=int)
    count[rows] -= 1
    intercept = np.divide(training.sum(axis=1), count, out=np.zeros(training.shape[0]), where=count > 0)
    training[rows, columns] = intercept[rows]
    centered = training - intercept[:, None]
    k = min(rank, *centered.shape)
    width = min(k + 8, *centered.shape)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(centered @ rng.standard_normal((centered.shape[1], width)), mode="reduced")
    for _ in range(2):
        z, _ = np.linalg.qr(centered.T @ q, mode="reduced")
        q, _ = np.linalg.qr(centered @ z, mode="reduced")
    small_u, sv, _ = np.linalg.svd(q.T @ centered, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(centered.shape) * max(float(sv[0]), 1.0)
    k = min(k, int(np.count_nonzero(sv > tolerance)))
    return np.asarray(q @ small_u[:, :k], dtype=np.float64), intercept, centered


def counts_log_cpm(counts):
    """log1p CPM over observed genes (row-max scaled before summing); NaN where unobserved or the row is empty."""
    observed = np.isfinite(counts)
    safe = np.where(observed, counts, 0.0)
    maximum = np.max(safe, axis=1, initial=0.0)
    valid = maximum > 0
    scaled = np.divide(safe, maximum[:, None], out=np.zeros_like(safe), where=valid[:, None])
    total = scaled.sum(axis=1)
    cpm = np.divide(scaled * 1e6, total[:, None], out=np.zeros_like(scaled), where=valid[:, None])
    return np.where(observed & valid[:, None], np.log1p(cpm), np.nan), valid


class ControlResidual:
    def __init__(self, t):
        self.t = t

    def fit(self, stack, counts, genes, perts):
        """stack: T x G x P record effects (NaN missing); counts: T x G summed record control counts (NaN missing)."""
        values = np.asarray(stack, dtype=np.float64).copy()
        counts = np.asarray(counts, dtype=np.float64)
        self.gene_index = {g: i for i, g in enumerate(genes)}
        t, g, p = values.shape
        pert_index = {name: j for j, name in enumerate(perts)}
        rows = np.asarray([i for i, name in enumerate(genes) if name in pert_index]).astype(int)
        columns = np.asarray([pert_index[genes[i]] for i in rows], dtype=int)
        for row, column in zip(rows, columns):
            block = values[:, row, :].copy()
            measured_diagonal = np.isfinite(block[:, column])
            block[:, column] = np.nan
            count = np.isfinite(block).sum(1)
            mean = np.divide(np.nansum(block, axis=1), count, out=np.zeros(len(block)), where=count > 0)
            values[measured_diagonal, row, column] = mean[measured_diagonal]
        baseline = ivw_baseline(values)
        self.U, self.intercept, centered = response_basis(baseline, rows, columns, min(self.t.residual_rank, g, p))
        self.response_scale = _rms(self.U.T @ centered)
        vtrain = self.U.T @ centered
        vtrain /= self.response_scale[:, None]
        k = self.U.shape[1]

        self.normalization_genes = np.isfinite(counts).all(axis=0) & (np.nansum(counts, axis=0) > 0)
        normalization_counts = counts.copy()
        normalization_counts[:, ~self.normalization_genes] = np.nan
        log_controls, valid_records = counts_log_cpm(normalization_counts)
        control_observed = np.isfinite(log_controls)
        ncontrol = control_observed.sum(axis=0)
        self.control_mean = np.divide(np.nansum(log_controls, axis=0), ncontrol, out=np.zeros(g), where=ncontrol > 0)
        deviation = np.where(control_observed, log_controls - self.control_mean, 0.0)
        variance = np.divide(np.sum(deviation**2, axis=0), ncontrol, out=np.zeros(g), where=ncontrol > 0)
        self.control_scale = np.maximum(np.sqrt(variance), self.t.residual_scale_floor)
        standardized = deviation / self.control_scale

        permitted = standardized[valid_records]
        _, singular, vt = np.linalg.svd(permitted, full_matrices=False)
        tolerance = np.finfo(float).eps * max(permitted.shape) * max(float(singular[0]), 1.0)
        width = min(self.t.residual_global, int(np.count_nonzero(singular > tolerance)))
        self.global_loadings = vt[:width].T.copy()
        if width:
            pivots = np.argmax(np.abs(self.global_loadings), axis=0)
            self.global_loadings *= np.where(self.global_loadings[pivots, np.arange(width)] < 0, -1.0, 1.0)
        self.global_scale = np.sqrt(np.mean((permitted @ self.global_loadings) ** 2, axis=0))
        global_scores = standardized @ self.global_loadings / self.global_scale
        f = 1 + self.global_loadings.shape[1]
        features = np.empty((t, g, f), dtype=np.float64)
        features[:, :, 0] = standardized
        features[:, :, 1:] = global_scores[:, None, :]

        measured_counts = np.zeros((t, g), dtype=np.int32)
        residual_coordinates = np.zeros((t, g, k), dtype=np.float64)
        projection = np.linalg.solve(vtrain @ vtrain.T + self.t.residual_proj_ridge * np.eye(k), vtrain).T
        for record in range(t):
            block = values[record]
            measured = np.isfinite(block)
            measured[rows, columns] = False
            measured_counts[record] = measured.sum(axis=1)
            residual = np.where(measured, block - baseline, 0.0)
            residual[rows, columns] = 0.0
            residual_coordinates[record] = residual @ projection
        record_support = (measured_counts >= self.t.residual_min_measured) & control_observed
        self.supported = np.any(record_support, axis=0) & (k > 0)
        self.feature_center = np.zeros((g, f), dtype=np.float64)
        self.coefficients = np.zeros((g, f, k), dtype=np.float64)
        penalty = self.t.residual_ridge * np.eye(f)
        for gene in np.flatnonzero(self.supported):
            supported = record_support[:, gene]
            x = features[supported, gene]
            self.feature_center[gene] = x.mean(axis=0)
            x = x - self.feature_center[gene]
            y = residual_coordinates[supported, gene]
            self.coefficients[gene] = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        return self

    def _target_features(self, counts, genes):
        fitted = np.asarray([self.gene_index.get(g, -1) for g in genes], dtype=int)
        qrows = np.flatnonzero(fitted >= 0)
        frows = fitted[qrows]
        aligned = np.full(len(self.gene_index), np.nan)
        aligned[frows] = counts[qrows]
        aligned[~self.normalization_genes] = np.nan
        logs, _ = counts_log_cpm(aligned[None, :])
        local = np.where(np.isfinite(logs[0]), logs[0] - self.control_mean, 0.0)
        local /= self.control_scale
        features = np.empty_like(self.feature_center)
        features[:, 0] = local
        features[:, 1:] = local @ self.global_loadings / self.global_scale
        features -= self.feature_center
        return qrows, frows, features

    def predict(self, query, counts, genes, perts):
        """query (G, Q) effects on `genes`; counts (G,) this context's summed controls. Ratio-bounded correction,
        own targets and null columns kept, sign crossings rolled back."""
        query = np.asarray(query, dtype=np.float64)
        out = query.copy()
        qrows, frows, features = self._target_features(np.asarray(counts, dtype=np.float64), genes)
        features = np.clip(features, -self.t.residual_clip, self.t.residual_clip)
        active = np.any(query != 0, axis=0)
        cols = np.flatnonzero(active)
        query_index = {name: row for row, name in enumerate(genes)}
        if qrows.size and cols.size and self.U.shape[1]:
            centered = query[np.ix_(qrows, cols)].copy() - self.intercept[frows, None]
            local_index = {genes[row]: i for i, row in enumerate(qrows)}
            for j, column in enumerate(cols):
                row = local_index.get(perts[column])
                if row is not None:
                    centered[row, j] = 0.0
            vquery = (self.U[frows].T @ centered) / self.response_scale[:, None]
            coordinates = np.einsum("gf,gfk->gk", features[frows], self.coefficients[frows], optimize=True)
            delta = coordinates @ vquery
            delta[~self.supported[frows]] = 0.0
            for j, column in enumerate(cols):
                row = local_index.get(perts[column])
                if row is not None:
                    delta[row, j] = 0.0
            with np.errstate(over="ignore", invalid="ignore"):
                correction = self.t.residual_alpha * delta
            out[np.ix_(qrows, cols)] = bounded_ratio(query[np.ix_(qrows, cols)], correction)
            for column, pert in enumerate(perts):
                if pert in query_index:
                    out[query_index[pert], column] = query[query_index[pert], column]
        out[:, ~active] = 0.0
        return preserve_signs(query, out, genes, perts)


def bounded_ratio(query, correction) -> np.ndarray:
    """query x exp(correction / query), the log-ratio saturating at +-ln 2 and the factor kept in [1/2, 2]."""
    out = query.copy()
    eligible = query != 0
    apply = eligible & (correction != 0)
    bound = np.log(2.0)
    saturated = apply & (np.abs(correction) >= bound * np.abs(query))
    same = (correction > 0) == (query > 0)
    ell = np.zeros_like(query)
    ell[saturated & same] = bound
    ell[saturated & ~same] = -bound
    inside = apply & ~saturated
    ell[inside] = correction[inside] / query[inside]
    with np.errstate(over="ignore", under="ignore"):
        out[apply] = query[apply] * np.exp(ell[apply])
    parent = np.abs(query[apply])
    magnitude = np.abs(out[apply])
    with np.errstate(under="ignore"):
        lower = 0.5 * parent
        below = lower / parent < 0.5
        lower[below] = np.nextafter(lower[below], np.inf)
    zero = magnitude == 0
    magnitude = np.maximum(magnitude, lower)
    upper = np.full_like(parent, np.finfo(float).max)
    can_double = parent <= np.finfo(float).max / 2
    upper[can_double] = 2 * parent[can_double]
    magnitude = np.minimum(magnitude, upper)
    magnitude[zero] = parent[zero]
    out[apply] = np.copysign(magnitude, query[apply])
    return out


def preserve_signs(query, adjusted, genes, perts) -> np.ndarray:
    """Entries that change sign or reach zero, and own targets, revert to the query."""
    keep = ((query > 0) & (adjusted > 0)) | ((query < 0) & (adjusted < 0))
    out = np.where(keep, adjusted, query)
    index = {gene: row for row, gene in enumerate(genes)}
    for column, pert in enumerate(perts):
        row = index.get(pert)
        if row is not None:
            out[row, column] = query[row, column]
    return out


def evidence_agreement(x, t, p) -> np.ndarray:
    """Per entry, the corrected profile p pulled back toward l2 x by w = 2uv/(u^2+v^2) of |x|, |t| (same sign only)."""
    x, t, p = [np.asarray(a, dtype=np.float64) for a in (x, t, p)]
    same = ((x > 0) & (t > 0)) | ((x < 0) & (t < 0))
    w = np.zeros_like(x)
    m = np.maximum(np.abs(x[same]), np.abs(t[same]))
    u, v = np.abs(x[same]) / m, np.abs(t[same]) / m
    w[same] = np.clip(2 * u * v / (u * u + v * v), 0, 1)
    a = x.copy()
    changed = p != x
    endpoint = changed & (w == 1)
    a[endpoint] = p[endpoint]
    middle = changed & (w > 0) & (w < 1)
    xx, pp, ww = x[middle], p[middle], w[middle]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        delta = pp - xx
        value = xx + ww * delta
        unsafe = ~np.isfinite(value) | ~np.isfinite(delta)
        scale = np.maximum(np.abs(xx[unsafe]), np.abs(pp[unsafe]))
        value[unsafe] = scale * ((1 - ww[unsafe]) * (xx[unsafe] / scale) + ww[unsafe] * (pp[unsafe] / scale))
    a[middle] = np.clip(value, np.minimum(xx, pp), np.maximum(xx, pp))
    rollback = ((x != 0) & (a == 0)) | ((x > 0) & (a < 0)) | ((x < 0) & (a > 0))
    a[rollback] = x[rollback]
    return a


def profile_effects(l2, l2_tot, stack, counts, src_genes, src_perts, ctrl_counts, genes, perts, t) -> np.ndarray:
    model = ControlResidual(t).fit(stack, counts, src_genes, src_perts)
    return evidence_agreement(l2, l2_tot, model.predict(l2, ctrl_counts, genes, perts))
