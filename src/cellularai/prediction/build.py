"""One context's predicted cells: n_cells integer cells per perturbation.

Pooled evidence and the training summaries give the calls, the placement plan and the design shape; the design's margin is solved so the
scorer's spread of the integer totals equals its expected credit, re-solved after each synthesis (`Design.solve`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .. import guards
from ..pooling.build import Evidence
from ..task import NCELL, control_cpm_mean
from . import finite, placement, plan, residual, synthesis, totals
from .inputs import GenerationInputs


@dataclass(frozen=True)
class Effects:
    """Pooled evidence as the design reads it (float64, z non-finite -> 0) and the control-residual profile effects."""

    l2: np.ndarray
    l2_tot: np.ndarray
    z: np.ndarray
    z_tot: np.ndarray
    profile: np.ndarray


class Design:
    """The per-perturbation totals of one context: the amplitude design amp x l2, and the shaped design at a margin
    of the credit, each as integer per-gene totals (perts x genes) of n_cells cells at the control depth."""

    def __init__(
        self, ctrl, eng: synthesis.Engine, fx: Effects, calls: dict, tidx, credit: float, m, n_cells: int = NCELL
    ):
        t = m.totals
        self.eng, self.fx, self.calls, self.credit, self.t = eng, fx, calls, credit, t
        self.mask = totals.own_mask(fx.l2.shape, tidx)
        self.w = 1.0 / (fx.l2.shape[0] - (tidx >= 0))
        self.spread = totals.make_spread(eng.fc, self.mask, m.guards.log2_max)
        self.amp, capped = totals.bisect_amp(self.spread, lambda a: a * fx.l2, t.amp_share * credit)
        guards.fired("amp_search_cap", capped, 1)
        self.s_tot = int(round(float(np.full(n_cells, float(eng.depth)).sum())))
        self.pcs = totals.control_pcs(ctrl, t.pw_k, t.pcs_cells, t.pcs_seed)

    def totals(self, Lg) -> np.ndarray:
        T = np.zeros((Lg.shape[1], Lg.shape[0]), dtype=np.int64)
        for k, q in enumerate(self.calls):
            lg = self.eng.guard_l2(np.asarray(Lg[:, k], dtype=np.float64), self.calls[q]["idx"], self.calls[q]["sign"])
            T[k] = synthesis.apportion(self.eng.design_frac(lg), self.s_tot)
        return T

    def shaped(self, margin: float, record: bool = False) -> np.ndarray:
        """The shaped design at a margin; `record` tallies its searches' caps (the synthesized design only: trial
        margins of the solve may cap, the solve's own reached flag covers them)."""
        fx = self.fx
        L, (c_amp, c_cp, c_pw) = totals.shaped_design(
            fx.l2_tot,
            fx.z_tot,
            fx.z,
            fx.profile,
            self.eng.fc,
            self.mask,
            self.pcs,
            self.spread,
            margin * self.credit,
            self.t,
        )
        if record:
            guards.fired("amp_search_cap", c_amp, 1)
            guards.fired("shape_search_cap", c_cp + c_pw, 2)
        return L

    def solve(self, jobs_at, draws, n_jobs: int) -> tuple[list, dict]:
        """(cells per perturbation, solve report): the margin solved to the expected credit, synthesis at its
        totals, the credit target re-taken from the synthesized moments, until it moves < fixed_rtol (at most
        fixed_steps syntheses)."""
        w = self.w
        c_hat, margin, targets, margins = np.full(self.mask.shape[1], np.inf), 1.0, [], []
        target = totals.expected_credit(c_hat, draws, w)
        for _ in range(self.t.fixed_steps):
            margin, reached = totals.solve_margin(
                lambda mg: totals.totals_spread(self.totals(self.shaped(mg)), self.mask, w), target, margin or 1.0
            )
            T_all = self.totals(self.shaped(margin, record=True))
            blocks = synthesis.synthesize_all(self.eng, jobs_at(T_all), n_jobs)
            c_hat = np.array([totals.jackknife_moment(b) for b in blocks])
            if not np.isfinite(c_hat).all():
                raise ValueError("non-finite jackknife moment of synthesized cells")
            targets.append(target)
            margins.append(margin)
            share = totals.totals_spread(T_all, self.mask, w) / target
            print(
                f"[generate] target {target:.6g} margin {margin:.6g} reached {reached} realized/target {share:.6g}",
                flush=True,
            )
            target = totals.expected_credit(c_hat, draws, w)
            converged = abs(target / targets[-1] - 1) < self.t.fixed_rtol
            if converged:
                break
        return blocks, {
            "margin": margin,
            "margin_reached": reached,
            "converged": converged,
            "targets": targets,
            "margins": margins,
            "next_target": target,
            "chat_over_proxy": float(np.mean(c_hat) / draws.mean()),
            "credit_share": share,
        }


def build(
    Y,
    genes,
    perts,
    ev: Evidence,
    inp: GenerationInputs,
    call_depth: float,
    median_degs: float,
    m,
    n_jobs: int = 30,
    n_cells: int = NCELL,
    seed: int | None = None,
) -> tuple[sp.csr_matrix, dict]:
    """(counts, report). Y: the context's control counts (cells x genes); ev: Pooled evidence on (genes, perts), perts sorted;
    m: the model."""
    genes, perts = list(genes), list(perts)
    g = m.guards
    guards.reset()
    ctrl = Y.toarray()
    mean_cpm = control_cpm_mean(Y)
    depth = float(np.asarray(Y.sum(axis=0)).ravel().sum() / Y.shape[0])
    l2, l2_tot = (np.asarray(a, np.float64) for a in (ev.l2, ev.l2_tot))
    z, z_tot = (np.where(np.isfinite(a), a, 0.0).astype(np.float64) for a in (ev.z, ev.z_tot))

    profile = residual.profile_effects(
        l2,
        l2_tot,
        inp.residual_stack,
        inp.residual_counts,
        genes,
        inp.residual_perts,
        np.asarray(Y.sum(axis=0), dtype=np.float64).ravel(),
        genes,
        perts,
        m.totals,
    )
    z_plan = finite(plan.plan_z(inp.mix_z, inp.mix_lines, inp.reference_ctrl, genes, mean_cpm, m.plan, g))
    zsim = finite(
        plan.similarity_z(inp.sim_z, inp.sim_meas, inp.sim_lines, inp.reference_ctrl, genes, mean_cpm, m.plan, g)
    )
    with np.errstate(invalid="ignore"):
        breadth = np.nanmean(inp.breadth_rel, axis=0)
    xs = plan.x_sim(inp.breadth_rel, inp.breadth_lines, inp.reference_ctrl, genes, mean_cpm, m.plan, g)

    gidx = {g: i for i, g in enumerate(genes)}
    tidx = np.array([gidx.get(q, -1) for q in perts])
    calls = plan.select_calls(l2, z, mean_cpm, genes, perts, depth, m, median_calls=float(call_depth))
    eng = synthesis.Engine(ctrl, m)
    draws = totals.control_moments(ctrl, m.totals, n_cells)
    credit = len(perts) * float(draws.mean())
    design = Design(ctrl, eng, Effects(l2, l2_tot, z, z_tot, profile), calls, tidx, credit, m, n_cells)

    sizes = plan.plan_sizes(breadth, xs, l2, z, z_plan, mean_cpm, tidx, m, float(median_degs))
    plans = plan.sized_plan(sizes, l2, z_plan, mean_cpm, genes, perts, depth, inp.common, m)
    with np.errstate(divide="ignore", invalid="ignore"):
        lq = np.log2(
            np.maximum(1e6 * design.totals(design.amp * l2).T / design.s_tot, 1e-300)
            / np.maximum(eng.mean_cpm[:, None], 1e-300)
        )
    zf = finite(z)
    tg = placement.Targets(
        l2=l2,
        z=zf,
        az=np.abs(zf),
        zsim=zsim,
        lq=np.clip(lq, -30, 30),
        cpm=eng.mean_cpm,
        elig=eng.eligible,
        log2_cg=eng.log2_cg,
        depth=float(eng.depth),
        amp=design.amp,
        tidx=tidx,
        lib=placement.library_profile(
            ctrl.sum(axis=1).astype(np.float64), design.s_tot, m.placement.lib_spread, n_cells
        ),
    )
    blocks, solved = design.solve(
        lambda T: [placement.job(k, q, plans[q], tg, T, m, seed) for k, q in enumerate(perts)], draws, n_jobs
    )

    report = {
        "credit": credit,
        **solved,
        "guards": {**ev.guards, **guards.snapshot()},
        "unmeasured": {
            "no_row": list(ev.unmeasured),
            "no_evidence": [q for k, q in enumerate(perts) if not np.nan_to_num(l2[:, k]).any() or not z[:, k].any()],
        },
    }
    print(f"[generate] {report}", flush=True)
    if not solved["margin_reached"]:
        print(
            f"[generate] WARNING: the credit target is out of reach; margin clamped to {solved['margin']:.6g}",
            flush=True,
        )
    if not solved["converged"]:
        print(f"[generate] WARNING: credit target not converged in {m.totals.fixed_steps} passes", flush=True)
    X = sp.vstack([sp.csr_matrix(b) for b in blocks], format="csr")
    X.eliminate_zeros()
    # float32 storage is exact only for integers below 2^24
    if not ((X.data >= 0) & (X.data == np.round(X.data)) & (X.data < 2**24)).all():
        raise ValueError("generated counts are not non-negative integers below 2^24")
    return X.astype(np.float32), report
