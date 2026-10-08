"""Certificates part: the weighted encoding on the two-gene toggle switch.

    python -m analysis.certificates toggle         every reference x weight family x schedule (about an hour)
    python -m analysis.certificates toggle_best    the best combination and its numerical validity checks
    python -m analysis.certificates toggle_exact   the exact solution used as the reference

For every reference the certified exponent K_T is reported for the two weights (sqrt(mu) and the resolvent weight)
and six window schedules, with the measured growth of the weighted state, the continuous estimate, the worst case W
over the horizon, and the split of the looseness into window penalty, non-normality and instance slack. The FSP
solution is used only for the measured growth and the worst case; certificates never see it.
"""
import sys
import time

import numpy as np

from .common import FAMILIES, LEAK_TOL, N_CONT, SUB, data, log_norm_weighted, num, save, schedules, settings
from qcme import certify, emulate, fsp, invariants as inv, lowrank, models, reference

PART = "certificates"
MODEL = "toggle"
THRESHOLD = 2.0  # K_T at which the certified cost would match the uncertified one
VALIDITY_TOL = {"I2": 1e-10, "I3": 1e-12, "I4": 1e-10}


def study_settings():
    return {**settings(), "pinv_rtol": reference.PINV_RTOL, "tt_ranks": reference.TT_RANKS,
            "tt_steps": lowrank.N_STEPS, "tt_fd_frac": lowrank.FD_FRAC, "threshold": THRESHOLD}


def model_setup(m, T=None):
    T = T if T is not None else models.relaxation_time(m)[0]
    widths, leak = fsp.select_box(m, lambda sp_: reference.initial_law(m, sp_), T, LEAK_TOL)
    return inv.setup(m, widths, T, families="extended"), leak


def evaluate(s, rn, P_at, sched):
    """Every reported quantity for one reference. P_at: t -> the FSP solution (measured growth and the R-TT
    validation only)."""
    ref, T = s.refs[rn], s.T
    lm0, lmT = ref.log_mu(0.0), ref.log_mu(T)
    log_n0 = log_norm_weighted(s.p0, lm0)
    pT = P_at(T)
    out = {"log10_psi0_norm": log_n0 / np.log(10), "schedules": {}}
    for sname, edges in sched.items():
        t0 = time.time()
        cert = certify.window_certificate(s.gen, ref, edges)
        log_meas = log_norm_weighted(pT, ref.log_mu(edges[-2])) - log_n0  # psi_T in the last window's mu
        out["schedules"][sname] = {
            "K": cert.K, "int_kappa": cert.int_kappa, "K_kappa": cert.K_kappa,
            "max_kappa": float(np.max(cert.kappa)), "sum_log_rho": float(np.sum(cert.log_rho)),
            "log10_Gamma": {f: (log_n0 + 0.5 * cert.K[f]) / np.log(10) for f in FAMILIES},
            "measured_log10_growth": log_meas / np.log(10),
            "log10_R": {f: (0.5 * cert.K[f] - log_meas) / np.log(10) for f in FAMILIES},
            "cw_per_window": cert.cw, "kappa_per_window": cert.kappa, "log_rho": cert.log_rho,
            "edges": edges, "seconds": time.time() - t0}
    cc = certify.continuous_certificate(s.gen, ref, np.linspace(0, T, N_CONT))
    meas_c = log_norm_weighted(pT, lmT) - log_n0
    out["K_continuous_estimate"] = cc.K
    out["int_kappa_continuous"] = cc.int_kappa
    out["measured_log10_growth_continuous"] = meas_c / np.log(10)
    out["log10_R_continuous"] = {f: (0.5 * cc.K[f] - meas_c) / np.log(10) for f in FAMILIES}
    if np.all(np.isfinite(lm0)) and np.all(np.isfinite(lmT)) and np.ptp(lm0) < 1400 and np.ptp(lmT) < 1400:
        W = emulate.horizon_worst_case(s.gen.A, lm0, lmT, T)
        out["log10_W_c"] = float(np.log10(W))
        out["log10_W_over_meas_c"] = out["log10_W_c"] - meas_c / np.log(10)
    else:
        out["log10_W_c"] = None  # ratios of mu beyond double precision: W not computable in this form
        out["log10_W_over_meas_c"] = None
    dec = {}
    for sname, row in out["schedules"].items():
        dec[sname] = {}
        for f in FAMILIES:
            Rc = out["log10_R_continuous"][f]
            slack = out["log10_W_over_meas_c"]
            dec[sname][f] = {"window_penalty": row["log10_R"][f] - Rc,
                             "non_normality": None if slack is None else Rc - slack, "instance_slack": slack}
    out["decomposition"] = dec
    if rn.startswith("R-TT"):
        core = ref.core
        tv = []
        for t in certify.uniform_edges(T, 20):
            p = P_at(t)
            tv.append(0.5 * float(np.abs(np.exp(core.log_mu(t)) - p / p.sum()).sum()))
        out["tt_validation_max_TV"] = max(tv)
        out["tt_rank_over_width"] = core.r / min(s.widths)
    return out


def best_combination(refs):
    """Smallest K_T over (reference, weight family, schedule), by certificate quantities only."""
    K, rn, f, sn = min((num(r["schedules"][sn]["K"][f]), rn, f, sn)
                       for rn, r in refs.items() for sn in r["schedules"] for f in FAMILIES)
    return {"K": K, "reference": rn, "family": f, "schedule": sn}


def stage_toggle():
    m = next(mm for mm in models.registry() if mm.name == MODEL)
    s, leak = model_setup(m)
    sched = schedules(s.T)
    times = sorted(set(np.concatenate([certify.uniform_edges(s.T, 20), [s.T]]).tolist()))
    P = dict(zip(times, fsp.propagate(s.gen.A, s.p0, times)))
    P_at = lambda t: P[min(P, key=lambda u: abs(u - t))]  # noqa: E731
    refs = {}
    for rn in s.refs:
        t0 = time.time()
        refs[rn] = evaluate(s, rn, P_at, sched)
        bestK = min(num(v["K"][f]) for v in refs[rn]["schedules"].values() for f in FAMILIES)
        print(f"  {rn:15s} min K={bestK:.4g} [{time.time() - t0:.0f}s]", flush=True)
    save(PART, "toggle_references", {**study_settings(), "model": MODEL, "T": s.T, "widths": s.widths},
         {"T": s.T, "widths": s.widths, "n_states": len(s.space), "leak": leak, "references": refs,
          "best": best_combination(refs)})


def stage_toggle_best():
    d = data(PART, "toggle_references")
    best = d["best"]
    m = next(mm for mm in models.registry() if mm.name == MODEL)
    s = inv.setup(m, tuple(d["widths"]), d["T"], families="extended")
    pc = inv.piecewise_checks(s, best["reference"], edges=schedules(s.T)[best["schedule"]], sub=SUB)
    validity = {"I2": pc.I2, "I3": max(pc.I3_violation.values()), "I4": pc.I4,
                "stabilized_max_lognorm": max(pc.stabilized_max_lognorm.values())}
    valid = all(validity[k] <= VALIDITY_TOL[k] for k in VALIDITY_TOL)
    K = num(best["K"])
    print(f"best K_T = {K:.4g} ({best['reference']}, {best['family']}, {best['schedule']}); validity {validity}", flush=True)
    save(PART, "toggle_best", {**study_settings(), "model": MODEL, "T": s.T, "widths": s.widths},
         {"K_gate": K, "threshold": THRESHOLD, "winner": best, "validity": validity, "winner_valid": bool(valid),
          "gate_pass": bool(valid and K <= THRESHOLD)})


class _FSPCache:
    """p_S(t) by uniformization from t = 0, cached (exact reference only)."""

    def __init__(self, A, p0):
        self.A, self.p0, self.c = A, p0, {}

    def __call__(self, t):
        t = float(t)
        if t not in self.c:
            self.c[t] = self.p0.copy() if t == 0.0 else fsp.propagate(self.A, self.p0, [t])[-1]
        return self.c[t]


def stage_toggle_exact():
    """The exact solution mu_t = p_t as the reference under every schedule (the window penalty alone, since the
    continuous certificate of the exact reference is 0), and the limit of fine windows,
    2 int CW(static p_t) dt + int max(-p_dot/p) dt."""
    m = next(mm for mm in models.registry() if mm.name == MODEL)
    d = data(PART, "toggle_references")
    s = inv.setup(m, tuple(d["widths"]), d["T"], families="core")
    t0 = time.time()
    P = _FSPCache(s.gen.A, s.p0)
    ref = reference.Exact(P, lambda t: s.gen.A @ P(t))
    rows = {}
    for sn, edges in schedules(s.T).items():
        cert = certify.window_certificate(s.gen, ref, edges)
        rows[sn] = {"K": cert.K, "sum_log_rho": float(np.sum(cert.log_rho)), "max_kappa": float(np.max(cert.kappa))}
    ts = np.linspace(0, s.T, N_CONT)
    static = certify.continuous_certificate(s.gen, inv.reference_static_view(ref), ts)
    jump = np.array([np.max(-ref.dlog_mu(t)) for t in ts])
    limit = {f: static.K[f] + float(np.trapezoid(jump, ts)) for f in FAMILIES}
    best = min((K, sn, f) for sn, r in rows.items() for f, K in r["K"].items())
    save(PART, "toggle_exact", {**study_settings(), "model": MODEL, "T": s.T, "widths": s.widths,
                                "reference": "exact solution"},
         {"schedules": rows, "riemann_limit": limit, "best": {"K": best[0], "schedule": best[1], "family": best[2]}})
    print(f"  exact reference: best K = {best[0]:.4g} ({best[1]}, {best[2]}) [{time.time() - t0:.0f}s]", flush=True)


STAGES = {"toggle": stage_toggle, "toggle_best": stage_toggle_best, "toggle_exact": stage_toggle_exact}

if __name__ == "__main__":
    for a in [a for a in sys.argv[1:] if a in STAGES]:
        STAGES[a]()
