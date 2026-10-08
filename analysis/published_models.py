"""Published-models part: certificates and the looseness decomposition on the twelve published instances.

    python -m analysis.published_models references [--only schlogl,...]  best cheap reference and decomposition
    python -m analysis.published_models exact [--only ...]               the exact solution as the reference
    python -m analysis.published_models exact_log [--only ...]           the same, with log p computed exactly
                                                                         where double precision cannot hold p
    python -m analysis.published_models exact_log_uniformization        log-domain uniformization (toggle_kuntz)
    python -m analysis.published_models overflow_bounds                  lower bounds where K_T overflows
    python -m analysis.published_models worst_case [--only ...]          W by scaled power iteration (large boxes)
    python -m analysis.published_models resolvent [--only ...]           resolvent weight on the best reference
                                                                         (the two largest instances)
    python -m analysis.published_models summary                          current values per instance

Instances: schlogl, toggle_hegland, toggle_kuntz, toggle_gardner, feedback_b1, feedback_b10, feedback_b100,
lambda_phage, toggle_tang, autoreg_tang_1, autoreg_tang_2, autoreg_tang_3. The largest instances take from hours to
about two days each; checkpoints let interrupted runs resume.
"""
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp

from .common import (CEILING, FAMILIES, N_CONT, OUT, SUB, TAIL_GROW, best_of, build_setup, certificate_table,
                     checkpoint_load, checkpoint_save, data, inf_if_nan, is_huge, is_large, log10, log_norm_weighted,
                     num, published_model, published_setup, save, schedules, settings, worst_case)
from qcme import certify, fsp, invariants as inv, models, reference

PART = "published_models"
VALIDITY_TOL = {"I2": 1e-10, "I3": 1e-12, "I4": 1e-10}
REPR_FLOOR = -745.0  # log of the smallest positive double: entries of p0 below this are 0 in double precision
ACC_FLOOR = 1e-270  # entrywise accuracy of uniformization
W_TOL = 1e-6  # power iteration for W: relative increase below this on three consecutive iterations
W_DROP = 700.0  # columns with weight below e^-700 of the largest are dropped (and bracketed)
TIMEBOX_S = 48 * 3600.0  # power iteration and the resolvent weight, per instance
LOGU_STOP = -40.0  # log-domain uniformization: stop once every newest term is below e^-40 of its sum
LOGU_MAX_STEP = 30.0


def instances(only):
    return [m for m in models.published() if not only or m.name in only]


# ----------------------------------------------------------------------------- FSP access
class _Stream:
    """p_S(t) by forward propagation from the last requested time (restarts from p0 for an earlier time)."""

    def __init__(self, A, p0):
        self.A, self.p0 = A, p0
        self.t, self.p = 0.0, p0.copy()

    def __call__(self, t):
        t = float(t)
        if t < self.t:
            self.t, self.p = 0.0, self.p0.copy()
        if t > self.t:
            self.p = fsp.propagate(self.A, self.p, [t - self.t])[-1]
            self.t = t
        return self.p


class _TimeCache:
    """p_S(t) at a fixed set of times, computed in one forward pass."""

    def __init__(self, A, p0, times):
        self.times = np.array(sorted(set(float(t) for t in times)))
        P = fsp.propagate(A, p0, self.times)
        self.c = {t: P[i] for i, t in enumerate(self.times)}

    def __call__(self, t):
        t = float(t)
        if t in self.c:
            return self.c[t]
        k = float(self.times[np.argmin(np.abs(self.times - t))])
        if abs(k - t) > 1e-12 * max(1.0, abs(t)):
            raise KeyError(f"time {t} not in the cache")
        return self.c[k]


def grown_widths(m, widths):
    fixed = np.array(m.binary) | fsp._bounded_by_conservation(m)
    return tuple(int(w) if fx else int(np.ceil(w * TAIL_GROW)) for w, fx in zip(widths, fixed))


# ----------------------------------------------------------------------------- references and decomposition
def references_model(m):
    s, leak, hinfo = published_setup(m)
    n = len(s.space)
    large = is_large(s)
    sched = schedules(s.T)
    t0 = time.time()
    pT = fsp.propagate(s.gen.A, s.p0, [s.T])[-1]
    print(f"-- {m.name}: T={s.T:.4g} widths={s.widths} |S|={n} leak={leak:.2e} large={large} "
          f"(FSP {time.time() - t0:.0f}s)", flush=True)
    times = np.linspace(0, s.T, N_CONT)
    refs = {}
    for rn, ref in s.refs.items():
        t1 = time.time()
        hit = checkpoint_load(PART, f"{m.name}__{rn}")
        if hit is not None:
            refs[rn] = {**hit, "_lm0": ref.log_mu(0.0), "_lmT": ref.log_mu(s.T)}
            continue
        tab = certificate_table(s, ref, sched, large)
        lm0, lmT = ref.log_mu(0.0), ref.log_mu(s.T)
        log_n0 = log_norm_weighted(s.p0, lm0)
        rows = {}
        for sn, c in tab.items():
            log_meas = log_norm_weighted(pT, ref.log_mu(sched[sn][-2])) - log_n0
            rows[sn] = {"K": c.K, "int_kappa": c.int_kappa, "K_kappa": c.K_kappa, "max_kappa": float(np.max(c.kappa)),
                        "sum_log_rho": float(np.sum(c.log_rho)), "measured_log10_growth": log10(log_meas),
                        "log10_Gamma": {f: log10(log_n0 + 0.5 * K) for f, K in c.K.items()},
                        "log10_R": {f: log10(0.5 * K - log_meas) for f, K in c.K.items()}}
        cc = certify.continuous_certificate(s.gen, ref, times, families=("sqrt_mu",) if large else FAMILIES)
        meas_c = log_norm_weighted(pT, lmT) - log_n0
        refs[rn] = {"log10_psi0": log10(log_n0), "schedules": rows, "K_continuous_estimate": dict(cc.K),
                    "int_kappa_continuous": cc.int_kappa, "measured_log10_growth_continuous": log10(meas_c),
                    "log10_R_continuous": {f: log10(0.5 * K - meas_c) for f, K in cc.K.items()},
                    "_lm0": lm0, "_lmT": lmT}
        checkpoint_save(PART, f"{m.name}__{rn}", {k: v for k, v in refs[rn].items() if not k.startswith("_")})
        K, sn, f = best_of(tab)
        print(f"   {rn:14s} best K={K:.4g} ({f}, {sn}) [{time.time() - t1:.0f}s]", flush=True)
    # best cheap reference, by certificate quantities only
    K, rn, fam, sn = min((num(v["K"][f]), rn, f, sn) for rn, r in refs.items()
                         for sn, v in r["schedules"].items() for f in v["K"])
    win, ref = refs[rn], s.refs[rn]
    if fam not in win["K_continuous_estimate"]:  # large box: resolvent continuous estimate for the best only
        cc = certify.continuous_certificate(s.gen, ref, times, families=(fam,))
        win["K_continuous_estimate"][fam] = cc.K[fam]
        win["log10_R_continuous"][fam] = log10(0.5 * cc.K[fam] - win["measured_log10_growth_continuous"] * np.log(10))
    log_W = worst_case(s, win["_lm0"], win["_lmT"])
    slack = None if log_W is None else log10(log_W) - win["measured_log10_growth_continuous"]
    Rc, Rs = win["log10_R_continuous"][fam], win["schedules"][sn]["log10_R"][fam]
    dec = {"window_penalty": Rs - Rc, "non_normality": None if slack is None else Rc - slack,
           "instance_slack": slack, "log10_W_c": None if log_W is None else log10(log_W)}
    # numerical validity of the best combination
    sv = s if not is_huge(s) else build_setup(s.model, tuple(s.model.widths), s.T)
    pc = inv.piecewise_checks(sv, rn, edges=sched[sn], sub=SUB)
    validity = {"I2": pc.I2, "I3": max(pc.I3_violation.values()), "I4": pc.I4,
                "stabilized_max_lognorm": max(pc.stabilized_max_lognorm.values()), "box": sv.widths}
    validity["valid"] = bool(all(validity[k] <= VALIDITY_TOL[k] for k in VALIDITY_TOL))
    # tail term: the same reference, schedule and weight on the box grown by 1.5 in every unbounded width
    gw = grown_widths(s.model, s.widths)
    n_g = len(s.model.space(gw))
    tail = {"widths": gw, "n_states": n_g}
    if n_g > CEILING:
        tail["skipped"] = f"grown box has {n_g} states, above the ceiling {CEILING}"
    elif is_huge(s):
        tail["skipped"] = "skipped: the grown box exceeds the size limit"
    else:
        t1 = time.time()
        sg = build_setup(s.model, gw, s.T)
        rg = sg.refs[rn]
        cg = certify.window_certificate(sg.gen, rg, sched[sn], families=(fam,))
        g_grown = log10(log_norm_weighted(sg.p0, rg.log_mu(0.0)) + 0.5 * cg.K[fam])
        tail.update({"K_grown": cg.K[fam], "log10_Gamma_grown": g_grown,
                     "log10_Gamma_box": win["schedules"][sn]["log10_Gamma"][fam],
                     "tail_term": g_grown - win["schedules"][sn]["log10_Gamma"][fam], "seconds": time.time() - t1})
    for r in refs.values():
        r.pop("_lm0"), r.pop("_lmT")
    return {"provenance": s.model.provenance, "T": s.T, "horizon": hinfo, "widths": s.widths, "n_states": n,
            "leak": leak, "leak_check": 1.0 - float(pT.sum()), "large_box_rule": large, "very_large_box_rule": is_huge(s),
            "references": refs,
            "best": {"K": K, "reference": rn, "family": fam, "schedule": sn,
                     "K_uniform_J20": win["schedules"]["uniform J=20"]["K"],
                     "K_continuous_estimate": win["K_continuous_estimate"][fam],
                     "int_kappa": win["schedules"][sn]["int_kappa"], "log10_psi0": win["log10_psi0"],
                     "log10_Gamma": win["schedules"][sn]["log10_Gamma"][fam], "log10_R": Rs},
            "decomposition": dec, "tail": tail, "validity": validity}


def stage_references(only=None):
    for m in instances(only):
        t0 = time.time()
        out = references_model(m)
        out["seconds"] = time.time() - t0
        save(PART, m.name, {**settings(), "model": m.name, "params": m.params, "x0": m.x0, "envelope": m.envelope}, out)


# ----------------------------------------------------------------------------- exact reference
def oracle_single_pass(s, sched, ts, log_domain=False):
    """The exact reference under every schedule with the sqrt(mu) weight, and the limit of fine windows, in one
    forward FSP pass (very large boxes). A window's CW depends only on its left-edge time and each jump only on
    consecutive edges, as in certify.window_certificate. With log_domain, log p at t = 0 comes from the log-domain
    initial law and the states not reachable from its representable support are propagated in scaled form."""
    A, leak = s.gen.A, s.gen.leak
    acc = None
    if log_domain:
        lp0 = log_p0(s.model, s.space)
        R = forward_closure(A, lp0 > REPR_FLOOR)
        iu = np.nonzero(~R)[0]
        if iu.size:
            AU = sp.csr_matrix(A[iu][:, iu])
            cU = float(lp0[iu].max())
            qU, lsU = np.exp(lp0[iu] - cU), 0.0
        acc = {"min_p_R_t_pos": np.inf, "times_below_floor": [], "zeros_after_U": 0, "n_U": int(iu.size)}
    pos = {sn: {float(t): j for j, t in enumerate(ed)} for sn, ed in sched.items()}
    times = sorted(set([float(t) for ed in sched.values() for t in ed] + [float(t) for t in ts]))
    J = {sn: len(ed) - 1 for sn, ed in sched.items()}
    cw = {sn: np.zeros(J[sn]) for sn in sched}
    kap = {sn: np.zeros(J[sn]) for sn in sched}
    lrho = {sn: np.zeros(max(J[sn] - 1, 0)) for sn in sched}
    prev = {sn: None for sn in sched}
    tsset = set(float(t) for t in ts)
    static, jump = {}, {}
    p, tc = s.p0.copy(), 0.0
    for t in times:
        if t > tc:
            p = fsp.propagate(A, p, [t - tc])[-1]
            if log_domain and iu.size:
                qs, ls_ = propagate_scaled(AU, qU, [t - tc])
                qU, lsU = qs[0], lsU + ls_[0]
            tc = t
        if log_domain and t == 0.0:
            lm = lp0.copy()
        else:
            with np.errstate(divide="ignore"):
                lm = np.log(p)
                if log_domain and iu.size:
                    lm[iu] = cU + lsU + np.log(qU)
            if log_domain:
                mn = float(p[R].min())
                acc["min_p_R_t_pos"] = min(acc["min_p_R_t_pos"], mn)
                if mn < ACC_FLOOR:
                    acc["times_below_floor"].append(t)
                acc["zeros_after_U"] = max(acc["zeros_after_U"], int(np.sum(~np.isfinite(lm))))
        with np.errstate(over="ignore", invalid="ignore"):
            r = certify.residual(A, lm)
        c = certify.cw_sqrt_mu(r, leak)
        if not np.isfinite(c):  # p underflows: the valid bound is +inf
            c = np.inf
        if t in tsset:
            static[t] = c
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                jump[t] = float(np.max(-r)) if log_domain else float(np.max(-(A @ p) / p))
        for sn in sched:
            j = pos[sn].get(t)
            if j is None or j >= J[sn]:
                continue
            if prev[sn] is not None:
                with np.errstate(invalid="ignore"):
                    lrho[sn][j - 1] = 0.5 * float(np.max(prev[sn] - lm))
            prev[sn] = lm
            cw[sn][j] = c
            kap[sn][j] = certify.kappa(r)
    rows = {}
    for sn, ed in sched.items():
        K = float(2 * np.sum(cw[sn] * np.diff(ed)) + 2 * np.sum(lrho[sn]))
        rows[sn] = {"K": {"sqrt_mu": np.inf if np.isnan(K) else K}, "sum_log_rho": float(np.sum(lrho[sn])),
                    "max_kappa": float(np.max(kap[sn]))}
    st = np.array([static[float(t)] for t in ts])
    jp = np.array([jump[float(t)] for t in ts])
    lim = float(2 * np.trapezoid(st, ts) + np.trapezoid(jp, ts))
    limit = {"sqrt_mu": np.inf if np.isnan(lim) else lim}
    K, sn = min((r["K"]["sqrt_mu"], sn) for sn, r in rows.items())
    share = 2 * rows[sn]["sum_log_rho"] / K if np.isfinite(K) and K > 0 else None
    best = {"K": K, "schedule": sn, "family": "sqrt_mu", "jump_share": share}
    return rows, limit, best, acc


def oracle_rows(s, ref, sched, large, ts):
    tab = inf_if_nan(certificate_table(s, ref, sched, large))
    rows = {sn: {"K": c.K, "sum_log_rho": float(np.sum(c.log_rho)), "max_kappa": float(np.max(c.kappa))}
            for sn, c in tab.items()}
    fams = ("sqrt_mu",) if (large or is_huge(s)) else FAMILIES
    static = certify.continuous_certificate(s.gen, inv.reference_static_view(ref), ts, families=fams)
    with np.errstate(over="ignore", invalid="ignore"):
        jump = np.array([np.max(-ref.dlog_mu(t)) for t in ts])
    limit = {f: static.K[f] + float(np.trapezoid(jump, ts)) for f in fams}
    limit = {f: np.inf if np.isnan(v) else v for f, v in limit.items()}
    K, sn, f = best_of(tab)
    share = 2 * rows[sn]["sum_log_rho"] / K if np.isfinite(K) and K > 0 else None
    return rows, limit, {"K": K, "schedule": sn, "family": f, "jump_share": share}


def stage_exact(only=None):
    """The exact solution as the reference under the same schedules, with the limit of fine windows
    2 int CW(static p_t) dt + int max(-p_dot/p) dt."""
    for m in instances(only):
        t0 = time.time()
        s, leak, hinfo = published_setup(m, refs=None)
        large = is_large(s)
        sched = schedules(s.T)
        ts = np.linspace(0, s.T, N_CONT)
        if is_huge(s):
            rows, limit, best, _ = oracle_single_pass(s, sched, ts)
            out = {"schedules": rows, "riemann_limit": limit, "single_pass": True, "best": best}
        else:
            P = _Stream(s.gen.A, s.p0) if len(s.space) > 1_000_000 else \
                _TimeCache(s.gen.A, s.p0, np.concatenate([ts] + list(sched.values())))
            ref = reference.Exact(P, lambda t: s.gen.A @ P(t))
            rows, limit, best = oracle_rows(s, ref, sched, large, ts)
            out = {"schedules": rows, "riemann_limit": limit, "best": best}
        out.update({"T": s.T, "widths": s.widths, "n_states": len(s.space), "large_box_rule": large,
                    "very_large_box_rule": is_huge(s), "seconds": time.time() - t0})
        save(PART, f"{m.name}_exact", {**settings(), "model": m.name, "reference": "exact solution"}, out)
        print(f"  exact {m.name:16s} best K = {best['K']:.4g} ({best['schedule']}) [{time.time() - t0:.0f}s]", flush=True)


# ----------------------------------------------------------------------------- exact reference in log form
def log_p0(m, space):
    """log of the normalized R-P initial law, computed in log space."""
    mom = reference.MomentTrajectory(m, 1e-9)
    lm = reference.Product(space, mom, m.binary).log_mu(0.0)
    return lm - logsumexp(lm)


def forward_closure(A, start):
    """States reachable from `start` in the transition graph y -> x (A[x, y] > 0, x != y)."""
    G = sp.csr_matrix(A, copy=True)
    G.setdiag(0)
    G.eliminate_zeros()
    G.data = np.ones_like(G.data)
    reach = start.copy()
    while True:
        nxt = reach | (G @ reach.astype(float) > 0)
        if nxt.sum() == reach.sum():
            return reach
        reach = nxt


def propagate_scaled(M, v, times):
    """e^{M t} v at sorted times in scaled form (x_i, log_scale_i) with x_i normalized."""
    from qcme import linalg
    xs, ls = [], []
    x, lsc, tp = np.asarray(v, float).copy(), 0.0, 0.0
    for t in times:
        if t > tp:
            x, l = linalg.expm_action(M, x, t - tp)
            lsc += l
            tot = x.sum()
            x, lsc = x / tot, lsc + np.log(tot)
            tp = t
        xs.append(x.copy())
        ls.append(lsc)
    return xs, ls


class ExactLog:
    """mu_t = p_t, with log p at t = 0 from the log-domain initial law, and the states not reachable from its
    representable support (set U, which receives no flux from the rest) propagated by e^{A_UU t} in scaled form.
    Other states: FSP (uniformization, entrywise accurate) from the usual p0."""

    name = "exact solution (log-domain initial law)"

    def __init__(self, s, times):
        A = s.gen.A
        self.A = A
        self.lp0 = log_p0(s.model, s.space)
        self.R = forward_closure(A, self.lp0 > REPR_FLOOR)
        self.U = ~self.R
        self.times = np.array(sorted(set(float(t) for t in times)))
        P = fsp.propagate(A, s.p0, self.times)
        self.lm = {}
        self.acc = {"min_p_R_t_pos": np.inf, "times_below_floor": [], "zeros_after_U": 0, "n_U": int(self.U.sum())}
        if self.U.any():
            iu = np.nonzero(self.U)[0]
            AU = sp.csr_matrix(A[iu][:, iu])
            cU = float(self.lp0[iu].max())
            qU, lsU = propagate_scaled(AU, np.exp(self.lp0[iu] - cU), self.times)
        for i, t in enumerate(self.times):
            if t == 0.0:
                self.lm[t] = self.lp0.copy()
                continue
            with np.errstate(divide="ignore"):
                lm = np.log(P[i])
            if self.U.any():
                with np.errstate(divide="ignore"):
                    lm[iu] = cU + lsU[i] + np.log(qU[i])
            pr = P[i][self.R]
            mn = float(pr.min()) if pr.size else np.inf
            self.acc["min_p_R_t_pos"] = min(self.acc["min_p_R_t_pos"], mn)
            if mn < ACC_FLOOR:
                self.acc["times_below_floor"].append(t)
            self.acc["zeros_after_U"] = max(self.acc["zeros_after_U"], int(np.sum(~np.isfinite(lm))))
            self.lm[t] = lm

    def _at(self, t):
        t = float(t)
        if t in self.lm:
            return self.lm[t]
        k = float(self.times[np.argmin(np.abs(self.times - t))])
        if abs(k - t) > 1e-12 * max(1.0, abs(t)):
            raise KeyError(f"time {t} not in the cache")
        return self.lm[k]

    def log_mu(self, t):
        return self._at(t)

    def dlog_mu(self, t):
        with np.errstate(over="ignore", invalid="ignore"):
            return certify.residual(self.A, self._at(t))


def stage_exact_log(only=None):
    names = only or ["toggle_kuntz", "feedback_b1", "feedback_b10", "feedback_b100", "toggle_tang", "schlogl",
                     "toggle_hegland", "autoreg_tang_1", "autoreg_tang_2", "autoreg_tang_3", "toggle_gardner"]
    for name in names:
        t0 = time.time()
        m = published_model(name)
        s, leak, hinfo = published_setup(m, refs=None)
        large = is_large(s)
        sched = schedules(s.T)
        ts = np.linspace(0, s.T, N_CONT)
        if is_huge(s):
            rows, limit, best, acc = oracle_single_pass(s, sched, ts, log_domain=True)
        else:
            ref = ExactLog(s, np.concatenate([ts] + list(sched.values())))
            rows, limit, best = oracle_rows(s, ref, sched, large, ts)
            acc = ref.acc
        save(PART, f"{name}_exact_log", {**settings(), "model": name, "acc_floor": ACC_FLOOR, "repr_floor": REPR_FLOOR},
             {"T": s.T, "widths": s.widths, "n_states": len(s.space), "n_U": acc["n_U"], "accuracy": acc,
              "entrywise_accurate": not acc["times_below_floor"], "schedules": rows, "riemann_limit": limit,
              "best": best, "very_large_box_rule": is_huge(s), "large_box_rule": large, "seconds": time.time() - t0})
        print(f"  exact (log) {name:16s} best K = {best['K']:.6g} ({best['schedule']}) [{time.time() - t0:.0f}s]",
              flush=True)


def log_matvec(P, logvals, lx):
    """log(P x) for P >= 0 (CSR, zero entries removed) given log x, by a per-row logsumexp."""
    n = P.shape[0]
    counts = np.diff(P.indptr)
    row_of = np.repeat(np.arange(n), counts)
    v = logvals + lx[P.indices]
    nonempty = counts > 0
    starts = P.indptr[:-1][nonempty]
    mm = np.full(n, -np.inf)
    mm[nonempty] = np.maximum.reduceat(v, starts)
    mrow = mm[row_of]
    fin = np.isfinite(mrow)
    with np.errstate(invalid="ignore", over="ignore"):
        ex = np.where(fin, np.exp(v - np.where(fin, mrow, 0.0)), 0.0)
    out = np.full(n, -np.inf)
    ssum = np.add.reduceat(ex, starts)
    with np.errstate(divide="ignore"):
        out[nonempty] = np.where(np.isfinite(mm[nonempty]), mm[nonempty] + np.log(ssum), -np.inf)
    return out


def log_uniformization(A, lp0, times, stop=LOGU_STOP, max_terms=200_000):
    """log p(t) at sorted times for p' = A p from log p(0) = lp0, entrywise exact in log form:
    p(t) = sum_k Pois(k; q dt) P^k p_prev with P = I + A/q >= 0, every term handled as a logarithm."""
    from scipy.special import gammaln
    A = sp.csr_matrix(A)
    q = float(np.max(np.abs(A.diagonal())))
    P = (sp.identity(A.shape[0], format="csr") + A / q).tocsr()
    P.eliminate_zeros()
    if P.data.min() < 0:
        raise ValueError("P = I + A/q must be non-negative")
    logvals = np.log(P.data)
    lp, tp, out = np.asarray(lp0, float).copy(), 0.0, {}
    for t in times:
        t = float(t)
        if t > tp:
            dt = t - tp
            nsub = max(1, int(np.ceil(q * dt / LOGU_MAX_STEP)))
            lam = q * dt / nsub
            for _ in range(nsub):
                acc = np.full_like(lp, -np.inf)
                term = lp.copy()
                small = 0
                for k in range(max_terms):
                    contrib = -lam + k * np.log(lam) - gammaln(k + 1) + term
                    acc = np.logaddexp(acc, contrib)
                    if k > lam:
                        with np.errstate(invalid="ignore"):
                            d = np.where(np.isneginf(contrib), -np.inf, contrib - acc)
                        small = small + 1 if np.max(d) < stop else 0
                        if small >= 3:
                            break
                    term = log_matvec(P, logvals, term)
                else:
                    raise RuntimeError("log-domain uniformization did not converge")
                lp = acc
            tp = t
        out[t] = lp.copy()
    return out


class ExactLogFull:
    """mu_t = p_t with log p from log-domain uniformization at every time."""

    name = "exact solution (log-domain uniformization)"

    def __init__(self, A, lp_by_t):
        self.A, self.lm = A, lp_by_t
        self.times = np.array(sorted(lp_by_t))

    def log_mu(self, t):
        t = float(t)
        if t in self.lm:
            return self.lm[t]
        k = float(self.times[np.argmin(np.abs(self.times - t))])
        if abs(k - t) > 1e-12 * max(1.0, abs(t)):
            raise KeyError(f"time {t} not in the cache")
        return self.lm[k]

    def dlog_mu(self, t):
        with np.errstate(over="ignore", invalid="ignore"):
            return certify.residual(self.A, self.log_mu(t))


def stage_exact_log_uniformization(only=None):
    """Validate log-domain uniformization against ordinary uniformization on feedback_b1 and autoreg_tang_2, then
    recompute the exact reference of toggle_kuntz with it at every time."""
    out = {"validation": {}}
    for name in ("feedback_b1", "autoreg_tang_2"):
        t0 = time.time()
        m = published_model(name)
        s, leak, hinfo = published_setup(m, refs=None)
        sched = schedules(s.T)
        times = sorted(set(float(x) for x in np.concatenate([np.linspace(0, s.T, N_CONT)] + list(sched.values()))))
        lp = log_uniformization(s.gen.A, log_p0(m, s.space), times)
        P = fsp.propagate(s.gen.A, s.p0, times)
        worst = 0.0
        for i, t in enumerate(times):
            ok = P[i] >= 1e-250
            if t > 0:
                worst = max(worst, float(np.max(np.abs(lp[t][ok] - np.log(P[i][ok])))))
        out["validation"][name] = {"max_abs_log_diff": worst, "pass": worst <= 1e-9, "seconds": time.time() - t0}
        print(f"  validation {name}: max |d log p| = {worst:.3g}", flush=True)
    cfg = {**settings(), "stop": LOGU_STOP, "max_step": LOGU_MAX_STEP}
    if all(v["pass"] for v in out["validation"].values()):
        t0 = time.time()
        m = published_model("toggle_kuntz")
        s, leak, hinfo = published_setup(m, refs=None)
        sched = schedules(s.T)
        ts = np.linspace(0, s.T, N_CONT)
        times = sorted(set(float(x) for x in np.concatenate([ts] + list(sched.values()))))
        lp = log_uniformization(s.gen.A, log_p0(m, s.space), times)
        rows, limit, best = oracle_rows(s, ExactLogFull(s.gen.A, lp), sched, is_large(s), ts)
        out["toggle_kuntz"] = {"all_log_p_finite": all(np.all(np.isfinite(v)) for v in lp.values()),
                               "min_log10_p": float(min(np.min(v) for v in lp.values()) / np.log(10)),
                               "schedules": rows, "riemann_limit": limit, "best": best, "seconds": time.time() - t0}
        print(f"  toggle_kuntz exact reference: K = {best['K']:.8g}", flush=True)
    save(PART, "exact_log_uniformization", cfg, out)


def log_cw_sqrt_lower(A, leak, lm):
    """log of a lower bound on CW_sqrt(mu) = (1/2) max_x (r_x - leak_x), in log form so that it survives when r
    overflows double precision."""
    A = sp.coo_matrix(A)
    off = A.row != A.col
    rows, cols, vals = A.row[off], A.col[off], A.data[off]
    lt = np.log(vals) + lm[cols] - lm[rows]
    order = np.argsort(rows, kind="stable")
    rows, lt = rows[order], lt[order]
    starts = np.r_[0, np.nonzero(np.diff(rows))[0] + 1]
    lsum = np.full(A.shape[0], -np.inf)
    lsum[rows[starts]] = np.logaddexp.reduceat(lt, starts)
    diag = np.asarray(sp.csr_matrix(A).diagonal())
    out = np.where(lsum > 700, lsum, np.log(np.maximum(np.exp(np.minimum(lsum, 700)) + diag - leak, 1e-300)))
    return float(np.log(0.5) + np.max(out))


def stage_overflow_bounds(only=None):
    """Where the exact-reference K_T is +inf in double precision: log10 of a lower bound from the first window alone,
    K >= 2 CW_0 (t_1 - t_0), for every schedule (the smallest over schedules is reported)."""
    out = {}
    for name in only or ["feedback_b10", "feedback_b100"]:
        m = published_model(name)
        s, leak, hinfo = published_setup(m, refs=None)
        lc = log_cw_sqrt_lower(s.gen.A, s.gen.leak, log_p0(m, s.space))
        per = {sn: (np.log(2) + lc + np.log(ed[1] - ed[0])) / np.log(10) for sn, ed in schedules(s.T).items()}
        out[name] = {"log10_CW0_lower": lc / np.log(10), "log10_K_lower_by_schedule": per,
                     "log10_K_lower_min": min(per.values())}
        print(f"  {name}: log10 K >= {min(per.values()):.2f}", flush=True)
    save(PART, "exact_overflow_bounds", settings(), out)


# ----------------------------------------------------------------------------- large boxes: W and the resolvent
def scaled_power_W(A, lm0, lmT, T, ckpt, timebox=TIMEBOX_S):
    """sigma_1(D_T^{-1/2} e^{A T} D_0^{1/2}) by power iteration on M^T M (M >= 0 entrywise), with the scale
    factored out: M = e^c diag(sT) e^{AT} diag(s0), s0, sT <= 1."""
    a0, aT = 0.5 * np.asarray(lm0, float), -0.5 * np.asarray(lmT, float)
    c0, cT = float(a0.max()), float(aT.max())
    r0, rT = a0 - c0, aT - cT
    drop = r0 < -W_DROP
    s0 = np.where(drop, 0.0, np.exp(np.maximum(r0, -W_DROP)))
    sT = np.exp(rT)
    logc = c0 + cT
    log_drop_F = logc + 0.5 * float(logsumexp(2 * r0[drop])) if drop.any() else -np.inf
    At = sp.csr_matrix(A.T)
    state = {"v": None, "sigmas": [], "elapsed": 0.0}
    if ckpt.exists():
        z = np.load(ckpt, allow_pickle=True)
        state = {"v": z["v"], "sigmas": list(z["sigmas"]), "elapsed": float(z["elapsed"])}
    v = state["v"] if state["v"] is not None else np.where(drop, 0.0, 1.0)
    v = v / np.linalg.norm(v)
    sig = state["sigmas"]

    def converged():
        return len(sig) >= 4 and all((sig[-k] - sig[-k - 1]) / sig[-k - 1] < W_TOL for k in (1, 2, 3))
    while not converged() and state["elapsed"] <= timebox:
        t0 = time.time()
        u = sT * fsp.propagate(A, s0 * v, [T])[-1]
        s_now = float(np.linalg.norm(u))
        w = s0 * fsp.propagate(At, sT * u, [T])[-1]
        v = w / np.linalg.norm(w)
        sig.append(s_now)
        state["elapsed"] += time.time() - t0
        np.savez(ckpt, v=v, sigmas=np.array(sig), elapsed=state["elapsed"])
        print(f"   iteration {len(sig)}: log10 W = {(logc + np.log(s_now)) / np.log(10):.10f}", flush=True)
    log_W = logc + np.log(sig[-1])
    upper = np.logaddexp(log_W, log_drop_F)
    return {"log_W": log_W, "log10_W": log_W / np.log(10), "converged": converged(), "iterations": len(sig),
            "hours": state["elapsed"] / 3600, "n_dropped_columns": int(drop.sum()),
            "log10_dropped_F_bound": log_drop_F / np.log(10) if np.isfinite(log_drop_F) else None,
            "log10_W_upper_if_converged": upper / np.log(10),
            "sigma_history_log10": [(logc + np.log(x)) / np.log(10) for x in sig]}


def stage_worst_case(only=None):
    for name in only or ["toggle_hegland", "lambda_phage", "toggle_gardner"]:
        t0 = time.time()
        d = data(PART, name)
        rn, sn, fam = d["best"]["reference"], d["best"]["schedule"], d["best"]["family"]
        s = build_setup(published_model(name), tuple(d["widths"]), d["T"])
        ref = s.refs[rn]
        ckpt = OUT / PART / "partial" / f"{name}_worst_case.npz"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        res = scaled_power_W(s.gen.A, ref.log_mu(0.0), ref.log_mu(s.T), s.T, ckpt)
        meas_c = num(d["references"][rn]["measured_log10_growth_continuous"])
        Rc, Rs = num(d["references"][rn]["log10_R_continuous"][fam]), num(d["best"]["log10_R"])
        slack = res["log10_W"] - meas_c
        res.update({"winner": rn, "family": fam, "schedule": sn, "measured_log10_growth_continuous": meas_c,
                    "decomposition": {"window_penalty": Rs - Rc, "non_normality": Rc - slack, "instance_slack": slack,
                                      "label": "exact" if res["converged"] else
                                      "W lower bound: non-normality is an upper bound, slack a lower bound"},
                    "seconds": time.time() - t0})
        save(PART, f"{name}_worst_case", {**settings(), "model": name, "w_tol": W_TOL, "w_drop": W_DROP,
                                          "timebox_s": TIMEBOX_S}, res)


def stage_resolvent(only=None):
    """The resolvent weight on the best sqrt(mu) reference and schedule, window by window (checkpointed)."""
    for name in only or ["lambda_phage", "toggle_gardner"]:
        t0 = time.time()
        d = data(PART, name)
        rn, sn = d["best"]["reference"], d["best"]["schedule"]
        s = build_setup(published_model(name), tuple(d["widths"]), d["T"])
        ref = s.refs[rn]
        edges = schedules(s.T)[sn]
        A, leak = s.gen.A, s.gen.leak
        J = len(edges) - 1
        ck = OUT / PART / "partial" / f"{name}_resolvent.json"
        ck.parent.mkdir(parents=True, exist_ok=True)
        st = json.loads(ck.read_text()) if ck.exists() else {"cw_sqrt": [], "cw_res": [], "elapsed": 0.0}
        timed_out = False
        for j in range(len(st["cw_res"]), J):
            t1 = time.time()
            lm = ref.log_mu(edges[j])
            with np.errstate(over="ignore"):
                r = certify.residual(A, lm)
            c_sqrt = certify.cw_sqrt_mu(r, leak)
            Sigma = certify.symmetric_part(certify.weighted_generator(A, lm))
            c_res = min(c_sqrt, certify.resolvent_cw(Sigma, c_sqrt, 1e-6).value)
            st["cw_sqrt"].append(float(c_sqrt))
            st["cw_res"].append(float(c_res))
            st["elapsed"] += time.time() - t1
            ck.write_text(json.dumps(st))
            print(f"   window {j}: CW sqrt_mu {c_sqrt:.6g}, resolvent {c_res:.6g}", flush=True)
            if st["elapsed"] > TIMEBOX_S:
                timed_out = True
                break
        log_rho = [0.5 * float(np.max(ref.log_mu(edges[j - 1]) - ref.log_mu(edges[j]))) for j in range(1, J)]
        dt = np.diff(edges)
        k = len(st["cw_res"])
        out = {"winner": rn, "schedule": sn, "windows_done": k, "windows": J, "timed_out": timed_out,
               "hours": st["elapsed"] / 3600, "cw_sqrt_mu": st["cw_sqrt"], "cw_resolvent": st["cw_res"],
               "K_sqrt_mu_best": num(d["best"]["K"]), "seconds": time.time() - t0}
        if k == J:
            K_res = float(2 * np.sum(np.array(st["cw_res"]) * dt) + 2 * np.sum(log_rho))
            K_sq = float(2 * np.sum(np.array(st["cw_sqrt"]) * dt) + 2 * np.sum(log_rho))
            out.update({"K_resolvent": K_res, "K_sqrt_mu_recomputed": K_sq, "K_best_min_family": min(K_res, K_sq)})
        save(PART, f"{name}_resolvent", {**settings(), "model": name, "timebox_s": TIMEBOX_S}, out)


# ----------------------------------------------------------------------------- summary
def stage_summary(only=None):
    """Current values per instance: the best cheap K_T (with the resolvent weight where it was computed only on the
    best reference), the exact-reference K_T (log form where it applies), and the W split of the large boxes."""
    over = data(PART, "exact_overflow_bounds") or {}
    logu = data(PART, "exact_log_uniformization") or {}
    rows = {}
    for m in models.published():
        name = m.name
        dec, ora, exl = data(PART, name), data(PART, f"{name}_exact"), data(PART, f"{name}_exact_log")
        cheap, cheap_note = num(dec["best"]["K"]), "best of all references"
        res = data(PART, f"{name}_resolvent")
        mixed = None
        if res and res.get("K_best_min_family") is not None and res["K_best_min_family"] < cheap:
            cheap, cheap_note = res["K_best_min_family"], "resolvent weight on the best sqrt(mu) reference"
        if res and res.get("timed_out"):
            cs, cr = np.array(res["cw_sqrt_mu"]), np.array(res["cw_resolvent"])
            dt = np.diff(schedules(dec["T"])[res["schedule"]])[:len(cs)]
            mixed = {"label": "resolvent weight on the finished windows, sqrt(mu) on the rest",
                     "windows_resolvent": len(cs), "windows": res["windows"],
                     "K": num(res["K_sqrt_mu_best"]) - 2 * float(np.sum((cs - cr) * dt))}
        ex, ex_note = (num(ora["best"]["K"]) if ora else None), "standard finite state projection"
        if exl is not None:
            ex, ex_note = num(exl["best"]["K"]), "log-domain initial law"
            if name == "toggle_kuntz" and "toggle_kuntz" in logu:
                ex, ex_note = num(logu["toggle_kuntz"]["best"]["K"]), "log-domain uniformization, entrywise exact"
            if not np.isfinite(ex):
                if name in over:
                    ex_note = f"beyond double range, log10 K >= {over[name]['log10_K_lower_min']:.1f}"
                elif exl["accuracy"]["zeros_after_U"]:
                    ex_note = f"not computable: p_t underflows on {exl['accuracy']['zeros_after_U']:,} states at t > 0"
        W = data(PART, f"{name}_worst_case")
        rows[name] = {"cheap_K": cheap, "cheap_note": cheap_note, "exact_K": ex, "exact_note": ex_note,
                      "W_split": None if W is None or name not in ("toggle_gardner", "lambda_phage")
                      else {**W["decomposition"], "log10_W": W["log10_W"], "converged": W["converged"]},
                      "mixed_certificate": mixed}
    save(PART, "summary", settings(), {"rows": rows})


STAGES = {"references": stage_references, "exact": stage_exact, "exact_log": stage_exact_log,
          "exact_log_uniformization": stage_exact_log_uniformization, "overflow_bounds": stage_overflow_bounds,
          "worst_case": stage_worst_case, "resolvent": stage_resolvent, "summary": stage_summary}

if __name__ == "__main__":
    args = sys.argv[1:]
    only = args[args.index("--only") + 1].split(",") if "--only" in args else None
    for a in [a for a in args if a in STAGES]:
        STAGES[a](only=only)
