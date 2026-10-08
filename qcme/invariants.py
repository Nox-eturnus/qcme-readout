"""Numerical identities I1-I7, I10 and I11 of the weighted encoding and its certificates.

Each function returns plain numbers so the tests can apply their
tolerances. `setup` builds everything for one model on a given box.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from . import certify, emulate, fsp, network, reference
from .models import relaxation_time

TOL = {"I1": 1e-10, "I2": 1e-10, "I3": 1e-12, "I4": 1e-10, "I5": 1e-9, "I6": 1e-12, "I10": 0.0}


@dataclass
class Setup:
    model: object
    widths: tuple
    T: float
    space: network.StateSpace
    gen: network.TruncatedGenerator
    p0: np.ndarray
    refs: dict


def setup(model, widths=None, T=None, families="core"):
    """families: "core" (R-P, R-NB and their mixtures) or "extended" (adds R-LNA, the E1/E2 mixtures and R-TT)."""
    widths = tuple(widths or model.widths)
    T = T if T is not None else relaxation_time(model)[0]
    space = model.space(widths)
    gen = network.generator(space, model.reactions(widths))
    p0 = reference.initial_law(model, space)
    if families == "core":
        refs = reference.standard_references(model, space, T, widths)
    elif families == "extended":
        refs = reference.extended_references(model, space, T, widths, gen=gen)
    else:
        raise ValueError(families)
    return Setup(model, widths, T, space, gen, p0, refs)


# ----------------------------------------------------------------------------- I1
def I1_power_traces(s, ref_name="R-mix(NB)", t=None, jmax=None):
    """tr(B^j) = tr(A_S^j), j = 1..|S|, static reference mu(t). Relative error."""
    ref = s.refs[ref_name]
    lm = ref.log_mu(s.T / 2 if t is None else t)
    A = s.gen.A.toarray()
    B = certify.weighted_generator(s.gen.A, lm).toarray()
    scale = np.abs(A).max()
    n = len(A)
    Aj, Bj = np.eye(n), np.eye(n)
    worst = 0.0
    for _ in range(jmax or n):
        Aj, Bj = Aj @ A / scale, Bj @ B / scale
        ta, tb = np.trace(Aj), np.trace(Bj)
        worst = max(worst, abs(ta - tb) / max(abs(ta), 1e-300))
    return worst


# ----------------------------------------------------------------------------- I2-I4
@dataclass
class PiecewiseCheck:
    I2: float  # max rel | ||psi||^2 - (m^2/Z)(1 + chi2) |, psi emulated with B, (m, chi2) from FSP
    I2_state: float  # max rel || psi_emul - p/sqrt(mu) || / ||p/sqrt(mu)||
    I3_violation: dict  # family -> max over t of (||psi_t|| / bound_t - 1); must be <= 1e-12
    I3_kappa_violation: float
    I4: float  # max | <sqrt mu, psi> - m_t |
    stabilized_max_lognorm: dict  # family -> max log-norm of the stabilized state (<= 1e-12)
    stabilized_recon: dict  # family -> rel error of psi_T reconstructed from the stabilized run
    #                         (None when |K_T| > 1e6: log ||psi_T|| = K_T/2 + (-K_T/2 + O(1)) then
    #                         cancels catastrophically in double precision; not an emulator error)
    cert: object
    run: object


def piecewise_checks(s, ref_name, J=20, sub=5, edges=None):
    ref = s.refs[ref_name]
    edges = np.linspace(0.0, s.T, J + 1) if edges is None else np.asarray(edges, float)
    cert = certify.window_certificate(s.gen, ref, edges)
    run = emulate.piecewise_psi(s.gen, ref, edges, s.p0, sub)
    P = fsp.propagate(s.gen.A, s.p0, run.times)
    i2 = i2s = i4 = 0.0
    viol = {f: -np.inf for f in cert.K}
    viol_k = -np.inf
    n0 = np.linalg.norm(run.psi[0])
    for psi, lm, p, t in zip(run.psi, run.logmu, P, run.times):
        mu = np.exp(lm)
        m, Z = p.sum(), mu.sum()
        chi2 = np.sum((p / m - mu / Z) ** 2 / (mu / Z))
        nsq = psi @ psi
        i2 = max(i2, abs(nsq - m**2 / Z * (1 + chi2)) / nsq)
        exact = p * np.exp(-0.5 * lm)
        i2s = max(i2s, np.linalg.norm(psi - exact) / np.linalg.norm(exact))
        i4 = max(i4, abs(emulate.readout(psi, lm) - m))
        lr = np.log(np.linalg.norm(psi) / n0)  # measured log growth
        for f in cert.K:
            viol[f] = max(viol[f], np.expm1(lr - 0.5 * cert.K_at(f, t)))
        viol_k = max(viol_k, np.expm1(lr - 0.5 * cert.K_kappa_at(t)))
    smax, srec = {}, {}
    psiT = run.psi[-1]
    nT = np.linalg.norm(psiT)
    for f in cert.K:
        lognorms, log_psiT, phi = emulate.stabilized_psi(s.gen, ref, cert, f, s.p0)
        smax[f] = float(lognorms.max())
        srec[f] = (float(max(abs(log_psiT - np.log(nT)), np.linalg.norm(phi - psiT / nT)))
                   if abs(cert.K[f]) <= 1e6 else None)
    return PiecewiseCheck(i2, i2s, viol, viol_k, i4, smax, srec, cert, run)


def continuous_I2(s, ref_name, n_t=11):
    """I2 with the continuous reference: psi integrated from B_t (DOP853) vs FSP p."""
    ref = s.refs[ref_name]
    ts = np.linspace(0.0, s.T, n_t)
    psis = emulate.continuous_psi(s.gen, ref, s.p0, ts)
    P = fsp.propagate(s.gen.A, s.p0, ts)
    worst = 0.0
    for psi, p, t in zip(psis, P, ts):
        mu = np.exp(ref.log_mu(t))
        m, Z = p.sum(), mu.sum()
        chi2 = np.sum((p / m - mu / Z) ** 2 / (mu / Z))
        worst = max(worst, abs(psi @ psi - m**2 / Z * (1 + chi2)) / (psi @ psi))
    return worst


# ----------------------------------------------------------------------------- I5, I6
def I5_exact_reference(s, n_t=11):
    """R-exact, mu_t = p_t unnormalized: psi = sqrt(p) and kappa = 0 (joint ODE for psi and p).

    Returns (max |psi - sqrt p| / max sqrt p, max r^+ relative to the residual's term scale)."""
    A = s.gen.A
    n = A.shape[0]
    Aabs = abs(A)

    def rhs(t, y):
        psi, p = y[:n], y[n:]
        pd = A @ p
        lm = np.log(p)
        return np.concatenate([certify.weighted_generator(A, lm, pd / p) @ psi, pd])

    ts = np.linspace(0.0, s.T, n_t)
    sol = solve_ivp(rhs, (0, s.T), np.concatenate([np.sqrt(s.p0), s.p0]), method="DOP853", t_eval=ts,
                    rtol=2.5e-14, atol=1e-300)  # pure relative control near the DOP853 floor (notes 2.1)
    e_psi = e_k = 0.0
    for j in range(len(ts)):
        psi, p = sol.y[:n, j], sol.y[n:, j]
        e_psi = max(e_psi, np.abs(psi - np.sqrt(p)).max() / np.sqrt(p).max())
        r = certify.residual(A, np.log(p), (A @ p) / p)
        term_scale = (Aabs @ p) / p
        e_k = max(e_k, float(np.max(np.maximum(r, 0.0) / term_scale)))
    return e_psi, e_k


def I6_uniform(s):
    """Uniform mu: B = A_S and psi_T = sqrt|S| p_T."""
    n = len(s.space)
    u = reference.Uniform(n)
    B = certify.weighted_generator(s.gen.A, u.log_mu(0.0), u.dlog_mu(0.0))
    run = emulate.piecewise_psi(s.gen, u, np.array([0.0, s.T]), s.p0, sub=1)
    pT = fsp.propagate(s.gen.A, s.p0, [s.T])[-1]
    return max(abs(B - s.gen.A).max(), np.abs(run.psi[-1] - np.sqrt(n) * pT).max())


# ----------------------------------------------------------------------------- I7
def I7_exact_product_poisson(s, n_t=26):
    """Monomolecular networks with a Poisson start: the time-dependent R-P reference is the exact
    law on N^d (Jahnke-Huisinga 2007), so interior r = 0 and r <= 0 on the boundary (Prop. 4).
    Returns (max |r| at interior states, max r over all states)."""
    ref = s.refs["R-P"]
    widths = np.array(s.widths)
    interior = np.all(s.space.states < widths - 1, axis=1)
    worst_int, worst_all = 0.0, -np.inf
    for t in np.linspace(0.0, s.T, n_t)[1:]:
        r = certify.residual(s.gen.A, ref.log_mu(t), ref.dlog_mu(t))
        worst_int = max(worst_int, np.abs(r[interior]).max())
        worst_all = max(worst_all, r.max())
    return worst_int, worst_all


# ----------------------------------------------------------------------------- I10
def I10_conservation(model, widths=None, T=None):
    """Full product box versus the conservation manifold.

    Returns (mass that leaves the manifold in the full box - must be exactly 0,
    max |p_full restricted - p_manifold|)."""
    widths = tuple(widths or model.widths)
    T = T if T is not None else relaxation_time(model)[0]
    L, c = model.conservation
    man = model.space(widths)
    full = network.StateSpace.box(widths)
    p0_man = reference.initial_law(model, man)
    p0_full = np.zeros(len(full))
    p0_full[full.lookup(man.states)] = p0_man
    rx = model.reactions(widths)
    A_full = network.generator(full, rx).A
    A_man = network.generator(man, rx).A
    ts = np.linspace(0, T, 6)
    P_full = fsp.propagate(A_full, p0_full, ts)
    P_man = fsp.propagate(A_man, p0_man, ts)
    off = ~np.all(full.states @ L.T == c, axis=1)
    idx = full.lookup(man.states)
    return float(np.abs(P_full[:, off]).max()), float(np.abs(P_full[:, idx] - P_man).max())


# ----------------------------------------------------------------------------- I11
def I11_windows(s, ref_name, Js=(1, 2, 4, 8, 16, 32, 64), n_cont=257):
    """Piecewise-constant versus continuous references as windows shrink.

    (a) state: psi_T from the piecewise emulation, re-expressed in mu(T), versus the continuous
        ODE psi_T (both are exact representations of p_S, so they must agree for every J);
    (b) certificate: K_T^(J) for each family, the continuous estimate 2 int CW_t, and the Riemann
        limit of the piecewise scheme, 2 int CW_t^static + int max(-mu_dot/mu), which is >= the
        continuous value because a max of a sum is <= the sum of maxima.
    """
    ref = s.refs[ref_name]
    lmT = ref.log_mu(s.T)
    psi_c = emulate.continuous_psi(s.gen, ref, s.p0, np.array([0.0, s.T]))[-1]
    state_err, K = [], {f: [] for f in certify.FAMILIES}
    for J in Js:
        edges = np.linspace(0, s.T, J + 1)
        run = emulate.piecewise_psi(s.gen, ref, edges, s.p0, sub=1)
        psi_pw = run.psi[-1] * np.exp(0.5 * (run.logmu[-1] - lmT))
        state_err.append(float(np.linalg.norm(psi_pw - psi_c) / np.linalg.norm(psi_c)))
        cert = certify.window_certificate(s.gen, ref, edges)
        for f in K:
            K[f].append(cert.K[f])
    ts = np.linspace(0, s.T, n_cont)
    cc = certify.continuous_certificate(s.gen, ref, ts)
    static = certify.continuous_certificate(s.gen, reference_static_view(ref), ts)
    jump_rate = np.array([np.max(-ref.dlog_mu(t)) for t in ts])
    limit = {f: static.K[f] + float(np.trapezoid(jump_rate, ts)) for f in K}
    return dict(Js=list(Js), state_err=state_err, K=K, K_cont=cc.K, K_pw_limit=limit)


class reference_static_view(reference.Reference):
    """The same mu_t with mu_dot dropped (the generator used inside each window)."""

    def __init__(self, ref):
        self.ref, self.name = ref, ref.name + " (static view)"

    def log_mu(self, t):
        return self.ref.log_mu(t)

    def dlog_mu(self, t):
        return np.zeros_like(self.ref.log_mu(t))
