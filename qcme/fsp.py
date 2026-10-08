"""Finite state projection: propagation, retained mass, box selection and observable intervals.

Munsky & Khammash (2006) Thm 2.2: if m_T >= 1 - eps then p_S <= p_T <= p_S + eps elementwise on S.
"""
import numpy as np
from scipy.sparse.linalg import expm_multiply

from .linalg import expm_apply
from .network import generator


def propagate(A, p0, times, method="uniformization"):
    """p_S(t) for sorted times (t >= 0), step by step. Shape (len(times), n).

    "uniformization" (default) is entrywise relatively accurate (qcme.linalg), which the weighted
    comparisons need; "krylov" is SciPy's normwise-accurate expm_multiply."""
    times = np.asarray(times, float)
    out = np.zeros((len(times), len(p0)))
    p, t_prev = np.asarray(p0, float).copy(), 0.0
    for i, t in enumerate(times):
        if t > t_prev:
            if method == "uniformization":
                p = expm_apply(A, p, t - t_prev)
            elif method == "krylov":
                p = expm_multiply(A * (t - t_prev), p)
            else:
                raise ValueError(method)
            t_prev = t
        out[i] = p
    return out


def retained_mass(p):
    return float(np.sum(p))


def select_box(model, p0_fn, T, tol=1e-6, grow=1.25, max_states=2_000_000):
    """Smallest box in the geometric schedule widths * grow^j (binary and conserved-bounded
    species fixed) whose FSP leak 1 - m_T is <= tol. Returns (widths, leak)."""
    widths = np.array(model.widths, dtype=int)
    fixed = np.array(model.binary) | _bounded_by_conservation(model)
    while True:
        space = model.space(tuple(widths))
        if len(space) > max_states:
            raise RuntimeError(f"{model.name}: box {tuple(widths)} exceeds {max_states} states")
        gen = generator(space, model.reactions(tuple(widths)))
        pT = propagate(gen.A, p0_fn(space), [T])[-1]
        leak = 1.0 - retained_mass(pT)
        if leak <= tol or np.all(fixed):
            return tuple(int(w) for w in widths), leak
        widths = np.where(fixed, widths, np.ceil(widths * grow).astype(int))


def _bounded_by_conservation(model):
    if model.conservation is None:
        return np.zeros(len(model.species), bool)
    L, _ = model.conservation
    # a species is bounded if it appears with a positive coefficient in a non-negative law
    bounded = np.zeros(L.shape[1], bool)
    for row in L:
        if np.all(row >= 0):
            bounded |= row > 0
    return bounded


def bounded_interval(fS, pS, f_max, f_min=0.0):
    """Interval for E_{p_T}[f] of a bounded observable f_min <= f <= f_max (T2, last display)."""
    m = pS.sum()
    E = float(fS @ pS)
    return E + (1 - m) * f_min, E + (1 - m) * f_max


def unbounded_interval(fS, pS, M2):
    """Interval for an observable of the full law: E_S f <= E_p f <= E_S f + (1 - m) max_S f + sqrt(M2 (1 - m)),
    for f >= 0 with E_{p_T}[V] <= M2 and V >= f^2."""
    m = pS.sum()
    E = float(fS @ pS)
    return E, E + (1 - m) * float(fS.max()) + np.sqrt(M2 * max(1 - m, 0.0))


def drift_M2_production_degradation(beta_max, gamma, EV0, T):
    """M2(T) for V = sum_i x_i^2 when species i is produced at total rate <= beta_max[i] by unit
    births and removed at rate gamma[i] x_i by unit deaths (other moves of i are absent).

    Per species: A x^2 <= beta (2x + 1) + gamma x (1 - 2x) <= c - gamma x^2 with
    c = beta + (beta + gamma/2)^2 / gamma. Summing with a common
    rate gamma_min = min gamma_i gives A V <= sum c_i - gamma_min V.
    """
    beta_max, gamma = np.asarray(beta_max, float), np.asarray(gamma, float)
    c = float(np.sum(beta_max + (beta_max + gamma / 2) ** 2 / gamma))
    g = float(gamma.min())
    return np.exp(-g * T) * EV0 + (c / g) * (1 - np.exp(-g * T))
