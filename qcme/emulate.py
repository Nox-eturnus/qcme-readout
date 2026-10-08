"""Algorithm-level emulation of the weighted encoding.

The weighted state psi = p_S / sqrt(mu) is propagated with the weighted generator B (not with
A_S), including the diagonal reweighting at window boundaries, so the invariants compare an
independent computation against FSP. LCHS quadrature and gate-level circuits are in `lchs` and `circuits`.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import LinearOperator, expm_multiply, svds

from .certify import weighted_generator
from .linalg import expm_action, expm_apply


@dataclass
class PiecewiseRun:
    times: np.ndarray  # sample times (window sub-steps), starting at 0
    psi: np.ndarray  # psi at each time, in the reference of the window containing it
    logmu: np.ndarray  # reference used at each time
    window: np.ndarray  # window index at each time


def piecewise_psi(gen, ref, edges, p0, sub=5):
    """Propagate psi through windows [t_j, t_{j+1}) with static mu^(j) = mu(t_j).

    At t_{j+1} the state is first propagated with B_j, recorded in mu^(j), then reweighted by
    diag(sqrt(mu^(j)/mu^(j+1))) for the next window. Returns samples including t = 0 and t = T.
    """
    A = gen.A
    lm = ref.log_mu(edges[0])
    psi = p0 * np.exp(-0.5 * lm)
    times, psis, lms, win = [edges[0]], [psi.copy()], [lm], [0]
    for j in range(len(edges) - 1):
        if j > 0:
            lm_new = ref.log_mu(edges[j])
            psi = psi * np.exp(0.5 * (lm - lm_new))
            lm = lm_new
        B = weighted_generator(A, lm)
        dt = (edges[j + 1] - edges[j]) / sub
        for s in range(sub):
            psi = expm_apply(B, psi, dt)
            times.append(edges[j] + (s + 1) * dt)
            psis.append(psi.copy())
            lms.append(lm)
            win.append(j)
    return PiecewiseRun(np.array(times), np.array(psis), np.array(lms), np.array(win))


def stabilized_psi(gen, ref, cert, family, p0):
    """Stabilized propagation: P_j = e^{(B_j - CW_j I) Delta_j} and
    R_j = diag(sqrt(mu^(j)/mu^(j+1)))/rho_j, both contractions, applied to psi_0/||psi_0||.

    Returns (log of the stabilized norm after each factor, which must stay <= 0 up to rounding;
    log ||psi_T|| reconstructed as log ||psi_0|| + K_T/2 + log ||stabilized final state||; the
    final stabilized direction). Logs keep astronomically loose certificates representable."""
    A, edges = gen.A, cert.edges
    lm = ref.log_mu(edges[0])
    psi0 = p0 * np.exp(-0.5 * lm)
    phi = psi0 / np.linalg.norm(psi0)
    log_s = 0.0  # stabilized state = phi * e^{log_s}
    lognorms = [0.0]
    for j in range(len(edges) - 1):
        if j > 0:
            lm_new = ref.log_mu(edges[j])
            phi = phi * np.exp(0.5 * (lm - lm_new) - cert.log_rho[j - 1])
            lognorms.append(log_s + np.log(np.linalg.norm(phi)))
            lm = lm_new
        dt = edges[j + 1] - edges[j]
        x, ls = expm_action(weighted_generator(A, lm), phi, dt)
        nx = np.linalg.norm(x)
        phi = x / nx
        log_s += ls + np.log(nx) - cert.cw[family][j] * dt
        lognorms.append(log_s)
    log_psiT = np.log(np.linalg.norm(psi0)) + 0.5 * cert.K[family] + log_s
    return np.array(lognorms), log_psiT, phi


def continuous_psi(gen, ref, p0, times, rtol=1e-12, atol=1e-300):
    """psi' = B_t psi with the continuous reference."""
    A = gen.A
    psi0 = p0 * np.exp(-0.5 * ref.log_mu(times[0]))
    rhs = lambda t, y: weighted_generator(A, ref.log_mu(t), ref.dlog_mu(t)) @ y
    sol = solve_ivp(rhs, (times[0], times[-1]), psi0, method="DOP853", t_eval=times, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def readout(psi, logmu, f=None):
    """E_{p_S}[f] = <f sqrt(mu), psi>; f = None means f = 1, which returns m_T."""
    w = np.exp(0.5 * logmu)
    return float((w if f is None else f * w) @ psi)


@dataclass
class ReadoutFactors:
    gamma: float  # ||psi_0|| e^{K_T/2}
    R_f: float  # ||f sqrt(mu_T)|| / |E f|
    amplitude: float  # E f / (gamma ||f sqrt(mu_T)||), <= 1 by the certificate
    value: float  # E_{p_S}[f]


def readout_factors(psi0_norm, K, psi_T, logmu_T, f=None):
    fw = np.exp(0.5 * logmu_T) * (1.0 if f is None else f)
    val = float(fw @ psi_T)
    gamma = psi0_norm * np.exp(0.5 * K)
    nf = float(np.linalg.norm(fw))
    return ReadoutFactors(gamma, nf / abs(val), val / (gamma * nf), val)


def horizon_worst_case(A, logmu0, logmuT, T, dense_limit=3000):
    """||D_T^{-1/2} e^{A_S T} D_0^{1/2}||_2: the largest growth of ||psi|| over all initial vectors
   . No certificate computed from mu alone can beat it. Diagnostic."""
    s0, sT = np.exp(0.5 * logmu0), np.exp(-0.5 * logmuT)
    n = A.shape[0]
    if n <= dense_limit:
        from scipy.linalg import expm
        M = expm(A.toarray() * T) * s0[None, :] * sT[:, None]
        return float(np.linalg.norm(M, 2))
    At = sp.csr_matrix(A.T)
    op = LinearOperator((n, n), dtype=float,
                        matvec=lambda v: sT * expm_multiply(A * T, s0 * np.ravel(v)),
                        rmatvec=lambda v: s0 * expm_multiply(At * T, sT * np.ravel(v)))
    return float(svds(op, k=1, return_singular_vectors=False, tol=1e-8)[0])
