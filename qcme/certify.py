"""Residual certificates and the certified growth exponent K_T.

Inputs are the truncated generator (network only) and a reference. This module never imports
`qcme.fsp` or `qcme.emulate`.

Weight families for K_T: {sqrt(mu_t); the resolvent weight
w = (omega' I - Sigma_t)^{-1} 1 with omega' found by bisection}. Every CW value returned here is
evaluated directly as max_x (Sigma w)(x)/w(x) with a floating-point rounding guard, so it is a valid
upper bound on the log-norm for any positive w, however the weight was obtained.
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, splu

EPS = np.finfo(float).eps


# ----------------------------------------------------------------------------- basic objects
def weighted_generator(A, logmu, dlog=None):
    """B = D^{-1/2} A_S D^{1/2} - (1/2) diag(mu_dot/mu), sparse."""
    A = sp.coo_matrix(A)
    data = A.data * np.exp(0.5 * (logmu[A.col] - logmu[A.row]))
    B = sp.csr_matrix((data, (A.row, A.col)), shape=A.shape)
    if dlog is not None:
        B = B - 0.5 * sp.diags(dlog)
    return B.tocsr()


def residual(A, logmu, dlog=None):
    """r = (A_S mu - mu_dot)/mu, evaluated as sum_y A_xy exp(logmu_y - logmu_x) - dlog."""
    A = sp.coo_matrix(A)
    contrib = A.data * np.exp(logmu[A.col] - logmu[A.row])
    r = np.bincount(A.row, weights=contrib, minlength=A.shape[0])
    return r if dlog is None else r - dlog


def kappa(r):
    return max(float(np.max(r)), 0.0)


def symmetric_part(B):
    return (0.5 * (B + B.T)).tocsr()


def cw(Sigma, w):
    """Collatz-Wielandt value max_x (Sigma w)(x)/w(x) plus a rounding guard."""
    w = np.asarray(w, float)
    if not np.all(w > 0):
        raise ValueError("CW weight must be strictly positive")
    num = Sigma @ w
    guard = 4 * EPS * (abs(Sigma) @ w) * max(1, int(np.diff(Sigma.indptr).max()))
    return float(np.max((num + guard) / w))


def cw_sqrt_mu(r, leak):
    """CW(sqrt mu) = (1/2) max(r - l).

    No explicit rounding guard is added here, unlike `cw`: the value is the exact identity evaluated in
    double precision. The guard `cw` would add is at most 1.1e-11 relative per window on the toggle and
    Schlogl settings, below every reported digit; it is left out so
    that stored results reproduce bit for bit."""
    return 0.5 * float(np.max(r - leak))


@dataclass
class ResolventResult:
    value: float  # certified CW(w) <= omega' (upper bound on the log-norm)
    lower: float  # largest rejected omega' (omega >= lower is NOT certified; diagnostic only)
    iterations: int
    weight: np.ndarray = field(repr=False)


LU_MAX_STATES = 50_000  # above this the resolvent solve uses conjugate gradients
LU_MAX_NNZ = 100_000  # and above this many nonzeros in Sigma
CG_RTOL, CG_MAXITER = 1e-10, 5_000


def resolvent_cw(Sigma, upper, rtol=1e-6, max_iter=80, solver=None):
    """Bisection on omega' with w = (omega' I - Sigma)^{-1} 1.

    `upper` must be a valid upper bound on omega (e.g. CW(sqrt mu)). Returns the smallest certified
    CW(w) found; never larger than `upper` + rounding guard.

    solver: "lu" (sparse LU, exact) or "cg" (Jacobi-preconditioned conjugate gradients; omega' I - Sigma
    is symmetric positive definite for omega' > omega). Default: "lu" up to LU_MAX_STATES, else "cg"
    (LU fill-in makes 3-D boxes of 2e5 states take > 15 min per value). Validity never depends on the
    solver: every returned value is CW(w) evaluated exactly for a positive w, which bounds the log-norm
    for any positive w; an inexact or failed solve can only loosen the bound.
    """
    n = Sigma.shape[0]
    solver = solver or ("lu" if n <= LU_MAX_STATES and Sigma.nnz <= LU_MAX_NNZ else "cg")
    diag = Sigma.diagonal()
    lo = float(diag.max())  # omega >= max_x Sigma_xx (Rayleigh quotient of a unit vector)
    scale = max(1.0, abs(upper), abs(lo))
    hi = upper + 1e-9 * scale
    I = sp.identity(n, format="csc")
    Sc = Sigma.tocsc()
    Sr = Sigma.tocsr()
    best_val, best_w = np.inf, None

    def attempt(om):
        if solver == "lu":
            try:
                w = splu((om * I - Sc).tocsc()).solve(np.ones(n))
            except RuntimeError:
                return None
        else:
            M = (om * sp.identity(n, format="csr") - Sr).tocsr()
            d = om - diag  # > 0 for om > max diag (every bisection point)
            w, info = cg(M, np.ones(n), rtol=CG_RTOL, maxiter=CG_MAXITER, M=sp.diags(1.0 / d))
            if info != 0:
                return None
        return w if np.all(np.isfinite(w)) and np.all(w > 0) else None

    it = 0
    w = attempt(hi)
    if w is not None:
        best_val, best_w = cw(Sigma, w), w
    while hi - lo > rtol * max(1.0, abs(hi), abs(lo)) and it < max_iter:
        it += 1
        gap, s_lo = hi - lo, max(1.0, abs(lo))
        # A bracket spanning many decades (CW(sqrt mu) up to 1e178 on near-point-mass references)
        # is bisected geometrically in the gap; arithmetic halving
        # would exhaust max_iter far above omega. Ordinary brackets keep the arithmetic midpoint.
        mid = lo + np.sqrt(gap * s_lo) if gap > 1e3 * s_lo else 0.5 * (lo + hi)
        w = attempt(mid)
        if w is None:
            lo = mid
        else:
            hi = mid
            v = cw(Sigma, w)
            if v < best_val:
                best_val, best_w = v, w
    if best_w is None:
        raise RuntimeError("resolvent weight failed at the upper bracket")
    return ResolventResult(best_val, lo, it, best_w)


def lognorm_dense(B):
    """Exact Euclidean log-norm lambda_max(Sigma) for small matrices (diagnostic and tests only)."""
    S = symmetric_part(sp.csr_matrix(B)).toarray()
    return float(np.linalg.eigvalsh(S)[-1])


# ----------------------------------------------------------------------------- K_T
FAMILIES = ("sqrt_mu", "resolvent")


def uniform_edges(T, J):
    return np.linspace(0.0, T, J + 1)


def geometric_edges(T, J=20, first_frac=2.0**-10):
    """Window edges with first window T*first_frac and a constant ratio q, sum_{k<J} q^k = 1/first_frac
   ."""
    from scipy.optimize import brentq
    target = 1.0 / first_frac
    q = brentq(lambda q: np.expm1(J * np.log(q)) / (q - 1) - target, 1 + 1e-12, 10.0, xtol=1e-15, rtol=1e-15)
    widths = first_frac * q ** np.arange(J)
    edges = T * np.concatenate([[0.0], np.cumsum(widths / widths.sum())])
    edges[-1] = T
    return edges


@dataclass
class WindowCertificate:
    """Rigorous piecewise-constant certificate.

    edges: window boundaries t_0 = 0 < ... < t_J = T. Window j uses the static reference
    mu^(j) = mu(t_j). K_T = 2 sum_j CW_j Delta_j + 2 sum_j ln rho_j. Values per family.
    """

    edges: np.ndarray
    kappa: np.ndarray  # per window
    cw: dict  # family -> per-window CW
    log_rho: np.ndarray  # per boundary j = 0..J-2: ln max sqrt(mu^(j)/mu^(j+1))
    K: dict  # family -> K_T
    int_kappa: float  # sum_j kappa_j Delta_j (the kappa-based exponent adds 2 sum ln rho)
    K_kappa: float  # kappa-based counterpart: sum kappa_j Delta_j + 2 sum ln rho_j

    def K_at(self, family, t):
        """Certified exponent K on [0, t] (bound ||psi_t|| <= ||psi_0|| e^{K/2}), for t in [0, T]."""
        return _partial_K(self.edges, self.cw[family], self.log_rho, t)

    def K_kappa_at(self, t):
        return _partial_K(self.edges, 0.5 * self.kappa, self.log_rho, t)


def _partial_K(edges, rate, log_rho, t):
    K = 0.0
    for j in range(len(edges) - 1):
        if t <= edges[j]:
            break
        if j > 0:
            K += 2 * log_rho[j - 1]
        K += 2 * rate[j] * (min(t, edges[j + 1]) - edges[j])
    return K


def window_certificate(gen, ref, edges, families=FAMILIES, rtol=1e-6):
    A, leak = gen.A, gen.leak
    J = len(edges) - 1
    kap = np.zeros(J)
    cws = {f: np.zeros(J) for f in families}
    log_rho = np.zeros(max(J - 1, 0))
    prev = None
    for j in range(J):
        lm = ref.log_mu(edges[j])
        if prev is not None:
            log_rho[j - 1] = 0.5 * float(np.max(prev - lm))
        prev = lm
        with np.errstate(over="ignore"):
            r = residual(A, lm)
        kap[j] = kappa(r)
        c_sqrt = cw_sqrt_mu(r, leak)
        if not np.isfinite(c_sqrt):  # mu ratios overflow double precision: the valid bound is +inf
            for f in cws:
                cws[f][j] = np.inf
            continue
        if "sqrt_mu" in cws:
            cws["sqrt_mu"][j] = c_sqrt
        if "resolvent" in cws:
            Sigma = symmetric_part(weighted_generator(A, lm))
            cws["resolvent"][j] = min(c_sqrt, resolvent_cw(Sigma, c_sqrt, rtol).value)
    dt = np.diff(edges)
    K = {f: float(2 * np.sum(cws[f] * dt) + 2 * np.sum(log_rho)) for f in families}
    return WindowCertificate(np.asarray(edges, float), kap, cws, log_rho, K,
                             float(np.sum(kap * dt)), float(np.sum(kap * dt) + 2 * np.sum(log_rho)))


@dataclass
class ContinuousCertificate:
    """Continuous-reference exponent 2 int CW_t(w_t) dt by the trapezoid rule on a time grid.

    Analysis version: the quadrature makes it an estimate, not a certificate.
    """

    times: np.ndarray
    kappa: np.ndarray
    cw: dict
    K: dict
    int_kappa: float


def continuous_certificate(gen, ref, times, families=FAMILIES, rtol=1e-6):
    A, leak = gen.A, gen.leak
    kap = np.zeros(len(times))
    cws = {f: np.zeros(len(times)) for f in families}
    for i, t in enumerate(times):
        lm, dl = ref.log_mu(t), ref.dlog_mu(t)
        with np.errstate(over="ignore", invalid="ignore"):
            r = residual(A, lm, dl)
        kap[i] = kappa(r)
        c_sqrt = cw_sqrt_mu(r, leak)
        if not np.isfinite(c_sqrt):  # overflow: the estimate is +inf at this time
            for f in cws:
                cws[f][i] = np.inf
            continue
        if "sqrt_mu" in cws:
            cws["sqrt_mu"][i] = c_sqrt
        if "resolvent" in cws:
            Sigma = symmetric_part(weighted_generator(A, lm, dl))
            cws["resolvent"][i] = min(c_sqrt, resolvent_cw(Sigma, c_sqrt, rtol).value)
    K = {f: float(2 * np.trapezoid(cws[f], times)) for f in families}
    return ContinuousCertificate(np.asarray(times), kap, cws, K, float(np.trapezoid(kap, times)))


# ----------------------------------------------------------------------------- rigorous continuous K_T
@dataclass
class RigorousCertificate:
    """K_T^rig for a continuous reference.

    pieces: boundaries of the time pieces; bound[p] >= sup over piece p of CW_t(w_p), so
    ||psi_t|| <= ||psi_0|| exp(sum_p bound_p |piece_p cap [0, t]|) for every t (Groenwall)."""

    family: str
    pieces: np.ndarray
    bound: np.ndarray
    K: float

    def K_at(self, t):
        lens = np.clip(np.minimum(self.pieces[1:], t) - self.pieces[:-1], 0.0, None)
        return float(2 * np.sum(self.bound * lens))


def rigorous_continuous_certificate(gen, ref, family, k=1, rtol=1e-6, weight_cache=None):
    """Rigorous 2 int CW_t(w) dt for a reference with `piece` enclosures (qcme.hermite).

    Each node interval of ref.nodes is split into k equal pieces. Weight family "sqrt_mu": w =
    sqrt(mu) at the piece midpoint; "resolvent": the Prop. 5e weight computed once per node
    interval at its midpoint. On a piece, for all t in it,
      (Sigma_t w)(x)/w(x) = A_xx - dlog_t(x)/2 + sum over transitions of
                            (1/2) A_rc sqrt(mu_t(c)/mu_t(r)) (w_c/w_r at row r, w_r/w_c at row c),
    which is bounded above by the enclosures D_hi >= log mu(c) - log mu(r) and dlog_lo.
    """
    A = sp.coo_matrix(gen.A)
    off = A.row != A.col
    er, ec, rate = A.row[off], A.col[off], A.data[off]
    diag = gen.A.diagonal()
    n = A.shape[0]
    nodes = ref.nodes
    pieces, bounds = [], []
    for j in range(len(nodes) - 1):
        t0, t1 = nodes[j], nodes[j + 1]
        logw_node = None
        if family == "resolvent" and weight_cache is not None and j in weight_cache:
            logw_node = weight_cache[j]
        elif family == "resolvent":
            tm = 0.5 * (t0 + t1)
            lm, dl = ref.log_mu(tm), ref.dlog_mu(tm)
            Sigma = symmetric_part(weighted_generator(gen.A, lm, dl))
            up = cw_sqrt_mu(residual(gen.A, lm, dl), gen.leak)
            logw_node = np.log(resolvent_cw(Sigma, up, rtol).weight)
            if weight_cache is not None:
                weight_cache[j] = logw_node
        edges = np.linspace(t0, t1, k + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            logw = logw_node if logw_node is not None else 0.5 * ref.log_mu(0.5 * (a + b))
            Dhi, dlog_lo, _, _ = ref.piece(a, b, er, ec)
            base = 0.5 * rate * np.exp(0.5 * Dhi)
            to_r = base * np.exp(logw[ec] - logw[er])
            to_c = base * np.exp(logw[er] - logw[ec])
            off_sum = np.bincount(er, to_r, n) + np.bincount(ec, to_c, n)
            U = diag - 0.5 * dlog_lo + off_sum
            U += 4 * EPS * (np.abs(diag) + 0.5 * np.abs(dlog_lo) + off_sum) * max(1, int(np.diff(gen.A.indptr).max()))
            pieces.append(a)
            bounds.append(float(U.max()))
    pieces.append(nodes[-1])
    pieces, bounds = np.array(pieces), np.array(bounds)
    return RigorousCertificate(family, pieces, bounds, float(2 * np.sum(bounds * np.diff(pieces))))


# ----------------------------------------------------------------------------- chi^2 and TV bounds
def chi2_bound(K, psi0_norm_sq, m_T, Z_T):
    """From the certificate: 1 + chi2(p_hat_T || mu_bar_T) <= Z_T ||psi_0||^2 e^{K} / m_T^2.

    m_T must be a certified lower bound on the retained mass. Returns (chi2 bound, TV bound on S),
    where the full-space TV adds (1 - m_T)."""
    chi2 = Z_T * psi0_norm_sq * np.exp(K) / m_T**2 - 1.0
    return chi2, 0.5 * np.sqrt(max(chi2, 0.0)) + (1.0 - m_T)
