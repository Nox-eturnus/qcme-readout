"""Linear combination of Hamiltonian simulations (LCHS) for the stabilized weighted generator.

Convention (unit time; time is absorbed into the matrix): for A = L + iH with L = (A + A^*)/2 >= 0,

    e^{-A} = int g(k) U(k) dk,   U(k) = e^{-i(H + kL)}   (unitary for real k).

For a window of the weighted encoding, A = -B~ Delta with B~ = B - c I and c >= the log-norm of B,
so L = -Sym(B~) Delta >= 0. A time-ordered product of such exponentials (several windows, and the
diagonal window reweightings R_j = e^{-D_j}, D_j >= 0) is handled by the same identity with U(k)
the time-ordered product of the per-piece unitaries (An-Liu-Lin 2023, Thm 1, time-dependent form).

Kernels:
  K1  An, Childs, Lin (arXiv:2312.03916): g(k) = f(k)/(1 - ik), f(z) = exp(-(1 + iz)^beta)/C_beta,
      C_beta = 2 pi e^{-2^beta}, 0 < beta < 1. Exact identity. Analytic for -1 < Im k < 1.
  K2  Low, Somma (arXiv:2508.19238): g(k) = e^{c(1 - ik)} e^{-(k^2 + 1)/(4 gamma^2)}/(pi (1 + k^2)),
      gamma = sqrt(c + log((1 + 1/(2 pi))/eps_lchs))/c. Approximates e^{-A}; analytic for |Im k| < 1.
  K0  the original Cauchy kernel 1/(pi(1 + k^2)) (An, Liu, Lin 2023), for tests only.

Discretization: the uniform trapezoid rule k_j = j h, truncated to |j| <= N. Every error term is a
matrix-independent bound, given only Lnorm >= int ||L(s)|| ds (all bounds use ||U(k + iy)|| <= 1 for
y <= 0 and <= e^{y Lnorm} for y > 0, which holds whenever every L(s) >= 0):
  approximation  (K2 only)  ||int (g - g_exact) U|| <= int |g - g_exact|(k - i y0) dk   (contour shift;
                            g - g_exact is entire because the Gaussian factor vanishes at k = +-i)
  trapezoid      (infinite sum vs integral; Trefethen and Weideman 2014, Thm 5.1, applied to the
                 operator-valued integrand)  <= M_+(b)/(e^{2 pi b/h} - 1) + M_-(a)/(e^{2 pi a/h} - 1),
                 M_+(b) = e^{b Lnorm} int |g(k + ib)| dk,  M_-(a) = int |g(k - ia)| dk
  node tail      h sum_{|j| > N} |g(jh)|
The scalar integrals are evaluated by adaptive quadrature with a safety factor SAFETY; the bounds are
therefore numerical, not interval-rigorous.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import integrate, special

SAFETY = 1.25  # multiplies every quadrature-evaluated bound


# ----------------------------------------------------------------------------- kernels
class Kernel:
    name = "kernel"
    strip = (np.inf, np.inf)  # (a_max, b_max): analytic for -a_max < Im k < b_max

    def g(self, k):
        raise NotImplementedError

    def log_abs_g(self, k):
        return np.log(np.abs(self.g(k)))

    def approx_error(self):
        """Bound on ||int g U - e^{-A}|| (0 for exact identities)."""
        return 0.0


class ACL(Kernel):
    """K1: An-Childs-Lin kernel with parameter beta in (0, 1)."""

    strip = (1.0, 1.0)

    def __init__(self, beta):
        if not 0 < beta < 1:
            raise ValueError("beta must lie in (0, 1)")
        self.beta = float(beta)
        self.logC = np.log(2 * np.pi) - 2.0**self.beta
        self.name = f"K1(beta={self.beta:g})"

    def log_g(self, k):
        k = np.asarray(k, dtype=complex)
        return -(1 + 1j * k) ** self.beta - self.logC - np.log(1 - 1j * k)

    def g(self, k):
        return np.exp(self.log_g(k))

    def log_abs_g(self, k):
        return np.real(self.log_g(k))


class LowSomma(Kernel):
    """K2: Low-Somma Gaussian-convolved exponential kernel with shift c and target eps_lchs.

    g has poles at k = +-i (from 1/(1 + k^2)), so the trapezoid strip is |Im k| < 1; only the
    difference g - g_exact used in `approx_error` is entire."""

    strip = (1.0, 1.0)

    def __init__(self, c, eps_lchs):
        self.c, self.eps_lchs = float(c), float(eps_lchs)
        self.gamma = np.sqrt(self.c + np.log((1 + 1 / (2 * np.pi)) / self.eps_lchs)) / self.c
        self.R = 2 * self.c * self.gamma**2
        self.name = f"K2(c={self.c:g})"
        self._approx = None

    def log_g(self, k):
        k = np.asarray(k, dtype=complex)
        return self.c * (1 - 1j * k) - (k * k + 1) / (4 * self.gamma**2) - np.log(np.pi * (1 + k * k))

    def g(self, k):
        return np.exp(self.log_g(k))

    def log_abs_g(self, k):
        return np.real(self.log_g(k))

    def _log_abs_dg(self, k):
        """log |g - g_exact|, g_exact = e^{c(1-ik)}/(pi(1+k^2)): = log|g_exact| + log|e^{-(k^2+1)/4g^2} - 1|."""
        k = np.asarray(k, dtype=complex)
        w = -(k * k + 1) / (4 * self.gamma**2)
        base = self.c * (1 - 1j * k) - np.log(np.pi * (1 + k * k))
        return np.real(base + np.log(np.expm1(w)))

    def approx_error(self):
        """min over y0 > 0 of int |g - g_exact|(k - i y0) dk (contour shift into Im k < 0)."""
        if self._approx is None:
            y_opt = 2 * self.c * self.gamma**2  # minimizes c(1 - y) + y^2/(4 gamma^2)
            best = np.inf
            for y0 in y_opt * np.array([0.8, 0.9, 1.0, 1.1, 1.25]):
                if abs(y0 - 1) < 1e-3:
                    continue
                best = min(best, line_integral(lambda k, y0=y0: self._log_abs_dg(k - 1j * y0)))
            self._approx = SAFETY * best
        return self._approx


class Cauchy(Kernel):
    """K0: the original LCHS kernel (tests only)."""

    strip = (1.0, 1.0)
    name = "K0"

    def g(self, k):
        k = np.asarray(k, dtype=complex)
        return 1.0 / (np.pi * (1 + k * k))


# ----------------------------------------------------------------------------- scalar integrals
def line_integral(log_abs, scale=None):
    """int_R exp(log_abs(k)) dk for a log-magnitude function, by adaptive quadrature on a shifted
    exponent (the result may be astronomically small or large)."""
    ks = np.concatenate([-np.logspace(6, -3, 400), [0.0], np.logspace(-3, 6, 400)])
    vals = log_abs(ks)
    m = float(np.max(vals[np.isfinite(vals)]))
    f = lambda k: float(np.exp(log_abs(np.array([k]))[0] - m))
    tot = 0.0
    for a, b in ((-np.inf, -1e3), (-1e3, -1), (-1, 0), (0, 1), (1, 1e3), (1e3, np.inf)):
        v, _ = integrate.quad(f, a, b, limit=400, epsabs=0.0, epsrel=1e-9)
        tot += v
    return float(np.exp(m + np.log(tot))) if tot > 0 else 0.0


def half_line_integral(log_abs, x0, sign):
    """int_{x0}^{inf} (sign=+1) or int_{-inf}^{x0} (sign=-1) of exp(log_abs(k)) dk."""
    x0 = float(x0)
    m = float(log_abs(np.array([x0]))[0])
    f = lambda s: float(np.exp(log_abs(np.array([x0 + sign * s]))[0] - m))
    tot = 0.0
    edges = [0.0, 1.0, 10.0, 100.0, 1e3, 1e4, np.inf]
    for a, b in zip(edges[:-1], edges[1:]):
        v, _ = integrate.quad(f, a, b, limit=400, epsabs=0.0, epsrel=1e-9)
        tot += v
    return float(np.exp(m + np.log(tot))) if tot > 0 else 0.0


# ----------------------------------------------------------------------------- discretization
@dataclass
class Discretization:
    kernel: Kernel
    h: float
    N: int  # nodes j = -N..N
    Lnorm: float
    err_approx: float
    err_trap: float
    err_tail: float
    nodes: np.ndarray = field(repr=False)
    coef: np.ndarray = field(repr=False)  # h g(k_j)

    @property
    def K(self):
        return self.N * self.h

    @property
    def M(self):
        return len(self.nodes)

    @property
    def l1(self):
        return float(np.sum(np.abs(self.coef)))

    @property
    def error(self):
        return self.err_approx + self.err_trap + self.err_tail


def _strip_integrals(kernel, n_grid=24):
    """log int |g(k - ia)| dk and log int |g(k + ib)| dk on grids in the analyticity strip (cached on
    the kernel; independent of h and of the matrix)."""
    cache = getattr(kernel, "_strip_cache", None)
    if cache is not None:
        return cache
    a_max, b_max = kernel.strip
    a_grid = np.linspace(0.05, 0.98, n_grid) * (a_max if np.isfinite(a_max) else 1.0)
    b_grid = np.linspace(0.05, 0.98, n_grid) * (b_max if np.isfinite(b_max) else 1.0)
    if not np.isfinite(a_max):
        a_grid = np.concatenate([a_grid, np.geomspace(1.0, 60.0, n_grid)])
    if not np.isfinite(b_max):
        b_grid = np.concatenate([b_grid, np.geomspace(1.0, 60.0, n_grid)])
    la = np.array([np.log(line_integral(lambda k, a=a: kernel.log_abs_g(k - 1j * a))) for a in a_grid])
    lb = np.array([np.log(line_integral(lambda k, b=b: kernel.log_abs_g(k + 1j * b))) for b in b_grid])
    kernel._strip_cache = (a_grid, la, b_grid, lb)
    return kernel._strip_cache


def trapezoid_error(kernel, h, Lnorm):
    """min over the strip of M_+(b)/(e^{2 pi b/h} - 1) + M_-(a)/(e^{2 pi a/h} - 1) (upper and lower
    half-planes optimized separately)."""
    a_grid, la, b_grid, lb = _strip_integrals(kernel)
    with np.errstate(over="ignore"):
        lo = np.min(la - _log_expm1(2 * np.pi * a_grid / h))
        up = np.min(b_grid * Lnorm + lb - _log_expm1(2 * np.pi * b_grid / h))
    return SAFETY * float(np.exp(lo) + np.exp(up))


def _log_expm1(x):
    x = np.asarray(x, float)
    return np.where(x > 30, x + np.log1p(-np.exp(-np.minimum(x, 700.0))), np.log(np.expm1(np.minimum(x, 30.0))))


def node_tail(kernel, h, N):
    """h sum_{|j| > N} |g(jh)|, bounded by the tail integrals from (N)h where |g| decreases (checked),
    plus the first omitted term on each side."""
    tot = 0.0
    for sign in (1, -1):
        x0 = sign * N * h
        tot += h * float(np.exp(kernel.log_abs_g(np.array([sign * (N + 1) * h]))[0]))
        tot += half_line_integral(kernel.log_abs_g, x0 + sign * h, sign)
    return SAFETY * tot


def discretize(kernel, Lnorm, delta, share=(1 / 3, 1 / 3, 1 / 3)):
    """Coarsest trapezoid grid and fewest nodes with approximation + trapezoid + tail error <= delta.

    `share` splits delta; the approximation share is only checked (K2 sets it through eps_lchs)."""
    ea = kernel.approx_error()
    if ea > share[0] * delta * 1.0000001:
        raise ValueError(f"{kernel.name}: approximation error {ea:.3g} exceeds its share of {delta:.3g}")
    # largest h with trapezoid error <= share[1] delta (bisection on log h)
    lo, hi = 1e-8, 4.0
    if trapezoid_error(kernel, lo, Lnorm) > share[1] * delta:
        raise RuntimeError("trapezoid bound not reachable")
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if trapezoid_error(kernel, mid, Lnorm) <= share[1] * delta:
            lo = mid
        else:
            hi = mid
    h = lo
    et = trapezoid_error(kernel, h, Lnorm)
    # smallest N with node tail <= share[2] delta (doubling then bisection)
    target = share[2] * delta
    N = 1
    while node_tail(kernel, h, N) > target:
        N *= 2
        if N > 2**40:
            raise RuntimeError("node tail not reachable")
    lo_n, hi_n = N // 2, N
    while hi_n - lo_n > 1:
        mid = (lo_n + hi_n) // 2
        if node_tail(kernel, h, mid) <= target:
            hi_n = mid
        else:
            lo_n = mid
    N = hi_n
    if N * h > getattr(kernel, "K0", np.inf):  # K3: nodes must lie where f is computed, not bounded
        raise RuntimeError(f"{kernel.name}: cutoff {N * h:.1f} beyond the quadrature range {kernel.K0:.1f}")
    nodes = h * np.arange(-N, N + 1)
    coef = h * kernel.g(nodes)
    return Discretization(kernel, h, N, float(Lnorm), float(ea), float(et), float(node_tail(kernel, h, N)),
                          nodes, coef)


# ----------------------------------------------------------------------------- Hamiltonian simulation
def jacobi_anger_degree(tau, delta):
    """Smallest q >= 0 with 2 sum_{l > q} |J_l(tau)| <= delta: the Laurent degree of the truncated
    Jacobi-Anger expansion e^{-i tau cos(theta)} = sum_l (-i)^l J_l(tau) e^{i l theta} (the error of
    the degree-q truncation on the unit circle is at most 2 sum_{l>q} |J_l(tau)|)."""
    tau = abs(float(tau))
    if tau == 0.0:
        return 0
    start = int(np.floor(tau))
    ls = np.arange(start, start + 200 + int(20 * tau ** (1 / 3)) + int(4 * np.log(1 / delta) + 50))
    terms = np.abs(special.jv(ls, tau))
    # tail[i] = sum_{l >= ls[i]} |J_l|; beyond the grid the terms are negligible (checked below)
    if terms[-1] > 1e-3 * delta:
        raise RuntimeError("Jacobi-Anger grid too short")
    tail = np.cumsum(terms[::-1])[::-1]
    # smallest q with 2 * sum_{l > q} |J_l| <= delta, i.e. 2 * tail[index of q+1] <= delta
    ok = np.nonzero(2 * tail <= delta)[0]
    q_plus_1 = ls[ok[0]]
    q = max(int(q_plus_1) - 1, 0)
    if q < start:  # the bulk below floor(tau) is never negligible; conservative guard
        q = start
    return q


# ----------------------------------------------------------------------------- dense emulation
def split(A):
    """A = L + iH with L = (A + A^*)/2, H = (A - A^*)/(2i)."""
    A = np.asarray(A, dtype=complex)
    L = 0.5 * (A + A.conj().T)
    H = (A - A.conj().T) / 2j
    return L, H


def emulate(disc, pieces):
    """sum_j c_j prod_pieces e^{-i(H_p + k_j L_p)} for dense pieces (time-ordered: the first piece acts
    first). pieces: list of dense matrices A_p = L_p + iH_p (time absorbed). Returns the dense operator."""
    from scipy.linalg import expm
    n = pieces[0].shape[0]
    LH = [split(P) for P in pieces]
    out = np.zeros((n, n), dtype=complex)
    for k, c in zip(disc.nodes, disc.coef):
        U = np.eye(n, dtype=complex)
        for L, H in LH:
            U = expm(-1j * (H + k * L)) @ U
        out += c * U
    return out


def exact_product(pieces):
    from scipy.linalg import expm
    n = pieces[0].shape[0]
    W = np.eye(n, dtype=complex)
    for P in pieces:
        W = expm(-np.asarray(P, dtype=complex)) @ W
    return W


# ----------------------------------------------------------------------------- K3
def _bump_derivative(s):
    """psi'(s) for the C-infinity transition psi(s) = h(s)/(h(s) + h(1-s)), h(s) = exp(-1/s), on (0, 1)."""
    s = np.asarray(s, float)
    out = np.zeros_like(s)
    ok = (s > 0) & (s < 1)
    t = s[ok]
    # psi' = (h'(t) h(1-t) + h(t) h'(1-t)) / (h(t) + h(1-t))^2 with h' = h/t^2; in logs for stability
    a, b = -1.0 / t, -1.0 / (1 - t)
    m = np.maximum(a, b)
    num = np.exp(a + b - 2 * m) * (1 / t**2 + 1 / (1 - t) ** 2)
    den = (np.exp(a - m) + np.exp(b - m)) ** 2
    out[ok] = num / den
    return out


class FTLCHS(Kernel):
    """K3: Fourier-transform LCHS (Huang, An; Defs. 7-8) with F(x) = e^{-x} on x >= 0,
    F(x) = psi((x + delta)/delta) e^{-x} on [-delta, 0] and 0 below, psi the standard C-infinity
    transition. Then Phi = F + F' = e^{-x} psi'((x + delta)/delta)/delta is supported on [-delta, 0],
    f(z) = (1/2 pi) int_0^1 psi'(s) e^{delta (1 - s)(1 - i z)} ds is entire, and
    g(k) = f(k)/(1 - ik) (pole at k = -i). f is evaluated by Gauss-Legendre quadrature for
    |Re z| <= K0, and beyond by the m-fold integration-by-parts envelope
    |f(z)| <= ||psi^(m+1)||_1 e^{delta max(1 + Im z, 0)} / (2 pi |delta (1 - i z)|^m)
    (psi' and all its derivatives vanish at 0 and 1), with the norms from a Chebyshev representation
    and a factor 2 margin. Used as a sensitivity check of the kernel choice."""

    strip = (1.0, np.inf)

    def __init__(self, delta, n_quad=3000, m_max=8):
        self.delta = float(delta)
        self.name = f"K3(delta={self.delta:g})"
        x, w = np.polynomial.legendre.leggauss(n_quad)
        self.s = 0.5 * (x + 1)
        self.w = 0.5 * w * _bump_derivative(self.s)
        self.K0 = 0.25 * n_quad / self.delta  # quadrature resolves delta * |k| well below n_quad
        # ||psi^(m+1)||_1 for m = 0..m_max from a Chebyshev fit of psi' on [0, 1]
        cheb = np.polynomial.chebyshev.Chebyshev.fit(self.s, _bump_derivative(self.s), 600, domain=[0, 1])
        grid = np.linspace(0, 1, 20001)
        self.norms = []
        c = cheb
        for m in range(m_max + 1):
            self.norms.append(2.0 * float(np.trapezoid(np.abs(c(grid)), grid)))
            c = c.deriv()

    def f(self, z):
        z = np.atleast_1d(np.asarray(z, dtype=complex))
        out = np.empty(z.shape, dtype=complex)
        near = np.abs(z.real) <= self.K0
        if np.any(near):
            zz = z[near]
            vals = np.empty(zz.shape, dtype=complex)
            for a in range(0, len(zz), 2000):  # chunks keep the quadrature matrix small
                E = np.exp(self.delta * np.outer(1 - self.s, 1 - 1j * zz[a:a + 2000]))
                vals[a:a + 2000] = self.w @ E
            out[near] = vals / (2 * np.pi)
        if np.any(~near):
            zz = z[~near]
            beta = np.abs(self.delta * (1 - 1j * zz))
            grow = np.exp(self.delta * np.maximum(1 + zz.imag, 0.0))
            env = np.min([self.norms[m] * grow / (2 * np.pi * beta**m) for m in range(len(self.norms))], axis=0)
            out[~near] = env  # an upper bound on |f| (phase irrelevant: used only in magnitude bounds)
        return out

    def g(self, k):
        k = np.asarray(k, dtype=complex)
        return self.f(k).reshape(k.shape) / (1 - 1j * k)

    def log_abs_g(self, k):
        with np.errstate(divide="ignore"):
            return np.log(np.abs(self.g(k)))
