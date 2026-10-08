"""R-TT core for two-species models: dynamical low-rank approximation of the truncated CME.

p is held as an n1 x n2 matrix Y = U S V^T (C order, the `StateSpace.box` order) and advanced by
the Lubich-Oseledets projector-splitting (KSL) integrator, Lie-Trotter order, fixed step
h = T / N_STEPS. Each substep is linear in the small factors and is solved with the matrix
exponential of the induced operator:

    K = U S:        K' = F(K V^T) V            vec_C: (I (x) V)^T A (I (x) V)
    S (backward):   S' = -U^T F(U S V^T) V     vec_C: -(U (x) V)^T A (U (x) V)
    L^T = S V^T:    (L^T)' = U^T F(U L^T)      vec_C: (U (x) I)^T A (U (x) I)

with F(Y) = mat(A_S vec(Y)). The input is the truncated generator (network only); this module
never imports the FSP solution. Y can be negative and does not conserve mass
(R12 Section 5.1), so the core is max(Y, 0) normalized, and it is used only inside an R-mix.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

N_STEPS = 200  # h = T / 200
FD_FRAC = 1e-4  # time-derivative stencil half-width, in units of h (continuous estimate only)


def _qr_pos(M):
    Q, R = np.linalg.qr(M)
    s = np.where(np.diag(R) < 0, -1.0, 1.0)
    return Q * s, R * s[:, None]


def _complete(u, r):
    """Deterministic orthonormal completion: QR of [u | DCT-II vectors 1..r-1]; first column is u."""
    n = len(u)
    i = np.arange(n)
    cols = [u] + [np.cos(np.pi * k * (i + 0.5) / n) for k in range(1, r)]
    Q, _ = _qr_pos(np.stack(cols, axis=1))
    return Q


class LowRankCME:
    """Rank-r KSL solution of dp/dt = A p on an n1 x n2 box, stored on the grid t_k = k h."""

    def __init__(self, A, shape, p0, r, T, n_steps=N_STEPS):
        self.A = sp.csr_matrix(A)
        self.n1, self.n2 = shape
        if self.A.shape[0] != self.n1 * self.n2:
            raise ValueError("R-TT needs a full two-species box")
        if r > min(shape):
            raise ValueError("rank exceeds the box width")
        self.r, self.T, self.h = r, float(T), float(T) / n_steps
        self.I1 = sp.identity(self.n1, format="csr")
        self.I2 = sp.identity(self.n2, format="csr")
        P0 = np.asarray(p0, float).reshape(shape)
        Uf, sv, Vt = np.linalg.svd(P0, full_matrices=False)
        sgn = 1.0 if Uf[:, 0].sum() >= 0 else -1.0  # fix the SVD sign convention
        U, V = _complete(sgn * Uf[:, 0], r), _complete(sgn * Vt[0], r)
        S = np.zeros((r, r))
        S[0, 0] = sv[0]
        self.grid = [(U, S, V)]
        for _ in range(n_steps):
            self.grid.append(self._step(*self.grid[-1], self.h))

    def _step(self, U, S, V, h):
        A, n1, n2, r = self.A, self.n1, self.n2, self.r
        # K-step
        IV = sp.csr_matrix(sp.kron(self.I1, sp.csr_matrix(V)))
        MK = (IV.T @ (A @ IV)).toarray()
        K = expm_multiply(h * MK, (U @ S).ravel()).reshape(n1, r)
        U1, Sh = _qr_pos(K)
        # S-step, backward in time
        UV = np.kron(U1, V)
        MS = -(UV.T @ (A @ UV))
        St = expm_multiply(h * MS, Sh.ravel()).reshape(r, r)
        # L-step on L^T = S V^T
        UI = sp.csr_matrix(sp.kron(sp.csr_matrix(U1), self.I2))
        ML = (UI.T @ (A @ UI)).toarray()
        Lt = expm_multiply(h * ML, (St @ V.T).ravel()).reshape(r, n2)
        V1, R = _qr_pos(Lt.T)
        return U1, R.T, V1

    def factors(self, t):
        """(U, S, V) at time t: grid value, plus one partial step when t is off the grid."""
        if not 0.0 <= t <= self.T * (1 + 1e-12):
            raise ValueError("t outside [0, T]")
        k = min(int(np.floor(t / self.h + 1e-9)), len(self.grid) - 1)
        rest = t - k * self.h
        U, S, V = self.grid[k]
        if rest > 1e-12 * max(self.T, 1.0):
            U, S, V = self._step(U, S, V, rest)
        return U, S, V

    def Y(self, t):
        U, S, V = self.factors(t)
        return (U @ S @ V.T).ravel()

    def Ydot(self, t):
        """Time derivative of the implemented Y_t (grid values plus partial steps), by a central
        difference of width 2*FD_FRAC*h (one-sided at 0 and T). This is the derivative of the mu_t
        the certificate actually uses; the projected tangent P_Y(A Y) belongs to the exact
        low-rank flow and differs by O(h)."""
        d = FD_FRAC * self.h
        a, b = max(t - d, 0.0), min(t + d, self.T)
        return (self.Y(b) - self.Y(a)) / (b - a)


class LowRankCore:
    """Core max(Y_t, 0)/||max(Y_t, 0)||_1 as a reference (log mu = -inf where Y <= 0); used only
    inside an R-mix, which makes it strictly positive."""

    def __init__(self, gen, space, p0, r, T, n_steps=N_STEPS):
        shape = tuple(int(w) for w in space.widths)
        if len(space) != int(np.prod(shape)):
            raise ValueError("R-TT needs a full box state space")
        self.lr = LowRankCME(gen.A, shape, p0, r, T, n_steps)
        self.r, self.name = r, f"R-TT core (r={r})"
        self._cache = {}

    def _terms(self, t):
        key = float(t)
        if key not in self._cache:
            Y = self.lr.Y(t)
            pos = Y > 0
            Z = Y[pos].sum()
            with np.errstate(divide="ignore"):
                lm = np.where(pos, np.log(np.where(pos, Y, 1.0) / Z), -np.inf)
            Yd = self.lr.Ydot(t)
            dl = np.zeros_like(Y)
            dl[pos] = Yd[pos] / Y[pos] - Yd[pos].sum() / Z
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[key] = (lm, dl)
        return self._cache[key]

    def log_mu(self, t):
        return self._terms(t)[0]

    def dlog_mu(self, t):
        return self._terms(t)[1]

    def mu(self, t):
        return np.exp(self.log_mu(t))
