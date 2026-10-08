"""R-H references: Hermite-interpolated moment references with interval enclosures in time.

The reference is an explicit C^1 function of t: per species, cubic Hermite interpolants of
log m (Poisson / NB species), e = var/mean - 1 clipped at 0 (NB species) and logit g (binary
species) through node values and node derivatives taken from the moment ODE. The ODE only
chooses the reference; certificates are proved for the interpolant itself.

`piece` returns, for a time piece [a, b] inside one node interval, enclosures valid for every t
in [a, b]:
  - D_hi per transition (r <- c): an upper bound of log mu_t(c) - log mu_t(r);
  - dlog_lo per state: a lower bound of mu_dot_t / mu_t;
  - ell_lo, ell_hi per state: bounds of log mu_t (needed for R-mix).
Rounding is covered by outward padding (PAD relative, plus a term proportional to the summed
magnitudes). No function here sees the CME solution.
"""
import numpy as np
from scipy.special import expit, gammaln

from .reference import MomentTrajectory, Reference, _nb_terms

EPS = np.finfo(float).eps
PAD = 1e-12
M_FLOOR = 1e-12


# ----------------------------------------------------------------------------- Hermite pieces
class Hermite:
    """Cubic Hermite interpolant of several parameters through (nodes, y, dy); y has shape (nodes, p)."""

    def __init__(self, nodes, y, dy):
        self.t = np.asarray(nodes, float)
        self.y, self.dy = np.asarray(y, float), np.asarray(dy, float)

    def _coef(self, j):
        h = self.t[j + 1] - self.t[j]
        y0, y1, d0, d1 = self.y[j], self.y[j + 1], h * self.dy[j], h * self.dy[j + 1]
        return h, (y0, d0, -3 * y0 - 2 * d0 + 3 * y1 - d1, 2 * y0 + d0 - 2 * y1 + d1)

    def locate(self, t):
        return int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 2))

    def __call__(self, t):
        """(value, time derivative) at t."""
        j = self.locate(t)
        h, (c0, c1, c2, c3) = self._coef(j)
        s = (t - self.t[j]) / h
        return c0 + s * (c1 + s * (c2 + s * c3)), (c1 + s * (2 * c2 + 3 * s * c3)) / h

    def ranges(self, j, sa, sb):
        """Exact (value range, derivative range) on s in [sa, sb] of node interval j, padded."""
        h, (c0, c1, c2, c3) = self._coef(j)
        p = len(c0)
        vlo, vhi, dlo, dhi = (np.empty(p) for _ in range(4))
        for i in range(p):
            f = np.polynomial.Polynomial([c0[i], c1[i], c2[i], c3[i]])
            df = f.deriv()
            cand = [sa, sb] + [r.real for r in df.roots() if abs(r.imag) < 1e-14 and sa < r.real < sb]
            v = f(np.array(cand))
            dcand = [sa, sb] + [r.real for r in df.deriv().roots() if sa < r.real < sb]
            d = df(np.array(dcand)) / h
            scale_v = abs(c0[i]) + abs(c1[i]) + abs(c2[i]) + abs(c3[i])
            scale_d = (abs(c1[i]) + 2 * abs(c2[i]) + 3 * abs(c3[i])) / h
            vlo[i], vhi[i] = v.min() - 16 * EPS * scale_v, v.max() + 16 * EPS * scale_v
            dlo[i], dhi[i] = d.min() - 16 * EPS * scale_d, d.max() + 16 * EPS * scale_d
        return (vlo, vhi), (dlo, dhi)


def imul(a, b):
    """Interval product of arrays of intervals a = (lo, hi), b = (lo, hi)."""
    c = np.stack([a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]])
    return c.min(axis=0), c.max(axis=0)


def _log1p_over(e):
    """log(1 + e)/e (decreasing in e) and its derivative (increasing in e), vectorized, stable at 0.

    log(1+e)/e = int_0^1 ds/(1 + e s), so the derivative -int s/(1+es)^2 < 0 and the second
    derivative 2 int s^2/(1+es)^3 > 0: the monotonicities used by the enclosures."""
    e = np.asarray(e, float)
    small = e < 1e-4
    es = np.where(small, 1.0, e)
    L = np.log1p(es)
    lo = np.where(small, 1 - e / 2 + e * e / 3 - e**3 / 4, L / es)
    dlo = np.where(small, -0.5 + 2 * e / 3 - 3 * e * e / 4, (es / (1 + es) - L) / es**2)
    return lo, dlo


# ----------------------------------------------------------------------------- R-H product
class HermiteProduct(Reference):
    def __init__(self, space, moments, binary, nb, nodes, name=None):
        self.space, self.X = space, space.X
        self.states = space.states
        self.binary, self.nb = np.asarray(binary, bool), nb
        self.name = name or ("R-H(NB)" if nb else "R-H(P)")
        self.nodes = np.asarray(nodes, float)
        d = self.X.shape[1]
        Y = np.zeros((len(nodes), 2 * d))  # per species: [log m | logit g, e_raw]
        dY = np.zeros_like(Y)
        for k, t in enumerate(self.nodes):
            m, md, v, vd = moments.at(t)
            for i in range(d):
                if self.binary[i]:
                    g = np.clip(m[i], M_FLOOR, 1 - M_FLOOR)
                    Y[k, i] = np.log(g) - np.log1p(-g)
                    dY[k, i] = md[i] / (g * (1 - g)) if M_FLOOR < m[i] < 1 - M_FLOOR else 0.0
                else:
                    mm = max(m[i], M_FLOOR)
                    Y[k, i] = np.log(mm)
                    dY[k, i] = md[i] / mm if m[i] > M_FLOOR else 0.0
                    if nb:
                        Y[k, d + i] = v[i] / mm - 1
                        dY[k, d + i] = (vd[i] * mm - v[i] * md[i]) / mm**2
        self.h = Hermite(self.nodes, Y, dY)
        self.d = d
        self.top = int(self.states.max()) + 1

    # ---- point evaluation
    def _params(self, t):
        y, dy = self.h(t)
        return y, dy

    def _terms(self, t):
        y, dy = self._params(t)
        d = self.d
        logmu = np.zeros(len(self.X))
        dlog = np.zeros(len(self.X))
        for i in range(d):
            x = self.X[:, i]
            if self.binary[i]:
                g = expit(y[i])
                logmu += np.where(x == 1, -np.logaddexp(0, -y[i]), -np.logaddexp(0, y[i]))
                dlog += (x - g) * dy[i]
                continue
            m = np.exp(y[i])
            md = m * dy[i]
            if self.nb and y[d + i] > 0:
                e, ed = y[d + i], dy[d + i]
            else:
                e, ed = 0.0, 0.0
            lm, dl = _nb_terms(x, m, e, md, ed)
            logmu += lm
            dlog += dl
        return logmu, dlog

    def log_mu(self, t):
        return self._terms(t)[0]

    def dlog_mu(self, t):
        return self._terms(t)[1]

    # ---- enclosures on a piece
    def piece(self, a, b, edge_r, edge_c):
        """Enclosures valid for all t in [a, b] (inside one node interval). See module docstring."""
        j = self.h.locate(0.5 * (a + b))
        t0, t1 = self.nodes[j], self.nodes[j + 1]
        if a < t0 - 1e-12 * max(1, abs(t0)) or b > t1 + 1e-12 * max(1, abs(t1)):
            raise ValueError("piece must lie inside one node interval")
        h = t1 - t0
        (ylo, yhi), (dlo, dhi) = self.h.ranges(j, max((a - t0) / h, 0.0), min((b - t0) / h, 1.0))
        d, n = self.d, len(self.states)
        Dhi = np.zeros(len(edge_r))
        dlog_lo, ell_lo, ell_hi = np.zeros(n), np.zeros(n), np.zeros(n)
        mag = np.zeros(len(edge_r))  # summed magnitudes for the rounding pad
        for i in range(d):
            xr, xc = self.states[edge_r, i], self.states[edge_c, i]
            x = self.states[:, i]
            if self.binary[i]:
                Q = (ylo[i], yhi[i])
                dQ = (dlo[i], dhi[i])
                # D = ell(c) - ell(r) = (xc - xr) * logit g
                step = xc - xr
                Dhi += np.where(step > 0, Q[1], np.where(step < 0, -Q[0], 0.0))
                glo, ghi = expit(Q[0]), expit(Q[1])
                ell_lo += np.where(x == 1, -np.logaddexp(0, -Q[0]), -np.logaddexp(0, Q[1]))
                ell_hi += np.where(x == 1, -np.logaddexp(0, -Q[1]), -np.logaddexp(0, Q[0]))
                xg = (np.where(x == 1, 1 - ghi, -ghi), np.where(x == 1, 1 - glo, -glo))
                dlog_lo += imul(xg, (np.full(n, dQ[0]), np.full(n, dQ[1])))[0]
                continue
            mlo, mhi = np.exp(ylo[i]), np.exp(yhi[i])
            if self.nb:
                elo, ehi = max(ylo[d + i], 0.0), max(yhi[d + i], 0.0)
                if ylo[d + i] > 0:
                    de = (dlo[d + i], dhi[d + i])
                elif yhi[d + i] <= 0:
                    de = (0.0, 0.0)
                else:
                    de = (min(dlo[d + i], 0.0), max(dhi[d + i], 0.0))
            else:
                elo = ehi = 0.0
                de = (0.0, 0.0)
            jj = np.arange(self.top, dtype=float)
            t_lo = np.log(mlo + jj * elo) - np.log1p(ehi) - np.log(jj + 1)
            t_hi = np.log(mhi + jj * ehi) - np.log1p(elo) - np.log(jj + 1)
            Clo = np.concatenate([[0.0], np.cumsum(t_lo)])
            Chi = np.concatenate([[0.0], np.cumsum(t_hi)])
            Cabs = np.concatenate([[0.0], np.cumsum(np.abs(t_lo) + np.abs(t_hi))])
            # D_i = ell_i(xc) - ell_i(xr): sum of terms j in [xr, xc) if xc > xr, minus the reverse sum otherwise
            up = xc > xr
            Dhi += np.where(up, Chi[xc] - Chi[xr], -(Clo[xr] - Clo[xc]))
            mag += np.abs(Cabs[np.maximum(xc, xr)]) + np.abs(Cabs[np.minimum(xc, xr)])
            lo_e = _log1p_over(np.array([elo, ehi]))
            lo_max, lo_min = lo_e[0][0], lo_e[0][1]  # lo(e) decreasing in e
            dlo_min, dlo_max = lo_e[1][0], lo_e[1][1]  # dlo(e) increasing in e
            ell_lo += Clo[x] - mhi * lo_max
            ell_hi += Chi[x] - mlo * lo_min
            # dlog = m_dot (S_inv - lo(e)) + e_dot (S_ion - x/(1+e) - m dlo(e))
            Sinv_lo = np.concatenate([[0.0], np.cumsum(1.0 / (mhi + jj * ehi))])[x]
            Sinv_hi = np.concatenate([[0.0], np.cumsum(1.0 / (mlo + jj * elo))])[x]
            Sion_lo = np.concatenate([[0.0], np.cumsum(jj / (mhi + jj * ehi))])[x]
            Sion_hi = np.concatenate([[0.0], np.cumsum(jj / (mlo + jj * elo))])[x]
            md = imul((mlo, mhi), (dlo[i], dhi[i]))
            F1 = (Sinv_lo - lo_max, Sinv_hi - lo_min)
            mdl = imul((mlo, mhi), (dlo_min, dlo_max))
            F2 = (Sion_lo - x / (1 + elo) - mdl[1], Sion_hi - x / (1 + ehi) - mdl[0])
            t1_ = imul((np.full(n, md[0]), np.full(n, md[1])), F1)
            t2_ = imul((np.full(n, de[0]), np.full(n, de[1])), F2)
            scale = np.abs(t1_[0]) + np.abs(t1_[1]) + np.abs(t2_[0]) + np.abs(t2_[1])
            dlog_lo += t1_[0] + t2_[0] - 64 * EPS * scale
        Dhi += PAD * np.abs(Dhi) + 64 * EPS * mag
        dlog_lo -= PAD * np.abs(dlog_lo)
        ell_lo -= PAD * np.abs(ell_lo) + 1e-300
        ell_hi += PAD * np.abs(ell_hi) + 1e-300
        return Dhi, dlog_lo, ell_lo, ell_hi


class HermiteMix(Reference):
    """R-mix with an R-H core and the static envelope E1: mu = (1 - eta) core + eta env."""

    def __init__(self, core, env, eta, name=None):
        self.core, self.eta = core, eta
        self.le = env.log_mu(0.0)  # static envelope, exact
        self.name = name or f"R-H mix({core.name})"
        self.nodes = core.nodes

    def log_mu(self, t):
        return np.logaddexp(np.log1p(-self.eta) + self.core.log_mu(t), np.log(self.eta) + self.le)

    def dlog_mu(self, t):
        lc = np.log1p(-self.eta) + self.core.log_mu(t)
        wc = np.exp(lc - np.logaddexp(lc, np.log(self.eta) + self.le))
        return wc * self.core.dlog_mu(t)

    def piece(self, a, b, edge_r, edge_c):
        Dc, dlc, lc_lo, lc_hi = self.core.piece(a, b, edge_r, edge_c)
        la_lo, la_hi = np.log1p(-self.eta) + lc_lo, np.log1p(-self.eta) + lc_hi  # log a(x)
        le = np.log(self.eta) + self.le
        # mu(c)/mu(r) = (a_r R + B_c)/(a_r + B_r): monotone in a_r, increasing in R
        def ratio(la_r):
            return np.logaddexp(la_r + Dc, le[edge_c]) - np.logaddexp(la_r, le[edge_r])
        Dhi = np.maximum(ratio(la_lo[edge_r]), ratio(la_hi[edge_r]))
        # dlog = w_c dlog_core, w_c = a/(a + B) in [w(a_lo), w(a_hi)]
        wlo = np.exp(la_lo - np.logaddexp(la_lo, le))
        whi = np.exp(la_hi - np.logaddexp(la_hi, le))
        # dlog_core >= dlc and w in [wlo, whi] with w >= 0, so w dlog_core >= min(wlo dlc, whi dlc)
        dlog_lo = np.where(dlc >= 0, wlo * dlc, whi * dlc)
        ell_lo = np.logaddexp(la_lo, le)
        ell_hi = np.logaddexp(la_hi, le)
        Dhi = Dhi + PAD * np.abs(Dhi) + 1e-14
        dlog_lo = dlog_lo - PAD * np.abs(dlog_lo)
        return Dhi, dlog_lo, ell_lo, ell_hi


def hermite_references(model, space, T, widths=None, n_nodes=257):
    """The R-H reference families."""
    from .reference import ETA, envelope

    mom = MomentTrajectory(model, T, widths)
    nodes = np.linspace(0.0, T, n_nodes)
    hp = HermiteProduct(space, mom, model.binary, False, nodes, name="R-H(P)")
    hnb = HermiteProduct(space, mom, model.binary, True, nodes, name="R-H(NB)")
    env = envelope(model, space)
    return {"R-P": hp, "R-NB": hnb,
            "R-mix(P)": HermiteMix(hp, env, ETA, name="R-H mix(P)"),
            "R-mix(NB)": HermiteMix(hnb, env, ETA, name="R-H mix(NB)")}
