"""Reference measures mu_t on a state space.

Every reference returns log mu_t and the log-derivative mu_dot/mu. Working in logs keeps tail ratios finite on large
boxes. References are computed from the network alone (rate and moment equations); only
`Exact` takes a CME solution, and it is a validation oracle.

Families: R-P (product Poisson, Bernoulli for gene states), R-NB (product
negative binomial fitted to the linear-noise moments), R-LNA, R-TT, R-mix, R-exact, uniform.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gammaln

from .models import _jacobian

MEAN_FLOOR = 1e-3  # initial Poisson means >= floor; initial Bernoulli probabilities in [floor, 1 - floor]
SAFETY_FLOOR = 1e-12  # later means are clipped only if the moment ODE drifts to the boundary


# ----------------------------------------------------------------------------- moment equations
def floored_x0(model):
    """Initial means with the floor applied once, at t = 0 (a Poisson start with a tiny positive
    mean keeps product Poisson exact for monomolecular networks, which a running floor would not)."""
    m0 = np.asarray(model.x0, float)
    return np.where(model.binary, np.clip(m0, MEAN_FLOOR, 1 - MEAN_FLOOR), np.maximum(m0, MEAN_FLOOR))


class MomentTrajectory:
    """Rate equations with linear-noise covariance: m' = f(m), S' = J S + S J^T + D(m),
    D = sum_k nu_k nu_k^T r_k(m). Initial covariance diag(x0) (Poisson) or g(1 - g) (Bernoulli),
    matching the product-Poisson initial law p0."""

    def __init__(self, model, T, widths=None):
        self.model, self.d = model, len(model.species)
        rx = model.reactions(widths)
        self._nus = np.array([r.nu for r in rx], dtype=float)
        self._rx = rx
        m0 = floored_x0(model)
        v0 = np.where(model.binary, m0 * (1 - m0), m0)
        y0 = np.concatenate([m0, np.diag(v0).ravel()])
        self.sol = solve_ivp(self._rhs, (0.0, T), y0, method="LSODA", dense_output=True, rtol=1e-10, atol=1e-12)
        if not self.sol.success:
            raise RuntimeError(self.sol.message)

    def _rates(self, m):
        m = np.maximum(m, 0.0)
        return np.array([r.deterministic_rate(m) for r in self._rx])

    def _drift(self, m):
        return self._nus.T @ self._rates(m)

    def _rhs(self, t, y):
        d = self.d
        m, S = y[:d], y[d:].reshape(d, d)
        J = _jacobian(self._drift, np.maximum(m, 0.0))
        D = (self._nus.T * self._rates(m)) @ self._nus
        return np.concatenate([self._drift(m), (J @ S + S @ J.T + D).ravel()])

    def at(self, t):
        """(mean, mean_dot, var, var_dot) at time t."""
        m, md, S, Sd = self.full(t)
        return m, md, np.diag(S).copy(), np.diag(Sd).copy()

    def full(self, t):
        """(mean, mean_dot, covariance, covariance_dot) at time t (R-LNA)."""
        d = self.d
        y = self.sol.sol(t)
        dy = self._rhs(t, y)
        S, Sd = y[d:].reshape(d, d), dy[d:].reshape(d, d)
        return y[:d], dy[:d], 0.5 * (S + S.T), 0.5 * (Sd + Sd.T)


class ConstantMoments:
    """Time-independent means (and optional NB tail parameter q) for static envelopes."""

    def __init__(self, means, q=None):
        self.means = np.asarray(means, float)
        self.q = None if q is None else np.asarray(q, float)

    def at(self, t):
        m = self.means
        var = m.copy() if self.q is None else m / (1 - self.q)  # NB(s, q): var = mean / (1 - q)
        z = np.zeros_like(m)
        return m, z, var, z


# ----------------------------------------------------------------------------- marginals
def _log_poisson(x, lam):
    return x * np.log(lam) - lam - gammaln(x + 1)


def _log1p_over(e):
    """log(1 + e)/e and its derivative in e, stable down to e = 0 (limits 1 and -1/2)."""
    e = float(e)
    if e < 1e-4:
        return 1 - e / 2 + e * e / 3 - e**3 / 4, -0.5 + 2 * e / 3 - 3 * e * e / 4
    L = np.log1p(e)
    return L / e, (e / (1 + e) - L) / e**2


def _nb_terms(x, m, e, md, ed):
    """NB with mean m and relative overdispersion e = var/mean - 1 >= 0, and its time derivative.

    log mu(x) = sum_{i<x} log(m + i e) - x log(1 + e) - log x! - m log(1 + e)/e, which is
    NB(s = m/e, q = e/(1 + e)) for e > 0 and exactly Poisson(m) at e = 0, so the family is
    continuous as the variance approaches the mean.
    x must be non-negative integers (as floats)."""
    xi = x.astype(np.int64)
    top = int(xi.max()) + 1
    i = np.arange(top, dtype=float)
    den = m + i * e
    c_log = np.concatenate([[0.0], np.cumsum(np.log(den))])
    c_inv = np.concatenate([[0.0], np.cumsum(1.0 / den)])
    c_ion = np.concatenate([[0.0], np.cumsum(i / den)])
    lo, dlo = _log1p_over(e)
    logmu = c_log[xi] - x * np.log1p(e) - gammaln(x + 1) - m * lo
    dm = c_inv[xi] - lo
    de = c_ion[xi] - x / (1 + e) - m * dlo
    return logmu, dm * md + de * ed


class Reference:
    name = "reference"

    def log_mu(self, t):
        raise NotImplementedError

    def dlog_mu(self, t):
        raise NotImplementedError

    def mu(self, t):
        return np.exp(self.log_mu(t))


class Product(Reference):
    """Product reference: Bernoulli for binary species, and per species either Poisson(mean) or,
    with nb=True, NB matched to the (mean, variance) pair where the variance exceeds the mean
    (Poisson where it does not; the NB family reaches Poisson continuously)."""

    def __init__(self, space, moments, binary, nb=False, name=None):
        self.X = space.X
        self.moments, self.binary, self.nb = moments, np.asarray(binary, bool), nb
        self.name = name or ("R-NB" if nb else "R-P")

    def _terms(self, t):
        m, md, v, vd = self.moments.at(t)
        logmu = np.zeros(len(self.X))
        dlog = np.zeros(len(self.X))
        for i in range(self.X.shape[1]):
            x = self.X[:, i]
            if self.binary[i]:
                g = np.clip(m[i], SAFETY_FLOOR, 1 - SAFETY_FLOOR)
                gd = md[i] if SAFETY_FLOOR < m[i] < 1 - SAFETY_FLOOR else 0.0
                logmu += x * np.log(g) + (1 - x) * np.log1p(-g)
                dlog += (x / g - (1 - x) / (1 - g)) * gd
                continue
            lam = max(m[i], SAFETY_FLOOR)
            lamd = md[i] if m[i] > SAFETY_FLOOR else 0.0
            if self.nb and v[i] > lam:
                e = v[i] / lam - 1
                ed = (vd[i] * lam - v[i] * lamd) / lam**2
                lmu, dl = _nb_terms(x, lam, e, lamd, ed)
                logmu += lmu
                dlog += dl
            else:
                logmu += _log_poisson(x, lam)
                dlog += (x / lam - 1) * lamd
        return logmu, dlog

    def log_mu(self, t):
        return self._terms(t)[0]

    def dlog_mu(self, t):
        return self._terms(t)[1]


def envelope(model, space):
    """Static heavy-tailed envelope E1 given with each model."""
    spec = model.envelope
    if spec["kind"] == "poisson":
        return Product(space, ConstantMoments(spec["means"]), model.binary, name="envelope")
    if spec["kind"] == "nb":
        return Product(space, ConstantMoments(spec["means"], spec["q"]), model.binary, nb=True, name="envelope")
    if spec["kind"] == "poisson_maxprod":
        return Product(space, ConstantMoments(maxprod_means(model, space, spec["removal"])), model.binary,
                       name="envelope")
    raise ValueError(spec["kind"])


def maxprod_means(model, space, removal):
    """E1 rule for published models: per count species, the maximum
    over the state space of its total production rate sum_k max(nu_k,i, 0) a_k(x), divided by its
    first-order removal constant (the T3(iii) supersolution tail parameter); 1/2 for gene states."""
    X = space.X
    prod = np.zeros(X.shape[1])
    rx = model.reactions(tuple(int(w) for w in space.widths))
    for i in range(X.shape[1]):
        tot = np.zeros(len(X))
        for r in rx:
            if r.nu[i] > 0:
                tot += r.nu[i] * np.asarray(r.prop(X), float)
        prod[i] = tot.max()
    binary = np.asarray(model.binary, bool)
    return np.where(binary, 0.5, np.maximum(prod / np.asarray(removal, float), MEAN_FLOOR))


class Mix(Reference):
    """R-mix: (1 - eta) core + eta envelope; mu_dot mixes the same way."""

    def __init__(self, core, env, eta, name=None):
        self.core, self.env, self.eta = core, env, eta
        self.name = name or f"R-mix({core.name}, eta={eta:g})"

    def log_mu(self, t):
        return np.logaddexp(np.log1p(-self.eta) + self.core.log_mu(t), np.log(self.eta) + self.env.log_mu(t))

    def dlog_mu(self, t):
        lc = np.log1p(-self.eta) + self.core.log_mu(t)
        lm = np.logaddexp(lc, np.log(self.eta) + self.env.log_mu(t))
        wc = np.exp(lc - lm)
        return wc * self.core.dlog_mu(t) + (1 - wc) * self.env.dlog_mu(t)


PINV_RTOL = 1e-10  # R-LNA pseudo-inverse eigenvalue cutoff, relative to the largest


class LNA(Reference):
    """R-LNA: discretized Gaussian exp(-(x - m)^T Sigma^+ (x - m)/2), normalized on the state space,
    with the rate-equation mean and the linear-noise covariance.

    On conservation-law models the state space is the manifold, so the law is the Gaussian
    restricted to it (the conditional Gaussian when Sigma is nonsingular off the manifold)."""

    def __init__(self, space, moments, name="R-LNA"):
        self.X, self.moments, self.name = space.X, moments, name

    def _terms(self, t):
        m, md, S, Sd = self.moments.full(t)
        lam, Q = np.linalg.eigh(S)
        keep = lam > PINV_RTOL * lam.max()
        Qk = Q[:, keep]
        P = (Qk / lam[keep]) @ Qk.T
        # derivative of the pseudo-inverse at constant rank (symmetric case)
        Pi = np.eye(len(m)) - Qk @ Qk.T
        Pd = -P @ Sd @ P + Pi @ Sd @ P @ P + P @ P @ Sd @ Pi
        dx = self.X - m
        q = np.einsum("ni,ij,nj->n", dx, P, dx)
        qd = -2 * (dx @ P) @ md + np.einsum("ni,ij,nj->n", dx, Pd, dx)
        lu = -0.5 * q
        logZ = lu.max() + np.log(np.sum(np.exp(lu - lu.max())))
        w = np.exp(lu - logZ)
        return lu - logZ, -0.5 * qd + 0.5 * float(w @ qd)

    def log_mu(self, t):
        return self._terms(t)[0]

    def dlog_mu(self, t):
        return self._terms(t)[1]


def envelope_e2(model, space, moments, T, n_grid=1025):
    """Envelope E2: product law with, per species, the maximum over
    [0, T] of the rate-equation mean (1025-point grid, floored at MEAN_FLOOR); Poisson for counts,
    Bernoulli clipped to [MEAN_FLOOR, 1 - MEAN_FLOOR] for binary species."""
    ts = np.linspace(0.0, T, n_grid)
    means = np.max([moments.at(t)[0] for t in ts], axis=0)
    binary = np.asarray(model.binary, bool)
    means = np.where(binary, np.clip(means, MEAN_FLOOR, 1 - MEAN_FLOOR), np.maximum(means, MEAN_FLOOR))
    return Product(space, ConstantMoments(means), model.binary, name="envelope E2")


class Uniform(Reference):
    name = "uniform"

    def __init__(self, n):
        self.n = n

    def log_mu(self, t):
        return np.full(self.n, -np.log(self.n))

    def dlog_mu(self, t):
        return np.zeros(self.n)


class Exact(Reference):
    """R-exact: mu_t = p_t (unnormalized FSP solution). Validation oracle only, never for claims."""

    name = "R-exact"

    def __init__(self, p_of_t, pdot_of_t):
        self.p, self.pd = p_of_t, pdot_of_t

    def log_mu(self, t):
        return np.log(self.p(t))

    def dlog_mu(self, t):
        return self.pd(t) / self.p(t)


class Static(Reference):
    """A fixed reference (piecewise-constant windows are built from these)."""

    def __init__(self, logmu, name="static"):
        self._lm, self.name = np.asarray(logmu, float), name

    def log_mu(self, t):
        return self._lm

    def dlog_mu(self, t):
        return np.zeros_like(self._lm)


ETA = 1e-3  # weight of the envelope in every R-mix


def standard_references(model, space, T, widths=None):
    """The core reference families for a model: R-P, R-NB and R-mix with R-P and R-NB cores."""
    mom = MomentTrajectory(model, T, widths)
    rp = Product(space, mom, model.binary, nb=False)
    rnb = Product(space, mom, model.binary, nb=True)
    env = envelope(model, space)
    return {
        "R-P": rp,
        "R-NB": rnb,
        "R-mix(P)": Mix(rp, env, ETA, name="R-mix(P)"),
        "R-mix(NB)": Mix(rnb, env, ETA, name="R-mix(NB)"),
    }


TT_RANKS = (2, 4, 8)  # two-species models only


def extended_references(model, space, T, widths=None, gen=None, ranks=TT_RANKS):
    """The reference families of the toggle-switch study: R-P, R-NB, R-LNA, every R-mix
    with envelopes E1 and E2, R-LNA and R-mix(LNA), and, on two-species box models, R-TT with
    rank r in `ranks` (needs the truncated generator `gen`; network only, never the FSP solution)."""
    mom = MomentTrajectory(model, T, widths)
    rp = Product(space, mom, model.binary, nb=False)
    rnb = Product(space, mom, model.binary, nb=True)
    lna = LNA(space, mom)
    e1 = envelope(model, space)
    e2 = envelope_e2(model, space, mom, T)
    refs = {"R-P": rp, "R-NB": rnb, "R-LNA": lna}
    for core_name, core in (("P", rp), ("NB", rnb), ("LNA", lna)):
        for env_name, env in (("E1", e1), ("E2", e2)):
            name = f"R-mix({core_name},{env_name})"
            refs[name] = Mix(core, env, ETA, name=name)
    two_count_species = len(model.species) == 2 and not any(model.binary) and model.conservation is None
    if gen is not None and two_count_species and ranks:
        from .lowrank import LowRankCore
        p0 = initial_law(model, space)
        for r in ranks:
            core = LowRankCore(gen, space, p0, r, T)
            refs[f"R-TT(r={r})"] = Mix(core, e2, ETA, name=f"R-TT(r={r})")
    return refs


def published_references(model, space, T, widths=None):
    """The reference families for the published models: R-P, R-NB and their mixtures with envelopes E1
    and E2."""
    mom = MomentTrajectory(model, T, widths)
    rp = Product(space, mom, model.binary, nb=False)
    rnb = Product(space, mom, model.binary, nb=True)
    e1 = envelope(model, space)
    e2 = envelope_e2(model, space, mom, T)
    refs = {"R-P": rp, "R-NB": rnb}
    for core_name, core in (("P", rp), ("NB", rnb)):
        for env_name, env in (("E1", e1), ("E2", e2)):
            name = f"R-mix({core_name},{env_name})"
            refs[name] = Mix(core, env, ETA, name=name)
    return refs



def initial_law(model, space, T=1e-9):
    """p0: the normalized R-P reference at t = 0."""
    mom = MomentTrajectory(model, max(T, 1e-9))
    lm = Product(space, mom, model.binary).log_mu(0.0)
    p = np.exp(lm - lm.max())
    return p / p.sum()
