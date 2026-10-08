"""Entrywise-accurate exponential actions for Metzler matrices (uniformization).

A_S and the weighted generator B have non-negative off-diagonal entries. With q >= max |M_xx|,
e^{Mt} v = e^{-qt} sum_k (qt)^k/k! (I + M/q)^k v and I + M/q >= 0 entrywise, so for v >= 0 every
term is non-negative: no cancellation, hence relative accuracy in every entry, including tail
entries far below machine epsilon times the norm. Krylov/Taylor methods (expm_multiply) are only
normwise accurate, which is not enough once the weighted state psi = p/sqrt(mu) divides tail
probabilities by tiny reference values. The cost depends on q t only, not on the size of the
off-diagonal entries (large ratios sqrt(mu(x)/mu(y)) cost nothing extra).
"""
import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

MAX_RATE_STEP = 30.0  # sub-step so that q * dt <= this (keeps e^{-q dt} and the series benign)


def expm_action(M, v, t, tail=1e-280, max_terms=100_000):
    """e^{M t} v for Metzler M and v >= 0, entrywise relatively accurate. Returns (result, log_scale)
    with e^{M t} v = result * e^{log_scale}; log_scale absorbs growth to avoid overflow."""
    M = sp.csr_matrix(M)
    v = np.asarray(v, float)
    if np.any(v < 0):
        raise ValueError("uniformization needs a non-negative vector")
    if M.nnz and (M - sp.diags(M.diagonal())).min() < 0:
        raise ValueError("uniformization needs a Metzler matrix (non-negative off-diagonal)")
    if t == 0:
        return v.copy(), 0.0
    q = max(float(np.max(np.abs(M.diagonal()))), 1e-300)
    n_sub = max(1, int(np.ceil(q * t / MAX_RATE_STEP)))
    dt = t / n_sub
    lam = q * dt
    P = (sp.identity(M.shape[0], format="csr") + M / q).tocsr()
    log_scale = 0.0
    x = v.copy()
    for _ in range(n_sub):
        s = x.sum()
        if s == 0:
            return x, -np.inf
        x = x / s
        log_scale += np.log(s)
        acc = np.zeros_like(x)
        term = x
        log_t = 0.0  # term_k = P^k x * e^{-log_t}; renormalized to avoid over/underflow
        for k in range(max_terms):
            w = np.exp(-lam + k * np.log(lam) - gammaln(k + 1) + log_t)
            acc += w * term
            # stop past the Poisson mode once the remaining series is negligible (terms decay
            # super-geometrically there because the weights do)
            if k > lam and w * term.sum() <= tail * acc.sum():
                break
            term = P @ term
            tn = term.sum()
            if tn == 0:
                break
            if tn > 1e100 or tn < 1e-100:
                term = term / tn
                log_t += np.log(tn)
        else:
            raise RuntimeError("uniformization did not converge")
        x = acc
    return x, log_scale


def expm_apply(M, v, t, **kw):
    """e^{M t} v as a plain vector (overflows to inf if the true result exceeds double range)."""
    x, ls = expm_action(M, v, t, **kw)
    with np.errstate(over="ignore"):
        return x * np.exp(ls)
