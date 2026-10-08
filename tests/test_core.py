"""Unit tests for qcme.network, qcme.linalg, qcme.reference and qcme.certify.

    .venv/Scripts/python -m pytest tests/test_qcme_core.py
"""
import itertools

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.stats import nbinom, poisson

from qcme import certify, fsp, linalg, models, network, reference
from qcme.network import Reaction, StateSpace

rng = np.random.default_rng(20261002)


def random_network(d, K):
    """Vectorized random mass-action network (order <= 1 per species), used for random-network checks."""
    rx = []
    for _ in range(K):
        nu = rng.integers(-1, 2, size=d)
        if not nu.any():
            nu[0] = 1
        c = rng.uniform(0.1, 3.0)
        order = rng.integers(0, 2, size=d) | (nu < 0)  # a consumed species must appear in the propensity
        rx.append(Reaction(tuple(int(v) for v in nu),
                           lambda X, c=c, o=order: c * np.prod(np.where(o[None, :] == 1, X, 1.0), axis=1)))
    return rx


# ----------------------------------------------------------------------------- network
def test_state_space_lookup_roundtrip_and_misses():
    S = StateSpace.box((3, 4, 2))
    assert len(S) == 24
    assert np.array_equal(S.lookup(S.states), np.arange(24))
    assert np.all(S.lookup(np.array([[3, 0, 0], [0, -1, 0], [0, 0, 2]])) == -1)


def box_states(widths):
    return list(itertools.product(*[range(w) for w in widths]))


def hill(x, K=10.0, n=2):
    return 1.0 / (1.0 + (x / K) ** n)


def truncated_generator(states, reactions):
    """Independent dense implementation: in-box transitions kept, all outflow (including leaving S) on the
    diagonal; column convention dp/dt = A p. Returns A, the propensity table, the leak and the index."""
    index = {s: i for i, s in enumerate(states)}
    n, K = len(states), len(reactions)
    A, a, leak = np.zeros((n, n)), np.zeros((K, n)), np.zeros(n)
    for i, x in enumerate(states):
        for k, (nu, prop) in enumerate(reactions):
            rate = prop(x)
            a[k, i] = rate
            A[i, i] -= rate
            y = tuple(xi + ni for xi, ni in zip(x, nu))
            if y in index:
                A[index[y], i] += rate
            else:
                leak[i] += rate
    return A, a, leak, index


def test_generator_matches_an_independent_dense_implementation():
    toggle_rx = [((1, 0), lambda x: 30.0 * hill(x[1])), ((-1, 0), lambda x: 1.0 * x[0]),
                 ((0, 1), lambda x: 30.0 * hill(x[0])), ((0, -1), lambda x: 1.0 * x[1])]
    states = box_states([9, 11])
    A_ref, _, leak_ref, index = truncated_generator(states, toggle_rx)
    m = models.toggle()
    gen = network.generator(StateSpace.box((9, 11)), m.reactions((9, 11)))
    perm = gen.space.lookup(np.array(states))
    A = gen.A.toarray()[np.ix_(perm, perm)]
    assert np.abs(A - A_ref).max() < 1e-12
    assert np.abs(gen.leak[perm] - leak_ref).max() < 1e-12


def test_column_sums_equal_minus_leak_on_every_model():
    for m in models.registry():
        gen = network.generator(m.space(), m.reactions())
        assert np.abs(np.asarray(gen.A.sum(axis=0)).ravel() + gen.leak).max() < 1e-9, m.name


def test_negative_count_propensity_is_rejected():
    bad = [Reaction((-1,), lambda X: np.ones(len(X)))]
    with pytest.raises(ValueError):
        network.generator(StateSpace.box((4,)), bad)


def test_conservation_laws():
    def span_contains(L, v):
        return np.linalg.matrix_rank(np.vstack([L, v])) == np.linalg.matrix_rank(L)

    mm = models.michaelis_menten()
    L = network.conservation_laws(mm.reactions())
    assert L.shape == (2, 4)
    assert all(span_contains(L, row) for row in mm.conservation[0])
    assert np.array_equal(network.conservation_laws(models.self_repressor().reactions()), [[1, 1, 0]])
    assert network.conservation_laws(models.toggle().reactions()).shape == (0, 2)


# ----------------------------------------------------------------------------- linalg / fsp
def test_uniformization_matches_dense_expm():
    for _ in range(5):
        n = 30
        M = rng.uniform(0, 1, (n, n)) * (rng.uniform(size=(n, n)) < 0.2)
        np.fill_diagonal(M, 0)
        M -= np.diag(rng.uniform(0, 3, n) + M.sum(axis=0))
        v = rng.uniform(0, 1, n)
        ref = expm(M * 0.7) @ v
        assert np.abs(linalg.expm_apply(sp.csr_matrix(M), v, 0.7) - ref).max() < 1e-12 * np.abs(ref).max()


def test_uniformization_is_relatively_accurate_in_the_tails():
    """Pure birth at rate lam from 0: p(x, t) = Poisson(lam t)(x) for x < N - 1, down to 1e-120."""
    lam, t, N = 2.0, 1.0, 90
    gen = network.generator(StateSpace.box((N,)), [Reaction((1,), lambda X: np.full(len(X), lam))])
    p0 = np.zeros(N)
    p0[0] = 1.0
    x = np.arange(N - 1)
    exact = poisson.pmf(x, lam * t)
    p_u = fsp.propagate(gen.A, p0, [t])[-1][: N - 1]
    assert exact.min() < 1e-100
    assert np.max(np.abs(p_u / exact - 1)) < 1e-11
    p_k = fsp.propagate(gen.A, p0, [t], method="krylov")[-1][: N - 1]
    assert np.max(np.abs(p_k / exact - 1)) > 1e-3  # why uniformization is the default (notes 1.2)


# ----------------------------------------------------------------------------- reference
def test_stable_nb_matches_scipy_and_reaches_poisson_continuously():
    x = np.arange(80.0)
    for m, e in ((5.0, 0.3), (8.0, 4.0), (0.5, 2.0)):
        lm, _ = reference._nb_terms(x, m, e, 0.0, 0.0)
        assert np.abs(lm - nbinom.logpmf(x, m / e, 1 / (1 + e))).max() < 1e-11
    lm0, _ = reference._nb_terms(x, 5.0, 0.0, 0.0, 0.0)
    assert np.abs(lm0 - poisson.logpmf(x, 5.0)).max() < 1e-11
    lm_small, _ = reference._nb_terms(x, 5.0, 1e-12, 0.0, 0.0)
    assert np.abs(lm_small - lm0).max() < 1e-7


@pytest.mark.parametrize("name", ["telegraph", "toggle", "self_repressor", "bursty"])
def test_reference_log_derivatives_match_finite_differences(name):
    m = next(mm for mm in models.registry() if mm.name == name)
    space = m.space()
    refs = reference.standard_references(m, space, 1.0)
    h = 1e-6
    for rn, ref in refs.items():
        for t in (0.2, 0.7):
            fd = (ref.log_mu(t + h) - ref.log_mu(t - h)) / (2 * h)
            assert np.abs(ref.dlog_mu(t) - fd).max() < 1e-5 * max(1.0, np.abs(fd).max()), (name, rn, t)


def test_rmix_residual_is_a_convex_combination_lemma_10():
    m = models.telegraph()
    space = m.space()
    gen = network.generator(space, m.reactions())
    refs = reference.standard_references(m, space, 1.0)
    core, mix = refs["R-P"], refs["R-mix(P)"]
    env = mix.env
    for t in (0.1, 0.5):
        r = [certify.residual(gen.A, ref.log_mu(t), ref.dlog_mu(t)) for ref in (core, env, mix)]
        assert np.all(r[2] <= np.maximum(r[0], r[1]) + 1e-9)


# ----------------------------------------------------------------------------- certify
def _random_case(d_max=3):
    d = int(rng.integers(1, d_max + 1))
    space = StateSpace.box([int(rng.integers(3, 7)) for _ in range(d)])
    gen = network.generator(space, random_network(d, int(rng.integers(2, 6))))
    n = len(space)
    lm = np.log(rng.uniform(0.05, 1.0, n))
    dl = rng.normal(size=n)
    return gen, lm, dl


def test_residual_and_weighted_generator_match_dense_formulas():
    for _ in range(50):
        gen, lm, dl = _random_case()
        A, mu = gen.A.toarray(), np.exp(lm)
        r = certify.residual(gen.A, lm, dl)
        assert np.abs(r - (A @ mu - dl * mu) / mu).max() < 1e-10
        B = certify.weighted_generator(gen.A, lm, dl).toarray()
        s = np.sqrt(mu)
        assert np.abs(B - (A * s[None, :] / s[:, None] - 0.5 * np.diag(dl))).max() < 1e-10


def test_chi2_identity_lemma_2_with_sparse_objects():
    worst = 0.0
    for _ in range(100):
        gen, lm, dl = _random_case()
        n = len(lm)
        mu, h = np.exp(lm), rng.normal(size=n)
        psi = np.sqrt(mu) * h
        lhs = psi @ (certify.weighted_generator(gen.A, lm, dl) @ psi)
        A = sp.coo_matrix(gen.A)
        off = A.row != A.col
        dirichlet = np.sum(mu[A.col[off]] * A.data[off] * (h[A.row[off]] - h[A.col[off]]) ** 2)
        r = certify.residual(gen.A, lm, dl)
        rhs = -0.5 * dirichlet - 0.5 * np.sum(mu * h**2 * gen.leak) + 0.5 * np.sum(mu * h**2 * r)
        worst = max(worst, abs(lhs - rhs) / max(1.0, abs(lhs)))
    assert worst < 1e-10


def test_certificate_ordering_omega_le_resolvent_le_sqrt_mu_le_half_kappa():
    for _ in range(60):
        gen, lm, dl = _random_case()
        B = certify.weighted_generator(gen.A, lm, dl)
        Sigma = certify.symmetric_part(B)
        om = certify.lognorm_dense(B)
        r = certify.residual(gen.A, lm, dl)
        c_sqrt = certify.cw_sqrt_mu(r, gen.leak)
        res = certify.resolvent_cw(Sigma, c_sqrt)
        assert om <= res.value + 1e-12
        assert res.value <= c_sqrt + 1e-9 * max(1, abs(c_sqrt))
        assert c_sqrt <= 0.5 * certify.kappa(r) + 1e-12
        assert res.value - om <= 2e-6 * max(1.0, abs(om))  # bisection reaches omega to its tolerance
        assert om <= certify.cw(Sigma, np.sqrt(np.exp(lm)) * np.exp(rng.normal(size=len(lm)))) + 1e-12


def test_window_certificate_partial_exponents():
    m = models.telegraph()
    space = m.space((2, 30))
    gen = network.generator(space, m.reactions())
    ref = reference.standard_references(m, space, 1.0)["R-mix(NB)"]
    edges = np.linspace(0, 1.0, 6)
    cert = certify.window_certificate(gen, ref, edges)
    for f in certify.FAMILIES:
        assert cert.K_at(f, 1.0) == pytest.approx(cert.K[f], rel=1e-12)
        assert cert.K_at(f, 0.0) == 0.0
    assert cert.K["resolvent"] <= cert.K["sqrt_mu"] + 1e-9 <= cert.K_kappa + 2e-9
    assert np.all(cert.log_rho >= 0)


def test_chi2_corollary_bounds_the_true_chi2():
    from qcme import emulate

    m = models.telegraph()
    space = m.space((2, 40))
    gen = network.generator(space, m.reactions())
    p0 = reference.initial_law(m, space)
    ref = reference.standard_references(m, space, 1.0)["R-mix(NB)"]
    cert = certify.window_certificate(gen, ref, np.linspace(0, 1.0, 11))
    pT = fsp.propagate(gen.A, p0, [1.0])[-1]
    muT = np.exp(ref.log_mu(cert.edges[-2]))  # last window's reference
    psi0_sq = np.sum(p0**2 / np.exp(ref.log_mu(0.0)))
    chi2_bd, tv_bd = certify.chi2_bound(cert.K["resolvent"], psi0_sq, pT.sum(), muT.sum())
    q, mb = pT / pT.sum(), muT / muT.sum()
    assert np.sum((q - mb) ** 2 / mb) <= chi2_bd
    assert 0.5 * np.abs(q - mb).sum() <= tv_bd
