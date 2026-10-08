"""Costs: LCHS kernels and bounds, Hamiltonian-simulation degree, gate-level block encodings (I8,
I9), the walk construction, resource formulas, AE formulas and the quantum baselines."""
import math

import numpy as np
import pytest
from scipy import special

from qcme import baselines as BL
from qcme import certify, circuits, fsp, lchs, models, network, reference
from qcme import resources as R


# ----------------------------------------------------------------------------- LCHS
@pytest.mark.parametrize("ker", [lchs.Cauchy(), lchs.ACL(0.7), lchs.LowSomma(1.0, 1e-8), lchs.LowSomma(2.0, 1e-8)])
def test_kernel_reproduces_scalar_exponential(ker):
    """int g(k) e^{-ika} dk = e^{-a} for a >= 0 (the kernel conventions of qcme.lchs)."""
    h = 0.002
    k = np.arange(-4000, 4000, h)
    g = ker.g(k)
    tol = 2e-4 if isinstance(ker, lchs.Cauchy) else 1e-9  # K0 has a slow 1/K tail
    for a in (0.0, 0.5, 3.0):
        v = np.sum(g * np.exp(-1j * k * a)) * h
        assert abs(v - np.exp(-a)) < tol


def test_discretization_bound_holds_on_time_ordered_products():
    rng = np.random.default_rng(1)
    pieces = []
    for _ in range(2):
        X = rng.normal(size=(5, 5))
        H = rng.normal(size=(5, 5))
        pieces.append(X @ X.T / 5 + 1j * (H + H.T) / 2)
    pieces.insert(1, np.diag(rng.uniform(0, 2, 5)).astype(complex))
    Lnorm = sum(np.linalg.norm(lchs.split(P)[0], 2) for P in pieces)
    W = lchs.exact_product(pieces)
    for ker in (lchs.ACL(0.8), lchs.LowSomma(1.0, 1e-6 / 3)):
        d = lchs.discretize(ker, Lnorm, 1e-6)
        assert np.linalg.norm(lchs.emulate(d, pieces) - W, 2) <= d.error <= 1e-6


def test_jacobi_anger_degree_is_the_smallest_rigorous_truncation():
    for tau in (0.5, 7.0, 60.0):
        for delta in (1e-6, 1e-14):
            q = lchs.jacobi_anger_degree(tau, delta)
            tail = lambda qq: 2 * sum(abs(special.jv(l, tau)) for l in range(qq + 1, qq + 400))
            assert tail(q) <= delta
            if q > math.floor(tau):
                assert tail(q - 1) > delta


# ----------------------------------------------------------------------------- circuits (I8, I9)
@pytest.mark.parametrize("name,w", [("toggle", (4, 4)), ("mm", None)])
def test_block_encoding_I8_I9(name, w):
    m = models.toggle() if name == "toggle" else models.michaelis_menten(E0=2, S0=3)
    w = tuple(w or m.widths)
    sp_ = m.space(w)
    rx = m.reactions(w)
    gen = network.generator(sp_, rx)
    ref = reference.extended_references(m, sp_, 1.0, w, ranks=())["R-mix(NB,E1)"]
    lm = ref.log_mu(0.3)
    c = certify.cw_sqrt_mu(certify.residual(gen.A, lm), gen.leak)
    inst = circuits.block_encoding_circuit(gen, [r.nu for r in rx], lm, c)
    assert inst.circuit.check_permutations()[0]
    Bk = circuits.block_of(inst)
    assert np.abs(Bk - inst.M / inst.alpha).max() <= 1e-10  # I8
    idx = circuits.system_index(sp_.states, inst.sys_bits)
    Bt = certify.weighted_generator(gen.A, lm).toarray() - c * np.eye(len(sp_))
    assert np.abs(inst.M[np.ix_(idx, idx)] - Bt).max() <= 1e-12
    out = np.setdiff1d(np.arange(Bk.shape[0]), idx)
    assert np.all(Bk[np.ix_(out, idx)] == 0)  # I9, exactly
    assert np.linalg.eigvalsh(0.5 * (inst.M + inst.M.T)).max() <= 1e-9  # log-norm of M <= 0


def test_hermitian_select_is_a_self_inverse_block_encoding():
    rng = np.random.default_rng(3)
    D = 8
    Q, _ = np.linalg.qr(rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)))
    B = Q[:4, :4] * 2.0  # a block of a unitary, alpha = 2
    ck = (-1.5 + 1j) / 2
    Lam = 2.0 * np.sqrt(1 + 3.0**2)
    V = circuits.hermitian_select(Q, 1, ck, Lam, 2.0)
    assert np.abs(V @ V - np.eye(2 * D)).max() < 1e-12
    assert np.abs(V - V.conj().T).max() < 1e-12
    assert np.abs(V[:4, :4] - (ck * B + np.conj(ck) * B.conj().T) / Lam).max() < 1e-12


# ----------------------------------------------------------------------------- resources
def test_primitive_formulas_and_clog2():
    assert [R.add(n) for n in (1, 4, 8)] == [0, 3, 7]
    assert R.mult(4, 4) == 28 and R.qrom(256) == 254 and R.phasegrad(8) == 6 and R.mcz(8) == 7
    assert R.alias(16, 8) == 14 + 15 + 4
    assert R.uniform_prep(12) == 3 * 2 - 1 and R.uniform_prep(16) == 0
    for n in (1, 2, 3, 4, 5, 1023, 1024, 1025, 10**30):
        c = R.clog2(n)
        assert 2**c >= n and (c == 0 or 2 ** (c - 1) < n)


@pytest.mark.parametrize("fname,lo,hi", [("exp", -14.0, 0.0), ("softplus", -14.0, 14.0)])
def test_function_evaluation_meets_its_error_budget(fname, lo, hi):
    b = 20
    fc = R.function_eval(fname, lo, hi, b)
    f = R.FUNCTIONS[fname]
    edges = np.linspace(lo, hi, fc.pieces + 1)
    nodes = np.cos(np.pi * (np.arange(fc.degree + 1) + 0.5) / (fc.degree + 1))
    t = np.linspace(-1, 1, 200)
    worst = 0.0
    for p in np.linspace(0, fc.pieces - 1, min(fc.pieces, 64)).astype(int):
        a, c = edges[p], edges[p + 1]
        xs = 0.5 * (a + c) + 0.5 * (c - a) * nodes
        coef = np.polynomial.chebyshev.chebfit(nodes, f(xs), fc.degree)
        xt = 0.5 * (a + c) + 0.5 * (c - a) * t
        worst = max(worst, np.max(np.abs(np.polynomial.chebyshev.chebval(t, coef) - f(xt))))
    assert worst <= 2.0 ** (-b - 1)


def test_sigmoid_bound_used_for_softplus_derivatives():
    """|sigmoid(z)| <= 8 on every disk of radius 3 centred on the real axis (Cauchy estimate input)."""
    th = np.linspace(0, 2 * np.pi, 4001)
    worst = 0.0
    for x0 in np.linspace(-10, 10, 401):
        z = x0 + 3 * np.exp(1j * th)
        worst = max(worst, np.max(np.abs(1 / (1 + np.exp(-z)))))
    assert worst <= 8.0


def test_T2_amplitude_estimation_constants():
    eta, dfail = 1e-3, 1e-2
    M, n = R.t2_grover(eta), R.median_runs(dfail)
    assert M == math.ceil(8 * math.pi / eta)
    assert n <= math.ceil(5.19 * math.log(1 / dfail))  # T2: n <= ceil(5.19 ln(1/delta_fail))
    for P, e in ((0.05, 2.5e-4), (0.999, 1e-3)):
        Mb = R.bhmt_M(P, e)
        err = lambda MM: 2 * math.pi * math.sqrt(P * (1 - P)) / MM + math.pi**2 / MM**2
        assert err(Mb) <= e * (1 + 1e-12) and err(Mb - 1) > e


# ----------------------------------------------------------------------------- baselines
def test_uniformization_is_exact_in_law():
    m = models.birth_death()
    w = (30,)
    sp_ = m.space(w)
    gen = network.generator(sp_, m.reactions(w))
    p0 = reference.initial_law(m, sp_)
    T = 1.0
    Lam = BL.uniformization_rate(gen)
    P = np.eye(len(sp_)) + gen.A.toarray() / Lam
    N = BL.poisson_truncation(Lam * T, 1e-14)
    from scipy.stats import poisson
    p, acc = p0.copy(), np.zeros(len(sp_))
    for n in range(N + 1):
        acc += poisson.pmf(n, Lam * T) * p
        p = P @ p
    pT = fsp.propagate(gen.A, p0, [0.0, T])[-1]
    assert np.abs(acc - pT).sum() < 1e-12


@pytest.mark.parametrize("order", [1, 2])
def test_lindblad_scheme_order(order):
    m = models.birth_death()
    w = (40,)
    sp_ = m.space(w)
    rx = m.reactions(w)
    gen = network.generator(sp_, rx)
    p0 = reference.initial_law(m, sp_)
    pT = fsp.propagate(gen.A, p0, [0.0, 1.0])[-1]
    errs = []
    for N in (128, 256, 512):
        Mx, _ = BL.lindblad_step_matrix(gen, [r.nu for r in rx], order, 1.0 / N)
        p = p0.copy()
        for _ in range(N):
            p = Mx @ p
        errs.append(np.abs(p - pT).sum())
    ratio = errs[1] / errs[2]
    assert abs(ratio / 2**order - 1) < 0.1


def test_lindblad_channel_is_substochastic():
    m = models.toggle()
    w = (10, 10)
    sp_ = m.space(w)
    rx = m.reactions(w)
    gen = network.generator(sp_, rx)
    for order in (1, 2):
        Mx, _ = BL.lindblad_step_matrix(gen, [r.nu for r in rx], order, 0.01)
        assert Mx.min() >= 0 and np.asarray(Mx.sum(axis=0)).max() <= 1 + 1e-12


def test_propensity_dependence_on_conservation_manifolds():
    """Regression: dependence is read on the register box, so manifold models
    keep their true propensity variables."""
    for m, want in ((models.michaelis_menten(), [{0, 1}, {2}, {2}]),
                    (models.self_repressor(), [{0, 2}, {1}, {0}, {1}, {2}])):
        w = m.widths
        st = R.ModelStructure.build(m, network.generator(m.space(w), m.reactions(w)), w)
        assert st.deps == want
    m = models.toggle()
    st = R.ModelStructure.build(m, network.generator(m.space((20, 20)), m.reactions((20, 20))), (20, 20))
    assert st.deps == [{1}, {0}, {0}, {1}]
