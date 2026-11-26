"""The hard instances of the readout barrier (Supplementary Section S2) and the stationary-law reference on a
reversible model (Supplementary Section S4.5), checked numerically.

    python -m pytest tests/test_hard_instances.py
"""
import numpy as np
from scipy.linalg import expm
from scipy.special import logsumexp

from qcme import certify, fsp, models, network, reference
from qcme.network import Reaction, StateSpace


def hidden_set_instance(N, hidden, lam=1.0):
    """Two species: X in 0..N-1 with no reactions, and Y produced at rate lam * g(x), g = 0 on the hidden states.
    The box holds the states with Y = 0, so producing Y leaves it."""
    g = np.ones(N)
    g[list(hidden)] = 0.0
    space = StateSpace.box((N, 1))
    gen = network.generator(space, [Reaction((0, 1), lambda X: lam * g[X[:, 0].astype(int)], name="0 -> Y")])
    return space, gen, g


def test_hard_family_is_a_truncated_master_equation_that_counts_the_hidden_states():
    N, K, lam, T = 64, 6, 1.0, 12.0
    hidden = [3, 10, 11, 29, 40, 63]
    space, gen, g = hidden_set_instance(N, hidden, lam)
    A = gen.A.toarray()
    order = space.states[:, 0].astype(int)
    assert np.allclose(A, np.diag(-lam * g[order]))  # diagonal generator: no reactions inside the box
    assert np.allclose(gen.leak, lam * g[order])  # every non-hidden state leaves the box at rate lam
    p0 = np.full(N, 1.0 / N)
    pT = expm(T * A) @ p0
    eta = np.exp(-lam * T)
    m_T = pT.sum()
    S_eff = m_T**2 / np.sum(pT**2)
    assert np.isclose(m_T, (K + eta * (N - K)) / N)
    assert np.isclose(S_eff, (K + eta * (N - K)) ** 2 / (K + eta**2 * (N - K)))
    assert K <= N * m_T <= K * (1 + 1e-3) and abs(S_eff - K) < 1e-2  # m_T counts the hidden states


def test_block_encoding_of_the_hard_instance_and_the_two_readout_amplitudes():
    N, lam, T = 32, 1.0, 9.0
    hidden = [1, 2, 20]
    space, gen, g = hidden_set_instance(N, hidden, lam)
    P = expm(T * gen.A.toarray())  # diagonal with entries 1 (hidden) and eta (others): c = 1
    D = np.diag(P)
    assert np.all(D <= 1.0 + 1e-15)
    U = np.block([[np.diag(D), np.diag(np.sqrt(1 - D**2))], [np.diag(np.sqrt(1 - D**2)), -np.diag(D)]])
    assert np.allclose(U @ U.T, np.eye(2 * N))  # unitary, built from one controlled rotation per state
    assert np.allclose(U[:N, :N], P)
    plus = np.full(N, 1.0 / np.sqrt(N))
    pT = P @ np.full(N, 1.0 / N)
    m_T, S_eff = pT.sum(), pT.sum() ** 2 / np.sum(pT**2)
    a_direct = plus @ U[:N, :N] @ plus  # direct overlap readout, initial state p0/||p0|| = |+>
    a_norm = plus @ (pT / np.linalg.norm(pT))  # normalize-then-overlap readout
    assert np.isclose(a_direct, m_T) and a_direct <= np.sqrt(S_eff / N) + 1e-12  # Lemma S1
    assert np.isclose(a_norm, np.sqrt(S_eff / N))


def test_observable_amplitude_bound_for_a_tail_set():
    m = models.cascade(2, width=12)
    space = m.space((12, 12))
    gen = network.generator(space, m.reactions((12, 12)))
    pT = fsp.propagate(gen.A, reference.initial_law(m, space), [1.0])[-1]
    B = space.states[:, 1] >= 6
    P_B, q = pT[B].sum(), pT[B]
    a_B = P_B / (np.sqrt(B.sum()) * np.linalg.norm(pT))  # the largest amplitude allowed by c >= ||p_T|| / ||p_0||
    assert a_B <= np.sqrt((q.sum() ** 2 / np.sum(q**2)) / B.sum()) + 1e-15  # Corollary S2


def test_stationary_law_certifies_the_reversible_schlogl_model():
    m = models.schlogl()
    w = (200,)
    space = m.space(w)
    gen = network.generator(space, m.reactions(w))
    X = space.states.astype(float)
    birth = sum(r.prop(X) for r in m.reactions(w) if tuple(r.nu) == (1,))
    death = sum(r.prop(X) for r in m.reactions(w) if tuple(r.nu) == (-1,))
    order = np.argsort(space.states[:, 0])
    lp = np.zeros(len(order))
    for k in range(1, len(order)):
        lp[order[k]] = lp[order[k - 1]] + np.log(birth[order[k - 1]]) - np.log(death[order[k]])
    lp -= logsumexp(lp)
    cert = certify.window_certificate(gen, reference.Static(lp), np.array([0.0, 0.5]))
    assert max(cert.K.values()) < 1e-8  # K_T = 0 up to rounding: detailed balance makes the weighted generator symmetric
    B = certify.weighted_generator(gen.A, lp).toarray()
    assert np.allclose(B, B.T, atol=1e-9 * np.abs(B).max())


def test_conversion_cascade_is_a_product_law_and_catalytic_cascade_is_not():
    """With a Poisson start the conversion chain stays a product of Poisson laws; the catalytic cascade does not."""
    out = {}
    for name, m in (("conversion", models.cascade(2, width=24)), ("catalytic", models.catalytic_cascade(2, width=24))):
        space = m.space((24, 24))
        gen = network.generator(space, m.reactions((24, 24)))
        p = fsp.propagate(gen.A, reference.initial_law(m, space), [1.0])[-1]
        p = p / p.sum()
        X = space.states
        marg = [np.bincount(X[:, i], weights=p, minlength=24) for i in (0, 1)]
        out[name] = (1.0 / np.sum(p**2)) / np.prod([1.0 / np.sum(q**2) for q in marg])
    assert abs(out["conversion"] - 1.0) < 1e-3  # S_eff multiplies across species
    assert out["catalytic"] < 0.95  # correlated species: S_eff is below the product of the marginal values


def test_stationary_law_of_the_whole_space_certifies_a_non_reversible_circuit():
    """The burst-with-feedback circuit has no detailed balance. Its stationary law on a wider box, restricted to the
    instance box, has a residual of at most zero there, so the certificate gives K_T <= 0 (Section S4.5)."""
    m = next(x for x in models.published() if x.name == "feedback_b1")
    w, wide = (2, 40), (2, 80)
    space_w = m.space(wide)
    gen_w = network.generator(space_w, m.reactions(wide))
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    A = (gen_w.A + sp.diags(gen_w.leak)).tolil()
    n = A.shape[0]
    A[0, :] = np.ones(n)
    b = np.zeros(n)
    b[0] = 1.0
    pi = np.maximum(spla.spsolve(A.tocsc(), b), 1e-300)
    index = {tuple(x): i for i, x in enumerate(space_w.states)}
    space = m.space(w)
    gen = network.generator(space, m.reactions(w))
    lp = np.log(pi[[index[tuple(x)] for x in space.states]])
    r = certify.residual(gen.A, lp)
    # the box only loses the inflow from outside, so r <= 0 up to round-off where pi is tiny, and r - leak <= 0
    assert np.max(r - gen.leak) < 1e-8
    cert = certify.window_certificate(gen, reference.Static(lp), np.array([0.0, 1.0]))
    assert min(cert.K.values()) < 1e-8
