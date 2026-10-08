"""R-H references and the rigorous continuous certificate.

    .venv/Scripts/python -m pytest tests/test_qcme_hermite.py
"""
import numpy as np
import pytest
import scipy.sparse as sp

from qcme import certify, hermite, models, network, reference

rng = np.random.default_rng(20261003)
CASES = {"telegraph": (2, 40), "self_repressor": (2, 2, 40), "bursty": (40,), "toggle": (14, 14)}


def _setup(name, T=1.0):
    m = next(mm for mm in models.registry() if mm.name == name)
    space = m.space(CASES[name])
    gen = network.generator(space, m.reactions(CASES[name]))
    return m, space, gen, hermite.hermite_references(m, space, T, CASES[name], n_nodes=17)


@pytest.mark.parametrize("name", list(CASES))
def test_hermite_log_derivative_matches_finite_differences(name):
    _, _, _, refs = _setup(name)
    h = 1e-6
    for rn, ref in refs.items():
        for t in (0.13, 0.52, 0.91):
            fd = (ref.log_mu(t + h) - ref.log_mu(t - h)) / (2 * h)
            assert np.abs(ref.dlog_mu(t) - fd).max() < 1e-5 * max(1.0, np.abs(fd).max()), (name, rn, t)


@pytest.mark.parametrize("name", list(CASES))
def test_hermite_matches_phase_c_reference_at_nodes(name):
    m, space, _, refs = _setup(name)
    old = reference.standard_references(m, space, 1.0, CASES[name])
    for rn, ref in refs.items():
        for t in ref.nodes[::4]:
            a, b = ref.log_mu(t), old[rn].log_mu(t)
            ok = b > np.log(1e-300)
            assert np.abs(a - b)[ok].max() < 1e-6, (name, rn, t)


@pytest.mark.parametrize("name", list(CASES))
def test_piece_enclosures_hold_at_every_sampled_time(name):
    _, _, gen, refs = _setup(name)
    A = sp.coo_matrix(gen.A)
    off = A.row != A.col
    er, ec = A.row[off], A.col[off]
    for rn, ref in refs.items():
        for _ in range(4):
            j = int(rng.integers(0, len(ref.nodes) - 1))
            t0, t1 = ref.nodes[j], ref.nodes[j + 1]
            a, b = np.sort(rng.uniform(t0, t1, 2))
            Dhi, dlog_lo, ell_lo, ell_hi = ref.piece(a, b, er, ec)
            for t in np.linspace(a, b, 7):
                lm, dl = ref.log_mu(t), ref.dlog_mu(t)
                assert np.all(lm[ec] - lm[er] <= Dhi), (name, rn)
                assert np.all(dl >= dlog_lo), (name, rn)
                assert np.all((lm >= ell_lo) & (lm <= ell_hi)), (name, rn)


def test_rigorous_certificate_dominates_pointwise_cw_and_converges():
    _, _, gen, refs = _setup("telegraph")
    ref = refs["R-mix(NB)"]
    cert = certify.rigorous_continuous_certificate(gen, ref, "sqrt_mu", k=2)
    for p in range(0, len(cert.bound), 5):
        a, b = cert.pieces[p], cert.pieces[p + 1]
        tm = 0.5 * (a + b)
        w = np.exp(0.5 * ref.log_mu(tm))
        for t in (a, tm, b):
            Sigma = certify.symmetric_part(certify.weighted_generator(gen.A, ref.log_mu(t), ref.dlog_mu(t)))
            assert certify.cw(Sigma, w) <= cert.bound[p] + 1e-9
    K2 = cert.K
    K8 = certify.rigorous_continuous_certificate(gen, ref, "sqrt_mu", k=8).K
    trap = certify.continuous_certificate(gen, ref, np.linspace(0, 1, 513), families=("sqrt_mu",)).K["sqrt_mu"]
    assert K2 >= K8 - 1e-9 >= trap - 0.5
    assert cert.K_at(1.0) == pytest.approx(cert.K)


def test_rigorous_certificate_bounds_the_emulated_state():
    from qcme import fsp

    m, space, gen, refs = _setup("toggle")
    p0 = reference.initial_law(m, space)
    for rn in ("R-P", "R-mix(NB)"):
        ref = refs[rn]
        ts = ref.nodes
        P = fsp.propagate(gen.A, p0, ts)
        n0 = np.linalg.norm(p0 * np.exp(-0.5 * ref.log_mu(0.0)))
        for f in certify.FAMILIES:
            cert = certify.rigorous_continuous_certificate(gen, ref, f, k=1)
            for p, t in zip(P, ts):
                meas = np.linalg.norm(p * np.exp(-0.5 * ref.log_mu(t))) / n0
                assert np.log(meas) <= 0.5 * cert.K_at(t) + 1e-12, (rn, f, t)
