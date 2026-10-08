"""Large instances: the log-domain initial law, the forward closure, the exact treatment of the states
not reachable from the representable support, the power iteration for W, and log-domain uniformization."""
import types
import pathlib

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import expm

from analysis import common, published_models
from qcme import certify, models, reference

ROOT = pathlib.Path(__file__).resolve().parents[1]




@pytest.fixture(scope="module")
def CB():
    return published_models


@pytest.fixture(scope="module")
def F():
    return types.SimpleNamespace(**{**vars(common), **vars(published_models)})


def test_log_p0_matches_initial_law_where_representable(CB):
    m = models.feedback_kuntz(10.0)
    sp_ = m.space((2, 157))
    lp = CB.log_p0(m, sp_)
    p0 = reference.initial_law(m, sp_)
    ok = p0 > 1e-300
    assert np.allclose(lp[ok], np.log(p0[ok]), rtol=0, atol=1e-10)
    assert np.all(np.isfinite(lp)) and (~ok).sum() > 0  # finite exactly where double precision gives 0


def test_forward_closure_on_a_chain(CB):
    # 0 -> 1 -> 2, and 3 -> 2 only: from {0}, the closure is {0, 1, 2}; state 3 is never entered
    A = sp.csr_matrix(np.array([[-1, 0, 0, 0], [1, -1, 0, 0], [0, 1, 0, 1], [0, 0, 0, -1]], float))
    r = CB.forward_closure(A, np.array([True, False, False, False]))
    assert r.tolist() == [True, True, True, False]


def test_exact_U_treatment_reproduces_plain_fsp(CB, F, monkeypatch):
    """Force an artificial representability floor on a small toggle_tang box, so that some states fall in U while
    plain FSP is still accurate there: the U treatment must reproduce log p to 1e-9 (change 1)."""
    m = models.toggle_tang()
    s = F.build_setup(m, (2, 2, 10, 10), 1.0, refs=None)
    monkeypatch.setattr(CB, "REPR_FLOOR", -30.0)
    times = [0.0, 0.25, 1.0]
    ex = CB.ExactLog(s, times)
    assert ex.acc["n_U"] > 0
    from qcme import fsp
    P = fsp.propagate(s.gen.A, reference.initial_law(m, s.space), times)
    for i, t in enumerate(times):
        lm = ex.log_mu(t)
        assert np.allclose(lm, np.log(P[i]), rtol=0, atol=1e-9)


def test_exact_log_matches_plain_exact_without_underflow(CB, F):
    m = models.autoreg_tang(2)
    s = F.build_setup(m, (2, 2, 60), 0.5, refs=None)
    edges = certify.uniform_edges(0.5, 5)
    ex = CB.ExactLog(s, edges)
    P = F._TimeCache(s.gen.A, s.p0, edges)
    old = reference.Exact(P, lambda t: s.gen.A @ P(t))
    c_new = certify.window_certificate(s.gen, ex, edges)
    c_old = certify.window_certificate(s.gen, old, edges)
    for f in c_old.K:
        assert c_new.K[f] == pytest.approx(c_old.K[f], rel=1e-9)


def test_scaled_power_W_matches_dense_svd(CB, F):
    m = models.toggle_hegland()
    s = F.build_setup(m, (14, 14), 3.0)
    ref = s.refs["R-mix(NB,E1)"]
    lm0, lmT = ref.log_mu(0.0), ref.log_mu(3.0)
    M = expm(s.gen.A.toarray() * 3.0) * np.exp(0.5 * lm0)[None, :] * np.exp(-0.5 * lmT)[:, None]
    want = np.log(np.linalg.norm(M, 2))
    ck = ROOT / "tests" / "_w_test.npz"
    ck.unlink(missing_ok=True)
    try:
        got = CB.scaled_power_W(s.gen.A, lm0, lmT, 3.0, ck, timebox=600)
    finally:
        ck.unlink(missing_ok=True)
    assert got["converged"]
    assert got["log_W"] == pytest.approx(want, abs=1e-5)


def test_log_cw_lower_matches_cw_sqrt_without_overflow(CB, F):
    m = models.feedback_kuntz(1.0)
    s = F.build_setup(m, (2, 40), 0.5, refs=None)
    lp0 = CB.log_p0(m, s.space)
    with np.errstate(over="ignore"):
        want = certify.cw_sqrt_mu(certify.residual(s.gen.A, lp0), s.gen.leak)
    assert np.isfinite(want) and want > 0
    assert np.exp(CB.log_cw_sqrt_lower(s.gen.A, s.gen.leak, lp0)) == pytest.approx(want, rel=1e-9)


def test_log_matvec_matches_dense(CB):
    rng = np.random.default_rng(0)
    M = sp.random(30, 30, density=0.2, random_state=1, format="csr")
    M.data = np.abs(M.data)
    x = rng.random(30) + 1e-3
    got = CB.log_matvec(M, np.log(M.data), np.log(x))
    want = M @ x
    ok = want > 0
    assert np.allclose(got[ok], np.log(want[ok]), atol=1e-12) and np.all(np.isneginf(got[~ok]))


def test_log_uniformization_matches_fsp_and_stays_finite(CB, F):
    from qcme import fsp
    m = models.feedback_kuntz(1.0)
    s = F.build_setup(m, (2, 40), 0.3, refs=None)
    times = [0.0, 0.01, 0.1, 0.3]
    lp = CB.log_uniformization(s.gen.A, CB.log_p0(m, s.space), times)
    P = fsp.propagate(s.gen.A, s.p0, times)
    for i, t in enumerate(times[1:], 1):
        ok = P[i] >= 1e-250
        assert np.allclose(lp[t][ok], np.log(P[i][ok]), atol=1e-9)
    assert np.all(np.isfinite(lp[0.3]))
