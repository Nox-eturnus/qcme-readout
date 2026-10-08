"""Window schedules and the extended references: R-LNA, envelope E2, R-TT."""
import numpy as np
import pytest

from qcme import certify, fsp, invariants as inv, models, network, reference
from qcme.lowrank import LowRankCME, LowRankCore


def small(model, widths):
    space = model.space(widths)
    gen = network.generator(space, model.reactions(widths))
    return space, gen


# ----------------------------------------------------------------------------- schedules
def test_geometric_edges():
    T = 3.0
    e = certify.geometric_edges(T)
    w = np.diff(e)
    assert len(e) == 21 and e[0] == 0.0 and e[-1] == T
    assert np.isclose(w[0], T / 2**10, rtol=1e-12)
    assert np.allclose(w[1:] / w[:-1], w[1] / w[0], rtol=1e-10)
    assert np.all(w > 0)


def test_certificate_is_inf_not_an_error_on_overflow():
    m = models.cascade(2)
    s = inv.setup(m, (8, 8), 1.0, families="extended")
    cert = certify.window_certificate(s.gen, s.refs["R-LNA"], certify.uniform_edges(1.0, 5))
    assert cert.K["resolvent"] == np.inf and cert.K["sqrt_mu"] == np.inf


# ----------------------------------------------------------------------------- R-LNA
@pytest.mark.parametrize("name,widths", [("toggle", (12, 12)), ("michaelis_menten", None), ("self_repressor", (2, 2, 20))])
def test_lna_normalized_and_derivative(name, widths):
    m = models.michaelis_menten(E0=4, S0=10) if name == "michaelis_menten" else next(
        mm for mm in models.registry() if mm.name == name)
    T = models.relaxation_time(m)[0]
    space, _ = small(m, widths or m.widths)
    lna = reference.LNA(space, reference.MomentTrajectory(m, T, widths))
    for t in (0.2 * T, 0.7 * T):
        lm = lna.log_mu(t)
        assert np.all(np.isfinite(lm))
        assert np.isclose(np.exp(lm).sum(), 1.0, rtol=1e-12)
        h = 1e-6
        fd = (lna.log_mu(t + h) - lna.log_mu(t - h)) / (2 * h)
        assert np.max(np.abs(fd - lna.dlog_mu(t)) / (1 + np.abs(fd))) < 1e-6


def test_lna_peak_follows_the_rate_equation_mean():
    m = models.toggle()
    space, _ = small(m, (30, 30))
    mom = reference.MomentTrajectory(m, 3.0)
    lna = reference.LNA(space, mom)
    x_peak = space.states[np.argmax(lna.log_mu(3.0))]
    assert np.all(np.abs(x_peak - mom.at(3.0)[0]) <= 1.0)


# ----------------------------------------------------------------------------- E2
def test_envelope_e2_means():
    m = models.self_repressor()
    T = models.relaxation_time(m)[0]
    space, _ = small(m, (2, 2, 30))
    mom = reference.MomentTrajectory(m, T)
    env = reference.envelope_e2(m, space, mom, T)
    grid_max = np.max([mom.at(t)[0] for t in np.linspace(0, T, 1025)], axis=0)
    want = np.where(m.binary, np.clip(grid_max, 1e-3, 1 - 1e-3), np.maximum(grid_max, 1e-3))
    assert np.allclose(env.moments.means, want)
    assert env.moments.means[0] == 1 - 1e-3  # G starts at 1: clipped


# ----------------------------------------------------------------------------- R-TT
def test_lowrank_full_rank_is_the_truncated_solution():
    m = models.toggle()
    space, gen = small(m, (8, 8))
    p0 = reference.initial_law(m, space)
    lr = LowRankCME(gen.A, (8, 8), p0, 8, 1.0)
    P = fsp.propagate(gen.A, p0, [0.3, 1.0])
    assert np.abs(lr.Y(0.3) - P[0]).max() < 1e-12
    assert np.abs(lr.Y(1.0) - P[1]).max() < 1e-12


def test_lowrank_rank2_tracks_product_poisson_on_the_cascade():
    m = models.cascade(2)
    space, gen = small(m, (15, 15))
    p0 = reference.initial_law(m, space)
    core = LowRankCore(gen, space, p0, 2, 1.0)
    rp = reference.standard_references(m, space, 1.0)["R-P"]
    tv = 0.5 * np.abs(np.exp(core.log_mu(1.0)) - np.exp(rp.log_mu(1.0))).sum()
    assert tv < 1e-3


def test_lowrank_query_order_does_not_matter():
    m = models.toggle()
    space, gen = small(m, (10, 10))
    p0 = reference.initial_law(m, space)
    a = LowRankCME(gen.A, (10, 10), p0, 4, 2.0)
    b = LowRankCME(gen.A, (10, 10), p0, 4, 2.0)
    ya = a.Y(1.2345)
    b.Y(0.5), b.Y(1.9)
    assert np.array_equal(ya, b.Y(1.2345))


def test_rtt_reference_is_strictly_positive_and_valid():
    m = models.toggle()
    s = inv.setup(m, (12, 12), 3.0, families="extended")
    for r in (2, 4, 8):
        ref = s.refs[f"R-TT(r={r})"]
        for t in (0.0, 1.1, 3.0):
            assert np.all(np.isfinite(ref.log_mu(t))) and np.all(np.isfinite(ref.dlog_mu(t)))
    pc = inv.piecewise_checks(s, "R-TT(r=2)", edges=certify.geometric_edges(3.0), sub=2)
    assert pc.I2 < 1e-10 and max(pc.I3_violation.values()) <= 1e-12 and pc.I4 < 1e-10


def test_extended_family_set():
    s = inv.setup(models.toggle(), (12, 12), 3.0, families="extended")
    assert set(s.refs) == {"R-P", "R-NB", "R-LNA", "R-mix(P,E1)", "R-mix(P,E2)", "R-mix(NB,E1)", "R-mix(NB,E2)",
                           "R-mix(LNA,E1)", "R-mix(LNA,E2)", "R-TT(r=2)", "R-TT(r=4)", "R-TT(r=8)"}
    s3 = inv.setup(models.cascade(3), (5, 5, 5), 1.0, families="extended")
    assert not any(k.startswith("R-TT") for k in s3.refs)  # two-species models only


# ----------------------------------------------------------------------------- resolvent bisection
def test_resolvent_wide_bracket_converges_to_a_valid_bound():
    """A bracket of ~100 decades (as on near-point-mass references) converges well inside
    max_iter, and the returned CW is a valid upper bound on the log-norm, within rtol of it."""
    import scipy.sparse as sp
    rng = np.random.default_rng(0)
    n = 40
    M = sp.random(n, n, density=0.2, random_state=1, data_rvs=rng.random).toarray()
    S = 0.5 * (M + M.T)
    np.fill_diagonal(S, -rng.random(n) * 5)
    lam = np.linalg.eigvalsh(S)[-1]
    res = certify.resolvent_cw(sp.csr_matrix(S), 1e100)
    assert res.iterations < 60
    assert lam - 1e-9 <= res.value <= lam + 1e-5 * max(1.0, abs(lam))


@pytest.mark.slow
def test_toggle_best_value_reproduces():
    """The best combination on the toggle switch (R-mix(NB,E1), resolvent, uniform J = 80) is unchanged."""
    import json
    import pathlib
    d = json.loads((pathlib.Path(__file__).resolve().parents[1] / "results/certificates/toggle_references.json").read_text())["data"]
    s = inv.setup(models.toggle(), tuple(d["widths"]), d["T"], families="extended")
    c = certify.window_certificate(s.gen, s.refs["R-mix(NB,E1)"], certify.uniform_edges(s.T, 80))
    assert c.K["resolvent"] == pytest.approx(36.8059248668344, rel=1e-12)


def test_resolvent_cg_matches_lu_and_is_valid():
    """The CG path (used above certify.LU_MAX_STATES) agrees with LU and stays a valid bound."""
    m = models.toggle()
    s = inv.setup(m, (25, 25), 3.0, families="core")
    lm = s.refs["R-mix(NB)"].log_mu(1.0)
    Sig = certify.symmetric_part(certify.weighted_generator(s.gen.A, lm))
    up = certify.cw_sqrt_mu(certify.residual(s.gen.A, lm), s.gen.leak)
    a = certify.resolvent_cw(Sig, up, solver="lu")
    b = certify.resolvent_cw(Sig, up, solver="cg")
    lam = np.linalg.eigvalsh(Sig.toarray())[-1]
    assert abs(a.value - b.value) <= 1e-9 * max(1.0, abs(a.value))
    assert b.value >= lam - 1e-9
