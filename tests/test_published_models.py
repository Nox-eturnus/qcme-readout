"""Published models: transcription of every instance, the E1 envelope rule, the horizon rule, and the
single-pass exact reference."""
import pathlib
import types

import numpy as np
import pytest

from analysis import common, published_models
from qcme import certify, fsp, models, network, reference

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _runner():
    """The analysis helpers (analysis.common and analysis.published_models) under one name."""
    return types.SimpleNamespace(**{**vars(common), **vars(published_models)})


def _props(m, x):
    X = np.asarray([x], float)
    return {r.name: float(np.asarray(r.prop(X))[0]) for r in m.reactions()}


# ----------------------------------------------------------------------------- transcription (hand values)
def test_toggle_hegland_transcription():
    a = _props(models.toggle_hegland(), (3, 7))  # b = 0.4, c = 0.05 (R12 eq. 21)
    assert a["S1 -> 0"] == pytest.approx(0.15)
    assert a["S2 -> 0"] == pytest.approx(0.35)
    assert a["0 -> S1"] == pytest.approx(0.4 / 7.4)
    assert a["0 -> S2"] == pytest.approx(0.4 / 3.4)


def test_toggle_kuntz_transcription():
    a = _props(models.toggle_kuntz(), (4, 2))  # theta = 1, k1 = 30, k2 = k4 = 1, k3 = 10 (R21 eq. 113)
    assert a["0 -> P1"] == pytest.approx(30 / 9)
    assert a["P1 -> 0"] == pytest.approx(4)
    assert a["0 -> P2"] == pytest.approx(10 / 5)
    assert a["P2 -> 0"] == pytest.approx(2)


def test_toggle_gardner_transcription():
    a = _props(models.toggle_gardner(), (9, 4))  # R11 Sec. 4.3.2: 500/(1 + x2^1.5), 200/(1 + x1)
    assert a["0 -> X1"] == pytest.approx(500 / 9)
    assert a["X1 -> 0"] == pytest.approx(2.7)
    assert a["0 -> X2"] == pytest.approx(20)
    assert a["X2 -> 0"] == pytest.approx(1.6)


def test_feedback_kuntz_transcription():
    m = models.feedback_kuntz(10.0)  # k1 = k2 = k5 = 1, k3 = k4 = 10, geometric p(k) = q^k p0, p0 = 1/11
    a = _props(m, (1, 3))
    q, p0 = 10 / 11, 1 / 11
    assert a["Goff -> Gon"] == pytest.approx(0.0)
    assert a["Gon -> Goff"] == pytest.approx(1.0)
    assert a["Gon + P -> Goff + P"] == pytest.approx(30.0)
    assert a["burst 1"] == pytest.approx(10 * p0 * q)
    assert a["burst 5"] == pytest.approx(10 * p0 * q**5)
    assert a["P -> 0"] == pytest.approx(3.0)
    # every non-empty burst is represented: total rate 10 (1 - p0) when on
    N = m.widths[1]
    tot = sum(v for k, v in a.items() if k.startswith("burst"))
    assert tot == pytest.approx(10 * (1 - p0))
    assert a[f"burst >= {N} (leaves S)"] == pytest.approx(10 * q**N)
    assert _props(m, (0, 3))["Goff -> Gon"] == pytest.approx(1.0)


def test_lambda_phage_transcription():
    a = _props(models.lambda_phage(), (2, 3, 1, 4, 5))  # R12 eq. 22 and the table p. 2298
    assert a["0 -> S1"] == pytest.approx(0.5 * 0.12 / 3.12)
    assert a["0 -> S2"] == pytest.approx((1 + 5) * 0.6 / 2.6)
    assert a["0 -> S3"] == pytest.approx(0.15 * 3 / 4)
    assert a["0 -> S4"] == pytest.approx(0.3 * 1 / 2)
    assert a["0 -> S5"] == pytest.approx(0.3 * 1 / 2)
    for i, (c, x) in enumerate(zip((0.0025, 0.0007, 0.0231, 0.01, 0.01), (2, 3, 1, 4, 5))):
        assert a[f"S{i + 1} -> 0"] == pytest.approx(c * x)


def test_toggle_tang_transcription():
    # Tang et al. SI eq. (S2.9): sx = sy = 50, dx = dy = 1, bx = by = 1e-4, ux = uy = 0.1; propensity c prod n^r
    m = models.toggle_tang()
    a = _props(m, (1, 0, 3, 5))  # Gx unbound, Gy bound
    assert a["Gx -> Gx + Px"] == pytest.approx(50.0)
    assert a["Gy -> Gy + Py"] == pytest.approx(0.0)
    assert a["Px -> 0"] == pytest.approx(3.0)
    assert a["Py -> 0"] == pytest.approx(5.0)
    assert a["2Px + Gy -> Gy_bar"] == pytest.approx(0.0)
    assert a["2Py + Gx -> Gx_bar"] == pytest.approx(1e-4 * 25)
    assert a["Gy_bar -> 2Px + Gy"] == pytest.approx(0.1)
    assert a["Gx_bar -> 2Py + Gx"] == pytest.approx(0.0)
    b = _props(m, (0, 1, 1, 4))  # Px = 1: the source's reflecting boundary blocks 2Px + Gy
    assert b["2Px + Gy -> Gy_bar"] == 0.0 and b["Gx_bar -> 2Py + Gx"] == pytest.approx(0.1)
    assert _props(m, (1, 1, 2, 0))["2Px + Gy -> Gy_bar"] == pytest.approx(1e-4 * 4)
    rx = {r.name: r for r in m.reactions()}
    assert rx["2Px + Gy -> Gy_bar"].deterministic_rate((1.0, 0.5, 1.5, 0.0)) == pytest.approx(1e-4 * 2.25 * 0.5)


def test_autoreg_tang_transcription():
    # Tang et al. SI eq. (S2.6), Supplementary Table I case 2: sigma_u = 0.69, sigma_b = 0.07, rho_u = 7.2, rho_b = 40.6
    m = models.autoreg_tang(2)
    a = _props(m, (1, 0, 4))
    assert a["Gu + P -> Gb"] == pytest.approx(0.28)
    assert a["Gb -> Gu + P"] == pytest.approx(0.0)
    assert a["Gu -> Gu + P"] == pytest.approx(7.2)
    assert a["Gb -> Gb + P"] == pytest.approx(0.0)
    assert a["P -> 0"] == pytest.approx(4.0)
    b = _props(m, (0, 1, 4))
    assert b["Gb -> Gu + P"] == pytest.approx(0.69) and b["Gb -> Gb + P"] == pytest.approx(40.6)
    assert models.TANG_AUTOREG_CASES == {1: (0.94, 0.01, 8.40, 28.1), 2: (0.69, 0.07, 7.20, 40.6),
                                         3: (0.44, 0.08, 0.94, 53.1)}


def test_published_set():
    names = [m.name for m in models.published()]
    assert names == ["schlogl", "toggle_hegland", "toggle_kuntz", "toggle_gardner", "feedback_b1",
                     "feedback_b10", "feedback_b100", "lambda_phage",
                     "toggle_tang", "autoreg_tang_1", "autoreg_tang_2", "autoreg_tang_3"]
    assert all(m.provenance for m in models.published())


# ----------------------------------------------------------------------------- E1 rule and horizon
def test_maxprod_envelope_matches_closed_forms():
    for m, want in ((models.toggle_hegland(), (20.0, 20.0)), (models.toggle_kuntz(), (30.0, 10.0)),
                    (models.toggle_gardner(), (500 / 0.3, 500.0)),
                    (models.toggle_tang(), (0.5, 0.5, 50.2, 50.2)),  # sx + 2 u (release of two proteins)
                    (models.autoreg_tang(3), (0.5, 0.5, 0.44 + 53.1))):  # Gb: sigma_u + rho_b
        means = reference.maxprod_means(m, m.space(), m.envelope["removal"])
        assert np.allclose(means, want)


def test_horizon_rule_and_fallback():
    run = _runner()
    T, info = run.horizon(models.lambda_phage())
    assert T == models.LAMBDA_PHAGE_T and info["rule"].startswith("source") and info["n_estimate"] > run.CEILING
    T2, info2 = run.horizon(models.toggle_hegland())
    assert info2["rule"] == "one relaxation time" and T2 == pytest.approx(models.relaxation_time(models.toggle_hegland())[0])


def test_single_pass_oracle_matches_window_certificate():
    """The single-pass exact reference (one FSP pass) equals certify.window_certificate on R-exact, sqrt(mu) family."""
    from qcme import invariants as inv
    run = _runner()
    m = models.toggle_kuntz()
    s = run.build_setup(m, (20, 12), 1.0, refs=None)
    sched = run.schedules(1.0)
    ts = np.linspace(0, 1.0, 17)
    rows, limit, best, _ = run.oracle_single_pass(s, sched, ts)
    P = run._TimeCache(s.gen.A, s.p0, np.concatenate([ts] + list(sched.values())))
    ref = reference.Exact(P, lambda t: s.gen.A @ P(t))
    for sn, ed in sched.items():
        c = certify.window_certificate(s.gen, ref, ed, families=("sqrt_mu",))
        assert rows[sn]["K"]["sqrt_mu"] == pytest.approx(c.K["sqrt_mu"], rel=1e-12, abs=1e-12)
    static = certify.continuous_certificate(s.gen, inv.reference_static_view(ref), ts, families=("sqrt_mu",))
    jump = np.array([np.max(-ref.dlog_mu(t)) for t in ts])
    assert limit["sqrt_mu"] == pytest.approx(static.K["sqrt_mu"] + np.trapezoid(jump, ts), rel=1e-12)


def test_single_pass_oracle_infinite_where_p_underflows():
    """Near-zero start (bursty feedback, <b> = 10, box (2, 157)): p_0 underflows in the tail, window_certificate gives
    +inf, and the single-pass oracle must give +inf too (not NaN)."""
    run = _runner()
    m = models.feedback_kuntz(10.0)
    s = run.build_setup(m, (2, 157), 0.5, refs=None)
    sched = {k: v for k, v in run.schedules(0.5).items() if k in ("uniform J=5", "geometric J=20")}
    ts = np.linspace(0, 0.5, 9)
    rows, limit, best, _ = run.oracle_single_pass(s, sched, ts)
    P = run._TimeCache(s.gen.A, s.p0, np.concatenate([ts] + list(sched.values())))
    ref = reference.Exact(P, lambda t: s.gen.A @ P(t))
    for sn, ed in sched.items():
        with np.errstate(all="ignore"):
            c = certify.window_certificate(s.gen, ref, ed, families=("sqrt_mu",))
        assert np.isinf(c.K["sqrt_mu"]) and np.isinf(rows[sn]["K"]["sqrt_mu"])
    assert np.isinf(limit["sqrt_mu"])


def test_standard_oracle_nan_jump_maps_to_inf():
    """A reference that is exactly 0 at the same state on two consecutive edges makes
    the jump (-inf) - (-inf) = NaN; window_certificate returns NaN, and inf_if_nan gives +inf
    (the CW of those windows is +inf already)."""
    run = _runner()
    m = models.birth_death()
    s = run.build_setup(m, (20,), 1.0, refs=None)

    class ZeroAtTail:
        def log_mu(self, t):
            lm = np.log(np.full(20, 1 / 20))
            lm[-1] = -np.inf
            return lm

    with np.errstate(all="ignore"):
        c = certify.window_certificate(s.gen, ZeroAtTail(), certify.uniform_edges(1.0, 3), families=("sqrt_mu",))
    assert np.isnan(c.K["sqrt_mu"])
    tab = run.inf_if_nan({"uniform J=3": c})
    assert np.isinf(tab["uniform J=3"].K["sqrt_mu"]) and tab["uniform J=3"].K["sqrt_mu"] > 0
