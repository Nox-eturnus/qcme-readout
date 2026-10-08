"""Numerical identities I1-I7, I10, I11 on every benchmark model.

Small boxes (I1 uses dense power traces up to |S|). J = 5 windows keeps the suite quick.
"""
import numpy as np
import pytest

from qcme import invariants as inv
from qcme import models

BOXES = {
    "birth_death": (30,), "telegraph": (2, 30), "bursty": (30,), "schlogl": (60,), "toggle": (12, 12),
    "michaelis_menten": None, "self_repressor": (2, 2, 30), "cascade2": (8, 8), "cascade3": (5, 5, 5),
}
TOL = inv.TOL


def _model(name):
    if name == "michaelis_menten":
        return models.michaelis_menten(E0=4, S0=10)
    return next(m for m in models.registry() if m.name == name)


@pytest.fixture(scope="module", params=list(BOXES))
def s(request):
    return inv.setup(_model(request.param), BOXES[request.param])


def test_I1_similarity_preserves_power_traces(s):
    assert inv.I1_power_traces(s) < TOL["I1"]


@pytest.mark.parametrize("ref_name", ["R-P", "R-NB", "R-mix(P)", "R-mix(NB)"])
def test_I2_I3_I4_piecewise_emulation(s, ref_name):
    pc = inv.piecewise_checks(s, ref_name, J=5, sub=3)
    assert pc.I2 < TOL["I2"]
    assert pc.I2_state < TOL["I2"]
    assert max(pc.I3_violation.values()) <= TOL["I3"]
    assert pc.I3_kappa_violation <= TOL["I3"]
    assert pc.I4 < TOL["I4"]
    assert max(pc.stabilized_max_lognorm.values()) <= 1e-12, "stabilized factors must be contractions"
    for f, e in pc.stabilized_recon.items():
        assert e is None or e < 1e-9, f


@pytest.mark.slow
def test_I2_continuous_reference(s):
    # ODE (DOP853, rtol 1e-12) accumulates ~1e-10 on the stiffer models; notes Section 2.1
    assert inv.continuous_I2(s, "R-mix(NB)") < 1e-8


def test_I5_exact_reference(s):
    e_psi, e_kappa = inv.I5_exact_reference(s)
    assert e_psi < TOL["I5"]
    assert e_kappa < TOL["I5"]


def test_I6_uniform_reference(s):
    assert inv.I6_uniform(s) < TOL["I6"]


@pytest.mark.parametrize("name", ["birth_death", "cascade2", "cascade3"])
def test_I7_exact_product_poisson(name):
    s = inv.setup(_model(name), BOXES[name])
    interior, max_r = inv.I7_exact_product_poisson(s)
    assert interior < 1e-10
    assert max_r < 1e-10  # boundary residuals are <= 0 (Proposition 4); rounding only


@pytest.mark.parametrize("name", ["michaelis_menten", "self_repressor"])
def test_I10_conservation(name):
    off, diff = inv.I10_conservation(_model(name), BOXES[name])
    assert off == 0.0, "states off the conserved manifold must stay exactly unpopulated"
    assert diff < 1e-12


@pytest.mark.slow
def test_I11_windows_toggle():
    s = inv.setup(models.toggle(), (16, 16))
    out = inv.I11_windows(s, "R-P", Js=(1, 2, 4, 8, 16, 32), n_cont=129)
    assert max(out["state_err"]) < 1e-8, "piecewise and continuous psi_T represent the same p_S"
    for f, Ks in out["K"].items():
        d = np.diff(Ks)
        assert np.all(d >= -1e-9) or np.all(d <= 1e-9), "K_T^(J) must converge monotonically"
        assert abs(Ks[-1] - out["K_pw_limit"][f]) < 0.05 * abs(out["K_pw_limit"][f])
        # the Riemann limit of the piecewise scheme can only be looser than the continuous estimate
        assert out["K_cont"][f] <= out["K_pw_limit"][f] + 1e-6 * max(1.0, abs(out["K_pw_limit"][f]))
