"""Checks added in the sixth revision: the closed-form cut box of the conversion cascade against the cut box that
fitted_factors takes from a finite state projection solve, and the stored gates of the cut-box stage."""
import numpy as np
import pytest

from analysis.common import build_setup, data
from analysis.readout import conversion_means, fitted_factors, product_cut
from qcme import fsp, models


@pytest.mark.parametrize("d,T,w", [(2, 1.0, 12), (3, 10.0, 24)])
def test_product_cut_matches_fitted_factors(d, T, w):
    """On a product law the closed-form full-box and cut-box factors equal those of the solved law on the same box."""
    s = build_setup(models.cascade(d, width=w), (w,) * d, T, refs=None)
    pT = fsp.propagate(s.gen.A, s.p0, [T])[-1]
    f = fitted_factors(s, pT)
    full, cut, _ = product_cut(conversion_means(d, T), (w,) * d)
    assert full == pytest.approx(f["full_box"], abs=1e-4)
    assert cut == pytest.approx(f["cut"]["log10_readout_factor"], abs=1e-4)


def test_conversion_means_match_the_solved_law():
    d, T, w = 3, 3.0, 20
    s = build_setup(models.cascade(d, width=w), (w,) * d, T, refs=None)
    pT = fsp.propagate(s.gen.A, s.p0, [T])[-1]
    X = s.space.states
    means = [float(X[:, i] @ (pT / pT.sum())) for i in range(d)]
    assert np.allclose(conversion_means(d, T), means, atol=1e-4)


def test_cut_horizons_stored_gates():
    out = data("readout", "cut_horizons")
    assert out["gates"]["passed"] and out["gates"]["a_max_gap"] <= 0.01 and out["gates"]["b_max_gap"] <= 0.01
    assert out["claim_test"]["persists_at_T10"]
