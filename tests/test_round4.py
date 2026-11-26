"""Checks of the fourth-revision calculations: the structure-only per-use counts, the signalling cascades and the
fitted boxes.

    .venv/Scripts/python -m pytest tests/test_round4.py
"""
import numpy as np
import pytest

from analysis import costs, readout
from analysis.common import build_setup
from qcme import fsp, models


@pytest.mark.parametrize("d", [1, 2])
def test_structure_counts_equal_pipeline(d):
    """The per-use counts made from the network structure equal those of the full pipeline."""
    a, b = costs.cascade_per_use(d), costs.pipeline_per_use(d)
    assert a["walk_steps"] == b["walk_steps"] and a["query"] == b["query"] and a["b"] == b["b"]
    assert a["toffoli_U"] == pytest.approx(b["toffoli_U"], rel=1e-9)
    assert a["ssa_per_A"] == b["ssa_per_A"] and a["Lambda"] == b["Lambda"]
    assert np.allclose(a["alphas"], b["alphas"], rtol=1e-10)


def test_signalling_cascade_is_feed_forward():
    """The first species of the two-species cascade has the law of the one-species cascade."""
    T = 1.0
    one = build_setup(models.signalling_cascade(1), (24,), T, refs=None)
    two = build_setup(models.signalling_cascade(2), (24, 90), T, refs=None)
    p1 = fsp.propagate(one.gen.A, one.p0, [T])[-1]
    p2 = fsp.propagate(two.gen.A, two.p0, [T])[-1]
    marg = np.bincount(two.space.states[:, 0], weights=p2, minlength=24)
    assert abs(p2.sum() - p1.sum()) < 1e-12  # only X1 leaks, and it leaks the same in both
    assert np.abs(marg - p1).max() < 1e-12


def test_fitted_boxes_keep_the_mass():
    """The cut box keeps all but CUT_TOL of the law, and restricting never raises S_eff above |S|."""
    s = build_setup(models.cascade(2, width=12), (12, 12), 1.0, refs=None)
    pT = fsp.propagate(s.gen.A, s.p0, [1.0])[-1]
    f = readout.fitted_factors(s, pT)
    assert f["cut"]["mass"] >= pT.sum() - readout.CUT_TOL - 1e-12
    assert f["cut"]["n_states"] <= len(s.space)
    assert f["cut"]["log10_readout_factor"] <= f["full_box"] + 1e-12
