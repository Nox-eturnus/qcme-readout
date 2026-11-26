"""Checks added in the fifth revision: the looseness split adds up, the amplitude-estimation cost of the swap-test
overlaps, and the cascade readout factor at several horizons against the exact product law."""
import math

import numpy as np
import pytest

from analysis import check_numbers as cn
from analysis.readout import ae_queries
from qcme import fsp, models, network, reference


def test_split_identity_on_stored_results():
    """Table S11: K / (2 ln 10) = window + non-normality + slack + log10 of the measured growth, on every row."""
    assert cn.s11_identity_gap() <= 1e-9


def test_ae_queries_formula():
    F1, F2 = 0.284, 0.00703
    expected = math.sqrt(2) * math.pi / 0.01 * (1 / math.sqrt(F1) + 1 / math.sqrt(F2))
    assert ae_queries(F1, F2) == pytest.approx(expected, rel=1e-12)
    # the cost of amplitude estimation scales as the inverse square root of the overlap, not as its inverse square
    assert ae_queries(F1, F2 / 4) / ae_queries(F1, F2) == pytest.approx(
        (1 / math.sqrt(F1) + 2 / math.sqrt(F2)) / (1 / math.sqrt(F1) + 1 / math.sqrt(F2)), rel=1e-12)


@pytest.mark.parametrize("T", [1.0, 3.0, 10.0])
def test_conversion_cascade_factor_matches_product_law(T):
    """The conversion cascade from zero copies has a product of Poisson laws with means 4 P(Gamma(i, 1) <= T), so
    the factor sqrt(|S| / S_eff) on a box is the product of one-species factors."""
    d, w = 2, 20
    m = models.cascade(d, width=w)
    space = m.space((w,) * d)
    gen = network.generator(space, m.reactions((w,) * d))
    pT = fsp.propagate(gen.A, reference.initial_law(m, space), [T])[-1]
    fsp_factor = 0.5 * math.log10(len(space) * np.sum(pT**2) / pT.sum() ** 2)
    exact = 0.0
    for i in range(1, d + 1):
        mean = 4.0 * (1 - math.exp(-T) * sum(T**j / math.factorial(j) for j in range(i)))
        q = np.array([math.exp(-mean) * mean**k / math.factorial(k) for k in range(w)])
        exact += 0.5 * math.log10(w * np.sum(q**2) / q.sum() ** 2)
    assert fsp_factor == pytest.approx(exact, abs=1e-3)
