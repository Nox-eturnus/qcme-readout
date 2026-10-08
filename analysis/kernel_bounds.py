"""Rigorous interval bounds on the LCHS discretization of the primary kernel (Low-Somma, c = 2), and 50-digit checks
of the Jacobi-Anger truncation, for every headline cost row. Needs mpmath (pip install mpmath).

    python -m analysis.kernel_bounds

The bound adds three parts: the approximation error of the kernel (by a contour shift), the trapezoid-rule error
(Trefethen-Weideman, best strip on a 19-point grid in each half-plane), and the tail beyond the last node. Each part
is an outward-rounded interval evaluation, so the total is an upper bound.
"""
import math
import time

import mpmath as mp
import numpy as np

from .common import data, save
from .costs import COST_MODELS, ENCS, cost_settings, headline
from qcme import lchs

iv = mp.iv
PART = "costs"


def _G(k, y, c, gam):
    """|g(k + iy)| for the Low-Somma kernel, as an interval."""
    y, c, gam = iv.mpf(y), iv.mpf(c), iv.mpf(gam)
    num = iv.exp(c * (1 + y) - (k * k - y * y + 1) / (4 * gam * gam))
    den = iv.pi * iv.sqrt((1 + k * k - y * y) ** 2 + 4 * k * k * y * y)
    return num / den


def _dG(k, y0, c, gam):
    """Upper envelope of |g - g_exact|(k - i y0): |g_exact| (e^{Re w} + 1)."""
    y0, c, gam = iv.mpf(y0), iv.mpf(c), iv.mpf(gam)
    base = iv.exp(c * (1 - y0)) / (iv.pi * iv.sqrt((1 + k * k - y0 * y0) ** 2 + 4 * k * k * y0 * y0))
    return base * (iv.exp(-(k * k - y0 * y0 + 1) / (4 * gam * gam)) + 1)


def _up(x):
    return mp.mpf(x.b) if isinstance(x, type(iv.mpf(0))) else x


def _riemann_ab(fun, lo, hi, cells):
    """Rigorous upper bound of int_lo^hi fun by interval evaluation on cells."""
    edges = [mp.mpf(lo) + (mp.mpf(hi) - mp.mpf(lo)) * i / cells for i in range(cells + 1)]
    tot = iv.mpf(0)
    for a, b in zip(edges[:-1], edges[1:]):
        tot += fun(iv.mpf([a, b])) * iv.mpf([b - a, b - a])
    return _up(tot)


def _riemann(fun, L, cells):
    return _riemann_ab(fun, -L, L, cells)


def _gauss_tail(y, c, gam, L):
    """int_{|k| > L} |g(k + iy)| dk, using sqrt((1 + k^2 - y^2)^2 + 4 k^2 y^2) >= k^2 (1 - max(y^2 - 1, 0)/L^2)."""
    y = mp.mpf(y)
    shrink = 1 - max(y * y - 1, 0) / (L * L)
    assert shrink > 0
    pref = mp.exp(c * (1 + y) + (y * y - 1) / (4 * gam * gam)) / (mp.pi * L * L * shrink)
    return 2 * pref * gam * mp.sqrt(mp.pi) * mp.erfc(L / (2 * gam)) * (1 + mp.mpf("1e-20"))


def rigorous_K2(c, eps_lchs, Lnorm, h, N, cells=1500):
    mp.mp.dps = 40
    ker = lchs.LowSomma(c, eps_lchs)
    c_, gam = mp.mpf(c), mp.mpf(ker.gamma)
    L = max(40 * gam, 2 * (1 + 3))

    def integral(y):
        y = mp.mpf(y)
        LL = max(L, 2 * (1 + abs(y)))
        return _riemann(lambda k: _G(k, y, c_, gam), LL, cells) + _gauss_tail(y, c_, gam, LL)
    y0 = 2 * c_ * gam * gam
    LL = max(L, 4 * y0)
    approx = _riemann(lambda k: _dG(k, y0, c_, gam), LL, 4 * cells)
    approx += (2 * mp.exp(c_ * (1 - y0)) * 2 / (mp.pi * LL)
               + 2 * mp.exp(c_ * (1 - y0) + (y0 * y0 - 1) / (4 * gam * gam)) * 2 / (mp.pi * LL * LL)
               * gam * mp.sqrt(mp.pi) * mp.erfc(LL / (2 * gam)))
    hh = mp.mpf(h)
    grid = [mp.mpf(i) / 20 for i in range(1, 20)]
    lo = min(integral(-a) / (mp.exp(2 * mp.pi * a / hh) - 1) for a in grid)
    up = min(mp.exp(b * mp.mpf(Lnorm)) * integral(b) / (mp.exp(2 * mp.pi * b / hh) - 1) for b in grid)
    k0 = N * hh
    k1 = max(k0 + 40 * gam, 2 * mp.mpf(1))
    tail = 2 * (_riemann_ab(lambda k: _G(k, mp.mpf(0), c_, gam), k0, k1, cells) + _gauss_tail(0, c_, gam, k1) / 2)
    return {"approx": float(approx), "trap": float(lo + up), "tail": float(tail), "total": float(approx + lo + up + tail)}


def _kapteyn(n, tau):
    """|J_n(n z)| <= z^n exp(n sqrt(1 - z^2)) / (1 + sqrt(1 - z^2))^n, 0 < z <= 1 (DLMF 10.14.8)."""
    z = tau / n
    s = mp.sqrt(1 - z * z)
    return z**n * mp.exp(n * s) / (1 + s) ** n


def ja_tail_50(tau, q, extra=800):
    """2 sum_{l > q} |J_l(tau)| to 60 digits by Miller's backward recurrence, normalized by J_0 + 2 sum J_{2k} = 1,
    with the remainder bounded by Kapteyn's inequality."""
    mp.mp.dps = 60
    tau = mp.mpf(tau)
    L = int(max(q, float(tau))) + extra
    jp1, j = mp.mpf(0), mp.mpf("1e-300")
    vals = {L: j}
    for l in range(L, 0, -1):
        jm1 = (2 * l / tau) * j - jp1
        jp1, j = j, jm1
        vals[l - 1] = j
        if abs(j) > mp.mpf("1e300"):
            for k in vals:
                vals[k] /= j
            jp1 /= j
            j = mp.mpf(1)
    norm = vals[0] + 2 * mp.fsum(vals[k] for k in range(2, L + 1, 2))
    tail = 2 * mp.fsum(abs(vals[k] / norm) for k in range(q + 1, L + 1))
    z = tau / (L + 1)
    rem = 2 * _kapteyn(L + 1, tau) / (1 - z)
    return tail * (1 + rem) + rem


def main():
    rows = []
    for name in COST_MODELS:
        c = data(PART, name)
        for enc in ENCS:
            info = c["encodings"][enc]
            al = np.asarray(info["alphas"], float)
            dts = np.diff(np.asarray(info["edges"], float))
            Lnorm = float(np.sum(al * dts) + np.sum(np.asarray(info["Dmax"], float)))
            for obs in ("one", "tail"):
                r = headline(c, enc, obs, "W")
                if r is None:
                    continue
                delta = r["eta"] / 2
                d_lchs = delta / 3
                t0 = time.time()
                rb = rigorous_K2(2.0, d_lchs / 3, Lnorm, float(r["h"]), (int(r["nodes"]) - 1) // 2)
                K = float(r["K_cut"])
                dhs = (delta / 3) / (float(r["s"]) * len(al))
                ja_ok, worst = True, 0.0
                for a_, dt in zip(al, dts):
                    tau = a_ * math.sqrt(1 + K * K) * dt
                    tail = float(ja_tail_50(tau, lchs.jacobi_anger_degree(tau, dhs)))
                    worst = max(worst, tail / dhs)
                    ja_ok &= tail <= dhs
                rows.append({"model": name, "encoding": enc, "observable": obs, "d_lchs": d_lchs, **rb,
                             "lchs_ok": rb["total"] <= d_lchs, "numerical_bound": r["lchs_error"],
                             "ja_ok": bool(ja_ok), "ja_worst_tail_over_share": worst, "seconds": time.time() - t0})
                print(f"  {name:16s} {enc:13s} {obs:4s} rigorous/allocated {rb['total'] / d_lchs:.3f}", flush=True)
    save(PART, "kernel_rigorous", cost_settings(), {"rows": rows, "pass": all(r["lchs_ok"] and r["ja_ok"] for r in rows)})


if __name__ == "__main__":
    main()
