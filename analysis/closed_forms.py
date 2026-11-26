"""Published closed forms of the stationary laws of the six one-gene instances, against the numerical laws.

    python -m analysis.closed_forms compute     the closed forms in high precision (needs mpmath; run it where the
                                                Qualtran cross-check runs, see README.md)
    python -m analysis.closed_forms check       total variation from the numerical stationary law and K_T with the
                                                closed form as the reference

Autoregulation (Tang et al.): the feedback loop of Grima, Schmidt and Newman, J. Chem. Phys. 137, 035104 (2012),
arXiv:1206.1571, Eqs. (21)-(22) and (26)-(29), with no decay of bound protein (theta = 0) and unit decay of free
protein. Gene state 0 is unbound (Gu), 1 is bound (Gb).

Burst-with-feedback (Kuntz et al.): the model of Kumar, Platini and Kulkarni, Phys. Rev. Lett. 113, 268105 (2014),
arXiv:1409.3499, with their state 0 the producing state (Kuntz's gene on, switched off at k2 + k3 n), their state 1
the silent state (switched on at k1), k_0 = k4, k_1 = 0, mu = k5, alpha = k2, alpha~ = k3, beta = k1 and b the mean
burst. Their generating function of the protein count is G(z) = 2F1(u, v; c; x(z)) / 2F1(u, v; c; 1 - phi) with
x(z) = 1 - phi (1 + b (1 - z)), so P(n) = (phi b)^n / n! (u)_n (v)_n / (c)_n 2F1(u + n, v + n; c + n; x(0)) / norm,
a closed form term by term. The producing state follows from the steady-state equation of G (k_1 = 0):
G_0(z) = mu (1 + b (1 - z)) G'(z) / (k_0 b), so P_0(n) = mu ((1 + b)(n + 1) P(n + 1) - b n P(n)) / (k_0 b).
"""
import sys
import time

import numpy as np

from .common import data, log10, log_norm_weighted, save
from qcme import certify, fsp, models, reference

PART = "published_models"
DPS = 60
GRIMA_TAIL = 1e-40  # the autoregulation law is normalized by summing until the terms fall below this


def _grima_logpi(su, sb, ru, rb, n_max):
    """log P(gene, n), gene in {0 (Gu), 1 (Gb)}, n = 0..n_max - 1, Grima et al. with theta = 0."""
    import mpmath as mp
    mp.mp.dps = DPS
    su, sb, ru, rb = (mp.mpf(str(x)) for x in (su, sb, ru, rb))
    Sb = 1 + sb
    R = ru - rb * Sb
    alpha = su * (ru - rb) / R
    beta = 1 + (su + ru - ru / Sb) / Sb
    w0 = -R / Sb**2
    Mc = {}

    def M(a_off, b_off):
        if (a_off, b_off) not in Mc:
            Mc[(a_off, b_off)] = mp.hyp1f1(alpha + a_off, beta + b_off, w0)
        return Mc[(a_off, b_off)]

    def coef(m):  # (R / Sb)^m (alpha)_m / (beta)_m
        return (R / Sb) ** m * mp.rf(alpha, m) / mp.rf(beta, m)

    def p1(n):
        return sum(mp.binomial(n, m) * rb ** (n - m) * coef(m) * M(m, m) for m in range(n + 1)) / mp.factorial(n)

    def p0(n):
        if n == 0:
            return Sb / (Sb - 1) * alpha / ru * M(1, 0) - su / R * M(0, 0)
        s = 0
        for m in range(n):
            s += mp.binomial(n - 1, m) * rb ** (n - 1 - m) * coef(m) * (
                Sb / (Sb - 1) * (m + alpha) * M(m + 1, m) - (m + alpha + su * rb / R) * M(m, m))
        return s / mp.factorial(n)

    P0, P1 = [], []
    n = 0
    while True:
        P0.append(p0(n))
        P1.append(p1(n))
        if n >= n_max and abs(P0[-1]) + abs(P1[-1]) < GRIMA_TAIL * (abs(P0[0]) + abs(P1[0]) + sum(P1[:n_max])):
            break
        n += 1
    Z = sum(P0) + sum(P1)
    neg = min(min(P0), min(P1))
    return ([float(mp.log(x / Z)) for x in P0[:n_max]], [float(mp.log(x / Z)) for x in P1[:n_max]],
            {"terms": n + 1, "most_negative_term": float(neg / Z), "mass_beyond_box": float((sum(P0[n_max:]) + sum(P1[n_max:])) / Z)})


def _kumar_logpi(b, k1, k2, k3, k4, k5, n_max):
    """log P(gene, n), gene in {0 (off), 1 (on)} as in the Kuntz model (G = 1 on), n = 0..n_max - 1."""
    import mpmath as mp
    mp.mp.dps = DPS
    b, k1, k2, k3, k4, k5 = (mp.mpf(str(x)) for x in (b, k1, k2, k3, k4, k5))
    k0, mu, al, alt, be = k4, k5, k2, k3, k1  # Kumar et al.: producing state 0, k_1 = 0
    s_uv = (k0 + al + be) / (mu + alt)  # u + v (Delta k = k0, the k_1 term vanishes)
    p_uv = be * k0 / (mu * (mu + alt))  # u v
    disc = mp.sqrt(s_uv**2 - 4 * p_uv)
    u, v = (s_uv + disc) / 2, (s_uv - disc) / 2
    w = (k0 + mu + alt * (1 + b)) / (mu + alt * (1 + b))
    phi = (mu + alt) / (mu + alt + b * alt)
    c = u + v + 1 - w
    x0 = 1 - phi * (1 + b)
    norm = mp.hyp2f1(u, v, c, 1 - phi)
    # u and v may be a complex-conjugate pair; every term is then real, and the imaginary parts are rounding
    Pc = [(phi * b) ** n / mp.factorial(n) * mp.rf(u, n) * mp.rf(v, n) / mp.rf(c, n)
          * mp.hyp2f1(u + n, v + n, c + n, x0) / norm for n in range(n_max + 1)]
    max_imag = max(abs(mp.im(x)) / max(abs(x), mp.mpf(10) ** -300) for x in Pc)
    P = [mp.re(x) for x in Pc]
    Pon = [mu * ((1 + b) * (n + 1) * P[n + 1] - b * n * P[n]) / (k0 * b) for n in range(n_max)]
    Poff = [P[n] - Pon[n] for n in range(n_max)]
    neg = min(min(Pon), min(Poff))
    return ([float(mp.log(x)) for x in Poff], [float(mp.log(x)) for x in Pon],
            {"u": str(u), "v": str(v), "c": str(c), "w": float(w), "phi": float(phi), "x0": float(x0),
             "max_relative_imaginary_part": float(max_imag),
             "most_negative_term": float(neg), "mass_beyond_box": float(1 - sum(P[:n_max]))})


def closed_form_logpi(m, widths):
    """log pi on the instance box, in the order of m.space(widths)."""
    space = m.space(widths)
    X = space.states
    if m.name.startswith("autoreg"):
        p = m.params
        n_max = int(widths[2])
        l0, l1, info = _grima_logpi(p["sigma_u"], p["sigma_b"], p["rho_u"], p["rho_b"], n_max)
        lp = np.where(X[:, 0] == 1, np.take(l0, X[:, 2]), np.take(l1, X[:, 2]))
    else:
        p = m.params
        n_max = int(widths[1])
        loff, lon, info = _kumar_logpi(p["b_mean"], p["k1"], p["k2"], p["k3"], p["k4"], p["k5"], n_max)
        lp = np.where(X[:, 0] == 1, np.take(lon, X[:, 1]), np.take(loff, X[:, 1]))
    return lp, info


INSTANCES = ("autoreg_tang_1", "autoreg_tang_2", "autoreg_tang_3", "feedback_b1", "feedback_b10", "feedback_b100")


def stage_compute(only=None):
    import mpmath as mp
    for name in (only or INSTANCES):
        t0 = time.time()
        m = next(x for x in models.published() if x.name == name)
        widths = tuple(data(PART, name)["widths"])
        lp, info = closed_form_logpi(m, widths)
        save(PART, f"{name}_closed_form", {"model": name, "widths": widths, "mpmath": mp.__version__, "dps": DPS,
                                           "source": __doc__.split("\n\n")[2 if name.startswith("autoreg") else 3]},
             {"widths": list(widths), "log_pi": lp.tolist(), **info, "seconds": time.time() - t0})
        print(f"  {name}: {len(lp)} states, mass beyond the box {info['mass_beyond_box']:.2e} [{time.time() - t0:.0f}s]",
              flush=True)


def stage_check(only=None):
    from .published_models import STATIONARY_GROW, stationary_on_box
    from .common import build_setup
    rows = []
    for name in (only or INSTANCES):
        t0 = time.time()
        m = next(x for x in models.published() if x.name == name)
        d = data(PART, name)
        cf = data(PART, f"{name}_closed_form")
        s = build_setup(m, tuple(d["widths"]), d["T"], refs=None)
        lcf = np.asarray(cf["log_pi"], float)
        lnum, chk = stationary_on_box(m, s.widths, STATIONARY_GROW)
        pc, pn = np.exp(lcf), np.exp(lnum)
        tv = 0.5 * float(np.abs(pc - pn).sum() + abs(cf["mass_beyond_box"] - chk["mass_outside_box"]))
        cert = certify.window_certificate(s.gen, reference.Static(lcf, name="closed form"), np.array([0.0, s.T]))
        K = min(cert.K.values())
        pT = fsp.propagate(s.gen.A, s.p0, [s.T])[-1]
        mT = float(pT.sum())
        from scipy.special import logsumexp
        log_rf = 0.5 * float(logsumexp(lcf)) - np.log(mT)
        row = {"instance": name, "total_variation": tv, "max_abs_log_ratio": float(np.max(np.abs(lcf - lnum))),
               "K": float(K), "log10_readout_cert": log10(log_norm_weighted(s.p0, lcf) + 0.5 * max(K, 0.0) + log_rf),
               "log10_readout_meas": log10(log_norm_weighted(pT, lcf) + log_rf),
               "mass_beyond_box_closed_form": cf["mass_beyond_box"], "mass_outside_box_numerical": chk["mass_outside_box"],
               "seconds": time.time() - t0}
        row["pass"] = bool(tv <= 1e-6 and abs(K) <= 1e-6)
        rows.append(row)
        print(f"  {name}: TV {tv:.2e}, max |log ratio| {row['max_abs_log_ratio']:.2e}, K_T {K:.2e}, pass {row['pass']}",
              flush=True)
    save(PART, "closed_form_check", {"criteria": "total variation <= 1e-6 and |K_T| <= 1e-6"},
         {"rows": rows, "all_pass": all(r["pass"] for r in rows)})


STAGES = {"compute": stage_compute, "check": stage_check}

if __name__ == "__main__":
    args = sys.argv[1:]
    only = args[args.index("--only") + 1].split(",") if "--only" in args else None
    for a in [a for a in args if a in STAGES]:
        STAGES[a](only=only)
