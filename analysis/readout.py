"""Readout part: the factor sqrt(|S| / S_eff) that uniform-encoding readouts omit, and its weighted counterparts.

    python -m analysis.readout cascade [--only 1,2,3]   linear cascades with d = 1..6 species (width 12, T = 1)
    python -m analysis.readout lambda_phage             the five-species lambda-phage model (about a day)
    python -m analysis.readout cascade_summary          growth per species (slopes) of every readout factor
    python -m analysis.readout published                the published instances (Table 1 of the article)
    python -m analysis.readout swap_test                the swap-test ratio estimator (Supplementary Table S4)
    python -m analysis.readout concept                  the data of Figure 1a

For each encoding the readout factor is reported with three growth factors of the weighted state: the certificate
(cert), the worst case over the horizon W, and the measured growth (meas). Only cert is a certified bound.
"""
import json
import sys
import time

import numpy as np
from scipy.special import logsumexp

from .common import (CEILING, FAMILIES, LEAK_TOL, best_of, build_setup, capped_box, certificate_table, data, is_huge,
                     is_large, log10, log_norm_weighted, observables, published_setup, save, schedules, settings,
                     worst_case)
from qcme import certify, emulate, fsp, invariants as inv, models, reference

PART = "readout"
CASCADE_WIDTH = 12
CASCADE_DS = (1, 2, 3, 4, 5, 6)
ENCODINGS = ("uniform", "R-P", "R-NB", "R-mix(NB,E1)")
# On the cascade R-NB coincides with R-P (variance = mean), and one weighted encoding at d = 6 takes about 12 h.
SKIP = {("cascade_6", "R-NB"): "skipped: R-NB coincides with R-P on the cascade (variance = mean)"}
REMOVE_SLOPE = 0.05  # a weighted encoding "removes" the growth if its slope is at most this (decades per species)
CONCEPT = {"model": "toggle", "widths": [50, 50], "T": 1.0, "reference": "R-mix(NB)", "J": 20,
           "p0": "normalized R-P at t = 0"}


def readout_setup(name):
    if name.startswith("cascade_"):
        d = int(name.split("_")[1])
        m = models.cascade(d, width=CASCADE_WIDTH)
        return build_setup(m, (CASCADE_WIDTH,) * d, 1.0), {"rule": "cascade family, T = 1, fixed width 12"}
    s, leak, hinfo = published_setup(models.lambda_phage())
    return s, hinfo


def readout_factors(name):
    s, hinfo = readout_setup(name)
    n = len(s.space)
    large = is_large(s)
    t0 = time.time()
    pT = fsp.propagate(s.gen.A, s.p0, [s.T])[-1]
    mT = float(pT.sum())
    S_eff = 1.0 / float(np.sum((pT / mT) ** 2))
    obs = observables(s, pT)
    Ef = {k: float(o["f"] @ pT) for k, o in obs.items()}
    print(f"-- {name}: T={s.T:.4g} widths={s.widths} |S|={n} leak={1 - mT:.2e} |S|/S_eff={n / S_eff:.4g} "
          f"(FSP {time.time() - t0:.0f}s)", flush=True)
    sched = schedules(s.T)
    enc = {}
    for e in ENCODINGS:
        t1 = time.time()
        if (name, e) in SKIP:
            ts = sorted(set([0.0, float(s.T)] + [float(t) for ed in sched.values() for t in ed]))
            dmax = max(float(np.max(np.abs(s.refs[e].log_mu(t) - s.refs["R-P"].log_mu(t)))) for t in ts)
            enc[e] = {"skipped": SKIP[(name, e)], "max_abs_dlogmu_vs_R-P": dmax, "n_times_compared": len(ts)}
            continue
        if e == "uniform":
            ref = reference.Uniform(n)
            cert = certify.window_certificate(s.gen, ref, np.array([0.0, s.T]),
                                              families=("sqrt_mu",) if is_huge(s) else FAMILIES)
            K, sn, fam = min((cert.K[f], "single window", f) for f in cert.K)
            table = {"single window": cert.K}
            lm0 = lmT = np.zeros(n)  # mu = 1: readout factors do not depend on the normalization of mu
        else:
            ref = s.refs[e]
            tab = certificate_table(s, ref, sched, large)
            K, sn, fam = best_of(tab)
            table = {k: v.K for k, v in tab.items()}
            lm0, lmT = ref.log_mu(0.0), ref.log_mu(sched[sn][-2])  # mu_T = mu of the last window
        log_psi0 = log_norm_weighted(s.p0, lm0)
        log_W = worst_case(s, lm0, lmT)
        growth = {"cert": log_psi0 + 0.5 * K, "W": None if log_W is None else log_psi0 + log_W,
                  "meas": log_norm_weighted(pT, lmT)}
        rf = {k: float(np.log(np.linalg.norm(o["f"] * np.exp(0.5 * lmT)) / abs(Ef[k]))) for k, o in obs.items()}
        enc[e] = {"K": K, "schedule": sn, "family": fam, "K_table": table, "large_box_rule": large,
                  "log10_psi0": log10(log_psi0),
                  "log10_Gamma": {g: None if v is None else log10(v) for g, v in growth.items()},
                  "log10_R_f": {k: log10(v) for k, v in rf.items()},
                  "log10_readout": {g: {k: None if v is None else log10(v + rf[k]) for k in obs}
                                    for g, v in growth.items()},
                  "seconds": time.time() - t1}
        r = enc[e]["log10_readout"]
        print(f"   {e:13s} K={K:.4g} ({fam}, {sn}) log10 readout f=1: cert {r['cert']['one']:.3f} "
              f"meas {r['meas']['one']:.3f} [{time.time() - t1:.0f}s]", flush=True)
    return {"T": s.T, "horizon": hinfo, "widths": s.widths, "n_states": n, "leak": 1 - mT, "S_eff": S_eff,
            "log10_S_over_Seff": log10(np.log(n / S_eff)),
            "observables": {k: {"E_f": Ef[k], "theta": o["theta"], "species": o["species"]} for k, o in obs.items()},
            "encodings": enc}


def consistency_check(d):
    """The uniform factor and the R-P certificate at J = 20 recomputed directly on 12^d (a cross-check)."""
    m = models.cascade(d, width=CASCADE_WIDTH)
    s = build_setup(m, (CASCADE_WIDTH,) * d, 1.0)
    pT = fsp.propagate(s.gen.A, s.p0, [1.0])[-1]
    uf = 0.5 * np.log10(len(s.space) * np.sum(pT**2) / pT.sum() ** 2)
    cert = certify.window_certificate(s.gen, s.refs["R-P"], certify.uniform_edges(1.0, 20))
    g = log10(log_norm_weighted(s.p0, s.refs["R-P"].log_mu(0.0)) + 0.5 * cert.K["resolvent"])
    return {"log10_uniform_factor": uf, "log10_Gamma_RP_J20_resolvent": g}


def stage_cascade(only=None):
    for d in CASCADE_DS:
        if only and str(d) not in only:
            continue
        t0 = time.time()
        out = readout_factors(f"cascade_{d}")
        if d <= 4:
            out["consistency_check"] = consistency_check(d)
        out["seconds"] = time.time() - t0
        save(PART, f"cascade_{d}", {**settings(), "model": f"cascade_{d}", "encodings": ENCODINGS,
                                    "cascade_width": CASCADE_WIDTH}, out)


def stage_lambda_phage(only=None):
    t0 = time.time()
    out = readout_factors("lambda_phage")
    out["seconds"] = time.time() - t0
    save(PART, "lambda_phage", {**settings(), "model": "lambda_phage", "encodings": ENCODINGS}, out)


def _slope(ds, ys):
    ds, ys = np.asarray(ds, float), np.asarray(ys, float)
    return float(np.polyfit(ds, ys, 1)[0]), [float(v) for v in np.diff(ys)]


def stage_cascade_summary(only=None):
    rows = {d: data(PART, f"cascade_{d}") for d in CASCADE_DS}
    rows = {d: r for d, r in rows.items() if r is not None}
    ds = sorted(rows)
    out = {"ds": ds, "slopes": {}, "statements": {}}
    uni = [rows[d]["encodings"]["uniform"]["log10_readout"]["meas"]["one"] for d in ds]
    sl_u, inc_u = _slope(ds, uni)
    out["statements"]["uniform_grows_exponentially"] = bool(all(v > 0 for v in inc_u)) if len(ds) == 6 else None
    out["uniform_meas_one"] = {"slope": sl_u, "increments": inc_u}
    for e in ENCODINGS:
        for g in ("cert", "W", "meas"):
            for o in ("one", "tail"):
                pts = [(d, rows[d]["encodings"][e].get("log10_readout", {}).get(g, {}).get(o)) for d in ds]
                pts = [(d, y) for d, y in pts if y is not None]
                if len(pts) < 2:
                    out["slopes"][f"{e}|{g}|{o}"] = None
                    continue
                sl, inc = _slope([d for d, _ in pts], [y for _, y in pts])
                out["slopes"][f"{e}|{g}|{o}"] = {"slope": sl, "increments": inc, "ds": [d for d, _ in pts]}
                if e != "uniform" and o == "one":
                    out["statements"][f"{e}|{g}"] = ("removes" if sl <= REMOVE_SLOPE else
                                                     "reduces" if sl < sl_u else "does not remove")
    save(PART, "cascade_summary", {**settings(), "ds": ds}, out)


def stage_published(only=None):
    """sqrt(|S| / S_eff) for f = 1 on each published instance, on the box and horizon of its certificates
    (results/published_models/<instance>.json). lambda-phage is read from readout/lambda_phage.json (same box and
    horizon). The Gardner toggle (3.66 million states, about an hour) is reused from its stored file unless it is
    named with --only."""
    rows = []
    for m in models.published():
        if only and m.name not in only:
            continue
        dec = data("published_models", m.name)
        s = build_setup(m, tuple(dec["widths"]), float(dec["T"]), refs=None)
        n_count = int(sum(not b for b in m.binary))
        if m.name == "lambda_phage":
            lam = data(PART, "lambda_phage")
            rows.append({"row": m.name, "n_states": lam["n_states"], "S_eff": lam["S_eff"],
                         "log10_readout_factor": 0.5 * lam["log10_S_over_Seff"], "n_count_species": n_count,
                         "source": "readout/lambda_phage.json (same box and horizon)"})
            continue
        stored = data(PART, f"published_{m.name}")
        if is_huge(s) and stored and not only:
            rows.append({k: stored[k] for k in ("row", "n_states", "leak", "S_eff", "readout_factor",
                                                "log10_readout_factor", "n_count_species")})
            continue
        t0 = time.time()
        pT = fsp.propagate(s.gen.A, s.p0, [s.T])[-1]
        mass = float(pT.sum())
        S_eff = mass**2 / float(np.sum(pT**2))
        n = len(s.space)
        out = {"row": m.name, "n_states": n, "widths": list(s.widths), "T": s.T, "leak": 1.0 - mass, "S_eff": S_eff,
               "readout_factor": float(np.sqrt(n / S_eff)), "log10_readout_factor": float(0.5 * np.log10(n / S_eff)),
               "species": list(m.species), "n_count_species": n_count, "seconds": time.time() - t0}
        print(f"-- {m.name}: |S|={n:,} leak={1 - mass:.2e} S_eff={S_eff:.4g} factor=10^{out['log10_readout_factor']:.3f}",
              flush=True)
        save(PART, f"published_{m.name}", {"observable": "f = 1 (total probability)",
                                           "quantity": "sqrt(|S| / S_eff(p_T)), S_eff = (sum p)^2 / sum p^2",
                                           "box_and_T": "published_models/<instance>.json"}, out)
        rows.append({k: out[k] for k in ("row", "n_states", "leak", "S_eff", "readout_factor", "log10_readout_factor",
                                          "n_count_species")})
    if only:
        return
    casc = []
    for d in CASCADE_DS:
        b = data(PART, f"cascade_{d}")
        casc.append({"d": d, "n_states": b["n_states"], "S_eff": b.get("S_eff"),
                     "log10_S_over_Seff_half": 0.5 * b["log10_S_over_Seff"]})
    lam = data(PART, "lambda_phage")
    save(PART, "published_summary", {"observable": "f = 1 (total probability)"},
         {"published_rows": rows, "cascade": casc,
          "lambda_phage_barrier_file": {"n_states": lam["n_states"], "S_eff": lam["S_eff"],
                                        "log10_readout_factor": 0.5 * lam["log10_S_over_Seff"]}})


def ae_queries(F1, F2, rel=1e-2):
    """Queries of canonical amplitude estimation for the same ratio, of the order of sqrt(2) pi / rel (1/sqrt(F1) +
    1/sqrt(F2)): M queries estimate an amplitude a = sqrt(F) to about pi / M, and the relative error rel of the ratio
    is split equally over its two amplitudes."""
    return float(np.sqrt(2) * np.pi / rel * (1 / np.sqrt(F1) + 1 / np.sqrt(F2)))


def stage_swap_test(only=None):
    """The swap-test ratio estimator V = <C|p>/<+|p> on a localized product Gaussian law (sigma = 8% of [0, 1] per
    asset, 32 grid points per asset, geometric-basket call payoff), d = 1..4. A swap test with squared overlap F
    returns 0 with probability (1 + F)/2; the two binomial variances are propagated to V by the delta method.
    No random numbers are used."""
    width, sigma, rows = 32, 0.08, []
    x1 = (np.arange(width) + 0.5) / width
    for d in range(1, 5):
        grids = np.meshgrid(*([x1] * d), indexing="ij")
        p = np.prod([np.exp(-0.5 * ((g - 0.5) / sigma) ** 2) for g in grids], axis=0).ravel()
        p /= p.sum()
        payoff = np.maximum(np.prod(grids, axis=0).ravel() ** (1.0 / d) - 0.5, 0.0)
        p_hat, c_hat = p / np.linalg.norm(p), payoff / np.linalg.norm(payoff)
        F1 = float((np.ones_like(p) / np.sqrt(p.size) @ p_hat) ** 2)
        F2 = float((c_hat @ p_hat) ** 2)
        shots_1pct = 0.25 * ((1 - F1**2) / F1**2 + (1 - F2**2) / F2**2) / 1e-4
        rows.append({"d": d, "F1_uniform_overlap_sq": F1, "F2_payoff_overlap_sq": F2,
                     "rel_error_at_1e6_shots": 0.5 * float(np.sqrt((1 - F1**2) / (1e6 * F1**2)
                                                                   + (1 - F2**2) / (1e6 * F2**2))),
                     "shots_for_1pct": float(shots_1pct), "ae_queries_for_1pct": ae_queries(F1, F2),
                     "sqrt_S_over_Seff": float(np.sqrt(p.size * np.sum(p**2)))})  # p sums to 1, S_eff = 1/sum p^2
    ratios = [rows[i + 1]["shots_for_1pct"] / rows[i]["shots_for_1pct"] for i in range(3)]
    ae = [rows[i + 1]["ae_queries_for_1pct"] / rows[i]["ae_queries_for_1pct"] for i in range(3)]
    save(PART, "swap_test", {"width": width, "sigma": sigma},
         {"source": "analytic evaluation (analysis/readout.py, swap_test)", "width": width, "sigma": sigma,
          "rows": rows, "shots_ratio_per_asset": ratios, "ae_ratio_per_asset": ae})


SWAP_VARIANTS = {"grid fitted to 4 standard deviations": {"half_width": 4.0, "rho": 0.0},
                 "correlated assets (correlation 0.5)": {"half_width": None, "rho": 0.5}}


def stage_swap_variants(only=None):
    """The swap-test example on a grid fitted to the law (0.5 +- 4 sigma per asset) and on correlated assets
    (equicorrelation 0.5, the grid of the main example), d = 1..4, with the same payoff and strike."""
    width, sigma, out = 32, 0.08, {}
    for name, v in SWAP_VARIANTS.items():
        lo, hi = (0.5 - v["half_width"] * sigma, 0.5 + v["half_width"] * sigma) if v["half_width"] else (0.0, 1.0)
        x1 = lo + (hi - lo) * (np.arange(width) + 0.5) / width
        rows = []
        for d in range(1, 5):
            grids = np.meshgrid(*([x1] * d), indexing="ij")
            Z = np.stack([(g.ravel() - 0.5) / sigma for g in grids], axis=1)
            C = (1 - v["rho"]) * np.eye(d) + v["rho"] * np.ones((d, d))
            lp = -0.5 * np.einsum("ij,jk,ik->i", Z, np.linalg.inv(C), Z)
            p = np.exp(lp - lp.max())
            p /= p.sum()
            payoff = np.maximum(np.prod(grids, axis=0).ravel() ** (1.0 / d) - 0.5, 0.0)
            p_hat, c_hat = p / np.linalg.norm(p), payoff / np.linalg.norm(payoff)
            F1 = float((np.ones_like(p) / np.sqrt(p.size) @ p_hat) ** 2)
            F2 = float((c_hat @ p_hat) ** 2)
            rows.append({"d": d, "F1_uniform_overlap_sq": F1, "F2_payoff_overlap_sq": F2,
                         "shots_for_1pct": float(0.25 * ((1 - F1**2) / F1**2 + (1 - F2**2) / F2**2) / 1e-4),
                         "ae_queries_for_1pct": ae_queries(F1, F2),
                         "sqrt_S_over_Seff": float(np.sqrt(p.size * np.sum(p**2)))})
        ratios = [rows[i + 1]["shots_for_1pct"] / rows[i]["shots_for_1pct"] for i in range(3)]
        factor = [rows[i + 1]["sqrt_S_over_Seff"] / rows[i]["sqrt_S_over_Seff"] for i in range(3)]
        ae = [rows[i + 1]["ae_queries_for_1pct"] / rows[i]["ae_queries_for_1pct"] for i in range(3)]
        out[name] = {"grid": [lo, hi], "rho": v["rho"], "rows": rows, "shots_ratio_per_asset": ratios,
                     "ae_ratio_per_asset": ae,
                     "readout_factor_ratio_per_asset": factor}
        print(f"-- {name}: shots " + ", ".join(f"{r['shots_for_1pct']:.3g}" for r in rows)
              + "; readout factor " + ", ".join(f"{r['sqrt_S_over_Seff']:.3g}" for r in rows), flush=True)
    save(PART, "swap_variants", {"width": width, "sigma": sigma, "variants": SWAP_VARIANTS}, out)


def stage_concept(only=None):
    """Toggle switch on 50 x 50 at T = 1: p_T, the reference mu_T and the two encoded states (Figure 1a)."""
    s = inv.setup(models.toggle(), tuple(CONCEPT["widths"]), CONCEPT["T"])
    pT = fsp.propagate(s.gen.A, s.p0, [CONCEPT["T"]])[-1]
    ref = s.refs[CONCEPT["reference"]]
    edges = certify.uniform_edges(CONCEPT["T"], CONCEPT["J"])
    cert = certify.window_certificate(s.gen, ref, edges)
    run = emulate.piecewise_psi(s.gen, ref, edges, s.p0)
    logmu_T = ref.log_mu(CONCEPT["T"])
    psi_T = pT * np.exp(-0.5 * logmu_T)
    n, S_eff = len(s.space), 1.0 / float(np.sum(pT**2))
    save(PART, "concept", CONCEPT,
         {"X": s.space.X.tolist(), "p_T": pT.tolist(), "log_mu_T": logmu_T.tolist(), "psi_T": psi_T.tolist(),
          "n_states": n, "S_eff": S_eff, "uniform_factor": float(np.sqrt(n / S_eff)),
          "psi_T_norm": float(np.linalg.norm(psi_T)), "psi0_norm": float(np.linalg.norm(run.psi[0])), "K": cert.K,
          "leak": float(1 - fsp.retained_mass(pT)),
          "measured_growth_piecewise": float(np.linalg.norm(run.psi[-1]) / np.linalg.norm(run.psi[0]))})


SENS_TOLS = (1e-4, 1e-6, 1e-8)
SENS_WIDTHS = (10, 12, 16, 20)
# left out: one solve takes hours (lambda_phage, toggle_gardner) or minutes, repeated for every box (the other two)
SENS_SKIP = ("lambda_phage", "toggle_gardner", "feedback_b100", "toggle_tang")


def _factor(m, widths, T):
    s = build_setup(m, tuple(widths), T, refs=None)
    pT = fsp.propagate(s.gen.A, s.p0, [T])[-1]
    mass = float(pT.sum())
    return len(s.space), 1.0 - mass, float(0.5 * np.log10(len(s.space) * np.sum(pT**2) / mass**2))


def stage_sensitivity(only=None):
    """How the factor sqrt(|S| / S_eff) depends on the truncation: the published instances with boxes grown to
    leak tolerances 1e-4, 1e-6 and 1e-8 (the eight instances that solve in seconds), and the cascade (d = 1..4) with
    box widths 10, 12, 16 and 20."""
    rows = []
    for m in models.published():
        if m.name in SENS_SKIP or (only and m.name not in only):
            continue
        T = float(data("published_models", m.name)["T"])
        row = {"row": m.name, "T": T, "by_tolerance": {}}
        for tol in SENS_TOLS:
            w, _ = fsp.select_box(m, lambda sp_: reference.initial_law(m, sp_), T, tol, max_states=2_000_000)
            n, leak, f = _factor(m, w, T)
            row["by_tolerance"][f"{tol:g}"] = {"widths": list(w), "n_states": n, "leak": leak, "log10_readout_factor": f}
        v = [x["log10_readout_factor"] for x in row["by_tolerance"].values()]
        row["spread"] = max(v) - min(v)
        rows.append(row)
        print(f"-- {m.name}: " + ", ".join(f"{k}: {x['log10_readout_factor']:.3f} (|S|={x['n_states']:,})"
                                           for k, x in row["by_tolerance"].items()), flush=True)
    casc = {}
    for w in SENS_WIDTHS:
        fs = []
        for d in (1, 2, 3, 4):
            n, leak, f = _factor(models.cascade(d, width=w), (w,) * d, 1.0)
            fs.append({"d": d, "n_states": n, "leak": leak, "log10_readout_factor": f})
        sl, _ = _slope([x["d"] for x in fs], [x["log10_readout_factor"] for x in fs])
        casc[str(w)] = {"rows": fs, "slope": sl}
        print(f"-- cascade width {w}: slope {sl:.3f} decades per species, leak at d = 4 {fs[-1]['leak']:.1e}", flush=True)
    save(PART, "sensitivity", {"tolerances": SENS_TOLS, "cascade_widths": SENS_WIDTHS, "skipped": SENS_SKIP},
         {"published": rows, "max_spread_published": max(r["spread"] for r in rows), "cascade": casc})


CORRELATED_DS = (1, 2, 3, 4, 5)
INITIAL_LAW_INSTANCES = ("schlogl", "toggle_hegland", "toggle_kuntz", "feedback_b1", "feedback_b10", "autoreg_tang_1",
                         "autoreg_tang_2", "autoreg_tang_3")


def stage_correlated(only=None):
    """sqrt(|S| / S_eff) on a catalytic cascade (each species produces the next), d = 1..5, T = 1, boxes grown to
    the leak tolerance. The law is not a product law; the last column compares S_eff with the product of the
    S_eff of the one-species marginals, which would be equal for a product law."""
    rows = []
    for d in CORRELATED_DS:
        if only and str(d) not in only:
            continue
        t0 = time.time()
        m = models.catalytic_cascade(d)
        w, _ = fsp.select_box(m, lambda sp_: reference.initial_law(m, sp_), 1.0, settings()["leak_tol"],
                              max_states=settings()["ceiling"])
        s = build_setup(m, w, 1.0, refs=None)
        pT = fsp.propagate(s.gen.A, s.p0, [1.0])[-1]
        mass = float(pT.sum())
        n = len(s.space)
        S_eff = mass**2 / float(np.sum(pT**2))
        X = s.space.states
        marg = [np.bincount(X[:, i], weights=pT / mass, minlength=w[i]) for i in range(d)]
        prod_seff = float(np.prod([1.0 / np.sum(q**2) for q in marg]))
        means = [float(np.arange(len(q)) @ q) for q in marg]
        fano = [float((np.arange(len(q)) ** 2) @ q - mu**2) / mu for q, mu in zip(marg, means)]
        corr = None
        if d > 1:
            xy = float((X[:, d - 2] * X[:, d - 1]) @ (pT / mass))
            v0 = float((X[:, d - 2] ** 2) @ (pT / mass)) - means[d - 2] ** 2
            v1 = float((X[:, d - 1] ** 2) @ (pT / mass)) - means[d - 1] ** 2
            corr = (xy - means[d - 2] * means[d - 1]) / np.sqrt(v0 * v1)
        rows.append({"d": d, "widths": list(w), "n_states": n, "leak": 1.0 - mass, "S_eff": S_eff,
                     "log10_readout_factor": float(0.5 * np.log10(n / S_eff)),
                     "S_eff_over_product_of_marginals": S_eff / prod_seff, "means": means, "fano": fano,
                     "correlation_last_two": corr, "seconds": time.time() - t0})
        print(f"-- catalytic cascade d={d}: widths={w} |S|={n:,} leak={1 - mass:.1e} factor=10^"
              f"{rows[-1]['log10_readout_factor']:.3f} S_eff/prod={S_eff / prod_seff:.3f} corr={corr}", flush=True)
    if only:
        return
    sl, inc = _slope([r["d"] for r in rows], [r["log10_readout_factor"] for r in rows])
    save(PART, "correlated_cascade", {"k": 4.0, "c": 1.0, "g": 1.0, "T": 1.0, "leak_tol": settings()["leak_tol"]},
         {"rows": rows, "slope": sl, "increments": inc})


def stage_initial_law(only=None):
    """sqrt(|S| / S_eff) at the same horizon for three initial laws: the default product of Poisson laws with the
    initial means of the model, a point mass at those means (rounded), and a product of Poisson laws at the
    attractor of the rate equations (on a box grown to the leak tolerance for that law)."""
    rows = []
    for m in models.published():
        if m.name not in INITIAL_LAW_INSTANCES or (only and m.name not in only):
            continue
        dec = data("published_models", m.name)
        T, w = float(dec["T"]), tuple(dec["widths"])
        s = build_setup(m, w, T, refs=None)
        X = s.space.states
        x0 = np.asarray(m.x0, float)
        x_star = np.maximum(models.relaxation_time(m)[1], 0.0)
        point = np.zeros(len(X))
        target = np.round(x0).astype(int)
        point[np.argmin(np.abs(X - target).sum(axis=1))] = 1.0
        m_star = models.Model(**{**m.__dict__, "x0": tuple(float(v) for v in x_star)})
        laws = {"poisson at the initial means": (s, s.p0), "point mass at the initial means": (s, point)}
        # the law started at the attractor can sit elsewhere, so its box is grown to the same leak tolerance
        w2, _ = fsp.select_box(m_star, lambda sp_: reference.initial_law(m_star, sp_), T, settings()["leak_tol"],
                               max_states=2_000_000)
        s2 = build_setup(m_star, w2, T, refs=None)
        laws["poisson at the attractor"] = (s2, s2.p0)
        out = {}
        for k, (sk, p0) in laws.items():
            pT = fsp.propagate(sk.gen.A, p0, [T])[-1]
            mass = float(pT.sum())
            out[k] = {"widths": list(sk.widths), "n_states": len(sk.space), "leak": 1.0 - mass,
                      "log10_readout_factor": float(0.5 * np.log10(len(sk.space) * np.sum(pT**2) / mass**2))}
        v = [o["log10_readout_factor"] for o in out.values()]
        rows.append({"row": m.name, "T": T, "widths": list(w), "initial_means": list(x0), "attractor": list(x_star),
                     "by_initial_law": out, "spread": max(v) - min(v)})
        print(f"-- {m.name}: " + ", ".join(f"{k}: {o['log10_readout_factor']:.3f} (leak {o['leak']:.1e})"
                                           for k, o in out.items()), flush=True)
    if only:
        return
    save(PART, "initial_law", {"instances": INITIAL_LAW_INSTANCES},
         {"rows": rows, "max_spread": max(r["spread"] for r in rows)})


def stage_cascades_matched(only=None):
    """The two cascades on the same boxes (the boxes of the catalytic cascade, width 15, which meet the leak
    tolerance for both), d = 1..5: the uniform factor of each, and for the catalytic cascade the measured readout
    factor of every cheap weighted encoding, ||sqrt(mu_T)|| ||p_T / sqrt(mu_T)|| / m_T."""
    cat = {r["d"]: r for r in data(PART, "correlated_cascade")["rows"]}
    rows = []
    for d in CORRELATED_DS:
        t0 = time.time()
        w = tuple(cat[d]["widths"])
        conv = _factor(models.cascade(d, width=w[0]), w, 1.0)
        s = build_setup(models.catalytic_cascade(d), w, 1.0)
        pT = fsp.propagate(s.gen.A, s.p0, [1.0])[-1]
        mT = float(pT.sum())
        weighted = {}
        for rn, ref in s.refs.items():
            lmT = ref.log_mu(1.0)
            weighted[rn] = log10(log_norm_weighted(pT, lmT) + 0.5 * float(logsumexp(lmT)) - np.log(mT))
        best = min(weighted, key=weighted.get)
        rows.append({"d": d, "widths": list(w), "conversion_uniform": conv[2], "conversion_leak": conv[1],
                     "catalytic_uniform": float(0.5 * np.log10(len(s.space) * np.sum(pT**2) / mT**2)),
                     "catalytic_weighted": weighted, "catalytic_weighted_smallest": weighted[best],
                     "catalytic_weighted_smallest_reference": best, "seconds": time.time() - t0})
        print(f"-- d={d}: conversion 10^{conv[2]:.3f}, catalytic 10^{rows[-1]['catalytic_uniform']:.3f}, "
              f"catalytic weighted {best} 10^{weighted[best]:.3f} [{time.time() - t0:.0f}s]", flush=True)
    ds = [r["d"] for r in rows]
    slopes = {k: _slope(ds, [r[k] for r in rows])[0]
              for k in ("conversion_uniform", "catalytic_uniform", "catalytic_weighted_smallest")}
    save(PART, "cascades_matched", {"width": 15, "T": 1.0}, {"rows": rows, "slopes": slopes})


# ----------------------------------------------------------------------------- fitted boxes and published cascades
CUT_TOL = 1e-6  # the cut box keeps all but this much of p_T
LNA_SD = 4.0  # the moment box: mean +- this many standard deviations at T
START_SD = 3.0  # signalling cascades: start box from the moments, mean + this many sd over [0, T]
SIGNALLING_DS = (1, 2, 3, 4)
FITTED_PUBLISHED = INITIAL_LAW_INSTANCES + ("toggle_tang",)
LAMBDA_BOXES = ((15, 40, 10, 10, 10), (19, 50, 13, 13, 13), (24, 63, 17, 17, 17))


def _restricted_factor(p, keep):
    q = p[keep]
    mass = float(q.sum())
    return {"n_states": int(keep.sum()), "mass": mass,
            "log10_readout_factor": float(0.5 * np.log10(int(keep.sum()) * np.sum(q**2) / mass**2))}


def fitted_factors(s, pT):
    """sqrt(|S| / S_eff) of p_T restricted to two boxes fitted to the law at T: (a) the cut box, the smallest box
    cut from the marginals that keeps all but CUT_TOL of p_T (the tolerance split equally over the two ends of every
    count species), and (b) the moment box, the linear-noise mean +- LNA_SD standard deviations at T. Both are
    restrictions of p_T on the original box, so they are optimistic for a computation that must also hold the laws
    at earlier times."""
    X = s.space.states
    count = [i for i, b in enumerate(s.model.binary) if not b]
    tau = CUT_TOL / (2 * len(count))
    keep_cut = np.ones(len(X), bool)
    cut = {}
    for i in count:
        marg = np.bincount(X[:, i], weights=pT, minlength=int(X[:, i].max()) + 1)
        lo = int(np.nonzero(np.cumsum(marg) > tau)[0][0])
        hi = int(np.nonzero(np.cumsum(marg[::-1]) > tau)[0][0])
        hi = len(marg) - 1 - hi
        cut[s.model.species[i]] = [lo, hi]
        keep_cut &= (X[:, i] >= lo) & (X[:, i] <= hi)
    mom = reference.MomentTrajectory(s.model, s.T, s.widths)
    m, _, v, _ = mom.at(s.T)
    keep_lna = np.ones(len(X), bool)
    lna = {}
    for i in count:
        sd = float(np.sqrt(max(v[i], 0.0)))
        lo, hi = max(0, int(np.floor(m[i] - LNA_SD * sd))), int(np.ceil(m[i] + LNA_SD * sd))
        lna[s.model.species[i]] = [lo, hi]
        keep_lna &= (X[:, i] >= lo) & (X[:, i] <= hi)
    mass = float(pT.sum())
    full = float(0.5 * np.log10(len(X) * np.sum(pT**2) / mass**2))
    return {"full_box": full, "cut": {**_restricted_factor(pT, keep_cut), "bounds": cut},
            "moment": {**_restricted_factor(pT, keep_lna), "bounds": lna}}


def signalling_model(kind, d, T=1.0):
    """The cascade with its start box from the rate equations and the linear-noise variances: per species the
    largest mean + START_SD sd over [0, T], plus one."""
    m0 = models.signalling_cascade(d, kind) if kind != "feedback" else models.signalling_feedback(d)
    mom = reference.MomentTrajectory(m0, T)
    hi = np.zeros(d)
    for t in np.linspace(0.0, T, 101):
        m, _, v, _ = mom.at(t)
        hi = np.maximum(hi, m + START_SD * np.sqrt(np.maximum(v, 0.0)))
    w = tuple(int(np.ceil(x)) + 1 for x in hi)
    return models.signalling_cascade(d, kind, widths=w) if kind != "feedback" else models.signalling_feedback(d, widths=w)


def _correlation_info(s, pT):
    X = s.space.states
    p = pT / pT.sum()
    seff_marg = 1.0
    for i in range(X.shape[1]):
        marg = np.bincount(X[:, i], weights=p)
        seff_marg *= 1.0 / float(np.sum(marg**2))
    out = {"S_eff_over_product_of_marginals": float((1.0 / np.sum(p**2)) / seff_marg)}
    if X.shape[1] >= 2:
        a, b = X[:, -2].astype(float), X[:, -1].astype(float)
        ma, mb = p @ a, p @ b
        cov = p @ ((a - ma) * (b - mb))
        out["correlation_last_two"] = float(cov / np.sqrt((p @ (a - ma) ** 2) * (p @ (b - mb) ** 2)))
    return out


def stage_signalling(only=None):
    """The published signalling cascades of Gupta, Schwab and Khammash 2021 (linear and nonlinear; the first d
    species, d = 1..4, which are the d-species cascades because the network is feed-forward) and their cascade with
    feedback (n = 2), at their final time T = 1 from zero copies: sqrt(|S| / S_eff) on boxes grown by the leak rule
    from a moment-based start box, the fitted-box factors and the correlation of the last two species."""
    jobs = [(k, d) for k in ("linear", "nonlinear") for d in SIGNALLING_DS] + [("feedback", 2)]
    for kind, d in jobs:
        m = signalling_model(kind, d)
        if only and m.name not in only:
            continue
        if data(PART, m.name) is not None and not only:
            continue
        t0 = time.time()
        T = 1.0
        widths, leak, capped = capped_box(m, T, part=PART)
        s = build_setup(m, widths, T, refs=None)
        pT = fsp.propagate(s.gen.A, s.p0, [T])[-1]
        mass = float(pT.sum())
        n = len(s.space)
        out = {"model": m.name, "kind": kind, "d": d, "T": T, "start_widths": list(m.widths), "widths": list(widths),
               "n_states": n, "leak": 1.0 - mass, "capped": capped,
               "log10_readout_factor": float(0.5 * np.log10(n * np.sum(pT**2) / mass**2)),
               "fitted": fitted_factors(s, pT), **_correlation_info(s, pT), "provenance": m.provenance,
               "params": m.params, "seconds": time.time() - t0}
        print(f"-- {m.name}: widths={widths} |S|={n:,} leak={1 - mass:.1e} factor 10^{out['log10_readout_factor']:.3f} "
              f"cut 10^{out['fitted']['cut']['log10_readout_factor']:.3f} [{time.time() - t0:.0f}s]", flush=True)
        save(PART, m.name, {"leak_tol": LEAK_TOL, "ceiling": CEILING, "start_sd": START_SD, "cut_tol": CUT_TOL,
                            "lna_sd": LNA_SD}, out)
    if only:
        return
    summary = {}
    for kind in ("linear", "nonlinear"):
        rows = [data(PART, f"signalling_{'lin' if kind == 'linear' else 'nonlin'}{d}") for d in SIGNALLING_DS]
        rows = [r for r in rows if r is not None]
        ds = [r["d"] for r in rows]
        fac = [r["log10_readout_factor"] for r in rows]
        cut = [r["fitted"]["cut"]["log10_readout_factor"] for r in rows]
        summary[kind] = {"ds": ds, "factor": fac, "cut": cut, "slope": _slope(ds, fac)[0], "increments": _slope(ds, fac)[1],
                         "cut_slope": _slope(ds, cut)[0], "cut_increments": _slope(ds, cut)[1],
                         "rises_with_every_species": bool(all(x > 0 for x in np.diff(fac)))}
    save(PART, "signalling_summary", {"ds": SIGNALLING_DS}, summary)


def stage_fitted_boxes(only=None):
    """Fitted-box factors on the conversion cascade (d = 1..6, width 12, the boxes of Figure 1), the catalytic
    cascade (d = 1..5, its leak-rule boxes) and the published instances that solve in minutes, on their stored boxes
    and horizons."""
    rows = []
    cat = {r["d"]: r for r in data(PART, "correlated_cascade")["rows"]}
    jobs = [("conversion", d) for d in CASCADE_DS] + [("catalytic", d) for d in CORRELATED_DS] + \
        [("published", n) for n in FITTED_PUBLISHED]
    for fam, x in jobs:
        t0 = time.time()
        if fam == "conversion":
            s = build_setup(models.cascade(x, width=CASCADE_WIDTH), (CASCADE_WIDTH,) * x, 1.0, refs=None)
        elif fam == "catalytic":
            s = build_setup(models.catalytic_cascade(x), tuple(cat[x]["widths"]), 1.0, refs=None)
        else:
            dec = data("published_models", x)
            s = build_setup(next(m for m in models.published() if m.name == x), tuple(dec["widths"]), float(dec["T"]),
                            refs=None)
        pT = fsp.propagate(s.gen.A, s.p0, [s.T])[-1]
        f = fitted_factors(s, pT)
        rows.append({"family": fam, "instance": x, "widths": list(s.widths), **f, "seconds": time.time() - t0})
        print(f"-- {fam} {x}: full 10^{f['full_box']:.3f} cut 10^{f['cut']['log10_readout_factor']:.3f} "
              f"(|S| {f['cut']['n_states']:,}) moment 10^{f['moment']['log10_readout_factor']:.3f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    slopes = {}
    for fam in ("conversion", "catalytic"):
        rr = [r for r in rows if r["family"] == fam]
        ds = [r["instance"] for r in rr]
        for key in ("full", "cut", "moment"):
            ys = [r["full_box"] if key == "full" else r[key]["log10_readout_factor"] for r in rr]
            sl, inc = _slope(ds, ys)
            slopes[f"{fam}|{key}"] = {"slope": sl, "increments": inc}
    save(PART, "fitted_boxes", {"cut_tol": CUT_TOL, "lna_sd": LNA_SD}, {"rows": rows, "slopes": slopes})


def stage_fitted_gardner(only=None):
    """The fitted-box factors of the Gardner toggle (3.66 million states; the solve takes about five hours), on its
    stored box and horizon. Run from a terminal: run_gardner_fitted.cmd."""
    dec = data("published_models", "toggle_gardner")
    m = next(x for x in models.published() if x.name == "toggle_gardner")
    t0 = time.time()
    s = build_setup(m, tuple(dec["widths"]), float(dec["T"]), refs=None)
    pT = fsp.propagate(s.gen.A, s.p0, [s.T])[-1]
    f = fitted_factors(s, pT)
    save(PART, "fitted_gardner", {"cut_tol": CUT_TOL, "lna_sd": LNA_SD},
         {"instance": "toggle_gardner", "widths": list(s.widths), **f, "seconds": time.time() - t0})


def stage_lambda_boxes(only=None):
    """lambda-phage on the last three boxes of its leak-rule growth schedule (the largest is the box of Table 1),
    T = 10: sqrt(|S| / S_eff), the leak and the fitted-box factors. Each box is saved as it finishes."""
    name = "lambda_phage_boxes"
    prev = data(PART, name) or {"rows": []}
    rows = list(prev["rows"])
    done = {tuple(r["widths"]) for r in rows}
    m = models.lambda_phage()
    for w in LAMBDA_BOXES:
        if tuple(w) in done:
            continue
        t0 = time.time()
        s = build_setup(m, w, 10.0, refs=None)
        pT = fsp.propagate(s.gen.A, s.p0, [10.0])[-1]
        mass = float(pT.sum())
        row = {"widths": list(w), "n_states": len(s.space), "leak": 1.0 - mass,
               "log10_readout_factor": float(0.5 * np.log10(len(s.space) * np.sum(pT**2) / mass**2)),
               "fitted": fitted_factors(s, pT), "seconds": time.time() - t0}
        rows.append(row)
        print(f"-- lambda-phage {w}: |S|={row['n_states']:,} leak={row['leak']:.2e} "
              f"factor 10^{row['log10_readout_factor']:.3f} [{row['seconds']:.0f}s]", flush=True)
        save(PART, name, {"T": 10.0, "boxes": LAMBDA_BOXES, "cut_tol": CUT_TOL, "lna_sd": LNA_SD}, {"rows": rows})


HORIZONS = (1.0, 3.0, 10.0)
HORIZON_DS = (1, 2, 3, 4, 5)


HORIZON_PART = "readout_horizons"  # one stored file per (family, T, d), so that interrupted runs resume per case
HORIZON_BUILD = {"conversion": lambda d, w=12: models.cascade(d, width=w),
                 "catalytic": lambda d, w=12: models.catalytic_cascade(d, width=w)}


def _hcase_name(fam, T, d):
    return f"{fam}_T{T:g}_d{d}"


def horizon_case(fam, T, d):
    """sqrt(|S| / S_eff) of one cascade at horizon T on the box grown by the leak rule (capped at the ceiling)."""
    t0 = time.time()
    mk = HORIZON_BUILD[fam]
    # the leak only grows with an added species, so the search can start at the width chosen for d - 1 (the same box
    # as a search from width 12, with fewer solves)
    prev = data(HORIZON_PART, _hcase_name(fam, T, d - 1)) if d > 1 else None
    m = mk(d, max(prev["widths"]) if prev else 12)
    if int(np.prod(m.widths)) > CEILING:
        # The cascades are feed-forward, so with d species the leak on any box is at least the leak of the first d - 1
        # species on the same widths. Every box of the schedule within the ceiling is narrower than the width the
        # (d - 1)-species case needed, so its leak exceeds the tolerance: the box is the largest within the ceiling.
        # Such a box would be truncated far beyond the tolerance and is left out of the slope (notes.md 11), so it is
        # recorded as over the size limit without a solve.
        need = max(prev["widths"])
        row = {"family": fam, "T": T, "d": d, "widths": None, "n_states": None, "leak": None, "capped": True,
               "means": None, "log10_readout_factor": None, "needs_width_at_least": need,
               "needs_states_at_least": need ** d, "note": "box over the size limit; not solved"}
        print(f"-- {fam} T={T:g} d={d}: needs width >= {need} ({need ** d:.2e} states): over the size limit",
              flush=True)
        save(HORIZON_PART, _hcase_name(fam, T, d), {"leak_tol": LEAK_TOL, "ceiling": CEILING}, row)
        return
    else:
        widths, leak, capped = capped_box(models.Model(**{**m.__dict__, "name": f"{fam}{d}_T{T:g}"}), T,
                                          part=f"{PART}_horizon_boxes")
    s = build_setup(m, widths, T, refs=None)
    pT = fsp.propagate(s.gen.A, s.p0, [T])[-1]
    mass = float(pT.sum())
    X = s.space.states
    row = {"family": fam, "T": T, "d": d, "widths": list(widths), "n_states": len(s.space), "leak": 1.0 - mass,
           "capped": capped, "means": [float(X[:, i] @ (pT / mass)) for i in range(d)],
           "log10_readout_factor": float(0.5 * np.log10(len(s.space) * np.sum(pT**2) / mass**2)),
           "seconds": time.time() - t0}
    print(f"-- {fam} T={T:g} d={d}: widths={widths} |S|={len(s.space):,} leak={1 - mass:.1e} capped={capped} "
          f"factor 10^{row['log10_readout_factor']:.3f} [{row['seconds']:.0f}s]", flush=True)
    save(HORIZON_PART, _hcase_name(fam, T, d), {"leak_tol": LEAK_TOL, "ceiling": CEILING}, row)


def horizons_summary():
    """Collect the cases into readout/horizons.json with the slopes and the pre-registered claim tests (notes.md 11)
    once every case is stored."""
    rows = [data(HORIZON_PART, _hcase_name(f, T, d)) for f in HORIZON_BUILD for T in HORIZONS for d in HORIZON_DS]
    if any(r is None for r in rows):
        print(f"horizons: {sum(r is None for r in rows)} cases still missing", flush=True)
        return
    slopes = {}
    for fam in HORIZON_BUILD:
        for T in HORIZONS:
            rr = sorted((r for r in rows if r["family"] == fam and r["T"] == T), key=lambda r: r["d"])
            rr = [r for r in rr if not r["capped"] and r["log10_readout_factor"] is not None]
            ds = [r["d"] for r in rr]
            sl, inc = _slope(ds, [r["log10_readout_factor"] for r in rr])
            r4 = [r for r in rr if r["d"] <= 4]  # the same range at every horizon (a diagnostic, not the test)
            slopes[f"{fam}|{T:g}"] = {"ds": ds, "slope": sl, "increments": inc,
                                      "slope_d1to4": _slope([r["d"] for r in r4],
                                                            [r["log10_readout_factor"] for r in r4])[0]}
    tests = {}
    for fam in HORIZON_BUILD:
        s1, s10 = slopes[f"{fam}|1"], slopes[f"{fam}|10"]
        tests[fam] = {"horizon_dependence": bool(s10["slope"] <= s1["slope"] - 0.1),
                      "exponential_at_T10": bool(s10["slope"] >= 0.05 and all(v > 0 for v in s10["increments"]))}
    save(PART, "horizons", {"horizons": HORIZONS, "ds": HORIZON_DS, "leak_tol": LEAK_TOL, "ceiling": CEILING},
         {"rows": rows, "slopes": slopes, "claim_tests": tests})
    print(json.dumps({"slopes": {k: round(v["slope"], 3) for k, v in slopes.items()}, "tests": tests}), flush=True)


def stage_horizons(only=None):
    """sqrt(|S| / S_eff) on the conversion and the catalytic cascade, d = 1..5, at T = 1, 3 and 10, each on the box
    grown by the leak rule at that horizon (as in stage_correlated; a box at the ceiling is marked capped). At T = 1
    the downstream species are nearly empty, so the growth per species depends on the horizon. Every case is stored
    on its own; the cases with up to four species run first. --only takes cases as family:T:d."""
    old = data(PART, "horizons")  # rows of the first, single-file version of this stage
    for r in (old or {}).get("rows", []):
        if data(HORIZON_PART, _hcase_name(r["family"], r["T"], r["d"])) is None:
            save(HORIZON_PART, _hcase_name(r["family"], r["T"], r["d"]), {"leak_tol": LEAK_TOL, "ceiling": CEILING}, r)
    cases = [(f, T, d) for d in HORIZON_DS for f in HORIZON_BUILD for T in HORIZONS]
    cases.sort(key=lambda c: c[2] == max(HORIZON_DS))
    for fam, T, d in cases:
        if only and f"{fam}:{T:g}:{d}" not in only:
            continue
        if only or data(HORIZON_PART, _hcase_name(fam, T, d)) is None:  # a named case is recomputed
            horizon_case(fam, T, d)
    horizons_summary()

CUT_T = (1.0, 3.0, 10.0)
CUT_DS = (1, 2, 3, 4, 5, 6)


def conversion_means(d, T, k=4.0):
    """Exact means of the conversion cascade (models.cascade, c = g = 1) at T from the floored initial means. The
    chain is monomolecular and starts from a product of Poisson laws, so its law stays a product of Poisson laws with
    these means (the rate equations)."""
    from scipy.linalg import expm
    A = np.zeros((d + 1, d + 1))
    for i in range(d):
        A[i, i] = -1.0
        if i:
            A[i, i - 1] = 1.0
    A[0, d] = k
    x = expm(A * T) @ np.r_[np.full(d, reference.MEAN_FLOOR), 1.0]
    return [float(v) for v in x[:d]]


def product_cut(means, widths, tol=CUT_TOL):
    """Full-box and cut-box factors (log10) of a product of Poisson laws on a product box. The cut follows
    fitted_factors: on every species, the smallest range whose two ends each hold at most tol / (2 d) of the law on the
    box. On a product law restricted to a product box the factor is the sum of one factor per species."""
    from scipy.stats import poisson
    d = len(means)
    tau = tol / (2 * d)
    q = [poisson.pmf(np.arange(w), m) for m, w in zip(means, widths)]
    mass = [float(qi.sum()) for qi in q]
    full = cut = 0.0
    bounds = []
    for i, qi in enumerate(q):
        marg = qi * float(np.prod([mass[j] for j in range(d) if j != i]))
        lo = int(np.nonzero(np.cumsum(marg) > tau)[0][0])
        hi = len(marg) - 1 - int(np.nonzero(np.cumsum(marg[::-1]) > tau)[0][0])
        qc = qi[lo:hi + 1]
        full += 0.5 * np.log10(len(qi) * np.sum(qi**2) / mass[i]**2)
        cut += 0.5 * np.log10(len(qc) * np.sum(qc**2) / qc.sum()**2)
        bounds.append([lo, hi])
    return float(full), float(cut), bounds


def stage_cut_horizons(only=None):
    """The cut-box factor of the conversion cascade at T = 1, 3 and 10, d = 1..6, from its closed-form law on the
    widest leak-rule box of each horizon (notes.md 13, S4). Two gates check the closed form against stored solves:
    (a) the cut box on the width-12 boxes at T = 1 against fitted_boxes, (b) the full box at T = 10 on the stored
    leak-rule widths against horizons. The claim test asks for a cut-box slope of at least 0.05 decades per species at
    T = 10 with every increment positive."""
    fitted = {r["instance"]: r for r in data(PART, "fitted_boxes")["rows"] if r["family"] == "conversion"}
    hz = {(r["T"], r["d"]): r for r in data(PART, "horizons")["rows"] if r["family"] == "conversion"}
    gap_a = {d: abs(product_cut(conversion_means(d, 1.0), (CASCADE_WIDTH,) * d)[1]
                    - fitted[d]["cut"]["log10_readout_factor"]) for d in CASCADE_DS}
    gap_b = {d: abs(product_cut(conversion_means(d, 10.0), tuple(hz[(10.0, d)]["widths"]))[0]
                    - hz[(10.0, d)]["log10_readout_factor"]) for d in HORIZON_DS}
    gates = {"a_max_gap": max(gap_a.values()), "b_max_gap": max(gap_b.values()), "tolerance": 0.01}
    gates["passed"] = bool(gates["a_max_gap"] <= 0.01 and gates["b_max_gap"] <= 0.01)
    rows, slopes = [], {}
    for T in CUT_T:
        w = max(max(r["widths"]) for (TT, _), r in hz.items() if TT == T)  # the widest leak-rule box at T
        rr = []
        for d in CUT_DS:
            means = conversion_means(d, T)
            full, cut, b = product_cut(means, (w,) * d)
            rr.append({"T": T, "d": d, "width": w, "means": means, "full_box": full, "cut": cut, "cut_bounds": b})
        sl, inc = _slope(CUT_DS, [r["cut"] for r in rr])
        slopes[f"{T:g}"] = {"slope": sl, "increments": inc}
        rows += rr
    s10 = slopes["10"]
    test = {"persists_at_T10": bool(s10["slope"] >= 0.05 and all(v > 0 for v in s10["increments"]))}
    save(PART, "cut_horizons", {"cut_tol": CUT_TOL, "horizons": CUT_T, "ds": CUT_DS, "law": "closed form"},
         {"rows": rows, "slopes": slopes, "gates": gates, "gap_a": gap_a, "gap_b": gap_b, "claim_test": test})
    print(json.dumps({"gates": gates, "slopes": {k: round(v["slope"], 3) for k, v in slopes.items()}, "test": test}),
          flush=True)


STAGES = {"horizons": stage_horizons, "cut_horizons": stage_cut_horizons,
          "signalling": stage_signalling, "fitted_boxes": stage_fitted_boxes, "lambda_boxes": stage_lambda_boxes,
          "fitted_gardner": stage_fitted_gardner,
          "cascades_matched": stage_cascades_matched, "swap_variants": stage_swap_variants, "correlated": stage_correlated, "initial_law": stage_initial_law, "sensitivity": stage_sensitivity,
          "cascade": stage_cascade, "lambda_phage": stage_lambda_phage, "cascade_summary": stage_cascade_summary,
          "published": stage_published, "swap_test": stage_swap_test, "concept": stage_concept}

if __name__ == "__main__":
    args = sys.argv[1:]
    only = args[args.index("--only") + 1].split(",") if "--only" in args else None
    for a in [a for a in args if a in STAGES]:
        STAGES[a](only=only)
