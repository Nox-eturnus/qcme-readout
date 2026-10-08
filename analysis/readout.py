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
import sys
import time

import numpy as np

from .common import (FAMILIES, best_of, build_setup, certificate_table, data, is_huge, is_large, log10,
                     log_norm_weighted, observables, published_setup, save, schedules, settings, worst_case)
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
    (results/published_models/<instance>.json). Instances too large for one direct solve are not recomputed:
    lambda-phage is read from readout/lambda_phage.json (same box and horizon), and the Gardner toggle is skipped."""
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
        if is_huge(s):
            rows.append({"row": m.name, "skipped": f"{len(s.space):,} states: too large for one direct solve"})
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
                     "shots_for_1pct": float(shots_1pct),
                     "sqrt_S_over_Seff": float(np.sqrt(p.size * np.sum(p**2)))})  # p sums to 1, S_eff = 1/sum p^2
    ratios = [rows[i + 1]["shots_for_1pct"] / rows[i]["shots_for_1pct"] for i in range(3)]
    save(PART, "swap_test", {"width": width, "sigma": sigma},
         {"source": "analytic evaluation (analysis/readout.py, swap_test)", "width": width, "sigma": sigma,
          "rows": rows, "shots_ratio_per_asset": ratios})


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


STAGES = {"cascade": stage_cascade, "lambda_phage": stage_lambda_phage, "cascade_summary": stage_cascade_summary,
          "published": stage_published, "swap_test": stage_swap_test, "concept": stage_concept}

if __name__ == "__main__":
    args = sys.argv[1:]
    only = args[args.index("--only") + 1].split(",") if "--only" in args else None
    for a in [a for a in args if a in STAGES]:
        STAGES[a](only=only)
