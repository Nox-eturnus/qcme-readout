"""Check every number quoted in the article and the Supplementary Information against the stored results.

    python -m analysis.check_numbers          -> outputs/number_check.md; exit code 1 if any number disagrees

Each entry gives the statement, the value as printed in the article, a function that reads it from results/, and the
rounding rule used to compare them.
"""
import math
import sys

from .common import ROOT, data, num

WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()
INSTANCES = ("schlogl", "toggle_hegland", "toggle_kuntz", "toggle_gardner", "feedback_b1", "feedback_b10",
             "feedback_b100", "lambda_phage", "toggle_tang", "autoreg_tang_1", "autoreg_tang_2", "autoreg_tang_3")


def d(part, name):
    return data(part, name)


def summary_row(m):
    return d("published_models", "summary")["rows"][m]


def cost_row(m, enc, obs, growth, variant="single", kernel="K2(c=2)"):
    return next(r for r in d("costs", m)["costs"][enc][obs]["rows"]
                if r["growth"] == growth and r["kernel"] == kernel and r["variant"] == variant and r["eps"] == 0.01)


def cost_readout(m, enc, growth="W"):
    """log10 Gamma R_f for f = 1 (primary solver: LCHS with the Low-Somma kernel, c = 2)."""
    return cost_row(m, enc, "one", growth)["log10_gamma"] + math.log10(d("costs", m)["costs"][enc]["one"]["R_f"])


def readout_rows(max_species=2):
    rows = [r for r in d("readout", "published_summary")["published_rows"] if "log10_readout_factor" in r]
    return [r for r in rows if r["n_count_species"] <= max_species]


def nn_largest():
    """Instances where non-normality is the largest looseness term (with the W split of the large boxes)."""
    n = 0
    for m in INSTANCES:
        terms = dict(d("published_models", m)["decomposition"])
        w = summary_row(m)["W_split"]
        if w:
            terms.update({k: w[k] for k in ("window_penalty", "non_normality", "instance_slack")})
        t = {k: num(terms[k]) for k in ("window_penalty", "non_normality", "instance_slack")}
        n += max(t, key=t.get) == "non_normality"
    return n


def exact_share():
    o = d("certificates", "toggle_exact")
    return 2 * o["schedules"][o["best"]["schedule"]]["sum_log_rho"] / o["best"]["K"]


def lit():
    return d("literature", "coding")


TOGGLE_SPLIT = "references/R-mix(NB,E1)/decomposition/uniform J=80/resolvent/"
NONREVERSIBLE = tuple(m for m in INSTANCES if m != "schlogl")
COSTED = ("toggle", "schlogl", "michaelis_menten", "self_repressor", "birth_death", "cascade1", "cascade2", "cascade3",
          "cascade4")
TOFFOLI_RATE = 1e6


def tail_rows():
    return [r for r in d("costs", "comparison")["table"] if r["observable"] == "tail"]


def splits():
    from .figures import cost_split
    return [cost_split(m) for m in COSTED]


def ssa_tail(m):
    return next(r for r in d("costs", f"{m}_baselines")["rows"]
                if r["kind"] == "ssa" and r["method"] == "uniformization" and r["observable"] == "tail")


def kernel_spread():
    """Largest log10 ratio of another kernel to the kernel used throughout (growth W, single oracle, eps = 0.01)."""
    worst = 0.0
    for m in COSTED:
        for enc, by_obs in d("costs", m)["costs"].items():
            for obs in by_obs.values():
                tot = {r["kernel"]: r["toffoli_total"] for r in obs["rows"]
                       if r["growth"] == "W" and r["variant"] == "single" and r["eps"] == 0.01}
                worst = max(worst, max(math.log10(v / tot["K2(c=2)"]) for v in tot.values()))
                assert min(tot.values()) == tot["K2(c=2)"]
    return worst


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"{m}e{int(e)}"


def path(part, name, p):
    x = d(part, name)
    for k in p.split("/"):
        x = x[int(k)] if isinstance(x, list) else x[k]
    return x


# (statement in the article, printed value, function, rule)
KNOWN = ("autoreg_tang_1", "autoreg_tang_2", "autoreg_tang_3", "feedback_b1", "feedback_b10", "feedback_b100")


def wf():
    return d("published_models", "weighted_factors")["rows"]


def wfr(m):
    return next(r for r in wf() if r["row"] == m)


def st(m):
    return d("published_models", f"{m}_stationary_known")


def swv(kind):
    v = d("readout", "swap_variants")
    return next(x for k, x in v.items() if k.startswith("grid fitted" if kind == "fitted" else "correlated"))


def lind_minus_enc(m):
    """log10 of the best Lindblad total over the cheapest amplitude encoding (cheaper oracle design), tail."""
    lind = next(r for r in d("costs", "comparison")["table"] if r["model"] == m and r["observable"] == "tail")
    enc = min(min(r["total_split"], r["total_single"]) for r in d("costs", "split_design")["rows"]
              if r["model"] == m and r["observable"] == "tail" and r["growth"] == "W"
              and r["encoding"] in ("uniform", "R-NB", "R-mix(NB,E1)"))
    return math.log10(lind["lindblad_best_total"]) - math.log10(enc)


def pu(m):
    from .tables import per_use_split
    return per_use_split(m)


def sg(tag, k):
    return d("readout", f"signalling_{tag}{k}")


def fb():
    return d("readout", "fitted_boxes")


def hz(fam, T):
    return d("readout", "horizons")["slopes"][f"{fam}|{T}"]["slope"]


def hz_means():
    return next(r for r in d("readout", "horizons")["rows"]
                if r["family"] == "conversion" and r["T"] == 1.0 and r["d"] == 5)["means"]


def ssa_times():
    return [r["seconds_for_needed"] for r in d("costs", "ssa_time")["rows"]]


def ae_over():
    """Coherent trajectories of canonical amplitude estimation over 1/(eps sqrt(P)), mean over the nine networks."""
    from .figures import COSTED, ssa_row
    v = [ssa_row(m)["uses"] * 1e-2 * math.sqrt(ssa_row(m)["P"]) for m, _ in COSTED]
    return sum(v) / len(v)


def s18_minus_s17():
    mu = {r["d"]: r["per_use_ratio"] for r in d("costs", "per_use_scaling")["rows"]}
    return [mu[k] - pu(f"cascade{k}")["per_use_ratio"] for k in (1, 2, 3, 4)]


def s18_walk_over_s17():
    mu = {r["d"]: r["walk_steps"] for r in d("costs", "per_use_scaling")["rows"]}
    return [mu[k] / pu(f"cascade{k}")["walk_steps"] for k in (1, 2, 3, 4)]


def s11_identity_gap():
    """Largest |K / (2 ln 10) - (window + non-normality + slack + log10 measured growth)| over Table S11."""
    gaps = []
    for m in INSTANCES:
        dd = d("published_models", m)
        row = summary_row(m)
        terms = dict(dd["decomposition"])
        rn = dd["best"]["reference"]
        meas = num(dd["references"][rn]["measured_log10_growth_continuous"])
        if row["W_split"]:
            terms.update({k: row["W_split"][k] for k in ("window_penalty", "non_normality", "instance_slack")})
            meas = row["W_split"]["log10_W"] - row["W_split"]["instance_slack"]
        gaps.append(abs(num(dd["best"]["K"]) / (2 * math.log(10)) - (terms["window_penalty"] + terms["non_normality"]
                                                                     + terms["instance_slack"] + meas)))
    return max(gaps)


ENTRIES = [
    ("Readout factor on the cascade, one to six species (Results, readout barrier)", "0.17 -> 2.45",
     lambda: f"{path('readout', 'cascade_1', 'encodings/uniform/log10_readout/meas/one'):.2f} -> "
             f"{path('readout', 'cascade_6', 'encodings/uniform/log10_readout/meas/one'):.2f}", ("eq",)),
    ("Readout factor on lambda-phage, log10 (Results, readout barrier)", 2.18,
     lambda: path("readout", "lambda_phage", "encodings/uniform/log10_readout/meas/one"), ("dec", 2)),
    ("Smallest readout factor, published one- and two-species instances (Results, readout barrier)", "10^0.34",
     lambda: "10^{:.2f}".format(min(r["log10_readout_factor"] for r in readout_rows())), ("eq",)),
    ("Largest readout factor, published one- and two-species instances (Results, readout barrier)", "10^1.55",
     lambda: "10^{:.2f}".format(max(r["log10_readout_factor"] for r in readout_rows())), ("eq",)),
    ("Published one- and two-species instances computed (Results, readout barrier)", "eleven", lambda: WORDS[len(readout_rows())], ("eq",)),
    ("Growth per species of the bare factor sqrt(|S|/S_eff) on the cascade, decades (Results, readout barrier)", 0.47,
     lambda: path("readout", "cascade_summary", "slopes/uniform|meas|one/slope"), ("dec", 2)),
    ("Growth per species of the tail-probability factor on the cascade, decades (SI S2.2)", 0.70,
     lambda: path("readout", "cascade_summary", "slopes/uniform|meas|tail/slope"), ("dec", 2)),
    ("Leaked probability of lambda-phage, the instance at the 10^7-state limit (SI S3.1; SI S6)", 7.7e-5,
     lambda: d("published_models", "lambda_phage")["leak"], ("sig", 2)),
    ("Leaked probability of the cascade at width 12 (Methods, models)", 3.0e-5,
     lambda: max(d("readout", f"cascade_{k}")["leak"] for k in range(1, 7)), ("sig", 2)),
    ("Truncation sensitivity: largest spread on the published instances, log10 (Methods, models; Table D10)", 0.15,
     lambda: d("readout", "sensitivity")["max_spread_published"], ("dec", 2)),
    ("Truncation sensitivity: published instances tested (Methods, models; SI S6)", "eight",
     lambda: WORDS[len(d("readout", "sensitivity")["published"])], ("eq",)),
    ("Truncation sensitivity: cascade growth per species over box widths 10 to 20, decades (Methods, models; Table D11)",
     "0.36 to 0.51", lambda: "{:.2f} to {:.2f}".format(*(f(c["slope"] for c in d("readout", "sensitivity")["cascade"].values())
                                                          for f in (min, max))), ("eq",)),
    ("Catalytic cascade: readout factor with one and with five species, log10 (Results, readout barrier)", "0.22 -> 2.28",
     lambda: "{:.2f} -> {:.2f}".format(d("readout", "correlated_cascade")["rows"][0]["log10_readout_factor"],
                                       d("readout", "correlated_cascade")["rows"][-1]["log10_readout_factor"]), ("eq",)),
    ("Catalytic cascade: growth per species, decades (Results, readout barrier; SI S3.4)", 0.52,
     lambda: d("readout", "correlated_cascade")["slope"], ("dec", 2)),
    ("Catalytic cascade: correlation of the last two species (Results, readout barrier; SI S3.4)", "0.41 to 0.47",
     lambda: "{:.2f} to {:.2f}".format(*(f(r["correlation_last_two"] for r in d("readout", "correlated_cascade")["rows"][1:])
                                         for f in (min, max))), ("eq",)),
    ("Catalytic cascade: S_eff over the product of the marginal values, d = 2 to 5 (SI S3.4)", "0.61 to 0.88",
     lambda: "{:.2f} to {:.2f}".format(*(f(r["S_eff_over_product_of_marginals"]
                                           for r in d("readout", "correlated_cascade")["rows"][1:]) for f in (min, max))),
     ("eq",)),
    ("Initial law: largest change of the readout factor on eight instances, log10 (Results, readout barrier; SI S6)", 0.48,
     lambda: d("readout", "initial_law")["max_spread"], ("dec", 2)),
    ("Initial law: largest change for a point mass, log10 (SI S6)", 0.03,
     lambda: max(abs(r["by_initial_law"]["point mass at the initial means"]["log10_readout_factor"]
                     - r["by_initial_law"]["poisson at the initial means"]["log10_readout_factor"])
                 for r in d("readout", "initial_law")["rows"]), ("dec", 2)),
    ("Conversion cascade, width 12: growth per species over d = 1 to 4, decades (SI S6)", 0.40,
     lambda: d("readout", "sensitivity")["cascade"]["12"]["slope"], ("dec", 2)),
    ("States of the Gardner toggle switch, millions (SI S4.5)", 3.66,
     lambda: d("published_models", "toggle_gardner")["n_states"] / 1e6, ("sig", 3)),
    ("Records from the literature searches (Methods, literature audit; SI S1.1)", 2335, lambda: lit()["screen"]["logged_hits"], ("int",)),
    ("Records after the title step (SI S1.1)", 176, lambda: lit()["screen"]["after_mechanical_title_step"], ("int",)),
    ("Works included (Abstract; Results, audit)", 14, lambda: lit()["n_included"], ("int",)),
    ("Accounted / omitted, bounded / omitted, growing / the same, conditional / unbounded parameter (Results, audit)",
     "7/2/1/3/1",
     lambda: "/".join(str(sum(e["verdict"] == v for e in lit()["included"]))
                      for v in ("accounted", "omitted, bounded", "omitted, growing", "omitted, growing (conditional)",
                                "unbounded parameter, growing")), ("eq",)),
    ("Works that claim an advantage in a variable in which the cost grows, without bounding it (Abstract)", 5,
     lambda: sum("growing" in e["verdict"] for e in lit()["included"]), ("int",)),
    ("Included works found by the searches / taken from works already cited (SI S1.1)", "11/3",
     lambda: "{}/{}".format(sum(e["entry_path"] == "search" for e in lit()["included"]),
                            sum(e["entry_path"] == "already cited" for e in lit()["included"])), ("eq",)),
    ("Word stems of the title filter (SI S1.1)", 25, lambda: len(lit()["screen"]["title_stems"]), ("int",)),
    ("Accounted works whose encoding removes the cost (Results, audit)", 5,
     lambda: sum(e["verdict"] == "accounted" and (e["encoding"].startswith("weighted") or e["encoding"].startswith("sqrt"))
                 for e in lit()["included"]), ("int",)),
    ("Swap-test shots for 1% relative error, one asset (Results, audit)", 5.1e7,
     lambda: path("readout", "swap_test", "rows/0/shots_for_1pct"), ("sig", 2)),
    ("Swap-test shots for 1% relative error, four assets (Results, audit)", 7.2e10,
     lambda: path("readout", "swap_test", "rows/3/shots_for_1pct"), ("sig", 2)),
    ("Swap-test shots, growth per added asset (SI S3.2)", "10 to 13",
     lambda: "{:.0f} to {:.0f}".format(min(d("readout", "swap_test")["shots_ratio_per_asset"]),
                                       max(d("readout", "swap_test")["shots_ratio_per_asset"])), ("eq",)),
    ("Best K_T on the toggle switch, twelve references (Results, reweighting)", 36.81,
     lambda: d("certificates", "toggle_best")["K_gate"], ("sig", 4)),
    ("K_T with the exact solution as the weight, toggle switch (Results, reweighting)", 217,
     lambda: d("certificates", "toggle_exact")["best"]["K"], ("sig", 3)),
    ("Share of the exact-reference K_T from the window penalty (Results, reweighting)", 0.92, exact_share, ("dec", 2)),
    ("Non-normal transient growth on the toggle, log10 (Results, reweighting)", 3.42,
     lambda: path("certificates", "toggle_references", TOGGLE_SPLIT + "non_normality"), ("dec", 2)),
    ("Actual excess growth (instance slack) on the toggle, log10 (Results, reweighting)", 0.44,
     lambda: path("certificates", "toggle_references", TOGGLE_SPLIT + "instance_slack"), ("dec", 2)),
    ("Best cheap K_T over the eleven non-reversible published instances (Results, reweighting)", "27.1 to 2.07e+04",
     lambda: "{:.3g} to {:.3g}".format(min(num(summary_row(m)["cheap_K"]) for m in NONREVERSIBLE),
                                       max(num(summary_row(m)["cheap_K"]) for m in NONREVERSIBLE)), ("eq",)),
    ("Smallest finite exact-reference K_T over the non-reversible instances (Results, reweighting)", 86.6,
     lambda: min(num(summary_row(m)["exact_K"]) for m in NONREVERSIBLE
                 if math.isfinite(num(summary_row(m)["exact_K"]))), ("sig", 3)),
    ("Non-reversible instances where non-normality is the largest looseness term (SI S4.4)", "eight",
     lambda: WORDS[nn_largest()], ("eq",)),
    ("Schlogl: non-normality is not the largest looseness term (SI S4.4)", "window_penalty",
     lambda: max(("window_penalty", "non_normality", "instance_slack"),
                 key=lambda k: num(d("published_models", "schlogl")["decomposition"][k])), ("eq",)),
    ("Schlogl, stationary law as the reference: K_T (Results, reweighting; SI S4.5)", 0,
     lambda: round(d("published_models", "schlogl_stationary")["best_K"], 6), ("int",)),
    ("Schlogl, stationary law: computed K_T, rounding level (SI S4.5)", 2e-11,
     lambda: d("published_models", "schlogl_stationary")["best_K"], ("sig", 1)),
    ("Schlogl, stationary law: certified readout factor, log10 (SI S4.5)", 1.30,
     lambda: d("published_models", "schlogl_stationary")["log10_readout_cert"], ("dec", 2)),
    ("Schlogl: certified readout factor of the uniform encoding, with normalization, log10 (SI S4.5)", 3.89,
     lambda: cost_readout("schlogl", "uniform", "cert"), ("dec", 2)),
    ("Schlogl: stationary weight below the uniform encoding on certified terms, decades (SI S4.5)", 2.6,
     lambda: cost_readout("schlogl", "uniform", "cert") - d("published_models", "schlogl_stationary")["log10_readout_cert"],
     ("dec", 1)),
    ("Schlogl: measured readout factor of the uniform encoding in the cost study, log10 (SI S4.5)", 0.34,
     lambda: cost_readout("schlogl", "uniform", "meas"), ("dec", 2)),
    ("Gardner toggle: exponent of the certificate that the three looseness terms split (Table S7)", 3.19e4,
     lambda: d("published_models", "toggle_gardner")["best"]["K"], ("sig", 3)),
    ("Measured growth in Table S7, smallest and largest, log10 (Table S7)", "0.01 to 3.8",
     lambda: "{:.2f} to {:.1f}".format(*(f(num((d("published_models", f"{m}_worst_case") or
                                                d("published_models", m)["references"][d("published_models", m)["best"]["reference"]])
                                               ["measured_log10_growth_continuous"]) for m in INSTANCES) for f in (min, max))),
     ("eq",)),
    ("Schlogl, stationary law: measured readout factor, log10 (SI S4.5)", 0.91,
     lambda: d("published_models", "schlogl_stationary")["log10_readout_meas"], ("dec", 2)),
    ("Schlogl, best cheap reference K_T (Table S7)", 12.6,
     lambda: summary_row("schlogl")["cheap_K"], ("sig", 3)),
    ("Exact reference on the toggle switch: limit of fine windows, resolvent weight (SI S4.4)", 215.6,
     lambda: d("certificates", "toggle_exact")["riemann_limit"]["resolvent"], ("dec", 1)),
    ("Exact reference on the toggle switch: K_T at 80 windows (SI S4.4)", 216.8,
     lambda: d("certificates", "toggle_exact")["best"]["K"], ("dec", 1)),
    ("Certified growth per species, best weighted reference, decades (Results, reweighting)", 0.85,
     lambda: path("readout", "cascade_summary", "slopes/R-mix(NB,E1)|cert|one/slope"), ("dec", 2)),
    ("Certified growth per species, uniform encoding, decades (Results, reweighting)", 0.65,
     lambda: path("readout", "cascade_summary", "slopes/uniform|cert|one/slope"), ("dec", 2)),
    ("Measured growth per species of the weighted state, decades (Table D7)", 0.02,
     lambda: path("readout", "cascade_summary", "slopes/R-P|W|one/slope"), ("dec", 2)),
    ("Smallest ratio to coherent SSA, cheaper oracle design (Results, trajectory sampling; SI S5)", 167,
     lambda: min(v["min_ratio_best_design"] for v in d("costs", "split_design")["cheaper_design"].values()), ("sig", 3)),
    ("Smallest ratio to coherent SSA, single-oracle design (Results, trajectory sampling)", 372,
     lambda: d("costs", "comparison")["verdict"]["R-NB|tail"]["min_ratio"], ("sig", 3)),
    ("Lindblad over coherent SSA, tail probability, nine networks (Results, trajectory sampling)", "123 to 12462",
     lambda: "{:.0f} to {:.0f}".format(min(r["lindblad_over_ssa"] for r in tail_rows()),
                                       max(r["lindblad_over_ssa"] for r in tail_rows())), ("eq",)),
    ("Networks counted gate by gate (Results, trajectory sampling)", "nine", lambda: WORDS[len(tail_rows())], ("eq",)),
    ("Encoding and network of the smallest certified ratio (Results, trajectory sampling; SI S5)", "uniform on cascade1",
     lambda: "{1} on {2}".format(*min((r[k], k.split("|")[0], r["model"]) for r in tail_rows() for k in r
                                      if k.endswith("|cert|ratio_to_ssa"))), ("eq",)),
    ("Smallest certified ratio of the mixture-weighted encoding, tail, single oracle (SI S5)", 5.5e6,
     lambda: min(r["R-mix(NB,E1)|cert|ratio_to_ssa"] for r in tail_rows()), ("sig", 2)),
    ("Counted tail probabilities, smallest and largest (Discussion; SI S5)", "0.003 to 0.05",
     lambda: "{:.1g} to {:.1g}".format(min(ssa_tail(m)["P"] for m in COSTED), max(ssa_tail(m)["P"] for m in COSTED)),
     ("eq",)),
    ("Coherent SSA uses on the tail probability, log10 (Methods, resource counts; SI S5)", "5.4 to 6.1",
     lambda: "{:.1f} to {:.1f}".format(*(f(math.log10(ssa_tail(m)["uses"]) for m in COSTED) for f in (min, max))), ("eq",)),
    ("Classical trajectories over coherent SSA uses (Methods, resource counts)", "4.8 to 21",
     lambda: "{:.1f} to {:.0f}".format(*(f(d("costs", m)["classical_ssa_samples"]["tail"] / ssa_tail(m)["uses"]
                                           for m in COSTED) for f in (min, max))), ("eq",)),
    ("1/(eps sqrt(P)) for the counted tail probabilities (SI S5)", "450 to 1900",
     lambda: "{} to {}".format(*(format(float(format(f(1 / (0.01 * math.sqrt(ssa_tail(m)["P"])) for m in COSTED), ".2g")), ".0f")
                                 for f in (min, max))), ("eq",)),
    ("Smallest ratio to coherent SSA with certified growth, tail, single oracle (Results, trajectory sampling; SI S5)", 2.4e5,
     lambda: min(r[k] for r in tail_rows() for k in r if k.endswith("|cert|ratio_to_ssa")), ("sig", 2)),
    ("Uses of the propagator against uses of a coherent trajectory: largest gap on eight networks, log10 (Results, trajectory sampling)",
     0.4, lambda: sorted(abs(math.log10(c["enc_uses"] / c["ssa_uses"])) for c in splits())[-2], ("dec", 1)),
    ("Toffolis per use, cheapest encoding over coherent SSA, log10 (Results, trajectory sampling)", "2.1 to 3.8",
     lambda: "{:.1f} to {:.1f}".format(*(f(math.log10(c["enc_per_use"] / c["ssa_per_use"]) for c in splits())
                                         for f in (min, max))), ("eq",)),
    ("Toggle: extra uses of the cheapest (uniform) encoding, log10 (Results, trajectory sampling)", 2.3,
     lambda: max(math.log10(c["enc_uses"] / c["ssa_uses"]) for c in splits()), ("dec", 1)),
    ("Toggle: the cheapest encoding is the uniform one (Results, trajectory sampling)", "uniform",
     lambda: splits()[0]["encoding"], ("eq",)),
    ("Finite state projection, longest solve of a counted network, seconds (Results, trajectory sampling)", "under one second",
     lambda: "under one second" if max(d("costs", m)["fsp_wall_time_s"]["median"] for m in COSTED) < 1 else "longer",
     ("eq",)),
    ("Classical SSA trajectories for the tail probability (Results, trajectory sampling)", "1.3e6 to 2.4e7",
     lambda: "{} to {}".format(sci(min(d("costs", m)["classical_ssa_samples"]["tail"] for m in COSTED)),
                               sci(max(d("costs", m)["classical_ssa_samples"]["tail"] for m in COSTED))), ("eq",)),
    ("Coherent SSA Toffolis for the tail probability, log10 (Results, trajectory sampling)", "9.4 to 12.4",
     lambda: "{:.1f} to {:.1f}".format(*(f(math.log10(ssa_tail(m)["total"]) for m in COSTED) for f in (min, max))),
     ("eq",)),
    ("Coherent SSA run time at 1e6 Toffolis per second: minutes, shortest (Results, trajectory sampling)", 43,
     lambda: min(ssa_tail(m)["total"] for m in COSTED) / TOFFOLI_RATE / 60, ("int_round",)),
    ("Coherent SSA run time at 1e6 Toffolis per second: days, longest (Results, trajectory sampling)", 32,
     lambda: max(ssa_tail(m)["total"] for m in COSTED) / TOFFOLI_RATE / 86400, ("int_round",)),
    ("Other propagation kernels: largest extra cost, log10 (Methods, resource counts)", 1.2, kernel_spread, ("dec", 1)),
    ("Gate-level checks: circuit sizes in qubits (Methods, resource counts)", "7 to 12",
     lambda: "{} to {}".format(min(r["qubits"] for r in d("costs", "circuit_checks")["I8_I9"]),
                               max(r["qubits"] for r in d("costs", "circuit_checks")["I8_I9"])), ("eq",)),
    ("Gate-level checks: block-encoding error below 1e-10 (Methods, resource counts)", "yes",
     lambda: "yes" if max(r["I8"] for r in d("costs", "circuit_checks")["I8_I9"]) < 1e-10 else "no", ("eq",)),
    ("Observable-matched trajectory steps for the total probability: none (SI S5)", 0,
     lambda: max(r["N_aware"] for r in d("costs", "observable_aware_ssa")["rows"] if r["observable"] == "one"), ("int",)),
    # ---- added in the third revision
    ("Weighted encoding, measured: smaller than uniform on this many published instances (Results, reweighting)", "nine",
     lambda: WORDS[sum(r["best_meas"] < r["uniform"] for r in wf())], ("eq",)),
    ("Weighted encoding, measured: larger than uniform on the toggles, log10 excess (Results, reweighting)", "0.03 to 2.3",
     lambda: "{:.2f} to {:.1f}".format(*(f(r["best_meas"] - r["uniform"] for r in wf() if r["best_meas"] > r["uniform"])
                                         for f in (min, max))), ("eq",)),
    ("Weighted encoding, measured, toggles raised: Hegland, Kuntz, Gardner excess (SI S4.6)", "0.03, 0.11, 2.3",
     lambda: "{:.2f}, {:.2f}, {:.1f}".format(*(wfr(m)["best_meas"] - wfr(m)["uniform"]
                                               for m in ("toggle_hegland", "toggle_kuntz", "toggle_gardner"))), ("eq",)),
    ("Weighted encoding, measured, bursty feedback <b> = 100, log10 (Results, reweighting)", 0.18,
     lambda: wfr("feedback_b100")["best_meas"], ("dec", 2)),
    ("Stationary law as the reference: instances with K_T = 0 (Results, reweighting)", "seven",
     lambda: WORDS[1 + sum(abs(st(m)["best_K"]) < 0.05 for m in KNOWN)], ("eq",)),
    ("Stationary law as the reference: computed K_T range (SI S4.5)", "-0.02 to 3e-14",
     lambda: "{:.2f} to {:.0e}".format(min(st(m)["best_K"] for m in KNOWN), max(st(m)["best_K"] for m in KNOWN))
     , ("eq",)),
    ("Stationary law: certified readout factor range, log10 (Results, reweighting)", "0.23 to 2.16",
     lambda: "{:.2f} to {:.2f}".format(*(f(st(m)["log10_readout_cert"] for m in KNOWN) for f in (min, max))), ("eq",)),
    ("Stationary law: certified factor below the measured uniform factor (Results, reweighting)", "four",
     lambda: WORDS[sum(st(m)["log10_readout_cert"] < st(m)["log10_uniform_factor"] for m in KNOWN)
                   + (d("published_models", "schlogl_stationary")["log10_readout_cert"] < cost_readout("schlogl", "uniform", "meas"))],
     ("eq",)),
    ("Stationary law: certified uniform factor on the burst-with-feedback instances, log10 (SI S4.5)", "17 to 697",
     lambda: "{:.0f} to {:.0f}".format(*(f(st(m)["log10_uniform_cert"] for m in KNOWN if m.startswith("feedback"))
                                         for f in (min, max))), ("eq",)),
    ("Factor at the stationary law, six stationary-source instances, log10 (Methods, models; SI S4.5)", "0.25 to 2.20",
     lambda: "{:.2f} to {:.2f}".format(*(f(st(m)["log10_readout_factor_at_stationary_law"]
                                         for m in ("schlogl", "toggle_kuntz", "toggle_gardner", "feedback_b1",
                                                   "feedback_b10", "feedback_b100")) for f in (min, max))), ("eq",)),
    ("Gardner toggle: stationary mass in the outer tenth of its box (SI S4.5)", 8e-7,
     lambda: st("toggle_gardner")["mass_outer_layer_wide_box"], ("sig", 1)),
    ("Catalytic cascade: growth per species, conversion cascade on the same boxes (Results, readout barrier)", 0.49,
     lambda: d("readout", "cascades_matched")["slopes"]["conversion_uniform"], ("dec", 2)),
    ("Catalytic cascade: correlation adds at five species, decades (SI S3.4)", 0.13,
     lambda: d("readout", "cascades_matched")["rows"][-1]["catalytic_uniform"]
     - d("readout", "cascades_matched")["rows"][-1]["conversion_uniform"], ("dec", 2)),
    ("Catalytic cascade: measured weighted factor, growth per species (Table S5)", 0.06,
     lambda: d("readout", "cascades_matched")["slopes"]["catalytic_weighted_smallest"], ("dec", 2)),
    ("Catalytic cascade: measured weighted factor at five species, log10 (Results, reweighting)", 0.24,
     lambda: max(r["catalytic_weighted_smallest"] for r in d("readout", "cascades_matched")["rows"]), ("dec", 2)),
    ("Swap test on a fitted grid: readout factor per asset (Methods, swap test; SI S3.2)", 1.5,
     lambda: sum(swv("fitted")["readout_factor_ratio_per_asset"]) / 3, ("dec", 1)),
    ("Swap test on a fitted grid: shots per added asset (Results, audit)", "4.7 to 5.6",
     lambda: "{:.1f} to {:.1f}".format(min(swv("fitted")["shots_ratio_per_asset"]),
                                       max(swv("fitted")["shots_ratio_per_asset"])), ("eq",)),
    ("Swap test with correlated assets: shots per added asset (Methods, swap test; SI S3.2)", "6.7 to 9.4",
     lambda: "{:.1f} to {:.1f}".format(min(swv("correlated")["shots_ratio_per_asset"]),
                                       max(swv("correlated")["shots_ratio_per_asset"])), ("eq",)),
    ("Swap test with correlated assets: readout factor per asset, about (SI S3.2)", 2,
     lambda: sum(swv("correlated")["readout_factor_ratio_per_asset"]) / 3, ("int_round",)),
    ("Lindblad cheaper than the cheapest amplitude encoding on this many networks (Results, trajectory sampling)", "seven",
     lambda: WORDS[sum(lind_minus_enc(m) < 0 for m in COSTED)], ("eq",)),
    ("Lindblad below the cheapest encoding: smallest and largest margin, log10 (SI S5)", "0.04 to 2.0",
     lambda: "{:.2f} to {:.1f}".format(*(f(-lind_minus_enc(m) for m in COSTED if lind_minus_enc(m) < 0)
                                         for f in (min, max))), ("eq",)),
    ("Per use: walk steps over trajectory steps, log10 (SI S5)", "1.4 to 1.7",
     lambda: "{:.1f} to {:.1f}".format(*(f(pu(m)["steps_ratio"] for m in COSTED) for f in (min, max))), ("eq",)),
    ("Per use: uniformization rate times horizon (SI S5)", "18 to 6,500",
     lambda: "{:.0f} to {:,.0f}".format(min(pu(m)["LT"] for m in COSTED), round(max(pu(m)["LT"] for m in COSTED), -2)),
     ("eq",)),
    ("Per use: cost of a walk step over a trajectory step, log10 (SI S5)", "1.0 to 2.1",
     lambda: "{:.1f} to {:.1f}".format(*(f(pu(m)["per_step_ratio"] for m in COSTED) for f in (min, max))), ("eq",)),
    ("Per use on the cascade: cost per step ratio, one and four species, log10 (Results, trajectory sampling)", "1.4 and 2.1",
     lambda: "{:.1f} and {:.1f}".format(pu("cascade1")["per_step_ratio"], pu("cascade4")["per_step_ratio"]), ("eq",)),
    ("Per use: steps ratio + cost per step ratio = cost per use ratio on every network (Table S12)", "yes",
     lambda: "yes" if all(abs(pu(m)["steps_ratio"] + pu(m)["per_step_ratio"] - pu(m)["per_use_ratio"]) < 1e-9
                          for m in COSTED) else "no", ("eq",)),
    ("Per-use scaling: structure-only counts equal the full pipeline for d = 1..4 (SI S5)", "yes",
     lambda: "yes" if all(v["pass"] for v in d("costs", "per_use_scaling")["validation"]) else "no", ("eq",)),
    ("Per-use scaling: rise of the per-use ratio from d = 4 to 12, decades (Results, trajectory sampling)", 0.08,
     lambda: d("costs", "per_use_scaling")["claim_test"]["rise_d4_to_d12"], ("dec", 2)),
    ("Per-use scaling: level of the per-use ratio for d >= 8, log10 (SI S5)", 4.1,
     lambda: (lambda v: sum(v) / len(v))([r["per_use_ratio"] for r in d("costs", "per_use_scaling")["rows"]
                                          if r["d"] >= 8]), ("dec", 1)),
    ("Per-use scaling: Toffolis added per query by the last species, about (SI S5)", 20000,
     lambda: round(d("costs", "per_use_scaling")["claim_test"]["query_increment_last"], -3), ("int",)),
    ("Per-use scaling: Toffolis added per trajectory step by the last species (SI S5)", 180,
     lambda: d("costs", "per_use_scaling")["claim_test"]["ssa_step_increment_last"], ("int",)),
    ("eps = 1e-3: smallest ratio of the cheapest encoding to coherent SSA (Results, trajectory sampling)", 210,
     lambda: d("costs", "eps_comparison")["claim_test"]["min_ratio_encoding"], ("int_round",)),
    ("eps = 1e-3: largest ratio of the cheapest encoding to coherent SSA (SI S5)", 4.3e4,
     lambda: max(r["ratio_cheapest_to_ssa"] for r in d("costs", "eps_comparison")["rows"]), ("sig", 2)),
    ("eps = 1e-3: Lindblad over coherent SSA (SI S5)", "319 to 36,759",
     lambda: "{:,.0f} to {:,.0f}".format(d("costs", "eps_comparison")["claim_test"]["min_ratio_lindblad"],
                                         d("costs", "eps_comparison")["claim_test"]["max_ratio_lindblad"]), ("eq",)),
    ("Closed-form stationary laws: largest total variation from the numerical law (Methods, weighted encoding)", 2e-13,
     lambda: max(r["total_variation"] for r in d("published_models", "closed_form_check")["rows"]), ("sig", 1)),
    ("Linear signalling cascade: factor at d = 1, 3, 4, log10 (Table S6)", "0.22, 0.94, 0.99",
     lambda: ", ".join(f"{sg('lin', k)['log10_readout_factor']:.2f}" for k in (1, 3, 4)), ("eq",)),
    ("Nonlinear signalling cascade: factor at d = 3, 4, log10 (Table S6)", "0.77, 0.86",
     lambda: ", ".join(f"{sg('nonlin', k)['log10_readout_factor']:.2f}" for k in (3, 4)), ("eq",)),
    ("Signalling cascades: growth per species over d = 1..3, linear and nonlinear (Results, readout barrier)", "0.36 and 0.28",
     lambda: "{:.2f} and {:.2f}".format(*((sg(t, 3)["log10_readout_factor"] - sg(t, 1)["log10_readout_factor"]) / 2
                                          for t in ("lin", "nonlin"))), ("eq",)),
    ("Signalling cascades: factor rises with every added species on both (SI S3.5)", "yes",
     lambda: "yes" if all(d("readout", "signalling_summary")[k]["rises_with_every_species"]
                          for k in ("linear", "nonlinear")) else "no", ("eq",)),
    ("Linear signalling cascade, d = 3: correlation of the last two species (Results, readout barrier)", 0.81,
     lambda: sg("lin", 3)["correlation_last_two"], ("dec", 2)),
    ("Linear signalling cascade, d = 3: S_eff over the product of the marginal values (SI S3.5)", 0.41,
     lambda: sg("lin", 3)["S_eff_over_product_of_marginals"], ("dec", 2)),
    ("Signalling cascades at d = 4: boxes at the ceiling, leaked probability (SI S3.5)", "2.1e-03 and 8.6e-05",
     lambda: "{:.1e} and {:.1e}".format(sg("lin", 4)["leak"], sg("nonlin", 4)["leak"])
     if sg("lin", 4)["capped"] and sg("nonlin", 4)["capped"] else "not capped", ("eq",)),
    ("Signalling cascade with feedback, two species: factor, log10 (Methods, models; SI S3.5)", 0.56,
     lambda: d("readout", "signalling_feedback2")["log10_readout_factor"], ("dec", 2)),
    ("Signalling cascades up to d = 3, with the feedback cascade: largest change from the cut box, log10 (SI S3.5)",
     0.09, lambda: max(sg(t, k)["log10_readout_factor"] - sg(t, k)["fitted"]["cut"]["log10_readout_factor"]
                       for t, k in [(t, k) for t in ("lin", "nonlin") for k in (1, 2, 3)] + [("feedback", 2)]),
     ("dec", 2)),
    ("Fitted boxes: growth per species on the cut box, conversion and catalytic (Methods, models)", "0.29 and 0.46",
     lambda: "{:.2f} and {:.2f}".format(fb()["slopes"]["conversion|cut"]["slope"], fb()["slopes"]["catalytic|cut"]["slope"]),
     ("eq",)),
    ("Fitted boxes: growth per species on the moment box, conversion and catalytic (Methods, models)", "0.17 and 0.23",
     lambda: "{:.2f} and {:.2f}".format(fb()["slopes"]["conversion|moment"]["slope"],
                                        fb()["slopes"]["catalytic|moment"]["slope"]), ("eq",)),
    ("Fitted boxes: largest lowering of the factor by the cut box on published instances, log10 (Results, readout barrier)", 0.17,
     lambda: max(r["full_box"] - r["cut"]["log10_readout_factor"] for r in fb()["rows"] if r["family"] == "published"),
     ("dec", 2)),
    ("Fitted boxes: largest probability dropped by the moment box on published instances, percent (Methods, models; SI S6)", 17,
     lambda: 100 * max(1 - r["moment"]["mass"] for r in fb()["rows"] if r["family"] == "published"), ("int_round",)),
    ("lambda-phage: cut-box factor on the largest box, log10 (SI S6)", 1.92,
     lambda: max(d("readout", "lambda_phage_boxes")["rows"], key=lambda r: r["n_states"])["fitted"]["cut"][
         "log10_readout_factor"], ("dec", 2)),
    ("lambda-phage: leaked probability on the three boxes (SI S6)", "5.6e-03, 9.4e-04, 7.7e-05",
     lambda: ", ".join(f"{r['leak']:.1e}" for r in sorted(d("readout", "lambda_phage_boxes")["rows"],
                                                          key=lambda r: r["n_states"])), ("eq",)),
    ("lambda-phage: factor on the three boxes of its growth schedule, log10 (SI S6)", "1.63, 1.90, 2.18",
     lambda: ", ".join(f"{r['log10_readout_factor']:.2f}" for r in
                       sorted(d("readout", "lambda_phage_boxes")["rows"], key=lambda r: r["n_states"])), ("eq",)),
    ("Normalization of the block encoding times T over Lambda T (SI S5)", "1.6 to 3.8",
     lambda: "{:.1f} to {:.1f}".format(*(f(pu(m)["alphaT_over_LT"] for m in COSTED) for f in (min, max))), ("eq",)),
    ("Trajectory steps on the tail probability (SI S5)", "36 to 6,802",
     lambda: "{} to {:,}".format(min(ssa_tail(m)["N_steps"] for m in COSTED), max(ssa_tail(m)["N_steps"] for m in COSTED)),
     ("eq",)),
    ("Uniform readout growth per species in the cost study, decades (Table D7)", 0.66,
     lambda: (cost_readout("cascade4", "uniform") - cost_readout("cascade1", "uniform")) / 3, ("dec", 2)),
    ("Weighted readout factor, cascade with one to four species, log10 (Table D7)", "0.02-0.11",
     lambda: "{:.2f}-{:.2f}".format(*(f(cost_readout(f"cascade{k}", e) for k in (1, 2, 3, 4)
                                        for e in ("R-NB", "R-mix(NB,E1)")) for f in (min, max))), ("eq",)),
    ("Negative-binomial encoding over coherent SSA on the cascade tail probability, single oracle, log10 (Table D7)",
     "3.1-3.8",
     lambda: "{:.1f}-{:.1f}".format(*(f(math.log10(r["R-NB|W|ratio_to_ssa"]) for r in d("costs", "comparison")["table"]
                                        if r["model"].startswith("cascade") and r["observable"] == "tail")
                                      for f in (min, max))), ("eq",)),
    ("Subroutine cases matching Qualtran 0.7.0 (Methods, resource counts)", "50 of 52",
     lambda: "{} of {}".format(sum(r["match"] for r in d("costs", "qualtran_crosscheck")["rows"]),
                               len(d("costs", "qualtran_crosscheck")["rows"])), ("eq",)),
    # ---- round 5
    ("Conversion cascade: growth per species at T = 1, leak-rule boxes, d = 1 to 5 (Results, readout barrier; SI S3.4)", 0.49,
     lambda: hz("conversion", "1"), ("dec", 2)),
    ("Conversion cascade: growth per species at T = 3 (SI S3.4)", 0.32, lambda: hz("conversion", "3"), ("dec", 2)),
    ("Conversion cascade: growth per species at T = 10 (Results, readout barrier; SI S3.4)", 0.27,
     lambda: hz("conversion", "10"), ("dec", 2)),
    ("Catalytic cascade: growth per species at T = 3 (SI S3.4)", 0.45, lambda: hz("catalytic", "3"),
     ("dec", 2)),
    ("Catalytic cascade: growth per species at T = 10, d = 1 to 4 (Results, readout barrier; SI S3.4)", 0.40,
     lambda: hz("catalytic", "10"), ("dec", 2)),
    ("Catalytic cascade: growth per species at T = 1 over d = 1 to 4 (Results, readout barrier; SI S3.4)", 0.50,
     lambda: d("readout", "horizons")["slopes"]["catalytic|1"]["slope_d1to4"], ("dec", 2)),
    ("Catalytic cascade at T = 10 with five species needs more than 1e7 states (SI S3.4)", "yes",
     lambda: "yes" if d("readout_horizons", "catalytic_T10_d5")["needs_states_at_least"] > 1e7 else "no", ("eq",)),
    ("Box widths at T = 10, catalytic / conversion (Table S5 caption)", "24-38/24",
     lambda: "/".join("-".join(str(w) for w in sorted({min(ws), max(ws)})) for ws in
                      ([r["widths"][0] for r in d("readout", "horizons")["rows"]
                        if r["family"] == f and r["T"] == 10.0 and r["widths"]] for f in ("catalytic", "conversion"))),
     ("eq",)),
    ("Cascades at T = 1: means of species four and five (Methods, models)", "0.077/0.016",
     lambda: "/".join(f"{x:.2g}" for x in hz_means()[3:5]), ("eq",)),
    ("Horizon tests: growth falls by at least 0.1 at T = 10 and stays exponential, both cascades (SI S3.4)",
     "yes", lambda: "yes" if all(v["horizon_dependence"] and v["exponential_at_T10"]
                                for v in d("readout", "horizons")["claim_tests"].values()) else "no", ("eq",)),
    ("Classical SSA time for the needed trajectories, shortest, seconds (Results, trajectory sampling)", 1.4, lambda: min(ssa_times()),
     ("dec", 1)),
    ("Classical SSA time for the needed trajectories, longest, minutes (Results, trajectory sampling)", 2.4,
     lambda: max(ssa_times()) / 60, ("dec", 1)),
    ("Classical SSA batch estimates within four standard errors of FSP, nine networks (SI S5)", "yes",
     lambda: "yes" if all(r["within_4_se"] for r in d("costs", "ssa_time")["rows"]) else "no", ("eq",)),
    ("Classical SSA time below coherent SSA at 1e6 Toffolis per second on every network (Results, trajectory sampling)", "yes",
     lambda: "yes" if all(r["seconds_for_needed"] < ssa_row_total(r["model"]) / 1e6
                          for r in d("costs", "ssa_time")["rows"]) else "no", ("eq",)),
    ("Swap test: amplitude estimation grows per asset, smallest (Results, audit; SI S3.2)", 1.8,
     lambda: min(d("readout", "swap_test")["ae_ratio_per_asset"]), ("dec", 1)),
    ("Swap test: amplitude estimation grows per asset, largest (Results, audit; SI S3.2)", 1.9,
     lambda: max(d("readout", "swap_test")["ae_ratio_per_asset"]), ("dec", 1)),
    ("Canonical amplitude estimation: coherent trajectories over 1/(eps sqrt P), about (Methods, resource counts; SI S5)", 600,
     lambda: round(ae_over(), -2), ("int_round",)),
    ("Table D6 over Table S12: per-use ratio difference at d = 1 to 4, smallest (SI S5)", 0.23,
     lambda: min(s18_minus_s17()), ("dec", 2)),
    ("Table D6 over Table S12: per-use ratio difference at d = 1 to 4, largest (SI S5)", 0.37,
     lambda: max(s18_minus_s17()), ("dec", 2)),
    ("Table D6 over Table S12: walk steps, smallest ratio (SI S5)", 1.6, lambda: min(s18_walk_over_s17()),
     ("dec", 1)),
    ("Table D6 over Table S12: walk steps, largest ratio (SI S5)", 2.2, lambda: max(s18_walk_over_s17()),
     ("dec", 1)),
    ("Per-use ratio: largest rise above d = 4 (Results, trajectory sampling; SI S5)", 0.13,
     lambda: max(r["per_use_ratio"] for r in d("costs", "per_use_scaling")["rows"] if r["d"] >= 4)
     - next(r["per_use_ratio"] for r in d("costs", "per_use_scaling")["rows"] if r["d"] == 4), ("dec", 2)),
    ("eps = 1e-3: networks where Lindblad is cheaper than the cheapest encoding (Results, trajectory sampling; SI S5)", "toggle",
     lambda: ",".join(r["model"] for r in d("costs", "eps_comparison")["rows"]
                      if r["lindblad_best"] < r["cheapest_total"]), ("eq",)),
    ("Table S7: caption identity holds, largest gap below 1e-9 (SI S4.4)", "yes",
     lambda: "yes" if s11_identity_gap() <= 1e-9 else "no", ("eq",)),
    ("Verdicts of growth that are conditional (Abstract; Discussion)", "three",
     lambda: WORDS[sum(e["verdict"] == "omitted, growing (conditional)" for e in lit()["included"])], ("eq",)),
    ("Autoregulation case 3: share of the stationary law inside the box, percent (SI S4.5)", 92,
     lambda: 100 * (1 - d("published_models", "autoreg_tang_3_stationary_known")["mass_outside_box"]),
     ("int_round",)),
    # round 6 (notes.md section 13)
    ("Conversion cascade: growth per species on the cut box at T = 1 and T = 10, closed-form law "
     "(Results, readout barrier; SI S3.4)", "0.29 and 0.22",
     lambda: " and ".join(f"{d('readout', 'cut_horizons')['slopes'][T]['slope']:.2f}" for T in ("1", "10")), ("eq",)),
    ("Conversion cascade: growth per species on the cut box at T = 3, closed-form law (SI S3.4)", 0.25,
     lambda: d("readout", "cut_horizons")["slopes"]["3"]["slope"], ("dec", 2)),
    ("Cut-box stage: the closed form reproduces the stored solves within 0.01 (validation gates)", "yes",
     lambda: "yes" if d("readout", "cut_horizons")["gates"]["passed"] else "no", ("eq",)),
    ("Cut-box stage: growth at T = 10 at least 0.05 decades per species, every increment positive (claim test)",
     "yes", lambda: "yes" if d("readout", "cut_horizons")["claim_test"]["persists_at_T10"] else "no", ("eq",)),
    ("lambda-phage: lowering of the factor by the cut box, log10 (Methods, models)", 0.26,
     lambda: (lambda r: r["log10_readout_factor"] - r["fitted"]["cut"]["log10_readout_factor"])(
         max(d("readout", "lambda_phage_boxes")["rows"], key=lambda r: r["n_states"])), ("dec", 2)),
    ("Published instances in the cut-box comparison besides lambda-phage (Methods, models)", "nine",
     lambda: WORDS[sum(r["family"] == "published" for r in d("readout", "fitted_boxes")["rows"])], ("eq",)),
    ("Audited works with the uniform encoding that read out by an overlap (Results, readout barrier)", "three",
     lambda: WORDS[sum(e["encoding"].startswith("L2 of p") and e["readout"].startswith("R1:")
                       for e in lit()["included"])], ("eq",)),
    ("Count species of the largest network counted gate by gate (Abstract)", "four",
     lambda: WORDS[max(sum(w > 2 for w in d("costs", m)["widths"]) for m in COSTED)], ("eq",)),
    ("Published instances whose initial law and horizon are chosen here, marked in Table 1 (Table 1; Table D3)", "six",
     lambda: WORDS[sum(v[1].startswith("chosen here") for v in __import__(
         "analysis.tables", fromlist=["PUBLISHED_START"]).PUBLISHED_START.values())], ("eq",)),
]


def ssa_row_total(m):
    from .figures import ssa_row
    return ssa_row(m)["total"]


def agrees(value, printed, rule):
    if rule[0] == "eq":
        return value == printed
    v = num(value)
    if rule[0] == "int":
        return v == float(printed)
    if rule[0] == "int_round":
        return round(v) == int(printed)
    if rule[0] == "sig":
        f = lambda x: float(f"{x:.{rule[1] - 1}e}")  # noqa: E731
        return f(v) == f(float(printed))
    if rule[0] == "dec":
        return round(v, rule[1]) == round(float(printed), rule[1])
    raise ValueError(rule)


def main():
    lines = ["# Numbers quoted in the article, checked against results/", "",
             "| Statement | Printed | Stored | Agrees |", "| --- | --- | --- | --- |"]
    bad = 0
    for statement, printed, fn, rule in ENTRIES:
        try:
            value = fn()
            ok = agrees(value, printed, rule)
        except Exception as exc:  # a missing result counts as a disagreement
            value, ok = f"{type(exc).__name__}: {exc}", False
        bad += not ok
        shown = f"{value:.6g}" if isinstance(value, float) else value
        lines.append(f"| {statement} | {printed} | {shown} | {'yes' if ok else '**NO**'} |")
        if not ok:
            print(f"DISAGREES: {statement}: printed {printed!r}, stored {value!r}")
    lines.insert(2, f"{len(ENTRIES) - bad} of {len(ENTRIES)} numbers agree with the stored results.\n")
    out = ROOT / "outputs" / "number_check.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(ENTRIES) - bad} of {len(ENTRIES)} numbers agree (outputs/number_check.md)")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
