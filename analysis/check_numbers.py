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


def path(part, name, p):
    x = d(part, name)
    for k in p.split("/"):
        x = x[int(k)] if isinstance(x, list) else x[k]
    return x


# (statement in the article, printed value, function, rule)
ENTRIES = [
    ("Readout factor on the cascade, one to six species (Section 3.1)", "0.17 -> 2.45",
     lambda: f"{path('readout', 'cascade_1', 'encodings/uniform/log10_readout/meas/one'):.2f} -> "
             f"{path('readout', 'cascade_6', 'encodings/uniform/log10_readout/meas/one'):.2f}", ("eq",)),
    ("Readout factor on lambda-phage, log10 (Section 3.1)", 2.18,
     lambda: path("readout", "lambda_phage", "encodings/uniform/log10_readout/meas/one"), ("dec", 2)),
    ("Smallest readout factor, published one- and two-species instances (Section 3.1)", "10^0.34",
     lambda: "10^{:.2f}".format(min(r["log10_readout_factor"] for r in readout_rows())), ("eq",)),
    ("Largest readout factor, published one- and two-species instances (Section 3.1)", "10^1.55",
     lambda: "10^{:.2f}".format(max(r["log10_readout_factor"] for r in readout_rows())), ("eq",)),
    ("Published one- and two-species instances computed (Section 3.1)", "ten", lambda: WORDS[len(readout_rows())], ("eq",)),
    ("Toggle instance too large for the direct solution, millions of states (Section 2.2)", 3.66,
     lambda: d("published_models", "toggle_gardner")["n_states"] / 1e6, ("sig", 3)),
    ("Records from the literature searches (Section 3.2, S1.1)", 2335, lambda: lit()["screen"]["logged_hits"], ("int",)),
    ("Records after the title step (S1.1)", 176, lambda: lit()["screen"]["after_mechanical_title_step"], ("int",)),
    ("Works included (Abstract, Section 3.2)", 14, lambda: lit()["n_included"], ("int",)),
    ("Accounted / omitted, bounded / omitted, growing (Section 3.2)", "7/2/5",
     lambda: "{accounted}/{omitted, bounded}/{omitted, growing}".format(**lit()["verdict_counts"]), ("eq",)),
    ("Accounted works whose encoding removes the cost (Section 3.2)", 5,
     lambda: sum(e["verdict"] == "accounted" and (e["encoding"].startswith("weighted") or e["encoding"].startswith("sqrt"))
                 for e in lit()["included"]), ("int",)),
    ("Swap-test relative error at 10^6 shots, one asset (Section 3.2)", 0.071,
     lambda: path("readout", "swap_test", "rows/0/rel_error_at_1e6_shots"), ("sig", 2)),
    ("Swap-test relative error at 10^6 shots, four assets (Section 3.2)", 2.7,
     lambda: path("readout", "swap_test", "rows/3/rel_error_at_1e6_shots"), ("sig", 2)),
    ("Best K_T on the toggle switch, twelve references (Section 3.3)", 36.81,
     lambda: d("certificates", "toggle_best")["K_gate"], ("sig", 4)),
    ("K_T with the exact solution as the weight, toggle switch (Section 3.3)", 217,
     lambda: d("certificates", "toggle_exact")["best"]["K"], ("sig", 3)),
    ("Share of the exact-reference K_T from the window penalty (Section 3.3)", 0.92, exact_share, ("dec", 2)),
    ("Non-normal transient growth on the toggle, log10 (Section 3.3)", 3.42,
     lambda: path("certificates", "toggle_references", TOGGLE_SPLIT + "non_normality"), ("dec", 2)),
    ("Actual excess growth (instance slack) on the toggle, log10 (Section 3.3)", 0.44,
     lambda: path("certificates", "toggle_references", TOGGLE_SPLIT + "instance_slack"), ("dec", 2)),
    ("Best cheap K_T over the published instances, smallest to largest (Section 3.3)", "12.6 to 2.07e+04",
     lambda: "{:.3g} to {:.3g}".format(min(num(summary_row(m)["cheap_K"]) for m in INSTANCES),
                                       max(num(summary_row(m)["cheap_K"]) for m in INSTANCES)), ("eq",)),
    ("Smallest finite exact-reference K_T over the published instances (Section 3.3)", 53.7,
     lambda: min(num(summary_row(m)["exact_K"]) for m in INSTANCES if math.isfinite(num(summary_row(m)["exact_K"]))),
     ("sig", 3)),
    ("Instances where non-normality is the largest looseness term (Section 3.3)", "eight",
     lambda: WORDS[nn_largest()], ("eq",)),
    ("Certified growth per species, best weighted reference, decades (Section 3.3)", 0.85,
     lambda: path("readout", "cascade_summary", "slopes/R-mix(NB,E1)|cert|one/slope"), ("dec", 2)),
    ("Certified growth per species, uniform encoding, decades (Section 3.3)", 0.65,
     lambda: path("readout", "cascade_summary", "slopes/uniform|cert|one/slope"), ("dec", 2)),
    ("Measured growth per species of the weighted state, decades (Section 3.3)", 0.02,
     lambda: path("readout", "cascade_summary", "slopes/R-P|W|one/slope"), ("dec", 2)),
    ("Smallest ratio to coherent SSA, cheaper oracle design (Section 3.4, S5)", 167,
     lambda: min(v["min_ratio_best_design"] for v in d("costs", "split_design")["cheaper_design"].values()), ("sig", 3)),
    ("Smallest ratio to coherent SSA, single-oracle design (Section 3.4)", 372,
     lambda: d("costs", "comparison")["verdict"]["R-NB|tail"]["min_ratio"], ("sig", 3)),
    ("Lindblad over coherent SSA, tail probabilities (Section 3.4)", "146 to 12462",
     lambda: "{:.0f} to {:.0f}".format(min(d("costs", "comparison")["lindblad_vs_ssa"]["tail"]["ratios"]),
                                       max(d("costs", "comparison")["lindblad_vs_ssa"]["tail"]["ratios"])), ("eq",)),
    ("Lindblad over observable-aware SSA, total probability (Section 3.4)", "42 to 504",
     lambda: "{:.0f} to {:.0f}".format(min(d("costs", "observable_aware_ssa")["lindblad_vs_ssa_aware"]["one"]["ratios"]),
                                       max(d("costs", "observable_aware_ssa")["lindblad_vs_ssa_aware"]["one"]["ratios"])),
     ("eq",)),
    ("Uniform readout growth per species in the cost study, decades (Section 3.4)", 0.66,
     lambda: (cost_readout("cascade4", "uniform") - cost_readout("cascade1", "uniform")) / 3, ("dec", 2)),
    ("Weighted readout factor, cascade with one to four species, log10 (Section 3.4)", "0.02-0.11",
     lambda: "{:.2f}-{:.2f}".format(*(f(cost_readout(f"cascade{k}", e) for k in (1, 2, 3, 4)
                                        for e in ("R-NB", "R-mix(NB,E1)")) for f in (min, max))), ("eq",)),
    ("Negative-binomial encoding over coherent SSA on the cascade tail probability, log10 (Section 3.4)", "3.1-3.8",
     lambda: "{:.1f}-{:.1f}".format(*(f(math.log10(r["R-NB|W|ratio_to_ssa"]) for r in d("costs", "comparison")["table"]
                                        if r["model"].startswith("cascade") and r["observable"] == "tail")
                                      for f in (min, max))), ("eq",)),
    ("Subroutine cases matching Qualtran 0.7.0 (Section 2.4, S5)", "50 of 52",
     lambda: "{} of {}".format(sum(r["match"] for r in d("costs", "qualtran_crosscheck")["rows"]),
                               len(d("costs", "qualtran_crosscheck")["rows"])), ("eq",)),
]


def agrees(value, printed, rule):
    if rule[0] == "eq":
        return value == printed
    v = num(value)
    if rule[0] == "int":
        return v == float(printed)
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
