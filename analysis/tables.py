"""Tables of the article and the Supplementary Information, from the stored results only.

    python -m analysis.tables      -> outputs/tables/*.csv and *.md

Table 1   readout factor of the uniform encoding on the published instances
Table 2   the literature audit (written by analysis.literature)
Table 3   five questions for a claimed speedup (text)
Table S3  the published instances (box, horizon, leak, species, readout factor, source)
Table S4  the swap-test estimator
Table S5  certified exponent and the looseness decomposition on the published instances
Table S6  Toffoli counts against coherent SSA and the Lindblad embedding (single-oracle design)
Table S7  uses and Toffolis per use: the cheapest encoding (cheaper oracle design) against coherent SSA
Table S8  classical reference: finite state projection time and classical SSA trajectories
Table S9  the readout factor under other truncations
Further tables (numbered by their order in the Supplementary Information):
  benchmark networks; initial laws and horizons of the published instances; the correlated (catalytic) cascade;
  the readout factor under other initial laws
"""
import csv
import math

from .common import ROOT, data, num

OUT = ROOT / "outputs" / "tables"
INSTANCES = ("schlogl", "toggle_hegland", "toggle_kuntz", "toggle_gardner", "feedback_b1", "feedback_b10",
             "feedback_b100", "lambda_phage", "toggle_tang", "autoreg_tang_1", "autoreg_tang_2", "autoreg_tang_3")
SOURCE = {"schlogl": "Kuntz et al. 2019, Fig. 3(b)",
          "toggle_hegland": "Jahnke and Huisinga 2008, Sec. 4.1, Eq. (21); model of Hegland et al.",
          "toggle_kuntz": "Kuntz et al. 2019, Sec. V.A, Eqs. (112)-(113)",
          "toggle_gardner": "Gupta et al. 2017, Sec. 4.3.2; toggle of Gardner et al.",
          "feedback_b1": "Kuntz et al. 2019, Sec. V.B, Eq. (114)", "feedback_b10": "Kuntz et al. 2019, Sec. V.B, Eq. (114)",
          "feedback_b100": "Kuntz et al. 2019, Sec. V.B, Eq. (114)",
          "lambda_phage": "Jahnke and Huisinga 2008, Sec. 4.2, Eq. (22); model of Hegland et al.",
          "toggle_tang": "Tang et al. 2023, Supplementary Note 2 F, Eq. (S2.9)",
          "autoreg_tang_1": "Tang et al. 2023, Supplementary Note 2 C, Eq. (S2.6), case 1",
          "autoreg_tang_2": "Tang et al. 2023, Supplementary Note 2 C, Eq. (S2.6), case 2",
          "autoreg_tang_3": "Tang et al. 2023, Supplementary Note 2 C, Eq. (S2.6), case 3"}
QUESTIONS = [
    "Which encoding is used (p/‖p‖, √p or a weighted variant), and what does the readout measure?",
    "Is the amplitude or norm that the readout estimates bounded in the claimed scaling variable, such as species, "
    "dimension or state-space size? Is its cost included in the stated complexity?",
    "If a classical guess or weight is used, is the growth of the encoded state certified, or only measured on small "
    "instances?",
    "Is the precision relative or additive, and is it adequate for the observable, for example a rare switching "
    "probability? For a rare event, how does the cost scale with its probability?",
    "Does the claim beat quantum-accelerated Monte Carlo over trajectories (amplitude estimation over stochastic "
    "simulation), and the best classical method, such as finite state projection or rare-event stochastic simulation, "
    "counted at the same precision?"]
CRITERIA = [
    "The encoding and the readout procedure are both named, down to the state the result is compared with.",
    "The quantity is bounded by an expression in the scaling variable, or computed over a stated range of it, and "
    "the bound enters the stated complexity.",
    "A certificate that does not use the solution is given. Measured growth on small instances is an acceptable "
    "interim answer only if it is labelled as uncertified.",
    "The precision is stated relative to the size of the observable, and the cost is given as a function of the "
    "probability of the event.",
    "The same observable, precision and failure probability are used on both sides, and the unit of cost (gates, "
    "samples or time at a stated gate rate) is stated."]
BENCHMARKS = [  # (key, name, reactions and propensities, initial means, where used)
    ("birth_death", "Birth–death", "0 → X at k = 10; X → 0 at g·x, g = 1", "2", "Fig. 3"),
    ("schlogl", "Schlögl (the published instance of Kuntz et al.)",
     "2X → 3X at k1·x(x−1); 3X → 2X at k2·x(x−1)(x−2); 0 → X at k3; X → 0 at k4·x; k1 = 1/9, k2 = 1/1215, "
     "k3 = 27/2, k4 = 59/20", "25", "Table 1, Figs 2b and 3"),
    ("toggle", "Toggle", "0 → X at α/(1 + (y/K)^n); 0 → Y at α/(1 + (x/K)^n); X → 0 at δ·x; Y → 0 at δ·y; "
     "α = 30, K = 10, n = 2, δ = 1", "(5, 20)", "Figs 1a, 2a and 3"),
    ("michaelis_menten", "Michaelis–Menten", "E + S → C at k1·e·s; C → E + S at k−1·c; C → E + P at k2·c; "
     "k1 = 1, k−1 = 5, k2 = 1; 10 enzymes and 40 substrates in total", "(10, 40, 0, 0)", "Fig. 3"),
    ("self_repressor", "Self-repressor", "G + P → Gb at kb·g·p; Gb → G + P at ku·gb; G → G + P at α·g; "
     "Gb → Gb + P at αb·gb; P → 0 at δ·p; α = 20, αb = 1, kb = 0.05, ku = 1, δ = 1", "gene free, no protein", "Fig. 3"),
]
PUBLISHED_START = {  # (initial means, where the initial law comes from)
    "schlogl": ("25", "chosen here; the source studies the stationary law"),
    "toggle_hegland": ("(30, 5)", "source (a discrete Gaussian at these means)"),
    "toggle_kuntz": ("(0, 0)", "chosen here; the source studies the stationary law"),
    "toggle_gardner": ("(0, 0)", "chosen here; the source studies the stationary law"),
    "feedback_b1": ("gene off, no protein", "chosen here; the source studies the stationary law"),
    "feedback_b10": ("gene off, no protein", "chosen here; the source studies the stationary law"),
    "feedback_b100": ("gene off, no protein", "chosen here; the source studies the stationary law"),
    "lambda_phage": ("0.15 for each species", "source (the mean of its multinomial initial law)"),
    "toggle_tang": ("genes free, no protein", "source"),
    "autoreg_tang_1": ("gene free, no protein", "source"),
    "autoreg_tang_2": ("gene free, no protein", "source"),
    "autoreg_tang_3": ("gene free, no protein", "source"),
}
TOFFOLI_RATE = 1e6  # Toffoli gates per second assumed when a count is converted to a run time
BENCH = ("toggle", "schlogl", "michaelis_menten", "self_repressor", "cascade2", "cascade3")


def g(v, fmt=".3g"):
    if v is None:
        return "n/a"
    v = num(v)
    if math.isnan(v):
        return "n/a"
    if math.isinf(v):
        return "inf"
    return format(v, fmt)


def write_table(name, head, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)
    md = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    md += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
    (OUT / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[table] outputs/tables/{name}", flush=True)


def table1():
    summ = {r["row"]: r for r in data("readout", "published_summary")["published_rows"]}
    rows = []
    for k in INSTANCES:
        dec, r = data("published_models", k), summ[k]
        rows.append([k, r.get("n_count_species", ""), dec["n_states"], g(dec["T"], ".4g"),
                     "not computed" if "skipped" in r else f"{r['log10_readout_factor']:.2f}"])
    write_table("table1_readout", ["Instance", "Count species", "|S|", "T", "log10 readout factor"], rows)


def table3():
    write_table("table3_checklist", ["No.", "Question", "What counts as an answer"],
                [[i + 1, q, c] for i, (q, c) in enumerate(zip(QUESTIONS, CRITERIA))])


def tableS3():
    summ = {r["row"]: r for r in data("readout", "published_summary")["published_rows"]}
    rows = []
    for k in INSTANCES:
        dec, r = data("published_models", k), summ[k]
        rows.append([k, r.get("n_count_species", ""), " x ".join(map(str, dec["widths"])), dec["n_states"],
                     g(dec["T"], ".4g"), g(dec["leak"], ".2g"),
                     "not computed" if "skipped" in r else f"{r['log10_readout_factor']:.2f}", SOURCE[k]])
    write_table("tableS3_models", ["Instance", "Count species", "Box", "|S|", "T", "Leak", "log10 readout factor",
                                   "Source"], rows)


def tableS4():
    d = data("readout", "swap_test")
    rows = [[x["d"], 32 ** x["d"], g(x["F1_uniform_overlap_sq"]), g(x["F2_payoff_overlap_sq"]), g(x["sqrt_S_over_Seff"]),
             g(x["shots_for_1pct"]), g(x["ae_queries_for_1pct"])] for x in d["rows"]]
    write_table("tableS4_swap_test", ["Assets d", "|S|", "F1 (uniform)", "F2 (payoff)", "sqrt(|S|/S_eff)",
                                      "Shots for 1%", "Amplitude estimation, queries for 1%"], rows)


def tableS5():
    from .figures import decomposition_rows
    rows = [[r["key"], r["n_states"], r["reference"], g(r["cheap_K"], ".4g"), g(r["decomposed_K"], ".4g"),
             g(r["window"]), g(r["nonnormal"]), g(r["slack"]), g(r["tail"]),
             "not computable" if not math.isfinite(r["exact_K"]) else g(r["exact_K"], ".4g")]
            for r in decomposition_rows()]
    # the three terms split the certificate of the K_T in the fifth column: K_T / (2 ln 10) = window + non-normality
    # + slack + log10 of the measured growth. It differs from the best K_T only where the resolvent weight was applied
    # to the best reference afterwards (the Gardner toggle).
    write_table("tableS5_decomposition", ["Instance", "|S|", "Best cheap reference", "K_T", "K_T of the split certificate",
                                          "Window", "Non-normality", "Slack", "Tail", "K_T, exact reference"], rows)


def tableS6():
    t = data("costs", "comparison")["table"]
    rows = []
    for m in BENCH:
        for obs in ("one", "tail"):
            r = next(x for x in t if x["model"] == m and x["observable"] == obs)
            mins = min(r[f"{e}|W|ratio_to_ssa"] for e in ("uniform", "R-NB", "R-mix(NB,E1)"))
            rows.append([m, "total probability" if obs == "one" else "tail probability", g(r["uniform|W"], ".2g"),
                         g(r["R-NB|W"], ".2g"), g(r["R-mix(NB,E1)|W"], ".2g"), g(r["R-mix(NB,E1)|cert"], ".2g"),
                         g(r["ssa_total"], ".2g"), g(r["lindblad_best_total"], ".2g"), g(mins, ".2g")])
    write_table("tableS6_costs", ["Model", "Observable", "Uniform [W]", "NB-weighted [W]", "Mixture [W]",
                                  "Mixture [cert]", "Coherent SSA", "Lindblad", "Smallest ratio [W]"], rows)


def tableS7():
    from .figures import COSTED, cost_split
    rows = []
    for m, _ in COSTED:
        c = cost_split(m)
        rows.append([m, c["encoding"], f"{math.log10(c['enc_uses']):.2f}", f"{math.log10(c['enc_per_use']):.2f}",
                     f"{math.log10(c['enc_total']):.2f}", f"{math.log10(c['ssa_uses']):.2f}",
                     f"{math.log10(c['ssa_per_use']):.2f}", f"{math.log10(c['ssa_total']):.2f}",
                     g(c["enc_total"] / c["ssa_total"], ".3g")])
    write_table("tableS7_cost_split", ["Network", "Cheapest encoding", "Uses (log10)", "Toffolis per use (log10)",
                                       "Total (log10)", "SSA uses (log10)", "SSA Toffolis per use (log10)",
                                       "SSA total (log10)", "Ratio"], rows)


def secs(x):
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


def tableS8():
    from .figures import COSTED, ssa_row
    st = {r["model"]: r for r in data("costs", "ssa_time")["rows"]}
    rows = []
    for m, _ in COSTED:
        c, s = data("costs", m), ssa_row(m)
        rows.append([m, c["n_states"], f"{c['fsp_wall_time_s']['median']:.3f}", g(s["P"], ".2g"),
                     g(c["classical_ssa_samples"]["tail"], ".2g"), secs(st[m]["seconds_for_needed"]),
                     g(s["total"], ".2g"), g(s["total"] / TOFFOLI_RATE, ".2g")])
    write_table("tableS8_classical", ["Network", "|S|", "FSP time (s)", "Tail probability", "Classical SSA trajectories",
                                      "Classical SSA time (s)", "Coherent SSA Toffolis",
                                      "Seconds at 1e6 Toffolis per second"], rows)


def tableS9():
    d = data("readout", "sensitivity")
    rows = []
    for r in d["published"]:
        b = r["by_tolerance"]
        rows.append([r["row"]] + [f"{b[k]['log10_readout_factor']:.2f} ({b[k]['n_states']})" for k in ("0.0001", "1e-06", "1e-08")]
                    + [f"{r['spread']:.2f}"])
    write_table("tableS9_sensitivity_published", ["Instance", "Leak 1e-4: factor (|S|)", "Leak 1e-6: factor (|S|)",
                                                  "Leak 1e-8: factor (|S|)", "Spread"], rows)
    rows = [[w] + [f"{x['log10_readout_factor']:.2f}" for x in c["rows"]] + [f"{c['slope']:.2f}", g(c["rows"][-1]["leak"], ".1g")]
            for w, c in d["cascade"].items()]
    write_table("tableS9_sensitivity_cascade", ["Box width", "d = 1", "d = 2", "d = 3", "d = 4", "Decades per species",
                                                "Leak at d = 4"], rows)


def table_benchmarks():
    from qcme import models
    rows = []
    for key, name, rx, x0, used in BENCHMARKS:
        c = data("costs", key)
        o = c["observables"]["tail"]
        box = " x ".join(map(str, c["widths"])) + f" ({c['n_states']} states)"
        if key == "toggle":
            box += "; 50 x 50 and T = 1 in Fig. 1a"
        rows.append([name, rx, x0, g(c["T"], ".3g"), box, f"last species at or above {o['theta']}", used])
    for d in (1, 2, 3, 4):
        c = data("costs", f"cascade{d}")
        rows.append([f"Conversion cascade, d = {d}",
                     "0 → X1 at k = 4; X_i → X_(i+1) at c·x_i; X_d → 0 at g·x_d; c = g = 1" if d == 1 else "as above",
                     "0", g(c["T"], ".3g"), f"15^{d} ({c['n_states']} states) in Fig. 3; 12^{d} in Fig. 1b",
                     f"last species at or above {c['observables']['tail']['theta']}", "Figs 1b and 3"])
    corr = data("readout", "correlated_cascade")
    if corr:
        rows.append(["Catalytic cascade, d = 1 to 5", "0 → X1 at k = 4; X_i → X_i + X_(i+1) at c·x_i; X_i → 0 at g·x_i; "
                     "c = g = 1", "0", "1", "15^d", "not counted", "Fig. 1b"])
    assert models.cascade(2).params == {"d": 2, "k": 4.0, "c": 1.0, "g": 1.0}
    write_table("table_benchmarks", ["Network", "Reactions and propensities", "Initial means", "T", "Box",
                                     "Tail observable", "Used in"], rows)


def table_initial_laws():
    rows = []
    for k in INSTANCES:
        dec = data("published_models", k)
        rule = "source horizon" if "source horizon" in dec["horizon"].get("rule", "") else "one relaxation time"
        rows.append([k, PUBLISHED_START[k][0], PUBLISHED_START[k][1], g(dec["T"], ".4g"), rule])
    write_table("table_initial_laws", ["Instance", "Initial means", "Initial law taken from", "T", "Horizon"], rows)


def table_correlated():
    d = data("readout", "correlated_cascade")
    mt = {r["d"]: r for r in (data("readout", "cascades_matched") or {"rows": []})["rows"]}
    hz = {(r["family"], r["T"], r["d"]): r for r in data("readout", "horizons")["rows"]}

    def at10(fam, k):
        r = hz[(fam, 10.0, k)]
        if r["log10_readout_factor"] is None:  # the box is over the size limit (not solved)
            return "—"
        return f"{r['log10_readout_factor']:.2f}" + ("*" if r["capped"] else "")
    rows = [[r["d"], r["n_states"], f"{r['log10_readout_factor']:.2f}", f"{r['S_eff_over_product_of_marginals']:.2f}",
             "" if r["correlation_last_two"] is None else f"{r['correlation_last_two']:.2f}",
             f"{mt[r['d']]['conversion_uniform']:.2f}", f"{mt[r['d']]['catalytic_weighted_smallest']:.2f}",
             at10("catalytic", r["d"]), at10("conversion", r["d"])]
            for r in d["rows"]]
    write_table("table_correlated_cascade", ["Species d", "|S|", "log10 readout factor",
                                             "S_eff over product of marginal S_eff", "Correlation of the last two species",
                                             "Conversion cascade, same box", "Weighted, measured", "Catalytic, T = 10",
                                             "Conversion, T = 10"], rows)


def table_initial_law_sensitivity():
    d = data("readout", "initial_law")
    keys = ("poisson at the initial means", "point mass at the initial means", "poisson at the attractor")
    rows = [[r["row"]] + [f"{r['by_initial_law'][k]['log10_readout_factor']:.2f}" for k in keys] + [f"{r['spread']:.2f}"]
            for r in d["rows"]]
    write_table("table_initial_law_sensitivity", ["Instance", "Poisson at the initial means (default)",
                                                  "Point mass at the initial means", "Poisson at the attractor",
                                                  "Spread"], rows)


def table_weighted_factors():
    """Measured readout factor of the weighted encoding on the published instances (f = 1)."""
    d = data("published_models", "weighted_factors")
    rows = [[r["row"], f"{r['uniform']:.2f}", r["best_reference"], f"{r['best_meas']:.2f}", f"{r['best_cert']:.3g}",
             r["smallest_reference"], f"{r['smallest_meas']:.2f}"] for r in d["rows"]]
    write_table("table_weighted_factors", ["Instance", "Uniform, measured", "Best cheap reference",
                                           "Weighted, measured", "Weighted, certified", "Smallest measured reference",
                                           "Weighted, smallest measured"], rows)


def table_stationary_known():
    """The stationary law as the reference (instances where it is known), and the factor at the stationary law."""
    rows = []
    for k in INSTANCES:
        s = data("published_models", f"{k}_stationary_known")
        if s is None:
            continue
        if k == "schlogl":
            st = data("published_models", "schlogl_stationary")
            cert, meas, K = st["log10_readout_cert"], st["log10_readout_meas"], max(st["best_K"], 0.0)
            known = ["0", f"{cert:.2f}", f"{meas:.2f}"]
        elif "best_K" in s:
            known = [f"{max(s['best_K'], 0.0):.0f}", f"{s['log10_readout_cert']:.2f}", f"{s['log10_readout_meas']:.2f}"]
        else:
            known = ["no known law", "", ""]
        uni = next(r["log10_readout_factor"] for r in data("readout", "published_summary")["published_rows"]
                   if r["row"] == k)
        rows.append([k, *known, f"{uni:.2f}", f"{s['log10_readout_factor_at_stationary_law']:.2f}",
                     g(s["mass_outside_box"], ".1g")])
    write_table("table_stationary_known", ["Instance", "K_T", "Weighted, certified", "Weighted, measured",
                                           "Uniform at T, measured", "Uniform at the stationary law",
                                           "Stationary mass outside the box"], rows)


def table_tail_factor():
    rows = []
    for dd in range(1, 7):
        r = data("readout", f"cascade_{dd}")
        o, u = r["observables"]["tail"], r["encodings"]["uniform"]["log10_readout"]["meas"]
        rows.append([dd, o["theta"], g(o["E_f"], ".2g"), f"{num(u['one']):.2f}", f"{num(u['tail']):.2f}"])
    write_table("table_tail_factor", ["Species d", "Threshold", "Tail probability", "Factor, total probability",
                                      "Factor, tail probability"], rows)


def table_cascade_costs():
    """The cascade in the cost study (width 15): readout factors credited with W, and totals on the tail."""
    from .check_numbers import cost_readout
    rows = []
    for dd in (1, 2, 3, 4):
        m = f"cascade{dd}"
        cmp_ = next(r for r in data("costs", "comparison")["table"] if r["model"] == m and r["observable"] == "tail")
        rows.append([dd, f"{cost_readout(m, 'uniform'):.2f}", f"{cost_readout(m, 'R-NB'):.2f}",
                     f"{cost_readout(m, 'R-mix(NB,E1)'):.2f}", f"{math.log10(cmp_['R-NB|W']):.2f}",
                     f"{math.log10(cmp_['ssa_total']):.2f}", f"{math.log10(cmp_['R-NB|W|ratio_to_ssa']):.2f}"])
    write_table("table_cascade_costs", ["Species d", "Uniform factor (with c)", "NB-weighted factor [W]",
                                        "Mixture-weighted factor [W]", "NB-weighted total, tail",
                                        "Coherent SSA total, tail", "Ratio"], rows)


COSTED = ("toggle", "schlogl", "michaelis_menten", "self_repressor", "birth_death", "cascade1", "cascade2", "cascade3",
          "cascade4")


def per_use_split(m):
    """Steps per use and Toffolis per step: the cheapest encoding (single-oracle design, headline kernel, growth W)
    against coherent SSA (uniformization), on the tail probability. The cost of a step is the cost per use divided
    by the steps per use on each side, so the three logs add up. A walk step makes two queries to the block encoding;
    the query ratio is the cost of one query against one trajectory step."""
    d, b = data("costs", m), data("costs", f"{m}_baselines")
    ssa = next(r for r in b["rows"] if r["observable"] == "tail" and r.get("method") == "uniformization")
    best = None
    for enc, c in d["costs"].items():
        hb = c["tail"]["best_headline"]
        for r in c["tail"]["rows"]:
            if (r["growth"] == "W" and r["variant"] == "single" and r["eps"] == 0.01 and r["kernel"] == hb["kernel"]
                    and (best is None or r["toffoli_total"] < best[1]["toffoli_total"])):
                best = (enc, r)
    enc, r = best
    return {"encoding": enc, "LT": ssa["Lambda"] * d["T"], "alphaT_over_LT": r["sum_alpha_dt"] / (ssa["Lambda"] * d["T"]),
            "N_ssa": ssa["N_steps"], "walk_steps": r["walk_steps"],
            "steps_ratio": math.log10(r["walk_steps"] / ssa["N_steps"]),
            "per_step_ratio": math.log10((r["toffoli_U"] / r["walk_steps"]) / (ssa["per_A"] / ssa["N_steps"])),
            "query_ratio": math.log10(r["query"] / ssa["step"]),
            "per_use_ratio": math.log10(r["toffoli_U"] / ssa["per_A"])}


def table_per_use():
    rows = []
    for m in COSTED:
        x = per_use_split(m)
        rows.append([m, x["encoding"], f"{x['LT']:,.0f}", f"{x['alphaT_over_LT']:.1f}", x["N_ssa"],
                     f"{x['walk_steps']:,}", f"{x['steps_ratio']:.2f}", f"{x['per_step_ratio']:.2f}",
                     f"{x['per_use_ratio']:.2f}", f"{x['query_ratio']:.2f}"])
    write_table("table_per_use", ["Network", "Cheapest encoding", "Lambda T", "Normalization over Lambda",
                                  "Trajectory steps", "Walk steps", "Steps ratio", "Cost per step ratio",
                                  "Cost per use ratio", "Query ratio"], rows)


def table_per_use_scaling():
    d = data("costs", "per_use_scaling")
    rows = [[r["d"], f"{r['Lambda']:.0f}", f"{r['sum_alpha_dt'] / (r['Lambda'] * r['T']):.2f}", r["ssa_N"],
             f"{r['walk_steps']:,}", f"{r['steps_ratio']:.2f}", f"{r['ssa_per_A'] / r['ssa_N']:,.0f}",
             f"{r['toffoli_U'] / r['walk_steps']:,.0f}", f"{r['per_step_ratio']:.2f}", f"{r['per_use_ratio']:.2f}"]
            for r in d["rows"]]
    write_table("table_per_use_scaling", ["Species d", "Lambda", "Normalization over Lambda", "Trajectory steps",
                                          "Walk steps", "Steps ratio", "Toffolis per trajectory step",
                                          "Toffolis per walk step", "Cost per step ratio", "Cost per use ratio"], rows)


def table_eps_comparison():
    d = data("costs", "eps_comparison")
    rows = [[r["model"], NET_SHORT.get(r["cheapest_encoding"], r["cheapest_encoding"]), f"{math.log10(r['ssa_total']):.2f}",
             f"{math.log10(r['cheapest_total']):.2f}", f"{math.log10(r['lindblad_best']):.2f}",
             g(r["ratio_cheapest_to_ssa"]), g(r["lindblad_over_ssa"])] for r in d["rows"]]
    order = {m: i for i, m in enumerate(COSTED)}
    rows.sort(key=lambda x: order[x[0]])
    write_table("table_eps_comparison", ["Network", "Cheapest encoding", "Coherent SSA", "Cheapest encoding total",
                                         "Lindblad", "Encoding over SSA", "Lindblad over SSA"], rows)


NET_SHORT = {"uniform": "uniform", "R-NB": "NB-weighted", "R-mix(NB,E1)": "mixture-weighted"}


def table_closed_forms():
    d = data("published_models", "closed_form_check")
    rows = []
    for r in d["rows"]:
        num_K = data("published_models", f"{r['instance']}_stationary_known")["best_K"]
        rows.append([r["instance"], g(r["total_variation"]), g(r["max_abs_log_ratio"]), g(r["K"]), g(num_K)])
    write_table("table_closed_forms", ["Instance", "Total variation", "Largest |difference of log pi|",
                                       "K_T, closed form", "K_T, numerical law"], rows)


def table_signalling():
    rows = []
    for kind, tag in (("linear", "lin"), ("nonlinear", "nonlin"), ("feedback", "feedback")):
        for dd in ((2,) if kind == "feedback" else (1, 2, 3, 4)):
            r = data("readout", f"signalling_{tag}{dd}")
            if r is None:
                continue
            rows.append([kind, dd, " x ".join(map(str, r["widths"])), f"{r['n_states']:,}", g(r["leak"]),
                         f"{r['log10_readout_factor']:.2f}" + ("*" if r["capped"] else ""),
                         f"{r['fitted']['cut']['log10_readout_factor']:.2f}",
                         "" if "correlation_last_two" not in r else f"{r['correlation_last_two']:.2f}"])
    write_table("table_signalling", ["Cascade", "Species d", "Box", "|S|", "Leak", "log10 readout factor",
                                     "Cut box", "Correlation of the last two species"], rows)


def table_fitted_boxes():
    d = data("readout", "fitted_boxes")
    lam = data("readout", "lambda_phage_boxes")
    rows = [[r["family"], r["instance"], f"{r['full_box']:.2f}", f"{r['cut']['log10_readout_factor']:.2f}",
             f"{r['cut']['n_states']:,}", f"{r['moment']['log10_readout_factor']:.2f}", f"{r['moment']['n_states']:,}"]
            for r in d["rows"]]
    for r in (lam or {"rows": []})["rows"]:
        rows.append(["lambda-phage", " x ".join(map(str, r["widths"])), f"{r['log10_readout_factor']:.2f}",
                     f"{r['fitted']['cut']['log10_readout_factor']:.2f}", f"{r['fitted']['cut']['n_states']:,}",
                     f"{r['fitted']['moment']['log10_readout_factor']:.2f}", f"{r['fitted']['moment']['n_states']:,}"])
    write_table("table_fitted_boxes", ["Family", "Instance", "Leak-rule box", "Cut box", "Cut box |S|", "Moment box",
                                       "Moment box |S|"], rows)


def table_swap_variants():
    d = data("readout", "swap_variants")
    rows = []
    for name, v in d.items():
        for r in v["rows"]:
            rows.append([name, r["d"], g(r["F1_uniform_overlap_sq"]), g(r["F2_payoff_overlap_sq"]),
                         g(r["sqrt_S_over_Seff"]), g(r["shots_for_1pct"]), g(r["ae_queries_for_1pct"])])
    write_table("table_swap_variants", ["Variant", "Assets d", "F1 (uniform)", "F2 (payoff)", "sqrt(|S|/S_eff)",
                                        "Shots for 1%", "Amplitude estimation, queries for 1%"], rows)


def round5_numbers():
    """Derived numbers quoted in the text from several stored results (round 5), written once so that the article
    and check_numbers.py read the same values: the per-use ratios of Table S18 over Table S17 at d = 1 to 4, and the
    coherent trajectories of canonical amplitude estimation over 1/(eps sqrt P)."""
    import json
    from .check_numbers import ae_over, s18_minus_s17
    out = {"s18_minus_s17": s18_minus_s17(), "ae_over_mean": ae_over()}
    f = ROOT / "outputs" / "tables" / "round5_numbers.json"
    f.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[table] {f.relative_to(ROOT).as_posix()}")


def main():
    from . import literature
    table1()
    literature.main()
    table3()
    tableS3()
    tableS4()
    tableS5()
    tableS6()
    tableS7()
    tableS8()
    if data("readout", "sensitivity"):
        tableS9()
    table_benchmarks()
    table_initial_laws()
    if data("readout", "correlated_cascade"):
        table_correlated()
    if data("readout", "initial_law"):
        table_initial_law_sensitivity()
    for name, fn in (("weighted_factors", table_weighted_factors), ("schlogl_stationary_known", table_stationary_known)):
        if data("published_models", name):
            fn()
    table_tail_factor()
    table_cascade_costs()
    table_per_use()
    if data("costs", "per_use_scaling"):
        table_per_use_scaling()
    if data("costs", "eps_comparison"):
        table_eps_comparison()
    if data("published_models", "closed_form_check"):
        table_closed_forms()
    if data("costs", "per_use_scaling"):
        round5_numbers()
    if data("readout", "signalling_lin1"):
        table_signalling()
    if data("readout", "fitted_boxes"):
        table_fitted_boxes()
    if data("readout", "swap_variants"):
        table_swap_variants()


if __name__ == "__main__":
    main()
