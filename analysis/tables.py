"""Tables of the article and the Supplementary Information, from the stored results only.

    python -m analysis.tables      -> outputs/tables/*.csv and *.md

Table 1   readout factor of the uniform encoding on the published instances
Table 2   the literature audit (written by analysis.literature)
Table 3   five questions for a claimed speedup (text)
Table S3  the published instances (box, horizon, leak, species, readout factor, source)
Table S4  the swap-test estimator
Table S5  certified exponent and the looseness decomposition on the published instances
Table S6  Toffoli counts against coherent SSA and the Lindblad embedding
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
    "probability?",
    "Does the claim beat quantum Monte Carlo over trajectories (amplitude estimation over stochastic simulation), "
    "counted at the same precision?"]
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
    write_table("table3_checklist", ["No.", "Question"], [[i + 1, q] for i, q in enumerate(QUESTIONS)])


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
             g(x["rel_error_at_1e6_shots"]), g(x["shots_for_1pct"])] for x in d["rows"]]
    write_table("tableS4_swap_test", ["Assets d", "|S|", "F1 (uniform)", "F2 (payoff)", "sqrt(|S|/S_eff)",
                                      "Relative error at 1e6 shots", "Shots for 1%"], rows)


def tableS5():
    from .figures import decomposition_rows
    rows = [[r["key"], r["n_states"], r["reference"], g(r["cheap_K"], ".4g"), g(r["window"]), g(r["nonnormal"]),
             g(r["slack"]), g(r["tail"]), "not computable" if not math.isfinite(r["exact_K"]) else g(r["exact_K"], ".4g")]
            for r in decomposition_rows()]
    write_table("tableS5_decomposition", ["Instance", "|S|", "Best cheap reference", "K_T", "Window", "Non-normality",
                                          "Slack", "Tail", "K_T, exact reference"], rows)


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


def main():
    from . import literature
    table1()
    literature.main()
    table3()
    tableS3()
    tableS4()
    tableS5()
    tableS6()


if __name__ == "__main__":
    main()
