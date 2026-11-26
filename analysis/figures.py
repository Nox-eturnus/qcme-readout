"""Figures 1-3 of the article, from the stored results only.

    python -m analysis.figures            -> outputs/figures/fig1.{png,svg}, fig2.{png,svg}, fig3.{png,svg}

Figure 1  a: a toggle-switch distribution, its uniform encoding, a cheap reference and the weighted encoding;
          b: the omitted readout factor against the number of species (cascades, published instances, lambda-phage)
Figure 2  a: K_T for every reference on the toggle switch; b: best cheap and exact references on the published instances
Figure 3  a: Toffoli counts of the master-equation encodings (cheaper oracle design), coherent SSA and the Lindblad
          embedding for the tail probability; b: uses and Toffolis per use, cheapest encoding against coherent SSA
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .common import ROOT, data, num  # noqa: E402

FIG = ROOT / "outputs" / "figures"
SURFACE, INK, INK2, MUTED, GRID = "#ffffff", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
BLUES = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
THRESH = 2.0
FAM = ("sqrt_mu", "resolvent")
LABEL = {"schlogl": "Schlögl (Kuntz et al.)", "toggle_hegland": "Toggle (Hegland et al.)",
         "toggle_kuntz": "Toggle (Kuntz et al.)", "toggle_gardner": "Toggle (Gardner et al.)",
         "feedback_b1": "Bursty feedback, ⟨b⟩ = 1", "feedback_b10": "Bursty feedback, ⟨b⟩ = 10",
         "feedback_b100": "Bursty feedback, ⟨b⟩ = 100", "lambda_phage": "λ-phage", "toggle_tang": "Toggle (Tang et al.)",
         "autoreg_tang_1": "Autoregulation, case 1", "autoreg_tang_2": "Autoregulation, case 2",
         "autoreg_tang_3": "Autoregulation, case 3"}
BENCH = [("toggle", "toggle"), ("schlogl", "Schlögl"), ("michaelis_menten", "Michaelis–\nMenten"),
         ("self_repressor", "self-\nrepressor"), ("cascade2", "cascade\nd = 2"), ("cascade3", "cascade\nd = 3")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 8, "axes.titlesize": 8.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 1.6,
    "lines.markersize": 5, "svg.fonttype": "none", "svg.hashsalt": "qcme-readout",
})


def panel(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, metadata={"Date": None} if ext == "svg" else None)
    plt.close(fig)
    print(f"[figure] outputs/figures/{name}", flush=True)


def fig1():
    d = data("readout", "concept")
    X = np.array(d["X"]).astype(int)
    w = X.max(axis=0) + 1

    def grid(v):
        img = np.zeros((w[1], w[0]))
        img[X[:, 1], X[:, 0]] = v
        return img
    pT, psi = grid(d["p_T"]), grid(d["psi_T"])
    mu = grid(np.exp(np.array(d["log_mu_T"])))
    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.35])
    panels = [(pT, "true law p"), (pT**2 / np.sum(pT**2), "uniform encoding"), (mu, "cheap reference μ"),
              (psi**2 / np.sum(psi**2), "weighted ψ = p/√μ")]
    for k, (img, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img / img.max(), origin="lower", cmap=BLUES, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=7, fontweight="normal")
        ax.grid(False)
        ax.set_xticks([0, 25, 50])
        ax.set_yticks([0, 25, 50] if k == 0 else [])
        ax.tick_params(labelsize=5.5, length=2)
        if k == 0:
            panel(ax, "a")
            ax.set_ylabel("copies of protein 2", fontsize=6.5)
        ax.set_xlabel("copies of protein 1", fontsize=6.5)
    bx = fig.add_subplot(gs[1, :])
    ds = list(range(1, 7))
    uni, cert, meas, ucert = [], [], [], []
    for dd in ds:
        b = data("readout", f"cascade_{dd}")
        uni.append(0.5 * b["log10_S_over_Seff"])
        ucert.append(num(b["encodings"]["uniform"]["log10_readout"]["cert"]["one"]))
        cert.append(num(b["encodings"]["R-mix(NB,E1)"]["log10_readout"]["cert"]["one"]))
        meas.append(num(b["encodings"]["R-mix(NB,E1)"]["log10_readout"]["meas"]["one"]))
    bx.plot(ds, uni, color=C1, marker="o", label="omitted readout factor √(|S|/S_eff), conversion cascade (width 12)")
    corr = data("readout", "correlated_cascade")
    if corr:
        bx.plot([r["d"] for r in corr["rows"]], [r["log10_readout_factor"] for r in corr["rows"]], color=C2, marker="s",
                markersize=3.5, label="the same factor, catalytic cascade (correlated species, width 15)")
    matched = data("readout", "cascades_matched")
    if matched:
        md = [r["d"] for r in matched["rows"]]
        bx.plot(md, [r["conversion_uniform"] for r in matched["rows"]], color=C1, marker="o", markersize=3, lw=0.8,
                alpha=0.55, label="the same factor, conversion cascade (width 15)")
        bx.plot(md, [r["catalytic_weighted_smallest"] for r in matched["rows"]], color=C2, ls=":", marker="^",
                markersize=3.5, label="catalytic cascade, weighted, measured (uncertified)")
    bx.plot(ds, ucert, color=C1, ls="--", marker="o", markersize=3, lw=1.0,
            label="uniform encoding, certified (with normalization c)")
    bx.plot(ds, cert, color=C4, ls="--", marker="D", markersize=3.5,
            label="conversion cascade, weighted, certified (piecewise $K_T$)")
    bx.plot(ds, meas, color=C3, ls=":", marker="^", markersize=3.5,
            label="conversion cascade, weighted, measured (uncertified)")
    rows = [r for r in data("readout", "published_summary")["published_rows"] if "log10_readout_factor" in r]
    small = sorted([r for r in rows if r["n_count_species"] <= 2], key=lambda r: (r["n_count_species"], r["row"]))
    xs, ys = [], []
    for nsp in (1, 2):
        grp = [r for r in small if r["n_count_species"] == nsp]
        jit = np.linspace(-0.16, 0.16, len(grp)) if len(grp) > 1 else [0.0]
        xs += [nsp + j for j in jit]
        ys += [r["log10_readout_factor"] for r in grp]
    bx.scatter(xs, ys, facecolors="none", edgecolors=INK, s=22, zorder=4,
               label=f"published gene-network models, one or two species ({len(small)} instances)")
    for tag, mk, lab in (("lin", "v", "linear"), ("nonlin", "^", "nonlinear")):
        sig = [data("readout", f"signalling_{tag}{dd}") for dd in range(1, 5)]
        sig = [r for r in sig if r is not None]
        if sig:  # a box capped at the ceiling is narrower than the smaller boxes: its point is a lower bound, not joined
            ok = [r for r in sig if not r["capped"]]
            bx.plot([r["d"] for r in ok], [r["log10_readout_factor"] for r in ok], color=C5, marker=mk, lw=1.0,
                    markersize=4, mfc="none", label=f"published {lab} signalling cascade (own boxes)")
            cap = [r for r in sig if r["capped"]]
            bx.plot([r["d"] for r in cap], [r["log10_readout_factor"] for r in cap], color=C5, marker=mk, lw=0,
                    markersize=4, mfc="none")
    lam = [r for r in rows if r["row"] == "lambda_phage"]
    if lam:
        bx.scatter([lam[0]["n_count_species"]], [lam[0]["log10_readout_factor"]], marker="s", facecolors="none",
                   edgecolors=INK, s=30, zorder=4, label="λ-phage (published, five species, own box)")
        bx.annotate("λ-phage\n(lower bound)", (lam[0]["n_count_species"] + 0.12, lam[0]["log10_readout_factor"] - 0.55),
                    fontsize=6.0, color=INK2, va="center")
    bx.set_ylim(0, 7.6)
    bx.set_xlabel("number of molecular species")
    bx.set_ylabel("log₁₀ readout factor")
    bx.set_xticks(range(1, 7))
    bx.legend(loc="upper left", fontsize=5.6, frameon=False, ncol=2)
    panel(bx, "b")
    save(fig, "fig1")


def _num(v):
    return np.nan if v is None else num(v)


def decomposition_rows():
    """Per published instance: the looseness terms of the best cheap reference (the W split of the large boxes where
    it was computed), the tail term, and the current best cheap and exact K_T."""
    summ = data("published_models", "summary")["rows"]
    out = []
    for key in LABEL:
        dec = data("published_models", key)
        terms = dict(dec["decomposition"])
        row = summ[key]
        if row["W_split"]:
            terms.update({k: row["W_split"][k] for k in ("window_penalty", "non_normality", "instance_slack")})
        out.append({"key": key, "label": LABEL[key], "window": _num(terms.get("window_penalty")),
                    "nonnormal": _num(terms.get("non_normality")), "slack": _num(terms.get("instance_slack")),
                    "tail": _num((dec.get("tail") or {}).get("tail_term")),
                    "cheap_K": num(row["cheap_K"]), "decomposed_K": num(dec["best"]["K"]),
                    "exact_K": np.inf if row["exact_K"] is None else num(row["exact_K"]),
                    "exact_note": row["exact_note"], "reference": dec["best"]["reference"], "n_states": dec["n_states"]})
    return out


def fig2():
    d, orc = data("certificates", "toggle_references"), data("certificates", "toggle_exact")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1]})
    names = list(d["references"])
    best = {n: {f: min(num(s["K"][f]) for s in d["references"][n]["schedules"].values()) for f in FAM} for n in names}
    cont = {n: min(num(d["references"][n]["K_continuous_estimate"][f]) for f in FAM) for n in names}
    names.append("exact solution")
    best["exact solution"] = {f: min(num(s["K"][f]) for s in orc["schedules"].values()) for f in FAM}
    cont["exact solution"] = np.nan
    order = sorted(names, key=lambda n: min(best[n].values()), reverse=True)
    y = np.arange(len(order))
    cap_a = 1e4  # the tensor-train references reach 10^36; values beyond the axis are given as numbers
    clip = lambda v: np.where(np.asarray(v, float) > cap_a, np.nan, v)  # noqa: E731
    ax.scatter(clip([best[n]["sqrt_mu"] for n in order]), y, color=C2, marker="s", s=14, label="√μ weight", zorder=3)
    ax.scatter(clip([best[n]["resolvent"] for n in order]), y, color=C1, marker="o", s=16, label="resolvent weight",
               zorder=3)
    # the two panels use different marker sets: squares and circles here, diamonds and crosses in (b)
    ax.scatter(clip([cont[n] for n in order]), y, facecolors="none", edgecolors=C3, marker="^", s=18,
               label="continuous estimate (not rigorous)", zorder=3)
    for yi, n in zip(y, order):
        lo = min(best[n].values())
        if lo > cap_a:
            ax.annotate(f"off scale: $10^{{{np.log10(lo):.0f}}}$", (cap_a, yi), xytext=(-2, 0),
                        textcoords="offset points", fontsize=5.3, color=INK2, va="center", ha="right")
    ax.axvline(THRESH, color=INK2, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(1, cap_a)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_yticks(y, order, fontsize=5.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("$K_T$, toggle switch")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3, fontsize=5.6)
    panel(ax, "a")
    rows = decomposition_rows()
    stat = {}
    for k in LABEL:  # the stationary law as the reference, where it is known (K_T = 0, drawn at the left edge)
        for name in (f"{k}_stationary_known", f"{k}_stationary"):
            v = data("published_models", name)
            if v and "best_K" in v:
                stat[k] = v
                break
    yy = np.arange(len(rows))[::-1]
    cap = 1e5
    for yi, r in zip(yy, rows):
        kc, ke = r["cheap_K"], r["exact_K"]
        bx.scatter([kc], [yi], color=C3, marker="D", s=14, zorder=3)
        if r["key"] in stat:
            bx.scatter([1.0], [yi], color=C5, marker="o", s=22, zorder=4, clip_on=False)
        if np.isfinite(ke) and ke <= cap:
            bx.scatter([ke], [yi], color=C4, marker="X", s=20, zorder=3)
            bx.plot([kc, ke], [yi, yi], color=GRID, lw=1.2, zorder=1)
        else:
            why = "exact: not computable" if not np.isfinite(ke) else f"exact: {ke:.1e} (off scale)"
            bx.annotate(why, (min(kc * 1.5, cap), yi), xytext=(3, 0), textcoords="offset points", fontsize=5.3,
                        color=INK2, va="center")
    bx.axvline(THRESH, color=INK2, ls=":", lw=1)
    bx.set_xscale("log")
    bx.set_xlim(1, cap * 4)
    bx.set_yticks(yy, [r["label"] for r in rows], fontsize=5.8)
    bx.grid(axis="y", visible=False)
    bx.scatter([], [], color=C3, marker="D", label="best cheap reference")
    bx.scatter([], [], color=C4, marker="X", label="exact solution")
    if stat:
        bx.scatter([], [], color=C5, marker="o", label="stationary law ($K_T$ = 0, at the axis)")
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=5.6)
    bx.set_xlabel("$K_T$, published models")
    panel(bx, "b")
    save(fig, "fig2")


COSTED = [("toggle", "toggle"), ("schlogl", "Schlögl"), ("michaelis_menten", "Michaelis–\nMenten"),
          ("self_repressor", "self-\nrepressor"), ("birth_death", "birth–\ndeath"), ("cascade1", "cascade\nd = 1"),
          ("cascade2", "cascade\nd = 2"), ("cascade3", "cascade\nd = 3"), ("cascade4", "cascade\nd = 4")]
ENCODINGS_3 = [("uniform", "uniform [W]", C1, "o"), ("R-NB", "NB-weighted [W]", C3, "^"),
               ("R-mix(NB,E1)", "mixture-weighted [W]", C4, "D")]


def cheaper_design(m, enc, obs="tail", growth="W"):
    """The split-design row of an encoding with the cheaper of the two oracle designs for the diagonal."""
    r = next(r for r in data("costs", "split_design")["rows"]
             if r["model"] == m and r["encoding"] == enc and r["observable"] == obs and r["growth"] == growth)
    return min(r["total_split"], r["total_single"]), r["N_U"]


def ssa_row(m, obs="tail"):
    return next(r for r in data("costs", f"{m}_baselines")["rows"]
                if r["kind"] == "ssa" and r["method"] == "uniformization" and r["observable"] == obs)


def cost_split(m):
    """Uses of the propagator (or of one coherent trajectory) and Toffolis per use, for the cheapest
    master-equation encoding (cheaper oracle design) and for coherent SSA, on the tail probability."""
    tot, nu, enc = min((*cheaper_design(m, e), e) for e, *_ in ENCODINGS_3)
    s = ssa_row(m)
    return {"encoding": enc, "enc_total": tot, "enc_uses": nu, "enc_per_use": tot / nu,
            "ssa_total": s["total"], "ssa_uses": s["uses"], "ssa_per_use": s["total"] / s["uses"]}


def fig3():
    table = data("costs", "comparison")["table"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(COSTED))
    for si, (enc, lab, col, mk) in enumerate(ENCODINGS_3):
        ys = [np.log10(cheaper_design(m, enc)[0]) for m, _ in COSTED]
        ax.scatter(x + (si - 2) * 0.13, ys, color=col, marker=mk, s=20, label=lab, zorder=3)
    lind = [np.log10(next(r for r in table if r["model"] == m and r["observable"] == "tail")["lindblad_best_total"])
            for m, _ in COSTED]
    ax.scatter(x + 0.13, lind, color=C5, marker="v", s=20, label="Lindblad (best order)", zorder=3)
    ax.scatter(x + 0.26, [np.log10(ssa_row(m)["total"]) for m, _ in COSTED], color=INK, marker="*", s=40,
               label="coherent SSA", zorder=4)
    ax.set_xticks(x, [lab for _, lab in COSTED], fontsize=5.6, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_ylabel("log₁₀ total Toffolis (ε = δ = 10⁻²)")
    ax.set_title("tail probability", fontsize=7.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=5.8)
    panel(ax, "a")
    w = 0.36
    for k, (m, _) in enumerate(COSTED):
        c = cost_split(m)
        for off, uses, per, col in ((-w / 2, c["enc_uses"], c["enc_per_use"], C3), (w / 2, c["ssa_uses"], c["ssa_per_use"], INK2)):
            lu, lp = np.log10(uses), np.log10(per)
            bx.bar(k + off, lu, w, color=col, alpha=0.45, edgecolor=col, lw=0.6, zorder=3)
            bx.bar(k + off, lp, w, bottom=lu, color=col, edgecolor=col, lw=0.6, zorder=3)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=C3, alpha=0.45, edgecolor=C3, label="cheapest encoding: uses"),
               Patch(facecolor=C3, edgecolor=C3, label="cheapest encoding: Toffolis per use"),
               Patch(facecolor=INK2, alpha=0.45, edgecolor=INK2, label="coherent SSA: uses"),
               Patch(facecolor=INK2, edgecolor=INK2, label="coherent SSA: Toffolis per use")]
    bx.set_xticks(x, [lab for _, lab in COSTED], fontsize=5.6, rotation=40, ha="right", rotation_mode="anchor")
    bx.set_ylabel("log₁₀ Toffolis (stacked)")
    bx.set_title("where the cost goes (tail probability)", fontsize=7.5)
    bx.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=5.8)
    panel(bx, "b")
    save(fig, "fig3")


def main():
    fig1()
    fig2()
    fig3()


if __name__ == "__main__":
    main()
