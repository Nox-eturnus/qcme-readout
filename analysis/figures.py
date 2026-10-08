"""Figures 1-3 of the article, from the stored results only.

    python -m analysis.figures            -> outputs/figures/fig1.{png,svg}, fig2.{png,svg}, fig3.{png,svg}

Figure 1  a: a toggle-switch distribution, its uniform encoding, a cheap reference and the weighted encoding;
          b: the omitted readout factor against the number of species (cascades, published instances, lambda-phage)
Figure 2  a: K_T for every reference on the toggle switch; b: best cheap and exact references on the published instances
Figure 3  Toffoli counts of the master-equation encodings, coherent SSA and the Lindblad embedding
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
        ax.set_xticks([])
        ax.set_yticks([])
        if k == 0:
            panel(ax, "a")
            ax.set_ylabel("protein 2", fontsize=6.5)
        ax.set_xlabel("protein 1", fontsize=6.5)
    bx = fig.add_subplot(gs[1, :])
    ds = list(range(1, 7))
    uni, cert, meas = [], [], []
    for dd in ds:
        b = data("readout", f"cascade_{dd}")
        uni.append(0.5 * b["log10_S_over_Seff"])
        cert.append(num(b["encodings"]["R-mix(NB,E1)"]["log10_readout"]["cert"]["one"]))
        meas.append(num(b["encodings"]["R-mix(NB,E1)"]["log10_readout"]["meas"]["one"]))
    bx.plot(ds, uni, color=C1, marker="o", label="omitted readout factor √(|S|/S_eff), linear cascade")
    bx.plot(ds, cert, color=C4, ls="--", marker="D", markersize=3.5, label="weighted encoding, certified (piecewise K_T)")
    bx.plot(ds, meas, color=C3, ls=":", marker="^", markersize=3.5, label="weighted encoding, measured (uncertified)")
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
    lam = [r for r in rows if r["row"] == "lambda_phage"]
    if lam:
        bx.scatter([lam[0]["n_count_species"]], [lam[0]["log10_readout_factor"]], marker="s", facecolors="none",
                   edgecolors=INK, s=30, zorder=4, label="λ-phage (published, five species)")
        bx.annotate("λ-phage", (lam[0]["n_count_species"] + 0.1, lam[0]["log10_readout_factor"] + 0.3), fontsize=6.5,
                    color=INK2, va="center")
    bx.set_ylim(0, 7.6)
    bx.set_xlabel("number of molecular species")
    bx.set_ylabel("log₁₀ readout factor")
    bx.set_xticks(range(1, 7))
    bx.legend(loc="upper left", fontsize=6.2, frameon=False, ncol=2)
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
                    "cheap_K": num(row["cheap_K"]), "exact_K": np.inf if row["exact_K"] is None else num(row["exact_K"]),
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
    ax.scatter([best[n]["sqrt_mu"] for n in order], y, color=C2, marker="s", s=14, label="√μ weight", zorder=3)
    ax.scatter([best[n]["resolvent"] for n in order], y, color=C1, marker="o", s=16, label="resolvent weight", zorder=3)
    ax.scatter([cont[n] for n in order], y, facecolors="none", edgecolors=C3, marker="^", s=18,
               label="continuous estimate (not rigorous)", zorder=3)
    ax.axvline(THRESH, color=INK2, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(y, order, fontsize=5.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("K_T, toggle switch")
    ax.legend(loc="upper right", fontsize=5.8)
    panel(ax, "a")
    rows = decomposition_rows()
    yy = np.arange(len(rows))[::-1]
    cap = 1e5
    for yi, r in zip(yy, rows):
        kc, ke = r["cheap_K"], r["exact_K"]
        bx.scatter([kc], [yi], color=C1, marker="o", s=16, zorder=3)
        if np.isfinite(ke) and ke <= cap:
            bx.scatter([ke], [yi], color=C2, marker="s", s=16, zorder=3)
            bx.plot([kc, ke], [yi, yi], color=GRID, lw=1.2, zorder=1)
        else:
            why = "beyond double range" if "beyond" in r["exact_note"] else \
                "not computable" if "not computable" in r["exact_note"] else f"exact {ke:.1e}"
            bx.annotate(why, (min(kc * 1.5, cap), yi), xytext=(3, 0), textcoords="offset points", fontsize=5.3,
                        color=INK2, va="center")
    bx.axvline(THRESH, color=INK2, ls=":", lw=1)
    bx.set_xscale("log")
    bx.set_xlim(1, cap * 4)
    bx.set_yticks(yy, [r["label"] for r in rows], fontsize=5.8)
    bx.grid(axis="y", visible=False)
    bx.scatter([], [], color=C1, marker="o", label="best cheap reference")
    bx.scatter([], [], color=C2, marker="s", label="exact solution")
    bx.legend(loc="lower right", fontsize=5.8)
    bx.set_xlabel("K_T, published models")
    panel(bx, "b")
    save(fig, "fig2")


def fig3():
    table = data("costs", "comparison")["table"]
    ser = [("uniform|W", "uniform [W]", C1, "o"), ("R-NB|W", "R-NB [W]", C3, "^"),
           ("R-mix(NB,E1)|W", "R-mix(NB, E1) [W]", C4, "D"), ("ssa_total", "coherent SSA", INK, "*"),
           ("lindblad_best_total", "Lindblad (best order)", C5, "v")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True, constrained_layout=True)
    for k, (ax, (obs, otitle)) in enumerate(zip(axes, [("one", "total probability, f = 1"), ("tail", "tail probability")])):
        x = np.arange(len(BENCH))
        for si, (key, lab, col, mk) in enumerate(ser):
            ys = [next(r for r in table if r["model"] == m and r["observable"] == obs)[key] for m, _ in BENCH]
            ax.scatter(x + (si - 2) * 0.12, np.log10(ys), color=col, marker=mk, s=22 if mk != "*" else 40, label=lab,
                       zorder=3)
        ax.set_xticks(x, [lab for _, lab in BENCH], fontsize=6.3)
        ax.set_title(otitle, fontsize=7.5)
        panel(ax, "ab"[k])
    axes[0].set_ylabel("log₁₀ total Toffolis (ε = δ = 10⁻²)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=5, fontsize=6.3)
    save(fig, "fig3")


def main():
    fig1()
    fig2()
    fig3()


if __name__ == "__main__":
    main()
