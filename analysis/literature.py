"""Literature part: the coded audit of published proposals (Table 2 of the article and Supplementary Tables S1-S2).

    python -m analysis.literature        -> outputs/tables/table2_literature.{csv,md}, tableS1_coding, tableS2_excluded

The coding itself (results/literature/coding.json) was done by hand from the full texts, with the location of every
coded statement. This script only arranges it into tables and counts the verdicts.
"""
from .common import data
from .tables import write_table

# Short wording for Table 2; the full coding with quotations and locations is in coding.json.
SHORT = {
    "R1": ("CME, 1 species (Schlögl), eigenproblem", "p/‖p‖ (zero mode)", "components of a 2-4 qubit state", "qubit count"),
    "R2": ("reaction–diffusion master equation", "p/‖p‖", "overlap with uniform state", "removes exponential post-processing"),
    "R3": ("Fokker–Planck (reversible)", "exact stationary weight", "basis sampling", "Schrödingerisation cost"),
    "R4": ("Fokker–Planck of low-dim. ODEs", "p/‖p‖", "post-selection (norm decay noted)", "potential advantage"),
    "R5": ("Kolmogorov forward (option baskets)", "p/‖p‖", "swap tests, cost O(1/V₀²)", "exponential in assets"),
    "R20": ("non-reversible Markov chains", "√π", "sampling", "up-to-exponential (mixing)"),
    "R25": ("Fokker–Planck reaction rates (reversible)", "exact √Boltzmann weight", "Hadamard test", "polynomial in particles"),
    "R27": ("Fokker–Planck via linear solvers", "p/‖p‖", "normalized state only", "exponential in dimension"),
    "R28": ("Gibbs sampling (Langevin)", "√σ", "sampling", "up to quartic"),
    "ZHANG": ("PDF transport, reacting flows", "p/‖p‖", "Hadamard test vs fixed state", "polylog in system size"),
    "GOPAL": ("whole-cell CME / SSA (perspective)", "p/‖p‖", "not specified", "log in state-space dimension"),
    "DEVEREUX": ("drift–diffusion, d dimensions", "p/‖p‖", "AE tomography, additive error", "advantage at fixed time"),
    "GOVAERS": ("Fokker–Planck prediction (Bayesian)", "√p", "sampling", "polynomial in grid points"),
    "HARIHARAN": ("multiscale kinetics (perspective)", "walk operators", "phase / amplitude estimation", "conditional on readout"),
}
VERDICT_ORDER = {"omitted, growing": 0, "omitted, bounded": 1, "accounted": 2}


def coding():
    return data("literature", "coding")


def main_table():
    c = coding()
    rows = []
    for e in sorted(c["included"], key=lambda r: (VERDICT_ORDER[r["verdict"]], r["key"])):
        task, enc, read, claim = SHORT[e["key"]]
        v = c["display"][e["key"]]
        rows.append([v["who"], v["ref"], task, enc, read, e["counted"], claim, e["verdict"]])
    return ["Work", "Reference key", "Task", "Encoding", "Readout", "Cost counted", "Scaling claimed", "Verdict"], rows


def coding_table():
    c = coding()
    rows = [[c["display"][e["key"]]["who"], c["display"][e["key"]]["encoding"], c["display"][e["key"]]["readout"],
             e["counted"], c["display"][e["key"]]["claim"], e["verdict"], e["where"]] for e in c["included"]]
    return ["Work", "Encoding", "Readout", "Cost counted", "Claim (quoted)", "Verdict", "Location"], rows


def excluded_table():
    c = coding()
    rows = [[e["ref"], e["reason"]] for e in c["excluded"]]
    rows += [[b["ref"], "baseline (trajectory sampling or density-matrix embedding), listed separately"]
             for b in c["baselines"]]
    return ["Work", "Reason"], rows


def counts():
    c = coding()
    return {"records": c["screen"]["logged_hits"], "after_title_step": c["screen"]["after_mechanical_title_step"],
            "included": c["n_included"], **c["verdict_counts"]}


def main():
    write_table("table2_literature", *main_table())
    write_table("tableS1_coding", *coding_table())
    write_table("tableS2_excluded", *excluded_table())
    print("verdicts:", counts())


if __name__ == "__main__":
    main()
