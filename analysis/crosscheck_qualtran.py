"""Cross-check of the subroutine Toffoli counts in qcme.resources against Qualtran 0.7.0 (QECGatesCost: AND +
Toffoli + controlled swap). Run in an environment with qualtran==0.7.0 installed:

    python -m analysis.crosscheck_qualtran

A row passes if the counts match exactly or differ for one of the two known reasons in KNOWN_DIFFERENCES.
"""
import platform
import warnings

import numpy as np

from .common import save
from qcme import lchs, resources as R

warnings.simplefilter("ignore")
SIZES = (4, 8, 16, 32)
KNOWN_DIFFERENCES = {
    "alias": "Qualtran's QROM prunes unary-iteration branches whose data bits are all zero, so its count depends on "
             "the loaded data (N - 3 at N = 512, N - 7 at N = 1024 for random alias tables). Our qrom(N) = N - 2 is "
             "the data-independent count and an upper bound; accepted when ours >= Qualtran's and the difference is "
             "at most 1%",
    "jacobi_anger_degree": "Qualtran truncates where |J_{d+1}(t)| <= eps/2 (a single-term criterion); our q uses the "
                           "rigorous tail 2 sum_{l>q} |J_l(t)| <= eps, so q >= d",
}


def known_ok(r):
    if r["subroutine"].startswith("alias") and r["qualtran"] is not None:
        return r["own"] >= r["qualtran"] and (r["own"] - r["qualtran"]) <= 0.01 * r["qualtran"]
    return False


def qec(bloq):
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    v = get_cost_value(bloq, QECGatesCost())
    return int(v.and_bloq + v.toffoli + v.cswap), int(v.rotation)


def gqsp_walk_calls():
    """Walk-operator calls of HamiltonianSimulationByGQSP against our rule 2 d (d the GQSP degree)."""
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import get_walk_operator_for_hubbard_model
    from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import HamiltonianSimulationByGQSP
    from qualtran.resource_counting import SympySymbolAllocator
    hs = HamiltonianSimulationByGQSP(get_walk_operator_for_hubbard_model(2, 2, 1, 1), t=0.5, precision=1e-6)
    d = int(hs.degree)
    calls = sum(int(cnt) for bloq, cnt in hs.gqsp.build_call_graph(SympySymbolAllocator()).items()
                if "walk" in str(bloq).lower())
    return {"degree": d, "walk_calls": calls, "own_rule": 2 * d, "match": calls == 2 * d}


def main():
    from importlib.metadata import version

    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add, AddK, LessThanConstant, LessThanEqual, Product
    from qualtran.bloqs.data_loading import QROM
    from qualtran.bloqs.mcmt import MultiControlZ
    from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
    from qualtran.bloqs.state_preparation import PrepareUniformSuperposition, StatePreparationAliasSampling
    from qualtran.linalg.polynomial.jacobi_anger_approximations import degree_jacobi_anger_approximation

    rows = []

    def row(name, size, own, bloq_fn):
        try:
            q, rot = qec(bloq_fn())
            err = ""
        except Exception as e:  # recorded; fails the check
            q, rot, err = None, None, repr(e)
        rows.append({"subroutine": name, "size": size, "own": int(own), "qualtran": q, "qualtran_rotations": rot,
                     "match": q == own, "error": err})

    for n in SIZES:
        row("add", n, R.add(n), lambda: Add(QUInt(n)))
        row("cadd_const", n, R.cadd_const(n), lambda: AddK(QUInt(n), 5).controlled())
        row("add_const", n, R.cadd_const(n), lambda: AddK(QUInt(n), 5))
        row("cmp_const", n, R.cmp_const(n), lambda: LessThanConstant(n, 5))
        row("cmp", n, R.cmp(n), lambda: LessThanEqual(n, n))
        row("mult", n, R.mult(n, n), lambda: Product(n, n))
        row("phasegrad", n, R.phasegrad(n), lambda: AddIntoPhaseGrad(n, n))
        row("mcz", n, R.mcz(n), lambda: MultiControlZ((1,) * n))
    for N in (4, 16, 64, 256, 1024):
        row("qrom", N, R.qrom(N), lambda: QROM.build_from_data(np.arange(N) % 5, target_bitsizes=(8,)))
        for mu in (6, 10):
            probs = np.random.default_rng(N).random(N)
            row(f"alias(mu={mu})", N, R.alias(N, mu),
                lambda: StatePreparationAliasSampling.from_probabilities(probs, precision=2.0**-mu))
    for N in (3, 5, 12, 100, 1000):
        row("uniform_prep", N, R.uniform_prep(N), lambda: PrepareUniformSuperposition(N))
    for r in rows:  # alias sampling: compare with the mu that Qualtran chose, so only the structure is tested
        if r["subroutine"].startswith("alias"):
            N = r["size"]
            mu_req = int(r["subroutine"].split("=")[1].rstrip(")"))
            sp = StatePreparationAliasSampling.from_probabilities(np.random.default_rng(N).random(N),
                                                                  precision=2.0**-mu_req)
            r["own"], r["mu_used"] = R.alias(N, int(sp.mu)), int(sp.mu)
            r["match"] = r["qualtran"] == r["own"]
    ja = []
    for tau in (1.0, 10.0, 100.0, 1000.0):
        for eps in (1e-6, 1e-12, 1e-30):
            d_q = int(degree_jacobi_anger_approximation(tau, precision=eps))
            q_own = lchs.jacobi_anger_degree(tau, eps)
            ja.append({"tau": tau, "eps": eps, "qualtran_degree": d_q, "own_degree": q_own, "own_ge_qualtran": q_own >= d_q})
    walk = gqsp_walk_calls()
    for r in rows:
        r["documented"] = (not r["match"]) and known_ok(r)
        if r["subroutine"] == "uniform_prep":
            L = r["size"]
            while L % 2 == 0:
                L //= 2
            r["rotations_match"] = r["qualtran_rotations"] == (0 if L == 1 else 2)
            r["match"] = r["match"] and r["rotations_match"]
    bad = [r for r in rows if not r["match"] and not r["documented"]]
    passed = (not bad) and all(j["own_ge_qualtran"] for j in ja) and walk["match"]
    save("costs", "qualtran_crosscheck", {"sizes": SIZES},
         {"qualtran": version("qualtran"), "python": platform.python_version(), "rows": rows, "jacobi_anger": ja,
          "gqsp_walk_calls": walk, "documented": KNOWN_DIFFERENCES, "unexplained_mismatches": bad, "pass": passed})
    print(f"Qualtran cross-check: {'PASS' if passed else 'FAIL'}; {sum(r['match'] for r in rows)} of {len(rows)} rows "
          f"match exactly; unexplained: {len(bad)}")


if __name__ == "__main__":
    main()
