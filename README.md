# qcme-readout

Code and data for the article *Readout costs decide whether quantum computers can speed up stochastic simulation of
gene networks* (Aarush Das Bansiwal).

Quantum algorithms for the chemical master equation (CME) return a quantum state, and reading a probability out of it
costs a factor that grows with the size of the state space relative to the number of occupied states. This repository
contains the Python package `qcme` and the analysis scripts that compute that factor on benchmark and published
gene-network models, test a reweighted encoding with a classically computable certificate, count fault-tolerant
resources for the quantum routes, and reproduce every figure, table and number in the article and its Supplementary
Information.

## Layout

| Path | Contents |
| --- | --- |
| `qcme/` | The package: reaction networks and models, finite state projection, reference measures, certificates, emulation of the weighted encoding, LCHS kernels, gate-level block encodings, resource counts and quantum baselines |
| `analysis/readout.py` | Readout factors on linear cascades, lambda-phage and the published instances; the swap-test example; the data of Figure 1a |
| `analysis/certificates.py` | The weighted encoding on the toggle switch: every reference, weight and window schedule, and the exact solution as the reference |
| `analysis/published_models.py` | Certificates, exact references and the looseness decomposition on the twelve published instances, including the computations needed for the largest boxes |
| `analysis/costs.py` | Toffoli counts of the uniform and weighted master-equation encodings, coherent stochastic simulation and the Lindblad embedding; gate-level circuit checks |
| `analysis/crosscheck_qualtran.py` | Subroutine counts compared with Qualtran 0.7.0 |
| `analysis/kernel_bounds.py` | Rigorous interval bounds on the LCHS kernel discretization |
| `analysis/literature.py` | Tables of the coded literature audit |
| `analysis/figures.py`, `analysis/tables.py` | Figures 1-3 and every table, from the stored results |
| `analysis/check_numbers.py` | Every number quoted in the article, checked against the stored results |
| `run_all.py` | Runs the parts together |
| `results/` | Stored results, one JSON file per calculation, grouped by part (`readout`, `certificates`, `published_models`, `costs`, `literature`) |
| `outputs/` | Generated figures, tables and check reports |
| `manuscript/` | The article and its Supplementary Information (Word) |
| `tests/` | Numerical identities of the encoding and the certificates, model transcriptions, circuits and resource formulas |

Each result file holds the settings it was computed with, the software environment and the data.

## Installation

Python 3.13 or later.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt      (Windows: .venv\Scripts\pip install -r requirements.txt)
.venv/bin/pip install -e .
```

## Running

```
python run_all.py            # figures, tables and the number check from the stored results (about a minute)
python run_all.py --quick    # also recompute the readout factors and the certificates of the smaller published
                             # instances, and compare them with the stored results (about fifteen minutes on a laptop)
python run_all.py --full     # recompute every result (about a week on a laptop, mostly the two largest published
                             # instances; stages resume from checkpoints)
python -m pytest -m "not slow"   # the test suite (a few minutes); drop -m "not slow" to include the slow checks
```

Single parts can be run on their own, for example

```
python -m analysis.readout cascade --only 1,2,3
python -m analysis.published_models references --only schlogl
python -m analysis.costs encodings --only toggle
```

The Qualtran cross-check and the rigorous kernel bounds need extra packages and are run separately:

```
pip install qualtran==0.7.0 mpmath
python -m analysis.crosscheck_qualtran
python -m analysis.kernel_bounds
```

`run_all.py --quick` writes the recomputed files to `outputs/recomputed/` and reports in
`outputs/recomputation_check.md` how many numbers were compared in each file and whether they agree with `results/`
(relative tolerance 1e-9; run times are not compared). `outputs/number_check.md` lists every number quoted in the
article with the stored value it comes from.

## Conventions

- The CME is truncated to a box S and solved by finite state projection with uniformization, which is accurate entry by
  entry. Boxes are grown until the leaked probability over the horizon is at most 1e-6, up to 10^7 states.
- The readout factor is sqrt(|S| / S_eff) with S_eff = (sum p)^2 / sum p^2.
- A weighted encoding psi = p / sqrt(mu) uses a reference mu built from the reaction network alone. Its certificate
  K_T never reads the CME solution (`tests/test_separation.py` enforces this). The worst case W over the horizon and
  the measured growth use the CME solution and are reported as uncertified.
- Resource counts are Toffoli gates at relative precision 1e-2 and failure probability 1e-2.
- All calculations are deterministic.

## Licence and citation

MIT licence (`LICENSE`). Please cite the article; `CITATION.cff` gives the details.
