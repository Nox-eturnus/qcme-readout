# qcme-readout

Code and data for the article *Readout costs limit claimed exponential speedups of amplitude-encoded master-equation solvers*
(Aarush Das Bansiwal).

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
| `analysis/readout.py` | Readout factors on linear cascades, lambda-phage and the published instances; their sensitivity to the truncation and to the initial law; a correlated (catalytic) cascade and the two cascades on the same boxes; the swap-test example on three grids; the data of Figure 1a; the published signalling cascades; the factor on boxes fitted to the law; lambda-phage on three boxes; the cascades at T = 1, 3 and 10; the cut box of the conversion cascade at these horizons from its closed-form law |
| `analysis/certificates.py` | The weighted encoding on the toggle switch: every reference, weight and window schedule, and the exact solution as the reference |
| `analysis/published_models.py` | Certificates, exact references and the looseness decomposition (`split` recomputes its window term from stored values) on the twelve published instances, including the computations needed for the largest boxes; the stationary law as a reference on the reversible Schlögl model and on the six instances whose stationary law is known; the measured readout factors of the weighted encoding |
| `analysis/costs.py` | Toffoli counts of the uniform and weighted master-equation encodings, coherent stochastic simulation and the Lindblad embedding; gate-level circuit checks; classical reference (finite state projection time, classical SSA trajectories); per-use counts on the cascade up to 12 species from the network structure; the comparison at eps = 1e-3; classical SSA run times |
| `analysis/crosscheck_qualtran.py` | Subroutine counts compared with Qualtran 0.7.0 |
| `analysis/closed_forms.py` | The published closed-form stationary laws of the six one-gene instances against the numerical laws (`compute` needs mpmath) |
| `analysis/kernel_bounds.py` | Rigorous interval bounds on the LCHS kernel discretization |
| `analysis/literature.py` | Tables of the coded literature audit (verdicts, entry path of each work, screening counts) |
| `analysis/figures.py`, `analysis/tables.py` | Figures 1-3 and every table, from the stored results |
| `analysis/check_numbers.py` | Every number quoted in the article, checked against the stored results |
| `run_all.py` | Runs the parts together |
| `results/` | Stored results, one JSON file per calculation, grouped by part (`readout`, `certificates`, `published_models`, `costs`, `literature`) |
| `outputs/` | Generated figures, tables and check reports |
| `manuscript/` | The article and its Supplementary Information (Word), and Supplementary Data 1 (Excel) |
| `tests/` | Numerical identities of the encoding and the certificates, the hard instances of the readout barrier, model transcriptions, circuits and resource formulas |

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
python -m analysis.published_models stationary
python -m analysis.readout sensitivity          # a few minutes
python -m analysis.readout initial_law
python -m analysis.readout correlated           # five species: a few minutes
python -m analysis.readout cascades_matched
python -m analysis.readout swap_variants
python -m analysis.readout cut_horizons         # seconds (closed form)
python -m analysis.published_models stationary_known   # the Gardner toggle: about 5 minutes and 9 GB of memory
python -m analysis.published_models weighted_factors   # builds the references on every box: a few minutes
python -m analysis.costs encodings --only toggle
```

The Qualtran cross-check and the rigorous kernel bounds need extra packages and are run separately:

```
pip install qualtran==0.7.0 mpmath
python -m analysis.crosscheck_qualtran
python -m analysis.kernel_bounds
python -m analysis.closed_forms compute     # then, in the main environment: python -m analysis.closed_forms check
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
- Resource counts are Toffoli gates at relative precision 1e-2 and failure probability 1e-2. Two oracle designs are
  counted for the diagonal of the generator; Figure 3 and Supplementary Table S11 use the cheaper one for each entry,
  Table D5 of Supplementary Data 1 the single-oracle design.
- Run times quoted for quantum routes assume 1e6 Toffoli gates per second (`analysis/tables.py`, `TOFFOLI_RATE`).
- All calculations are deterministic.

## Where each table of the article comes from

The files in `outputs/tables/` keep the names they had when they were first written. Tables S1 to S12 are in the
Supplementary Information, Tables D1 to D13 are the sheets of Supplementary Data 1.

| Article | File | Article | File |
| --- | --- | --- | --- |
| Table 1 | `table1_readout` | Table D1 | `tableS2_excluded` |
| Table 2 | `table2_literature` | Table D2 | `tableS3_models` |
| Table 3 | `table3_checklist` | Table D3 | `table_initial_laws` |
| Table S1 | `tableS1_coding` | Table D4 | `table_benchmarks` |
| Table S2 | `table_tail_factor` | Table D5 | `tableS6_costs` |
| Table S3 | `tableS4_swap_test` | Table D6 | `table_per_use_scaling` |
| Table S4 | `table_swap_variants` | Table D7 | `table_cascade_costs` |
| Table S5 | `table_correlated_cascade` | Table D8 | `tableS8_classical` |
| Table S6 | `table_signalling` | Table D9 | `table_eps_comparison` |
| Table S7 | `tableS5_decomposition` | Table D10 | `tableS9_sensitivity_published` |
| Table S8 | `table_closed_forms` | Table D11 | `tableS9_sensitivity_cascade` |
| Table S9 | `table_stationary_known` | Table D12 | `table_fitted_boxes` |
| Table S10 | `table_weighted_factors` | Table D13 | `table_initial_law_sensitivity` |
| Table S11 | `tableS7_cost_split` | | |
| Table S12 | `table_per_use` | | |

The statements in `outputs/number_check.md` name the part of the article where each number is printed: a Results or Methods subsection, an SI section, or a table.

## Licence and citation

MIT licence (`LICENSE`). Please cite the article; `CITATION.cff` gives the details.
