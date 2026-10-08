"""qcme: reference-weighted encodings of the chemical master equation.

Core:
  network    reactions, state spaces (boxes and conservation manifolds), truncated generator A_S
  models     benchmark networks and the published gene-network models
  linalg     entrywise-accurate exponential actions for Metzler matrices (uniformization)
  fsp        FSP propagation, box selection, retained mass, observable intervals
  reference  reference measures mu_t: R-P, R-NB, R-mix (E1, E2), R-LNA, R-exact, uniform
  hermite    R-H Hermite references with interval enclosures (rigorous continuous certificate)
  lowrank    R-TT dynamical low-rank references for two-species models
  certify    residual, kappa, CW(w), resolvent weight, piecewise and continuous certificates K_T
  emulate    algorithm-level emulation of the weighted state psi, stabilized propagation, readout
  invariants numerical identities I1-I7, I10, I11 used by the tests

Costs:
  lchs       linear combination of Hamiltonian simulations kernels and discretizations
  circuits   gate-level block encodings for small instances (invariants I8, I9)
  resources  logical resource counts (Toffoli-equivalents)
  baselines  amplitude estimation over coherent SSA, and the Lindblad embedding

Separation rule: `certify` and `reference` never import `fsp` or `emulate`,
so certificates cannot see the CME solution. tests/test_separation.py enforces this.
"""

__version__ = "1.0.0"
