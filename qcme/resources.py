"""Logical resource counts.

Unit: Toffoli-equivalents in the convention of Qualtran 0.7.0's QECGatesCost, i.e. the number of
logical-AND computations (Gidney 2018; measurement-based uncomputation of an AND is free) plus
explicit Toffolis and controlled swaps. Rotations are applied by phase-gradient addition, so they
are Toffolis too; the one-off phase-gradient state is reported separately. Every primitive formula
below is derived from the cited construction and cross-checked against Qualtran 0.7.0
(analysis/crosscheck_qualtran.py), where two documented differences remain.

Separation rule: this module never imports qcme.fsp. Growth factors that use the FSP solution
(Gamma_W, Gamma_meas) are computed in the runner and passed in as numbers.
"""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import lchs


# ----------------------------------------------------------------------------- primitives
def clog2(n):
    """ceil(log2 n) for n >= 1 (floats allowed; exact for integers)."""
    n = float(n)
    if n <= 1:
        return 0
    c = int(np.ceil(np.log2(n)))
    return c - 1 if 2.0 ** (c - 1) >= n else c


def add(n):
    """In-place n-bit addition mod 2^n (Gidney 2018): n - 1 ANDs."""
    return max(int(n) - 1, 0)


def cadd_const(n):
    """(Controlled) addition of a classical constant: the constant is loaded by X or CNOT gates, then add."""
    return add(n)


def cmp_const(n):
    """x < K for a classical K: the carry chain of x - K into a fresh bit (n ANDs)."""
    return int(n)


def cmp(n):
    """a <= b for two n-bit registers, output computed and later uncomputed coherently: 2n - 1."""
    return 2 * int(n) - 1


def qrom(N):
    """Unary-iteration QROM over N entries (Babbush et al. 2018): N - 2 ANDs (N >= 2); width-free."""
    N = int(N)
    return max(N - 2, 0)


def qrom_uncompute(N):
    """Measurement-based QROM uncomputation (Berry et al. 2019, App. C): about 2 sqrt(N)."""
    N = int(N)
    return 0 if N <= 2 else int(np.ceil(2 * np.sqrt(N)))


def mult(n, m):
    """Schoolbook n x m multiplication by m controlled additions: 2nm - m."""
    return 2 * int(n) * int(m) - int(m)


def phasegrad(b):
    """Rotation by a b-bit angle register, by addition into a phase-gradient state: b - 2."""
    return max(int(b) - 2, 0)


def mcz(m):
    """Z controlled on m qubits: m - 1 ANDs."""
    return max(int(m) - 1, 0)


def uniform_prep(N, b=None):
    """Uniform superposition over N = 2^k L states (L odd): nothing for L = 1; otherwise the Babbush
    et al. 2018 Fig. 12 construction (an inequality test on log2 L qubits with one round of amplitude
    amplification): 3 ceil(log2 L) - 1 ANDs plus two single-qubit rotations, applied here by phase
    gradient when b is given."""
    N = int(N)
    L = N
    while L % 2 == 0:
        L //= 2
    if L == 1:
        return 0
    return 3 * clog2(L) - 1 + (2 * phasegrad(b) if b is not None else 0)


def alias(N, mu):
    """Alias-sampling state preparation over N amplitudes with mu-bit keep probabilities (Babbush et
    al. 2018): uniform prep + QROM(N) + mu-bit comparator + ceil(log2 N) controlled swaps."""
    return uniform_prep(N, mu) + qrom(N) + cmp(mu) + clog2(N)


def rotation_tree(n_bits, b):
    """Arbitrary state on an n-qubit register by a tree of multiplexed rotations: level l has 2^l
    angles loaded by QROM (compute and measurement-based uncompute) and applied by phase gradient."""
    tot = 0
    for l in range(n_bits):
        tot += qrom(2**l) + qrom_uncompute(2**l) + phasegrad(b)
    return tot


# ----------------------------------------------------------------------------- function evaluation
@dataclass(frozen=True)
class FunctionCost:
    toffoli: int
    pieces: int
    degree: int


@lru_cache(maxsize=None)
def function_eval(fname, lo, hi, b):
    """Piecewise-polynomial evaluation of a named function on [lo, hi] to absolute error 2^{-b}
    with b-bit fixed point (Haner et al. 2018): coefficients loaded by QROM over 2^m equal pieces,
    then Horner with D b x b multiplications and D additions.

    Half the budget, 2^{-b-1}, goes to interpolation. The Chebyshev-node interpolation error on a
    piece of width w is at most 2 (w/4)^{D+1} max|f^(D+1)|/(D+1)!, with the derivative bounds of
    DERIVATIVE_BOUNDS; the other half covers fixed-point rounding (b bits plus guard bits in the
    caller). For each D the smallest m meeting the bound is taken; the cheapest (m, D) is returned.
    Functions with a singularity on [lo, hi] (arcsin at 1) return a lower bound that is never
    competitive (see FUNCTIONS)."""
    from math import lgamma, log
    if fname == "arcsin_exp":  # sqrt singularity at z = 0: uniform pieces need w <~ 2^{-2b}
        m = min(2 * b, 62)
        return FunctionCost(qrom(2**m), 2**m, 1)
    logM = DERIVATIVE_BOUNDS[fname]
    log_tol = -(b + 1) * log(2.0)
    rng = float(hi - lo)
    best = None
    for D in range(1, 64):
        # 2 (w/4)^{D+1} e^{logM(D+1)}/(D+1)! <= tol  ->  log w <= log 4 + (log_tol - log 2 - logM + lgamma(D+2))/(D+1)
        log_w = log(4.0) + (log_tol - log(2.0) - logM(D + 1) + lgamma(D + 2)) / (D + 1)
        m = max(0, int(np.ceil(np.log2(rng) - log_w / log(2.0))))
        if m > 62:
            continue
        P = 2**m
        cost = qrom(P) + qrom_uncompute(P) + D * (mult(b, b) + add(b))
        if best is None or cost < best.toffoli:
            best = FunctionCost(cost, P, D)
    return best if best is not None else FunctionCost(10**18, 0, 0)


FUNCTIONS = {  # the functions evaluated in circuits (numerical forms, used by tests)
    "exp": np.exp,  # z <= 0 -> e^z (inequality-test amplitude)
    "exp_half": lambda z: np.exp(0.5 * z),
    "softplus": lambda u: np.logaddexp(0.0, u),
    "arcsin_exp": lambda z: np.arcsin(np.exp(np.minimum(z, 0.0))) / (np.pi / 2),  # singular at z = 0
}

# log of a bound on max |f^(n)| over the evaluation range, as a function of n
DERIVATIVE_BOUNDS = {
    "exp": lambda n: 0.0,  # e^z <= 1 on z <= 0
    "exp_half": lambda n: -n * np.log(2.0),
    # softplus' = sigmoid (<= 1); sigmoid is analytic in |Im z| < pi with |sigmoid| <= 8 on |Im z| = 3,
    # so by Cauchy's estimate |sigmoid^(n-1)| <= 8 (n-1)!/3^(n-1)
    "softplus": lambda n: 0.0 if n <= 1 else float(np.log(8.0) + __import__("math").lgamma(n) - (n - 1) * np.log(3.0)),
}


def eval_cost(fname, lo, hi, b):
    return function_eval(fname, float(lo), float(hi), int(b)).toffoli


# ----------------------------------------------------------------------------- network structure
def species_bits(widths):
    return [max(clog2(w), 1) for w in widths]


def dependence(gen, reactions=None, tol=1e-12):
    """For each reaction, the species its propensity depends on.

    With `reactions`, dependence is tested on the enclosing register box (every combination of
    register values), which is what an oracle reads: vary one species with the others fixed. Without
    it, variation is tested on the state space only, which on a
    conservation manifold misses every dependence, because no species can move alone there
   ."""
    if reactions is not None:
        widths = tuple(int(w) for w in gen.space.widths)
        from .network import StateSpace
        X = StateSpace.box(widths).X
        deps = []
        for rxn in reactions:
            A = np.broadcast_to(np.asarray(rxn.prop(X), dtype=float), (len(X),)).reshape(widths)
            scale = max(1.0, float(np.abs(A).max()))
            deps.append({i for i in range(len(widths)) if np.any(np.ptp(A, axis=i) > tol * scale)})
        return deps
    X, a = gen.space.states, gen.a
    deps = []
    for k in range(a.shape[0]):
        dk = set()
        for i in range(X.shape[1]):
            others = np.delete(X, i, axis=1)
            _, inv = np.unique(others, axis=0, return_inverse=True)
            inv = np.ravel(inv)
            lo = np.full(inv.max() + 1, np.inf)
            hi = np.full(inv.max() + 1, -np.inf)
            np.minimum.at(lo, inv, a[k])
            np.maximum.at(hi, inv, a[k])
            if np.any(hi - lo > tol * max(1.0, np.abs(a[k]).max())):
                dk.add(i)
        deps.append(dk)
    return deps


@dataclass
class ModelStructure:
    """What the circuits need to know about a model: registers and per-reaction factor sets."""

    widths: tuple
    bits: list
    deps: list  # species each propensity depends on
    supp: list  # species each reaction changes
    K: int
    conservation: object
    n_states: int

    @classmethod
    def build(cls, model, gen, widths):
        rx = model.reactions(widths)
        supp = [set(np.nonzero(np.asarray(r.nu))[0].tolist()) for r in rx]
        return cls(tuple(widths), species_bits(widths), dependence(gen, rx), supp, len(supp), model.conservation,
                   len(gen.space))

    @property
    def d(self):
        return len(self.widths)

    @property
    def n_sys(self):
        return int(sum(self.bits))

    def box_padding(self):
        """True for species whose width is not a power of two (an in-box test is needed)."""
        return [w != 2**n for w, n in zip(self.widths, self.bits)]


# ----------------------------------------------------------------------------- oracles
@dataclass
class OracleCost:
    """Toffolis of one query to the block encoding U_B of B~/alpha, by part."""

    prep: int = 0
    select: int = 0
    amplitude: int = 0
    shift: int = 0
    boundary: int = 0
    diagonal: int = 0
    shared: int = 0
    parts: dict = field(default_factory=dict)

    @property
    def total(self):
        return self.prep + self.select + self.amplitude + self.shift + self.boundary + self.diagonal + self.shared


def in_box_cost(st):
    """Flag x in S: comparators for padded species, conservation-law equality checks (compute and
    uncompute)."""
    c = sum(cmp_const(n) for n, pad in zip(st.bits, st.box_padding()) if pad)
    if st.conservation is not None:
        L, _ = st.conservation
        for row in np.atleast_2d(L):
            sup = np.nonzero(row)[0]
            nb = max(st.bits[i] for i in sup) + clog2(len(sup))
            c += (len(sup) - 1) * add(nb) + cmp_const(nb) + cmp_const(nb)  # sum, then equality as two tests
    return 2 * c


def amplitude_cost(n_factor_bits, b, extra_log_terms=0, softplus_pairs=0, zlo=None):
    """Amplitude oracle for v(x) = product of univariate factors (x a list of factor-register widths).

    One factor and no extra terms: method A, an angle QROM and a phase-gradient rotation.
    Otherwise both methods are costed in the log domain (factor QROMs, adders, softplus pairs) and
    the cheaper is taken: (A) arcsin(exp z) evaluation + rotation, (B) exp z evaluation + comparator
    (inequality test). Arithmetic is uncomputed (x2); QROMs are uncomputed by measurement."""
    zlo = -b * np.log(2.0) if zlo is None else zlo
    lookups = sum(qrom(2**n) + qrom_uncompute(2**n) for n in n_factor_bits)
    if len(n_factor_bits) == 1 and extra_log_terms == 0 and softplus_pairs == 0:
        return lookups + phasegrad(b), "A(lookup)"
    nterms = len(n_factor_bits) + extra_log_terms + 2 * softplus_pairs
    arith = max(nterms - 1, 0) * add(b) + 2 * softplus_pairs * eval_cost("softplus", -b * np.log(2), b * np.log(2), b)
    a_cost = lookups + 2 * (arith + eval_cost("arcsin_exp", zlo, 0.0, b)) + phasegrad(b)
    b_cost = lookups + 2 * (arith + eval_cost("exp", zlo, 0.0, b)) + cmp(b)
    return (a_cost, "A(arith)") if a_cost < b_cost else (b_cost, "B(ineq)")


def oracle_cost(st, reftype, b, diag_scale=True, diag="value"):
    """Toffolis of one query to U_B for the uniform (reftype='uniform'), product ('product') or
    mixture ('mix') reference; the same count applies to every window (only table contents change).

    diag: "value" (the single diagonal oracle: a_0 in value arithmetic, scaled, inequality test) or "split"
    (the split diagonal; see split_diagonal_cost)."""
    oc = OracleCost()
    nterm = st.K + 1 if diag == "value" else st.K + split_diagonal_terms(st)
    oc.prep = 2 * alias(2 ** clog2(nterm), b)  # PREPARE and PREPARE^dagger over reactions + diagonal
    oc.select = qrom(nterm)  # unary iteration over the term index (controls each O_k, S_k)
    methods = []
    if reftype == "mix":
        # u(x) = log(eta/(1-eta)) + sum_i l_i(x_i): d lookups and d - 1 adders, computed once per query
        oc.shared = sum(qrom(2**n) + qrom_uncompute(2**n) for n in st.bits) + 2 * (st.d - 1) * add(b)
    for k in range(st.K):
        if reftype == "uniform":
            F = sorted(st.deps[k])
            extra, sp = 0, 0
        elif reftype == "product":
            F = sorted(st.deps[k] | st.supp[k])
            extra, sp = 0, 0
        elif reftype == "mix":
            F = sorted(st.deps[k] | st.supp[k])
            extra, sp = len(st.supp[k]), 1  # u(x + nu) - u(x) difference lookups; one softplus pair
        else:
            raise ValueError(reftype)
        if not F:  # constant propensity, uniform reference: a fixed rotation
            c, meth = phasegrad(b), "A(const)"
        else:
            c, meth = amplitude_cost([st.bits[i] for i in F], b, extra, sp)
            if reftype == "mix":
                c += sum(qrom(2 ** st.bits[i]) + qrom_uncompute(2 ** st.bits[i]) for i in st.supp[k])
        methods.append(meth)
        oc.amplitude += c + 1  # +1 AND: zero-propensity flag
        oc.shift += sum(cadd_const(st.bits[i]) for i in st.supp[k])
        oc.boundary += 2 * sum(cmp_const(st.bits[i]) for i in st.supp[k]) + 1
    # diagonal term d(x) = -(a0(x) + c): a0 from value tables (univariate propensities grouped per
    # species; multivariate ones by multiplying their factors), plus c, scaled, inequality test.
    groups = {}
    multi = 0
    for k in range(st.K):
        dk = sorted(st.deps[k])
        if len(dk) == 1:
            groups.setdefault(dk[0], 0)
        elif len(dk) >= 2:
            multi += len(dk) - 1
            for i in dk:
                groups.setdefault(("m", k, i), 0)
    tables = sum(qrom(2 ** st.bits[g if isinstance(g, int) else g[2]]) +
                 qrom_uncompute(2 ** st.bits[g if isinstance(g, int) else g[2]]) for g in groups)
    n_sum = len(groups) - multi
    arith = multi * mult(b, b) + max(n_sum, 0) * add(b) + add(b) + (mult(b, b) if diag_scale else 0)
    oc.diagonal = tables + 2 * arith + cmp(b) + in_box_cost(st)
    if diag == "split":
        oc.diagonal = split_diagonal_cost(st, b)
    elif diag != "value":
        raise ValueError(diag)
    oc.parts = {"methods": methods, "diag": diag, "nterm": nterm}
    return oc


def diagonal_groups(st):
    """(species with univariate propensities, multivariate reactions, constant reactions)."""
    uni = sorted({next(iter(st.deps[k])) for k in range(st.K) if len(st.deps[k]) == 1})
    multi = [k for k in range(st.K) if len(st.deps[k]) >= 2]
    const = [k for k in range(st.K) if not st.deps[k]]
    return uni, multi, const


def needs_out_term(st):
    return any(st.box_padding()) or st.conservation is not None


def split_diagonal_terms(st):
    uni, multi, _ = diagonal_groups(st)
    return len(uni) + len(multi) + 1 + (1 if needs_out_term(st) else 0)


def split_diagonal_cost(st, b):
    """The split diagonal design: the diagonal d(x) = -(a_0(x) + c) as separate LCU terms: one per
    species with univariate propensities (their sum stored as an angle table, method A), one per
    multivariate propensity (log-domain amplitude oracle), a constant term -(sum of constant
    propensities + c) (a fixed rotation), and, when S is not the whole register box, a term
    -(|c| + max a_0) active only off S. Every term is controlled on the in-S flag (+1 AND); the flag is
    computed once per query."""
    uni, multi, _ = diagonal_groups(st)
    c = sum(qrom(2 ** st.bits[i]) + qrom_uncompute(2 ** st.bits[i]) + phasegrad(b) + 1 for i in uni)
    for k in multi:
        c += amplitude_cost([st.bits[i] for i in sorted(st.deps[k])], b)[0] + 1
    c += phasegrad(b) + 1  # constant term
    if needs_out_term(st):
        c += phasegrad(b) + 1
    return c + in_box_cost(st)


def reweight_cost(st, reftype, b):
    """D_j(x) = log rho_j - (log mu^(j) - log mu^(j+1))/2 in fixed point: product references need one
    lookup per species (difference tables) and d - 1 adders; mixtures also two softplus pairs."""
    look = sum(qrom(2**n) + qrom_uncompute(2**n) for n in st.bits)
    arith = (st.d - 1) * add(b) + add(b)
    if reftype == "mix":
        look *= 2
        arith += 2 * st.d * add(b) + 2 * eval_cost("softplus", -b * np.log(2), b * np.log(2), b) + 2 * add(b)
    return look, arith


# ----------------------------------------------------------------------------- walk and LCHS
def ancilla_qubits(st, b):
    """Rough logical-qubit count of the U_B workspace (select, flags, b-bit arithmetic registers)."""
    return clog2(st.K + 1) + 4 + 4 * b


@dataclass
class LCHSCost:
    """Cost of one use of U, the block encoding of W/s, for one encoding."""

    variant: str  # "single" (one time-dependent LCHS across windows) or "per_window"
    kernel: str
    s: float  # subnormalization (l1 norm of LCHS coefficients; product over windows if per_window)
    toffoli_U: float
    walk_steps: int
    q: list  # Jacobi-Anger degree per window
    nodes: int
    K: float
    h: float
    b: int
    query: int  # Toffolis of one U_B query
    breakdown: dict
    lchs_error: float
    qubits: int


def lchs_cost(st, reftype, alphas, dts, Dmax, delta, kernel, variant="single", b=None, oc=None,
              diag="value"):
    """Toffolis of one use of U for window normalizations `alphas`, lengths `dts` and jump sizes
    `Dmax` (max_x D_j(x), one per boundary), with total error budget `delta` split equally between
    LCHS discretization, Hamiltonian simulation and oracle rounding.

    kernel: a callable (delta_lchs) -> lchs.Kernel, so K2's eps_lchs follows the budget."""
    alphas, dts, Dmax = np.asarray(alphas, float), np.asarray(dts, float), np.asarray(Dmax, float)
    J = len(alphas)
    d_lchs = d_hs = d_rnd = delta / 3.0
    if variant == "single":
        blocks = [(alphas, dts, Dmax)]
    elif variant == "per_window":
        blocks = [(alphas[j:j + 1], dts[j:j + 1], np.zeros(0)) for j in range(J)]
    else:
        raise ValueError(variant)
    nb = len(blocks)
    discs = []
    ker = kernel(d_lchs / nb / 3.0)  # one kernel object: its strip integrals are cached across blocks
    for al, dt, dm in blocks:
        Lnorm = float(np.sum(al * dt) + np.sum(dm))
        discs.append(lchs.discretize(ker, Lnorm, d_lchs / nb))
    s = float(np.prod([dc.l1 for dc in discs]))
    Kmax = max(dc.K for dc in discs)
    if b is None:
        b = clog2(s * np.sqrt(1 + Kmax**2) * (st.K + 1) * float(np.sum(alphas * dts)) / d_rnd) + 2
    if oc is None:
        oc = oracle_cost(st, reftype, b, diag=diag)
    q_list, steps = [], 0
    n_anc = ancilla_qubits(st, b) + 1  # + the self-inverse select qubit
    per_step = 2 * oc.total + 2 * 2 * phasegrad(b) + mcz(n_anc) + phasegrad(b) + 2
    tof = 0.0
    for dc, (al, dt, dm) in zip(discs, blocks):
        Lam = al * np.sqrt(1 + dc.K**2)
        dhs = d_hs / (s * nb * len(al))
        qs = [lchs.jacobi_anger_degree(L * t, dhs) for L, t in zip(Lam, dt)]
        q_list += qs
        st_steps = 2 * int(np.sum(qs))
        steps += st_steps
        nodes = dc.M
        node_prep = 2 * alias(2 ** clog2(nodes), b)
        angles = 2 * (qrom(nodes) + qrom_uncompute(nodes))
        jumps = 0
        if len(dm):
            look, arith = reweight_cost(st, reftype, b)
            jumps = len(dm) * (look + 2 * (arith + mult(b, b)) + phasegrad(b))
        tof += st_steps * per_step + node_prep + angles + jumps
    if variant == "per_window" and J > 1:  # exact reweighting block encodings V_j between windows
        look, arith = reweight_cost(st, reftype, b)
        tof += (J - 1) * (look + 2 * (arith + eval_cost("exp", -b * np.log(2), 0.0, b)) + cmp(b))
    breakdown = {"walk_query": steps * 2 * oc.total, "walk_other": steps * (per_step - 2 * oc.total),
                 "lchs_overhead": tof - steps * per_step}
    qubits = st.n_sys + n_anc + clog2(max(dc.M for dc in discs)) + 1 + 2 * b + b
    return LCHSCost(variant, discs[0].kernel.name, s, tof, steps, q_list, max(dc.M for dc in discs), Kmax,
                    discs[0].h, b, oc.total, breakdown, sum(dc.error for dc in discs), qubits)


# ----------------------------------------------------------------------------- amplitude estimation
AE_GAP = 8 / np.pi**2 - 0.5


def median_runs(delta_fail):
    return int(np.ceil(np.log(1 / delta_fail) / (2 * AE_GAP**2)))


def t2_grover(eta):
    """T2: canonical AE with M = ceil(8 pi/eta) (the Hadamard-test success probability is near 1/2)."""
    return int(np.ceil(8 * np.pi / eta))


def bhmt_M(P, eps_abs):
    """Smallest M with 2 pi sqrt(P(1-P))/M + pi^2/M^2 <= eps_abs (BHMT 2002, Thm 12, k = 1)."""
    a = 2 * np.pi * np.sqrt(max(P * (1 - P), 0.0))
    M = (a + np.sqrt(a * a + 4 * eps_abs * np.pi**2)) / (2 * eps_abs)
    return int(np.ceil(M))


@dataclass
class ReadoutCost:
    label: str
    gamma: float
    R_f: float
    eta: float
    M: int
    n: int
    N_U: int
    toffoli_U: float
    toffoli_prep: float
    toffoli_total: float
    qubits: int
    lchs: object = None


def weighted_readout(st, reftype, gamma_unsub, R_f, eps, delta_fail, alphas, dts, Dmax, kernel,
                     variant, prep_cost, include_s=True, label="", diag="value"):
    """T2 total for one growth factor. gamma_unsub = ||psi_0|| x growth (without s); the solver's s is
    multiplied in when include_s (certified path); the idealized growth-aware variants pass False.
    prep_cost: Toffolis of the two state preparations per use (a number, or a function of b).

    The solver error budget is delta = eta/2 with eta = eps/(Gamma R_f); since s enters Gamma and s
    depends on the discretization (weakly, through eps_lchs), the budget is solved by fixed point."""
    s = 1.0
    for _ in range(6):
        gamma = gamma_unsub * (s if include_s else 1.0)
        eta = eps / (gamma * R_f)
        cost = lchs_cost(st, reftype, alphas, dts, Dmax, eta / 2, kernel, variant, diag=diag)
        if abs(cost.s - s) <= 1e-9 * s:
            break
        s = cost.s
    gamma = gamma_unsub * (cost.s if include_s else 1.0)
    eta = eps / (gamma * R_f)
    M, n = t2_grover(eta), median_runs(delta_fail)
    N_U = 2 * M * n
    prep = prep_cost(cost.b) if callable(prep_cost) else prep_cost
    refl = M * n * mcz(cost.qubits)
    total = N_U * (cost.toffoli_U + prep) + refl
    return ReadoutCost(label, gamma, R_f, eta, M, n, N_U, cost.toffoli_U, prep, float(total), cost.qubits,
                       cost)


def product_state_prep(bits, b):
    """Register-by-register preparation of a product state (two per Hadamard-test use: psi and phi,
    and their inverses inside AE are part of N_U's count of uses)."""
    return sum(rotation_tree(n, b) for n in bits)


def generic_state_prep(n_states, b):
    """Non-product state over |S| amplitudes by alias sampling (index -> state loaded by the QROM)."""
    return alias(2 ** clog2(n_states), b)


# ----------------------------------------------------------------------------- register inventory
# Exact register inventory, rotation counts and T-counts of both oracle designs (the totals use the
# `ancilla_qubits` formula; the exact inventory changes them by a negligible amount).
def eval_workspace(fname, lo, hi, b):
    """Qubits live during one piecewise-polynomial evaluation: (D + 1) b coefficient qubits, a b-bit
    accumulator, a 2b-bit product and log2(pieces) - 1 unary-iteration ancillas."""
    fc = function_eval(fname, float(lo), float(hi), int(b))
    return (fc.degree + 1) * b + b + 2 * b + max(clog2(fc.pieces) - 1, 0)


def oracle_workspace(st, reftype, b, diag="value"):
    """Maximum transient qubits of one U_B query (terms act one after another under SELECT)."""
    z = b * np.log(2.0)
    ex = eval_workspace("exp", -z, 0.0, b)
    sp = eval_workspace("softplus", -z, z, b)
    best = 0
    for k in range(st.K):
        F = sorted(st.deps[k]) if reftype == "uniform" else sorted(st.deps[k] | st.supp[k])
        una = max([st.bits[i] - 1 for i in F] or [0])
        if len(F) <= 1 and reftype != "mix":
            w = b + una  # angle register + unary ancillas
        else:
            w = len(F) * b + b + una + max(ex, b)  # factor registers, sum, then exp and comparator
            if reftype == "mix":
                w += b + 2 * b + sp  # u difference, two softplus outputs, softplus workspace
        best = max(best, w + 2)  # + boundary-flag scratch and a carry
    if diag == "value":
        dw = b + b + 2 * b + b  # accumulator, lookup, product, comparator carries
    else:
        uni, multi, _ = diagonal_groups(st)
        dw = b + max([st.bits[i] for i in uni] or [0])
        for k in multi:
            dw = max(dw, len(st.deps[k]) * b + b + max(ex, b))
    return max(best, dw)


def register_inventory(st, reftype, b, nodes, M_ae, prep_kind, n_states, diag="value", n_windows=1):
    """Exact logical qubits (maximum simultaneously live) of the weighted or uniform solver
    inside canonical amplitude estimation. Returns (total, itemized dict)."""
    nterm = st.K + 1 if diag == "value" else st.K + split_diagonal_terms(st)
    persistent = {
        "system": st.n_sys,
        "AE phase register": clog2(M_ae),
        "Hadamard-test qubit": 1,
        "phase-gradient register": b,
        "LCHS node register": clog2(nodes),
        "node PREPARE junk (keep, alt, flag)": b + clog2(nodes) + 1,
        "LCHS angle registers (phi, chi)": 2 * b,
        "GQSP signal qubit": 1,
        "self-inverse select qubit": 1,
        "term select register": clog2(nterm),
        "term PREPARE junk": b + clog2(nterm) + 1,
        "block-encoding flags (amplitude, boundary, zero, in-S)": 4,
        "inequality-test register": b,
        "shared u register (R-mix)": b if reftype == "mix" else 0,
    }
    transient = {"oracle workspace (max over terms)": oracle_workspace(st, reftype, b, diag)}
    if n_windows > 1:
        transient["jump workspace"] = st.d * b + b + 2 * b + (2 * b + eval_workspace(
            "softplus", -b * np.log(2), b * np.log(2), b) if reftype == "mix" else 0)
    if prep_kind.startswith("generic"):
        transient["state preparation"] = b + 2 * clog2(n_states) + 1
    else:
        transient["state preparation"] = b + max(st.bits)
    total = sum(persistent.values()) + max(transient.values())
    return int(total), {"persistent": persistent, "transient_max_of": transient}


def rotation_counts(st, b, walk_steps, n_jumps, M_ae, n_runs, N_U, prep_kind, diag="value", methods=()):
    """Phase-gradient rotations (already counted as Toffolis) and directly synthesized rotations
    (the phase-gradient state and AE's inverse QFT), with the T-count of the synthesized ones
    (Ross-Selinger: 3 log2(1/eps) + 2 T per rotation at eps = 2^-b)."""
    a_terms = sum(1 for m in methods if m.startswith("A"))
    if diag == "split":
        uni, _, _ = diagonal_groups(st)
        a_terms += len(uni) + 1 + (1 if needs_out_term(st) else 0)
    per_U = walk_steps * (4 + 1 + 2 * a_terms) + n_jumps
    prep = 0 if prep_kind.startswith("generic") else 2 * sum(st.bits)
    pg = N_U * (per_U + prep)
    m = clog2(M_ae)
    synth = n_runs * (b + m * (m - 1))
    return {"phase_gradient_rotations": float(pg), "synthesized_rotations": int(synth),
            "T_synthesis": int(synth * (3 * b + 2))}
