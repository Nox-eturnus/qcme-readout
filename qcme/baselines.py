"""Quantum baselines: amplitude estimation (AE) over
a coherent stochastic simulation algorithm (SSA), and the Lindblad embedding of the CME.

Both estimate P = E[f(X_T)]/f_max for 0 <= f <= f_max by AE on a probability, with the same BHMT
(2002, Thm 12) accounting as T2. Error budget for relative precision eps on E f (eps P absolute on P):
  AE statistical error          eps P / 2
  law bias: truncation of the event count (SSA) or time discretization (Lindblad)   eps P / 4
  fixed-point rounding of probabilities / amplitudes                                eps P / 8
  Hamiltonian simulation (Lindblad only; for SSA this share is unused)              eps P / 8

The Lindblad schemes are those of Ding, Li, Lin (arXiv:2311.15533): the circuit applies
e^{-i sqrt(dt) H} to |0>_anc (x) rho and traces the ancilla, with the star-shaped dilated Hamiltonian
H = sum_a (|a><0| (x) H_a + h.c.). For CME jump operators every block H_a is a shift times a diagonal
(net stoichiometry fixed per block), so C^dagger C = sum_a H_a^dagger H_a = diag(q) and the channel
acts on diagonal rho as the linear map
  p'(x) = cos^2(sqrt(dt q(x))) p(x) + sum_a |h_a(y)|^2 sin^2(sqrt(dt q(y)))/q(y) p(y),   y -> x by block a,
which is emulated exactly here to choose the step count.
  first order:  H_j = J_j (one block per reaction; h_j(y) = sqrt(a_j(y))).
  second order: H_{1,j} = J_j + dt (1/2 V0 J_j + 1/2 J_j V0 + 1/6 J_j sum_j' J_j'^dagger J_j'),
                V0 = -(1/2) sum_j J_j^dagger J_j; H_{2,j,k} = sqrt(dt/2) J_j J_k (their Appendix A,
                with no system Hamiltonian; the O(dt^{3/2}) correction of H_2 changes only O(dt^3) terms).
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats

from . import lchs
from . import resources as R


# ----------------------------------------------------------------------------- AE on a probability
def ae_cost(P, eps, delta_fail):
    """(Grover budget M, median runs n) for relative precision eps on a probability P (AE share eps P/2)."""
    return R.bhmt_M(P, 0.5 * eps * P), R.median_runs(delta_fail)


# ----------------------------------------------------------------------------- coherent SSA
def uniformization_rate(gen):
    return float(gen.a.sum(axis=0).max())


def poisson_truncation(lam_T, tail):
    """Smallest N with P(Poisson(lam_T) > N) <= tail."""
    N = int(stats.poisson.ppf(1.0 - tail, lam_T))
    while stats.poisson.sf(N, lam_T) > tail:
        N += 1
    return N


@dataclass
class SSACost:
    label: str
    Lambda: float
    N_steps: int
    b: int
    step: float
    per_A: float
    M: int
    n: int
    uses: int
    total: float
    qubits: int
    P: float


def propensity_value_cost(st, b):
    """Toffolis to compute all K propensities in b-bit fixed point (and later uncompute): univariate
    ones by value QROM, multivariate ones as products of univariate factor lookups."""
    look, arith = 0, 0
    for k in range(st.K):
        F = sorted(st.deps[k])
        look += sum(R.qrom(2 ** st.bits[i]) + R.qrom_uncompute(2 ** st.bits[i]) for i in F)
        arith += max(len(F) - 1, 0) * R.mult(b, b)
    return look, arith


def coherent_ssa_cost(st, gen, T, P, eps, delta_fail, f_cost=0, method="uniformization", label="",
                      N=None):
    """AE over a coherent SSA with a fixed number of steps.

    uniformization: Lambda = max_S a_0; jump count ~ Poisson(Lambda T) drawn coherently by alias
      sampling over 0..N; each step compares a fresh b-bit uniform u against the K cumulative
      propensities/Lambda, applies nu_k controlled on the one-hot choice, and is active only while
      step index < jump count.
    direct: the same N (the event count of the direct method is stochastically dominated by
      Poisson(max a_0 T)), plus a coherent waiting time -ln(u)/a_0 per step: one ln evaluation, one
      reciprocal (as exp(-ln a_0) with a ln evaluation) and one multiplication, time accumulation and a
      comparison with T.
    The step circuits keep the uniform draws and choice bits as garbage (AE uses the circuit and its
    inverse), and uncompute the propensity arithmetic before the shift."""
    Lam = uniformization_rate(gen)
    tail = eps * P / 4
    if N is None:  # default rule; costs.observable_aware_ssa passes an N matched to the observable
        N = poisson_truncation(Lam * T, tail)
    b = R.clog2(8 * N * (st.K + 1) / (eps * P)) + 2
    look, arith = propensity_value_cost(st, b)
    cum = st.K * R.add(b)
    comps = st.K * R.cmp(b) + st.K  # K comparisons and one-hot decoding
    shifts = sum(R.cadd_const(st.bits[i]) + 1 for k in range(st.K) for i in st.supp[k])
    bound = 2 * sum(R.cmp_const(st.bits[i]) for k in range(st.K) for i in st.supp[k])
    active = 2 * R.cmp_const(R.clog2(N + 1)) + 1
    step = look + 2 * (arith + cum) + comps + shifts + bound + active
    if method == "direct":
        lnc = R.eval_cost("softplus", -b * np.log(2), b * np.log(2), b)  # ln via the same fit class (smooth)
        step += 2 * (2 * lnc + R.mult(b, b) + R.add(b)) + R.cmp(b)
    elif method != "uniformization":
        raise ValueError(method)
    count_prep = R.alias(2 ** R.clog2(N + 1), b)
    per_A = N * step + count_prep + f_cost
    M, n = ae_cost(P, eps, delta_fail)
    qubits = st.n_sys + N * (b + st.K + 1) + R.clog2(N + 1) + 4 * b
    uses = 2 * M * n
    total = uses * per_A + M * n * R.mcz(qubits)
    return SSACost(label or f"coherent SSA ({method})", Lam, N, b, float(step), float(per_A), M, n, uses,
                   float(total), qubits, P)


# ----------------------------------------------------------------------------- Lindblad embedding
def _blocks(gen, nus, order, dt):
    """Blocks (source index, target index or -1 for leaving S, coefficient h_a(source)) of the star
    Hamiltonian, as a list of arrays per block. Coefficients use the truncated generator's propensities."""
    space, a = gen.space, gen.a
    X = space.states
    tgt = [space.lookup(X + nu) for nu in nus]
    a0 = a.sum(axis=0)
    blocks = []
    for j in range(len(nus)):
        h = np.sqrt(a[j])
        if order == 2:
            a0_t = np.where(tgt[j] >= 0, a0[np.maximum(tgt[j], 0)], 0.0)  # a0 at the target; 0 in the sink
            h = h * (1 + dt * (-a0 / 12 - a0_t / 4))
        blocks.append((tgt[j], h))
    if order == 2:
        for j in range(len(nus)):
            for k in range(len(nus)):
                mid = tgt[k]
                ok = mid >= 0
                h = np.zeros(len(X))
                t2 = np.full(len(X), -1)
                h[ok] = np.sqrt(dt / 2) * np.sqrt(a[k][ok]) * np.sqrt(a[j][mid[ok]])
                t2[ok] = tgt[j][mid[ok]]
                # if jump k leaves S the path ends in the sink, where no further jump acts: h = 0
                blocks.append((t2, h))
    return blocks


def lindblad_step_matrix(gen, nus, order, dt):
    """The diagonal-preserving channel of one Ding-Li-Lin step as a sparse substochastic matrix on p."""
    import scipy.sparse as sp
    blocks = _blocks(gen, nus, order, dt)
    n = len(gen.space)
    q = np.zeros(n)
    for _, h in blocks:
        q += h * h
    s2 = np.where(q > 0, np.sin(np.sqrt(dt * q)) ** 2 / np.where(q > 0, q, 1.0), 0.0)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.cos(np.sqrt(dt * q)) ** 2]
    for tgt, h in blocks:
        ok = (tgt >= 0) & (h != 0)
        rows.append(tgt[ok])
        cols.append(np.nonzero(ok)[0])
        vals.append((h * h * s2)[ok])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)), blocks


def lindblad_steps(gen, nus, order, T, p0, f, target, bias, N_max=2**24):
    """Step count N (dt = T/N) with |E_{p_N} f - target| <= bias for the order-`order` scheme, within
    5% of the smallest such N. `target` is E_{p_S,T} f from FSP (the runner supplies it).

    The error is emulated exactly at doubling N from 16 until two successive ratios agree with the
    scheme's order (2^order) within 15% (the asymptotic regime); N is then extrapolated from the
    last error, and verified by direct emulation, increasing N by 5% until the bias is met. Returns
    (N, error at N, diagnostics)."""
    def err(N):
        Mx, _ = lindblad_step_matrix(gen, nus, order, T / N)
        p = p0.copy()
        for _ in range(N):
            p = Mx @ p
        return abs(float(f @ p) - target)
    hist = []
    N = 16
    while True:
        e = err(N)
        hist.append((N, e))
        if e <= bias:
            # already met: bisect down within the last doubling
            lo, hi = (hist[-2][0] if len(hist) > 1 else 0), N
            while hi - lo > max(1, hi // 20):
                mid = (lo + hi) // 2
                if mid > 0 and err(mid) <= bias:
                    hi = mid
                else:
                    lo = mid
            return hi, err(hi), {"history": hist, "method": "bisection"}
        if len(hist) >= 3:
            r1 = hist[-3][1] / hist[-2][1]
            r2 = hist[-2][1] / hist[-1][1]
            want = 2.0**order
            if abs(r1 / want - 1) < 0.15 and abs(r2 / want - 1) < 0.15:
                break
        N *= 2
        if N > N_max:
            return None, None, {"history": hist, "method": "not asymptotic below N_max"}
    N_pred = int(np.ceil(N * (e / bias) ** (1.0 / order)))
    if N_pred > N_max:
        return None, None, {"history": hist, "method": "extrapolated N above N_max", "N_pred": N_pred}
    Nv = N_pred
    ev = err(Nv)
    checks = [(Nv, ev)]
    while ev > bias:
        Nv = int(np.ceil(1.05 * Nv))
        ev = err(Nv)
        checks.append((Nv, ev))
    return Nv, ev, {"history": hist, "method": "extrapolated then verified", "N_pred": N_pred, "checks": checks}


@dataclass
class LindbladCost:
    label: str
    order: int
    N_steps: int
    emulated_error: float
    Lambda_H: float
    q: int
    b: int
    step: float
    per_A: float
    M: int
    n: int
    uses: int
    total: float
    qubits: int
    P: float


def lindblad_cost(st, gen, nus, order, T, N, P, eps, delta_fail, f_cost=0, err=np.nan):
    """Toffolis of AE over the Ding-Li-Lin circuit with N steps (block encoding of the star
    Hamiltonian by LCU over its blocks; qubitization and GQSP for e^{-i sqrt(dt) H} per step)."""
    dt = T / N
    blocks = _blocks(gen, nus, order, dt)
    Lam = float(sum(np.max(np.abs(h)) for _, h in blocks))
    nterm = len(blocks)
    b = R.clog2(16 * N * nterm / (eps * P)) + 2
    q = lchs.jacobi_anger_degree(np.sqrt(dt) * Lam, eps * P / (8 * N))
    # one query to the block encoding of C = sum_a |a><0| (x) H_a
    amp = 0
    for idx in range(nterm):
        if idx < st.K:
            F = sorted(st.deps[idx])
            if order == 2:  # the dt correction multiplies by a function of a0 at source and target
                F = sorted(st.deps[idx] | st.supp[idx] | set(range(st.d)))
        else:
            j, k = divmod(idx - st.K, st.K)
            F = sorted(st.deps[j] | st.deps[k] | st.supp[k])
        c, _ = R.amplitude_cost([st.bits[i] for i in F], b) if F else (R.phasegrad(b), "")
        amp += c + 1
    shifts = sum(R.cadd_const(st.bits[i]) for k in range(st.K) for i in st.supp[k])
    if order == 2:
        shifts += st.K * shifts
    query = 2 * R.alias(2 ** R.clog2(nterm), b) + R.qrom(nterm) + amp + shifts + 2 * shifts // max(st.K, 1)
    n_anc = R.clog2(nterm) + 4 + 4 * b
    walk = 2 * query + R.mcz(n_anc) + R.phasegrad(b) + 2
    step = 2 * q * walk
    per_A = N * step + f_cost
    M, n = ae_cost(P, eps, delta_fail)
    qubits = st.n_sys + N * (R.clog2(nterm) + 1) + n_anc + b
    uses = 2 * M * n
    total = uses * per_A + M * n * R.mcz(qubits)
    return LindbladCost(f"Lindblad (DLL order {order})", order, N, float(err), Lam, q, b, float(step), float(per_A),
                        M, n, uses, float(total), qubits, P)
