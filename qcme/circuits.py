"""Gate-level block encodings for small instances.

A circuit acts on named registers of qubits. Operations are
  * reversible classical maps on basis states (X, CNOT, Toffoli, multi-controlled X, modular constant
    addition, comparators: each is a permutation of basis states, checked to be one), and
  * multiplexed single-qubit rotations whose 2 x 2 matrix depends on the values of control registers
    (PREPARE trees, amplitude oracles). Amplitudes are set from their cosine directly, so a zero
    amplitude is exactly zero (I9 is checked exactly).

`block_encoding_circuit` builds the LCU block encoding for the stabilized weighted
generator of one window,
    M = B~ on S,   M = -(max_S a_0 + c) I on register states outside S (conservation manifolds and
                   box padding), so the log-norm of M is <= 0 whenever c >= the log-norm of B,
with PREPARE over the K reactions and the diagonal term, and SELECT applying (flag, amplitude
rotation, flag uncompute, controlled shift) per reaction. Arithmetic oracles are represented by exact
multiplexed rotations here; their fixed-point circuits are checked separately (ReversibleProgram).
"""
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------- registers and state
class Layout:
    def __init__(self, sizes):
        """sizes: ordered dict-like list of (name, n_qubits); the first register holds the lowest bits."""
        self.regs, off = {}, 0
        for name, n in sizes:
            self.regs[name] = (off, int(n))
            off += int(n)
        self.n = off

    def get(self, idx, name):
        off, n = self.regs[name]
        return (idx >> off) & ((1 << n) - 1)

    def set(self, idx, name, val):
        off, n = self.regs[name]
        mask = ((1 << n) - 1) << off
        return (idx & ~mask) | ((val & ((1 << n) - 1)) << off)

    def bit(self, name, j=0):
        off, n = self.regs[name]
        assert j < n
        return off + j


@dataclass
class Circuit:
    layout: Layout
    ops: list = field(default_factory=list)

    # --- classical reversible maps
    def perm(self, fn, name=""):
        """fn: vectorized map of basis indices (int64 array) to basis indices; must be a bijection."""
        self.ops.append(("perm", fn, name))

    def rot(self, target_bit, amp_fn, name=""):
        """Multiplexed rotation on `target_bit`: for basis index i (with the target bit cleared),
        amp_fn(i) = v in [-1, 1] gives the matrix [[v, -sqrt(1-v^2)], [sqrt(1-v^2), v]]."""
        self.ops.append(("rot", target_bit, amp_fn, name))

    def unitary2(self, target_bit, mat_fn, name=""):
        """Multiplexed arbitrary 2 x 2 unitary: mat_fn(i) -> array (n, 2, 2)."""
        self.ops.append(("u2", target_bit, mat_fn, name))

    def inverse(self):
        inv = Circuit(self.layout)
        for op in reversed(self.ops):
            if op[0] == "perm":
                fn = op[1]
                inv.ops.append(("perm_inv", fn, op[2]))
            elif op[0] == "perm_inv":
                inv.ops.append(("perm", op[1], op[2]))
            elif op[0] == "rot":
                t, f = op[1], op[2]
                inv.ops.append(("u2", t, (lambda i, f=f: _rot_mats(f(i)).conj().transpose(0, 2, 1)), op[3]))
            elif op[0] == "u2":
                t, f = op[1], op[2]
                inv.ops.append(("u2", t, (lambda i, f=f: f(i).conj().transpose(0, 2, 1)), op[3]))
        return inv

    def apply(self, psi):
        n = self.layout.n
        N = 1 << n
        idx = np.arange(N, dtype=np.int64)
        for op in self.ops:
            kind = op[0]
            if kind in ("perm", "perm_inv"):
                img = op[1](idx)
                if kind == "perm":
                    out = np.zeros_like(psi)
                    out[img] = psi
                else:
                    out = psi[img]
                psi = out
            elif kind in ("rot", "u2"):
                t = op[1]
                i0 = idx[(idx >> t) & 1 == 0]
                i1 = i0 | (1 << t)
                mats = _rot_mats(op[2](i0)) if kind == "rot" else op[2](i0)
                a0, a1 = psi[i0], psi[i1]
                psi = psi.copy()
                psi[i0] = mats[:, 0, 0] * a0 + mats[:, 0, 1] * a1
                psi[i1] = mats[:, 1, 0] * a0 + mats[:, 1, 1] * a1
        return psi

    def check_permutations(self):
        """Every classical op is a bijection of basis states (reversibility)."""
        N = 1 << self.layout.n
        idx = np.arange(N, dtype=np.int64)
        for op in self.ops:
            if op[0] in ("perm", "perm_inv"):
                img = op[1](idx)
                if len(np.unique(img)) != N or img.min() < 0 or img.max() >= N:
                    return False, op[2]
        return True, ""


def _rot_mats(v):
    v = np.asarray(v, float)
    if np.any(np.abs(v) > 1 + 1e-12):
        raise ValueError("rotation amplitude outside [-1, 1]")
    v = np.clip(v, -1, 1)
    s = np.sqrt(1 - v * v)
    m = np.empty((len(v), 2, 2))
    m[:, 0, 0], m[:, 0, 1], m[:, 1, 0], m[:, 1, 1] = v, -s, s, v
    return m


# ----------------------------------------------------------------------------- block encoding of B~
@dataclass
class BlockEncodingInstance:
    circuit: Circuit
    layout: Layout
    alpha: float
    M: np.ndarray  # target matrix on the full system register (2^n_sys x 2^n_sys)
    in_S: np.ndarray  # system basis index -> index in S or -1
    sys_bits: list


def system_index(x, bits):
    """Pack a count vector (or rows of counts) into a system basis index (species 0 in the lowest bits)."""
    x = np.atleast_2d(x)
    out = np.zeros(len(x), dtype=np.int64)
    off = 0
    for i, n in enumerate(bits):
        out |= (x[:, i].astype(np.int64) & ((1 << n) - 1)) << off
        off += n
    return out


def system_counts(idx, bits):
    idx = np.asarray(idx, dtype=np.int64)
    cols, off = [], 0
    for n in bits:
        cols.append((idx >> off) & ((1 << n) - 1))
        off += n
    return np.stack(cols, axis=1)


def block_encoding_circuit(gen, nus, logmu, c, diag="value", reactions=None):
    """LCU block encoding of M (module docstring) for one static window with reference log mu and
    shift c. Returns the circuit and the target matrix M on the system register.

    diag="split": the diagonal as separate terms: per
    species with univariate propensities, per multivariate propensity, a constant -(C + c) acting on
    every register state (C = sum of constant propensities), and -(|c| + max a_0) acting off S only.
    Then M = B~ on S and M = -(C + c + |c| + max a_0) I off S."""
    space = gen.space
    X = space.states
    d = X.shape[1]
    bits = [max(int(np.ceil(np.log2(w))), 1) for w in space.widths]
    n_sys = sum(bits)
    K = len(nus)
    Nsys = 1 << n_sys
    all_x = system_counts(np.arange(Nsys), bits)
    in_S = space.lookup(all_x)  # -1 outside S (padding or off the manifold)
    dterms = _diagonal_terms(gen, reactions, all_x, in_S, c, diag)
    nsel = max(int(np.ceil(np.log2(K + len(dterms)))), 1)
    lay = Layout([("sys", n_sys), ("sel", nsel), ("amp", 1), ("flag", 1)])
    a = gen.a  # (K, |S|)
    a0 = a.sum(axis=0)
    # b_k(x) = a_k(x) sqrt(mu(x)/mu(x + nu_k)) for x, x + nu_k in S
    bk = np.zeros((K, Nsys))
    tgt_sys = np.zeros((K, Nsys), dtype=np.int64)
    for k, nu in enumerate(nus):
        y = all_x + np.asarray(nu)
        tgt_sys[k] = system_index(np.mod(y, [1 << n for n in bits]), bits)
        jy = space.lookup(y)
        ok = (in_S >= 0) & (jy >= 0)
        src = in_S[ok]
        bk[k, ok] = a[k, src] * np.exp(0.5 * (logmu[src] - logmu[jy[ok]]))
    bmax = bk.max(axis=1)
    dmaxs = [float(np.max(np.abs(dv))) for dv in dterms]
    w = np.concatenate([bmax, dmaxs])
    alpha = float(w.sum())
    # target matrix on the system register
    M = np.zeros((Nsys, Nsys))
    for k in range(K):
        cols = np.nonzero(bk[k])[0]
        M[tgt_sys[k, cols], cols] += bk[k, cols]
    for dv in dterms:
        M[np.arange(Nsys), np.arange(Nsys)] += dv
    circ = Circuit(lay)
    # PREPARE: rotation tree on sel with amplitudes sqrt(w_t / alpha), padded with zeros
    probs = np.zeros(1 << nsel)
    probs[:K + len(dterms)] = w / alpha
    _prep_tree(circ, lay, "sel", probs)
    prep_ops = list(circ.ops)
    get_sys = lambda i: (i >> lay.regs["sys"][0]) & (Nsys - 1)
    get_sel = lambda i: (i >> lay.regs["sel"][0]) & ((1 << nsel) - 1)
    get_flag = lambda i: (i >> lay.regs["flag"][0]) & 1
    fbit, abit = lay.bit("flag"), lay.bit("amp")
    for k in range(K):
        bad = (in_S < 0) | ~_target_in_S(space, all_x, nus[k], in_S)

        def flag_fn(i, k=k, bad=bad):
            hit = (get_sel(i) == k) & bad[get_sys(i)]
            return i ^ (hit.astype(np.int64) << fbit)
        circ.perm(flag_fn, f"flag[{k}]")

        def amp_fn(i, k=k):
            v = np.ones(len(i))
            on = get_sel(i) == k
            xs = get_sys(i)
            vv = np.where(get_flag(i) == 1, 0.0, bk[k][xs] / bmax[k] if bmax[k] > 0 else 0.0)
            v[on] = vv[on]
            return v
        circ.rot(abit, amp_fn, f"amp[{k}]")
        circ.perm(flag_fn, f"unflag[{k}]")

        def shift_fn(i, k=k):
            on = get_sel(i) == k
            xs = get_sys(i)
            new = np.where(on, tgt_sys[k][xs], xs)
            return (i & ~np.int64(Nsys - 1)) | new
        circ.perm(shift_fn, f"shift[{k}]")

    for tix, (dv, dm) in enumerate(zip(dterms, dmaxs)):
        def diag_fn(i, tix=tix, dv=dv, dm=dm):
            v = np.ones(len(i))
            on = get_sel(i) == K + tix
            v[on] = (dv[get_sys(i)] / dm if dm > 0 else 0.0 * dv[get_sys(i)])[on]
            return v
        circ.rot(abit, diag_fn, f"diag[{tix}]")
    unprep = Circuit(lay, prep_ops).inverse()
    circ.ops += unprep.ops
    return BlockEncodingInstance(circ, lay, alpha, M, in_S, bits)


def _diagonal_terms(gen, reactions, all_x, in_S, c, diag):
    """Diagonal term values on every system-register state (see block_encoding_circuit)."""
    a0 = gen.a.sum(axis=0)
    if diag == "value":
        return [np.where(in_S >= 0, -(a0[np.maximum(in_S, 0)] + c), -(a0.max() + c))]
    if diag != "split" or reactions is None:
        raise ValueError("diag='split' needs the reactions")
    inS = in_S >= 0
    X = all_x.astype(float)
    groups, terms, const = {}, [], 0.0
    for rxn in reactions:
        av = np.broadcast_to(np.asarray(rxn.prop(X), dtype=float), (len(X),))
        dep = [i for i in range(X.shape[1]) if np.any(np.abs(_vary(rxn, X, i) - av) > 1e-12 * max(1, av.max()))]
        if not dep:
            const += float(av[0])
        elif len(dep) == 1:
            groups[dep[0]] = groups.get(dep[0], 0.0) + av
        else:
            terms.append(-np.where(inS, av, 0.0))
    out = [-np.where(inS, g, 0.0) for _, g in sorted(groups.items())] + terms
    out.append(np.full(len(X), -(const + c)))
    if not np.all(inS):
        out.append(np.where(inS, 0.0, -(abs(c) + a0.max())))
    return out


def _vary(rxn, X, i):
    """Propensity with species i replaced by a different register value (dependence probe)."""
    Y = X.copy()
    Y[:, i] = (Y[:, i] + 1) % max(2, int(X[:, i].max()) + 1)
    return np.broadcast_to(np.asarray(rxn.prop(Y), dtype=float), (len(X),))


def _target_in_S(space, all_x, nu, in_S):
    y = all_x + np.asarray(nu)
    return space.lookup(y) >= 0


def _prep_tree(circ, lay, reg, probs):
    """State sum_t sqrt(probs_t)|t> on register `reg` by a binary tree of multiplexed Ry rotations
    (most significant bit first), exact up to rounding."""
    off, n = lay.regs[reg]
    probs = np.asarray(probs, float)
    for level in range(n):
        bit = off + n - 1 - level  # qubit set at this level
        span = 1 << (n - level)  # block size of values sharing the higher bits

        def amp_fn(i, level=level, bit=bit, span=span):
            val = (i >> off) & ((1 << n) - 1)
            hi_mask = ~((1 << (n - level)) - 1) & ((1 << n) - 1)
            base = val & hi_mask
            tot = np.array([probs[b:b + span].sum() for b in base])
            left = np.array([probs[b:b + span // 2].sum() for b in base])
            with np.errstate(invalid="ignore", divide="ignore"):
                v = np.where(tot > 0, np.sqrt(np.clip(left / np.where(tot > 0, tot, 1), 0, 1)), 1.0)
            return v
        circ.rot(bit, amp_fn, f"prep L{level}")


def block_of(inst):
    """<0_anc| U |0_anc> on the system register, by simulating U on every system basis input."""
    lay, circ = inst.layout, inst.circuit
    Nsys = 1 << lay.regs["sys"][1]
    Nfull = 1 << lay.n
    out = np.zeros((Nsys, Nsys), dtype=complex)
    for x in range(Nsys):
        psi = np.zeros(Nfull, dtype=complex)
        psi[x] = 1.0  # all ancillas zero, sys = x (sys is the lowest register)
        phi = circ.apply(psi)
        out[:, x] = phi[:Nsys]
    return out


# ----------------------------------------------------------------------------- self-inverse select (walk)
def hermitian_select(U, n_anc_block, ck, Lam, alpha):
    """Given a dense unitary U whose top-left block (ancilla register of n_anc_block qubits in |0>,
    lowest bits = system) is B/alpha, build V = (P^dag (x) I) (|0><1| (x) U + |1><0| (x) U^dag) (P (x) I)
    on one extra qubit (the highest), with P|0> = cos(phi)|0> + e^{i chi} sin(phi)|1>, so that the
    top-left block is (ck B + conj(ck) B^T)/Lam (requires |ck| alpha / Lam <= 1/2). Returns V (Hermitian, V^2 = I)."""
    D = U.shape[0]
    r = abs(ck) * alpha / Lam
    if r > 0.5 + 1e-12:
        raise ValueError("normalization too small")
    phi = np.arcsin(min(2 * r, 1.0)) / 2  # sin(2 phi)/2 = r
    chi = np.angle(ck)
    p0, p1 = np.cos(phi), np.sin(phi) * np.exp(1j * chi)
    P = np.array([[p0, -np.conj(p1)], [p1, np.conj(p0)]])  # P|0> = p0|0> + p1|1>
    sel = np.zeros((2 * D, 2 * D), dtype=complex)
    sel[:D, D:] = U  # |0><1| (x) U   (extra qubit is the highest: blocks [q=0 | q=1])
    sel[D:, :D] = U.conj().T  # |1><0| (x) U^dag
    PI = np.kron(P, np.eye(D))
    return PI.conj().T @ sel @ PI


# ----------------------------------------------------------------------------- reversible fixed point
class ReversibleProgram:
    """Registers of unsigned integers modified by in-place reversible steps (xor-lookups into a
    register, modular addition of one register into another, xor of a product into a register), run
    forward then in reverse. Used to check that an oracle's arithmetic returns its workspace to zero
    (compute-copy-uncompute) and is a bijection on the tested inputs."""

    def __init__(self, widths):
        self.widths = dict(widths)
        self.steps = []

    def lookup_xor(self, dst, src, table):
        self.steps.append(("lookup", dst, src, np.asarray(table, dtype=object)))

    def add_into(self, dst, src, sign=1):
        self.steps.append(("add", dst, src, sign))

    def mul_xor(self, dst, a, b, shift):
        """dst ^= (a * b) >> shift (mod 2^width(dst)): a fixed-point product."""
        self.steps.append(("mul", dst, a, b, shift))

    def run(self, regs, reverse=False):
        regs = dict(regs)
        seq = reversed(self.steps) if reverse else self.steps
        for s in seq:
            kind = s[0]
            if kind == "lookup":
                _, dst, src, tab = s
                regs[dst] ^= int(tab[regs[src]]) & ((1 << self.widths[dst]) - 1)
            elif kind == "add":
                _, dst, src, sign = s
                sgn = -sign if reverse else sign
                regs[dst] = (regs[dst] + sgn * regs[src]) % (1 << self.widths[dst])
            elif kind == "mul":
                _, dst, a, b, shift = s
                regs[dst] ^= ((regs[a] * regs[b]) >> shift) & ((1 << self.widths[dst]) - 1)
        return regs
