"""Reaction networks, state spaces and the truncated CME generator A_S.

Conventions: column generator dp/dt = A_S p; A_S keeps every state's
full outflow on the diagonal, so transitions that leave S are dropped and the leak
l(x) = sum_{k: x + nu_k not in S} a_k(x) equals minus the column sum. Propensities must vanish
whenever a reaction would make a count negative; `generator` raises if they do not.
"""
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class Reaction:
    """One reaction channel.

    nu:   stoichiometric change (tuple of ints, one per species)
    prop: vectorized stochastic propensity, X of shape (n, d) (float) -> (n,)
    rate: deterministic (rate-equation) rate at a real mean vector m of shape (d,). Defaults to
          prop evaluated at m, which is right for zeroth- and first-order reactions and for Hill
          terms; falling-factorial mass action (k x(x-1)) must supply its macroscopic form.
    """

    nu: tuple
    prop: Callable
    rate: Optional[Callable] = None
    name: str = ""

    def deterministic_rate(self, m):
        m = np.asarray(m, dtype=float)
        if self.rate is not None:
            return float(self.rate(m))
        return float(np.asarray(self.prop(m[None, :]))[0])


class StateSpace:
    """A finite set S of count vectors with O(log n) vectorized lookup.

    States are encoded by mixed-radix keys over an enclosing box with the given widths.
    """

    def __init__(self, states, widths):
        states = np.asarray(states, dtype=np.int64)
        if states.ndim != 2:
            raise ValueError("states must have shape (n, d)")
        self.widths = np.asarray(widths, dtype=np.int64)
        self.d = states.shape[1]
        self._radix = np.concatenate([[1], np.cumprod(self.widths[:0:-1])])[::-1]  # C order
        keys = states @ self._radix
        order = np.argsort(keys, kind="stable")
        self.states = states[order]
        self._keys = keys[order]
        if np.any(np.diff(self._keys) == 0):
            raise ValueError("duplicate states")

    @classmethod
    def box(cls, widths):
        widths = [int(w) for w in widths]
        grids = np.meshgrid(*[np.arange(w) for w in widths], indexing="ij")
        return cls(np.stack([g.ravel() for g in grids], axis=1), widths)

    @classmethod
    def manifold(cls, widths, L, c):
        """States of the box with L x = c."""
        full = cls.box(widths)
        keep = np.all(full.states @ np.asarray(L, dtype=np.int64).T == np.asarray(c), axis=1)
        return cls(full.states[keep], widths)

    def __len__(self):
        return len(self.states)

    @property
    def X(self):
        """States as floats, the argument convention of Reaction.prop."""
        return self.states.astype(float)

    def lookup(self, Y):
        """Indices of the rows of Y in S, or -1 where a row is not in S."""
        Y = np.asarray(Y, dtype=np.int64)
        inside = np.all((Y >= 0) & (Y < self.widths), axis=1)
        keys = np.where(inside, Y @ self._radix, -1)
        pos = np.searchsorted(self._keys, keys)
        pos = np.clip(pos, 0, len(self._keys) - 1)
        found = inside & (self._keys[pos] == keys)
        return np.where(found, pos, -1)

    def index(self, x):
        i = int(self.lookup(np.asarray(x)[None, :])[0])
        if i < 0:
            raise KeyError(f"{tuple(x)} not in the state space")
        return i


@dataclass
class TruncatedGenerator:
    """A_S in CSR form, the propensity table a[k, x] and the leak l(x)."""

    A: sp.csr_matrix
    a: np.ndarray
    leak: np.ndarray
    space: StateSpace


def generator(space, reactions):
    X = space.X
    n = len(space)
    rows, cols, vals = [], [], []
    a = np.zeros((len(reactions), n))
    leak = np.zeros(n)
    for k, rxn in enumerate(reactions):
        ak = np.broadcast_to(np.asarray(rxn.prop(X), dtype=float), (n,)).copy()
        if np.any(ak < 0) or not np.all(np.isfinite(ak)):
            raise ValueError(f"reaction {k} ({rxn.name}): propensities must be finite and >= 0")
        Y = space.states + np.asarray(rxn.nu, dtype=np.int64)
        negative = np.any(Y < 0, axis=1)
        if np.any(ak[negative] > 0):
            raise ValueError(f"reaction {k} ({rxn.name}): positive propensity into negative counts")
        a[k] = ak
        tgt = space.lookup(Y)
        ok = (tgt >= 0) & (ak > 0)
        rows.append(tgt[ok])
        cols.append(np.nonzero(ok)[0])
        vals.append(ak[ok])
        out = (tgt < 0) & (ak > 0)
        leak[out] += ak[out]
    diag = -a.sum(axis=0)
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    A.sum_duplicates()
    return TruncatedGenerator(A=A, a=a, leak=leak, space=space)


def stoichiometry(reactions):
    """d x K stoichiometric matrix."""
    return np.array([r.nu for r in reactions], dtype=np.int64).T


def conservation_laws(reactions):
    """Integer basis L (rows) of the left null space of the stoichiometric matrix: L nu_k = 0.

    Exact rational Gaussian elimination, fine for the small matrices of CME models.
    """
    N = stoichiometry(reactions)  # d x K
    d, K = N.shape
    M = [[Fraction(int(N[i, k])) for i in range(d)] for k in range(K)]  # K x d; solve M y = 0
    pivots, row = [], 0
    for col in range(d):
        piv = next((r for r in range(row, K) if M[r][col] != 0), None)
        if piv is None:
            continue
        M[row], M[piv] = M[piv], M[row]
        pv = M[row][col]
        M[row] = [v / pv for v in M[row]]
        for r in range(K):
            if r != row and M[r][col] != 0:
                f = M[r][col]
                M[r] = [a - f * b for a, b in zip(M[r], M[row])]
        pivots.append(col)
        row += 1
        if row == K:
            break
    free = [c for c in range(d) if c not in pivots]
    basis = []
    for fc in free:
        y = [Fraction(0)] * d
        y[fc] = Fraction(1)
        for r, pc in enumerate(pivots):
            y[pc] = -M[r][fc]
        den = np.lcm.reduce([v.denominator for v in y])
        vec = np.array([int(v * den) for v in y], dtype=np.int64)
        g = np.gcd.reduce(np.abs(vec[vec != 0]))
        vec //= g
        if vec[np.nonzero(vec)[0][0]] < 0:
            vec = -vec
        basis.append(vec)
    return np.array(basis, dtype=np.int64).reshape(len(basis), d)
