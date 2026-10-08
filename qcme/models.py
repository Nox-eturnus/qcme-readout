"""Benchmark networks and the published gene-network models.

Every model records its parameters, provenance, initial mean x0, reference envelope for R-mix
and which species are binary (gene states, Bernoulli marginals in product references).
The horizon T is not stored: it is computed by `relaxation_time` (one deterministic relaxation time).
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from .network import Reaction, StateSpace


def hill_rep(x, K, n):
    return 1.0 / (1.0 + (np.maximum(x, 0.0) / K) ** n)


@dataclass
class Model:
    name: str
    level: str
    species: tuple
    build: Callable  # widths -> list[Reaction]
    widths: tuple  # default box
    x0: tuple  # initial mean (p0 is the normalized R-P reference at t = 0)
    binary: tuple  # per species: True for {0, 1} gene states
    envelope: dict  # R-mix envelope spec: {"kind": "poisson", "means": (...)} or {"kind": "nb", "means", "q"}
    provenance: str
    conservation: Optional[tuple] = None  # (L, c): states satisfy L x = c
    params: dict = field(default_factory=dict)

    def reactions(self, widths=None):
        return self.build(tuple(widths or self.widths))

    def space(self, widths=None):
        widths = tuple(widths or self.widths)
        if self.conservation is None:
            return StateSpace.box(widths)
        L, c = self.conservation
        return StateSpace.manifold(widths, L, c)

    def drift(self, m, widths=None):
        return sum(np.asarray(r.nu, float) * r.deterministic_rate(np.maximum(m, 0.0))
                   for r in self.reactions(widths))


def _jacobian(f, m, h=1e-6):
    d = len(m)
    J = np.zeros((d, d))
    for j in range(d):
        e = np.zeros(d)
        e[j] = h * max(1.0, abs(m[j]))
        J[:, j] = (f(m + e) - f(m - e)) / (2 * e[j])
    return J


def relaxation_time(model, t_settle=200.0):
    """One deterministic relaxation time: 1/|Re lambda_slow| of the rate-equation Jacobian at the
    attractor reached from x0 (zero modes from conservation laws dropped)."""
    f = lambda m: model.drift(m)
    sol = solve_ivp(lambda t, y: f(y), (0, t_settle), np.asarray(model.x0, float), method="LSODA",
                    rtol=1e-10, atol=1e-12)
    x_star = sol.y[:, -1]
    ev = np.linalg.eigvals(_jacobian(f, x_star))
    scale = np.abs(ev).max()
    re = np.abs(ev.real[np.abs(ev) > 1e-7 * scale])
    return 1.0 / re.min(), x_star


# ----------------------------------------------------------------------------- L0
def birth_death(k=10.0, g=1.0):
    build = lambda w: [
        Reaction((1,), lambda X: np.full(len(X), k), name="0 -> X"),
        Reaction((-1,), lambda X: g * X[:, 0], name="X -> 0"),
    ]
    return Model("birth_death", "L0", ("X",), build, (40,), (2.0,), (False,),
                 {"kind": "poisson", "means": (k / g,)}, "synthetic", params=dict(k=k, g=g))


def telegraph(lon=0.7, loff=1.3, nu=20.0, delta=1.0):
    build = lambda w: [
        Reaction((1, 0), lambda X: lon * (X[:, 0] == 0), rate=lambda m: lon * (1 - m[0]), name="I -> A"),
        Reaction((-1, 0), lambda X: loff * (X[:, 0] == 1), rate=lambda m: loff * m[0], name="A -> I"),
        Reaction((0, 1), lambda X: nu * (X[:, 0] == 1), rate=lambda m: nu * m[0], name="A -> A + M"),
        Reaction((0, -1), lambda X: delta * X[:, 1], name="M -> 0"),
    ]
    return Model("telegraph", "L0", ("gene", "M"), build, (2, 60), (0.2, 2.0), (True, False),
                 {"kind": "poisson", "means": (0.35, nu / delta)},
                 "Peccoud & Ycart 1995 form",
                 params=dict(lon=lon, loff=loff, nu=nu, delta=delta))


def bursty(k=2.0, theta=0.8, g=1.0):
    def build(w):
        N = w[0]
        rx = [Reaction((j,), lambda X, j=j: np.full(len(X), k * (1 - theta) * theta**j), name=f"burst {j}")
              for j in range(1, N)]
        rx.append(Reaction((N,), lambda X: np.full(len(X), k * theta**N), name=f"burst >= {N} (leaves S)"))
        rx.append(Reaction((-1,), lambda X: g * X[:, 0], name="M -> 0"))
        return rx
    return Model("bursty", "L0", ("M",), build, (80,), (1.0,), (False,),
                 {"kind": "nb", "means": (k * theta / ((1 - theta) * g),), "q": (theta,)},
                 "geometric bursts", params=dict(k=k, theta=theta, g=g))


# ----------------------------------------------------------------------------- L1
def schlogl(k1=1 / 9, k2=1 / 1215, k3=27 / 2, k4=59 / 20):
    """Kuntz et al. 2019 (R21), bimodal parameter set (b): a1 = k1 x(x-1), a2 = k2 x(x-1)(x-2)."""
    build = lambda w: [
        Reaction((1,), lambda X: k1 * X[:, 0] * (X[:, 0] - 1), rate=lambda m: k1 * m[0] ** 2, name="2X -> 3X"),
        Reaction((-1,), lambda X: k2 * X[:, 0] * (X[:, 0] - 1) * (X[:, 0] - 2), rate=lambda m: k2 * m[0] ** 3,
                 name="3X -> 2X"),
        Reaction((1,), lambda X: np.full(len(X), k3), name="0 -> X"),
        Reaction((-1,), lambda X: k4 * X[:, 0], name="X -> 0"),
    ]
    return Model("schlogl", "L1", ("X",), build, (200,), (25.0,), (False,),
                 {"kind": "poisson", "means": (103.6,)},
                 "Kuntz, Thomas, Stan, Barahona, JCP 2019, Fig. 3(b) bimodal set",
                 params=dict(k1=k1, k2=k2, k3=k3, k4=k4))


def toggle(alpha=30.0, K=10.0, n=2.0, delta=1.0):
    build = lambda w: [
        Reaction((1, 0), lambda X: alpha * hill_rep(X[:, 1], K, n), name="0 -> X"),
        Reaction((-1, 0), lambda X: delta * X[:, 0], name="X -> 0"),
        Reaction((0, 1), lambda X: alpha * hill_rep(X[:, 0], K, n), name="0 -> Y"),
        Reaction((0, -1), lambda X: delta * X[:, 1], name="Y -> 0"),
    ]
    return Model("toggle", "L1", ("X", "Y"), build, (60, 60), (5.0, 20.0), (False, False),
                 {"kind": "poisson", "means": (alpha / delta, alpha / delta)},
                 "synthetic (Hill n = 2)",
                 params=dict(alpha=alpha, K=K, n=n, delta=delta))


def michaelis_menten(E0=10, S0=40, k1=1.0, km1=5.0, k2=1.0):
    build = lambda w: [
        Reaction((-1, -1, 1, 0), lambda X: k1 * X[:, 0] * X[:, 1], name="E + S -> C"),
        Reaction((1, 1, -1, 0), lambda X: km1 * X[:, 2], name="C -> E + S"),
        Reaction((1, 0, -1, 1), lambda X: k2 * X[:, 2], name="C -> E + P"),
    ]
    L = np.array([[1, 0, 1, 0], [0, 1, 1, 1]])
    return Model("michaelis_menten", "L1", ("E", "S", "C", "P"), build, (E0 + 1, S0 + 1, E0 + 1, S0 + 1),
                 (float(E0), float(S0), 0.0, 0.0), (False,) * 4,
                 {"kind": "poisson", "means": (float(E0), float(S0), float(E0), float(S0))},
                 "synthetic", conservation=(L, np.array([E0, S0])),
                 params=dict(E0=E0, S0=S0, k1=k1, km1=km1, k2=k2))


def self_repressor(alpha=20.0, alpha_b=1.0, kb=0.05, ku=1.0, delta=1.0):
    """Promoter G binds its own protein P: G + P <-> Gb; G and Gb transcribe-translate P."""
    build = lambda w: [
        Reaction((-1, 1, -1), lambda X: kb * X[:, 0] * X[:, 2], name="G + P -> Gb"),
        Reaction((1, -1, 1), lambda X: ku * X[:, 1], name="Gb -> G + P"),
        Reaction((0, 0, 1), lambda X: alpha * X[:, 0], name="G -> G + P"),
        Reaction((0, 0, 1), lambda X: alpha_b * X[:, 1], name="Gb -> Gb + P"),
        Reaction((0, 0, -1), lambda X: delta * X[:, 2], name="P -> 0"),
    ]
    L = np.array([[1, 1, 0]])
    return Model("self_repressor", "L1", ("G", "Gb", "P"), build, (2, 2, 60), (1.0, 0.0, 0.0),
                 (True, True, False), {"kind": "poisson", "means": (0.5, 0.5, alpha / delta)}, "synthetic",
                 conservation=(L, np.array([1])),
                 params=dict(alpha=alpha, alpha_b=alpha_b, kb=kb, ku=ku, delta=delta))


def cascade(d, k=4.0, c=1.0, g=1.0, width=12):
    def build(w):
        rx = [Reaction(tuple([1] + [0] * (d - 1)), lambda X: np.full(len(X), k), name="0 -> X1")]
        for i in range(d - 1):
            nu = [0] * d
            nu[i], nu[i + 1] = -1, 1
            rx.append(Reaction(tuple(nu), lambda X, i=i: c * X[:, i], name=f"X{i + 1} -> X{i + 2}"))
        nu = [0] * d
        nu[-1] = -1
        rx.append(Reaction(tuple(nu), lambda X: g * X[:, d - 1], name=f"X{d} -> 0"))
        return rx
    return Model(f"cascade{d}", "L1", tuple(f"X{i + 1}" for i in range(d)), build, (width,) * d, (0.0,) * d,
                 (False,) * d, {"kind": "poisson", "means": (k / min(c, g),) * d},
                 "synthetic linear cascade", params=dict(d=d, k=k, c=c, g=g))


L0_MODELS = (birth_death, telegraph, bursty)
L1_MODELS = (schlogl, toggle, michaelis_menten, self_repressor, lambda: cascade(2), lambda: cascade(3))


def registry():
    return [f() for f in L0_MODELS + L1_MODELS]


# ----------------------------------------------------------------------------- published set
# Every value is transcribed from its source (the provenance string gives section, equation and page).
# Initial means: the source's where it gives a transient study, otherwise x0 = 0 with genes inactive. The E1 envelope "poisson_maxprod"
# is Poisson with mean max_{x in S}(production of i)/(removal constant of i) (T3(iii) supersolution;
# Bernoulli(1/2) for gene states), evaluated on the box in use (reference.envelope).
def toggle_hegland(b=0.4, c=0.05):
    """Hegland et al. 2007 toggle as given in Jahnke & Huisinga 2008 Sec. 4.1, eq. (21), p. 2296."""
    build = lambda w: [
        Reaction((-1, 0), lambda X: c * X[:, 0], name="S1 -> 0"),
        Reaction((0, -1), lambda X: c * X[:, 1], name="S2 -> 0"),
        Reaction((1, 0), lambda X: b / (b + X[:, 1]), name="0 -> S1"),
        Reaction((0, 1), lambda X: b / (b + X[:, 0]), name="0 -> S2"),
    ]
    return Model("toggle_hegland", "P", ("S1", "S2"), build, (51, 51), (30.0, 5.0), (False, False),
                 {"kind": "poisson_maxprod", "removal": (c, c)},
                 "Jahnke & Huisinga, Bull. Math. Biol. 70 (2008), Sec. 4.1, eq. (21), p. 2296 "
                 "(model of Hegland et al. 2007); b = 0.4, c = 0.05; source initial law: discrete Gaussian at (30, 5)",
                 params=dict(b=b, c=c))


def toggle_kuntz(k1=30.0, k2=1.0, k3=10.0, k4=1.0, theta=1.0):
    """Kuntz et al. 2019 Sec. V.A, eqs. (112)-(113), p. 17; parameters from Fig. 4, p. 18."""
    build = lambda w: [
        Reaction((1, 0), lambda X: k1 / (1.0 + (np.maximum(X[:, 1], 0.0) / theta) ** 3), name="0 -> P1"),
        Reaction((-1, 0), lambda X: k2 * X[:, 0], name="P1 -> 0"),
        Reaction((0, 1), lambda X: k3 / (1.0 + np.maximum(X[:, 0], 0.0)), name="0 -> P2"),
        Reaction((0, -1), lambda X: k4 * X[:, 1], name="P2 -> 0"),
    ]
    return Model("toggle_kuntz", "P", ("P1", "P2"), build, (48, 24), (0.0, 0.0), (False, False),
                 {"kind": "poisson_maxprod", "removal": (k2, k4)},
                 "Kuntz, Thomas, Stan, Barahona, JCP 2019 (arXiv:1702.05468v4), Sec. V.A, eqs. (112)-(113), "
                 "p. 17; theta = 1, k1 = 30, k2 = k4 = 1, k3 = 10 (Fig. 4 caption, p. 18); stationary study, x0 = 0",
                 params=dict(k1=k1, k2=k2, k3=k3, k4=k4, theta=theta))


def toggle_gardner(a1=500.0, a2=0.3, a3=200.0, a4=0.4, beta=1.5, gamma=1.0):
    """Gardner toggle as parameterized in Gupta, Mikelson, Khammash 2017 Sec. 4.3.2, p. 21."""
    build = lambda w: [
        Reaction((1, 0), lambda X: a1 / (1.0 + np.maximum(X[:, 1], 0.0) ** beta), name="0 -> X1"),
        Reaction((-1, 0), lambda X: a2 * X[:, 0], name="X1 -> 0"),
        Reaction((0, 1), lambda X: a3 / (1.0 + np.maximum(X[:, 0], 0.0) ** gamma), name="0 -> X2"),
        Reaction((0, -1), lambda X: a4 * X[:, 1], name="X2 -> 0"),
    ]
    return Model("toggle_gardner", "P", ("X1", "X2"), build, (256, 256), (0.0, 0.0), (False, False),
                 {"kind": "poisson_maxprod", "removal": (a2, a4)},
                 "Gupta, Mikelson, Khammash 2017 (arXiv:1704.07259v3), Sec. 4.3.2, p. 21 (Gardner et al. 2000 "
                 "toggle); alpha1 = 500, alpha2 = 0.3, alpha3 = 200, alpha4 = 0.4, beta = 1.5, gamma = 1; "
                 "stationary study, x0 = 0",
                 params=dict(a1=a1, a2=a2, a3=a3, a4=a4, beta=beta, gamma=gamma))


def feedback_kuntz(b_mean=10.0, k1=1.0, k2=1.0, k3=10.0, k4=10.0, k5=1.0):
    """Bursty expression with negative feedback, Kuntz et al. 2019 Sec. V.B, eq. (114), pp. 17-18;
    parameters from Fig. 7 caption, p. 20; geometric bursts p(k) = (1 - p0)^k p0, p0 = 1/(1 + <b>),
    k >= 0 (App. B). State (gene on, protein); a burst of size 0 is a self-loop."""
    q = b_mean / (1.0 + b_mean)
    p0 = 1.0 - q

    def build(w):
        N = w[1]
        rx = [Reaction((1, 0), lambda X: k1 * (1 - X[:, 0]), name="Goff -> Gon"),
              Reaction((-1, 0), lambda X: k2 * X[:, 0], name="Gon -> Goff"),
              Reaction((-1, 0), lambda X: k3 * X[:, 1] * X[:, 0], name="Gon + P -> Goff + P")]
        rx += [Reaction((0, j), lambda X, j=j: k4 * X[:, 0] * p0 * q**j, name=f"burst {j}") for j in range(1, N)]
        rx.append(Reaction((0, N), lambda X: k4 * X[:, 0] * q**N, name=f"burst >= {N} (leaves S)"))
        rx.append(Reaction((0, -1), lambda X: k5 * X[:, 1], name="P -> 0"))
        return rx
    width = int(max(40, 8 * b_mean))
    return Model(f"feedback_b{int(b_mean)}", "P", ("G", "P"), build, (2, width), (0.0, 0.0), (True, False),
                 {"kind": "nb", "means": (0.5, k4 * b_mean / k5), "q": (0.0, q)},
                 f"Kuntz, Thomas, Stan, Barahona, JCP 2019 (arXiv:1702.05468v4), Sec. V.B, eq. (114), pp. 17-18; "
                 f"k3 = k4 = 10, k1 = k2 = k5 = 1, <b> = {b_mean:g} (Fig. 7 caption, p. 20; model of their ref. 58); "
                 f"stationary study, x0 = 0 with the gene off",
                 params=dict(b_mean=b_mean, k1=k1, k2=k2, k3=k3, k4=k4, k5=k5))


LAMBDA_PHAGE_T = 10.0  # the source's horizon


def lambda_phage(a=(0.5, 1.0, 0.15, 0.3, 0.3), b=(0.12, 0.6, 1.0, 1.0, 1.0),
                 c=(0.0025, 0.0007, 0.0231, 0.01, 0.01)):
    """Hegland et al. 2007 lambda-phage model as given in Jahnke & Huisinga 2008 Sec. 4.2, eq. (22),
    p. 2297, parameter table p. 2298; source initial law multinomial(n = 3, p = 0.05 each), eq. (23)."""
    a1, a2, a3, a4, a5 = a
    b1, b2, b3, b4, b5 = b
    build = lambda w: [
        Reaction((1, 0, 0, 0, 0), lambda X: a1 * b1 / (b1 + X[:, 1]), name="0 -> S1"),
        Reaction((0, 1, 0, 0, 0), lambda X: (a2 + X[:, 4]) * b2 / (b2 + X[:, 0]), name="0 -> S2"),
        Reaction((0, 0, 1, 0, 0), lambda X: a3 * b3 * X[:, 1] / (b3 * X[:, 1] + 1), name="0 -> S3"),
        Reaction((0, 0, 0, 1, 0), lambda X: a4 * b4 * X[:, 2] / (b4 * X[:, 2] + 1), name="0 -> S4"),
        Reaction((0, 0, 0, 0, 1), lambda X: a5 * b5 * X[:, 2] / (b5 * X[:, 2] + 1), name="0 -> S5"),
    ] + [Reaction(tuple(-1 if j == i else 0 for j in range(5)), lambda X, i=i: c[i] * X[:, i], name=f"S{i + 1} -> 0")
         for i in range(5)]
    return Model("lambda_phage", "P", ("S1", "S2", "S3", "S4", "S5"), build, (15, 40, 10, 10, 10),
                 (0.15,) * 5, (False,) * 5, {"kind": "poisson_maxprod", "removal": tuple(c)},
                 "Jahnke & Huisinga, Bull. Math. Biol. 70 (2008), Sec. 4.2, eq. (22), p. 2297; parameters "
                 "p. 2298 (from Hegland et al. 2007); box (15, 40, 10, 10, 10) and T = 10 from the source; initial "
                 "means 0.15 = E of the source's multinomial(3, 0.05) law, eq. (23)",
                 params=dict(a=a, b=b, c=c, T_source=LAMBDA_PHAGE_T))


def schlogl_published():
    """P1: the registry Schlogl model (Kuntz et al. 2019 bimodal set), unchanged."""
    return schlogl()


TANG_TOGGLE_T = 40.0  # source horizon: 8e3 steps x 5e-3 h (Tang et al. main text Table I)


def toggle_tang(sx=50.0, sy=50.0, dx=1.0, dy=1.0, bx=1e-4, by=1e-4, ux=0.1, uy=0.1):
    """P7: genetic toggle with explicit gene states, Tang, Weng, Zhang, Nat. Mach. Intell. 2023,
    SI Supplementary Note 2 F, eq. (S2.9); values from the Supplementary Fig. 7 caption. Propensities
    c_k prod n_i^r_ki (main text, Methods), so 2P + G binds at b P^2 G; the source's reflecting boundary at
    zero blocks it when P < 2. State (Gx, Gy, Px, Py), G = 1 unbound, G_bar = 1 - G."""
    build = lambda w: [
        Reaction((0, 0, 1, 0), lambda X: sx * X[:, 0], name="Gx -> Gx + Px"),
        Reaction((0, 0, 0, 1), lambda X: sy * X[:, 1], name="Gy -> Gy + Py"),
        Reaction((0, 0, -1, 0), lambda X: dx * X[:, 2], name="Px -> 0"),
        Reaction((0, 0, 0, -1), lambda X: dy * X[:, 3], name="Py -> 0"),
        Reaction((0, -1, -2, 0), lambda X: by * X[:, 2] ** 2 * X[:, 1] * (X[:, 2] >= 2),
                 rate=lambda m: by * m[2] ** 2 * m[1], name="2Px + Gy -> Gy_bar"),
        Reaction((-1, 0, 0, -2), lambda X: bx * X[:, 3] ** 2 * X[:, 0] * (X[:, 3] >= 2),
                 rate=lambda m: bx * m[3] ** 2 * m[0], name="2Py + Gx -> Gx_bar"),
        Reaction((0, 1, 2, 0), lambda X: uy * (1 - X[:, 1]), name="Gy_bar -> 2Px + Gy"),
        Reaction((1, 0, 0, 2), lambda X: ux * (1 - X[:, 0]), name="Gx_bar -> 2Py + Gx"),
    ]
    return Model("toggle_tang", "P", ("Gx", "Gy", "Px", "Py"), build, (2, 2, 64, 64), (1.0, 1.0, 0.0, 0.0),
                 (True, True, False, False), {"kind": "poisson_maxprod", "removal": (1.0, 1.0, dx, dy)},
                 "Tang, Weng, Zhang, Nat. Mach. Intell. 2023 (doi:10.1038/s42256-023-00632-6), SI Supplementary "
                 "Note 2 F, eq. (S2.9), printed pp. 7-8; sx = sy = 50, dx = dy = 1, bx = by = 1e-4, ux = uy = 0.1 "
                 "(Supp. Fig. 7 caption, hours); propensity rule and reflecting boundaries: main text Methods "
                 "(arXiv:2210.01169v2); initial delta: one unbound form of each gene, zero proteins (Fig. 2 caption); "
                 "t in [0, 40] (Table I)",
                 params=dict(sx=sx, sy=sy, dx=dx, dy=dy, bx=bx, by=by, ux=ux, uy=uy, T_source=TANG_TOGGLE_T))


TANG_AUTOREG_T = 10.0  # source horizon: 2e3 steps x 5e-3 h (Tang et al. main text Table I; Supp. Fig. 4)
TANG_AUTOREG_CASES = {1: (0.94, 0.01, 8.40, 28.1), 2: (0.69, 0.07, 7.20, 40.6), 3: (0.44, 0.08, 0.94, 53.1)}


def autoreg_tang(case=1):
    """P8: autoregulatory feedback loop, Tang et al. SI Supplementary Note 2 C, eq. (S2.6),
    Supplementary Table I (sigma_u, sigma_b, rho_u, rho_b); P decays at rate 1 (hours). The registry
    self_repressor form: state (Gu, Gb, P) with Gu + Gb = 1."""
    su, sb, ru, rb = TANG_AUTOREG_CASES[case]
    build = lambda w: [
        Reaction((-1, 1, -1), lambda X: sb * X[:, 0] * X[:, 2], name="Gu + P -> Gb"),
        Reaction((1, -1, 1), lambda X: su * X[:, 1], name="Gb -> Gu + P"),
        Reaction((0, 0, 1), lambda X: ru * X[:, 0], name="Gu -> Gu + P"),
        Reaction((0, 0, 1), lambda X: rb * X[:, 1], name="Gb -> Gb + P"),
        Reaction((0, 0, -1), lambda X: 1.0 * X[:, 2], name="P -> 0"),
    ]
    L = np.array([[1, 1, 0]])
    return Model(f"autoreg_tang_{case}", "P", ("Gu", "Gb", "P"), build, (2, 2, 60), (1.0, 0.0, 0.0),
                 (True, True, False), {"kind": "poisson_maxprod", "removal": (1.0, 1.0, 1.0)},
                 f"Tang, Weng, Zhang, Nat. Mach. Intell. 2023 (doi:10.1038/s42256-023-00632-6), SI Supplementary "
                 f"Note 2 C, eq. (S2.6), printed p. 4, Supplementary Table I case {case}: sigma_u = {su}, "
                 f"sigma_b = {sb}, rho_u = {ru}, rho_b = {rb} (hours; model of Sukys, Ocal, Grima, iScience 2022); "
                 f"initial delta: one Gu, zero protein (Supp. Fig. 4 caption); t in [0, 10] (Table I)",
                 conservation=(L, np.array([1])),
                 params=dict(case=case, sigma_u=su, sigma_b=sb, rho_u=ru, rho_b=rb, T_source=TANG_AUTOREG_T))


PUBLISHED_MODELS = (schlogl_published, toggle_hegland, toggle_kuntz, toggle_gardner,
                    lambda: feedback_kuntz(1.0), lambda: feedback_kuntz(10.0), lambda: feedback_kuntz(100.0),
                    lambda_phage, toggle_tang, lambda: autoreg_tang(1), lambda: autoreg_tang(2),
                    lambda: autoreg_tang(3))


def published():
    """The twelve published gene-network instances."""
    return [f() for f in PUBLISHED_MODELS]
