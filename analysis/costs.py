"""Costs part: fault-tolerant resource counts of the quantum routes.

    python -m analysis.costs circuit_checks          block encodings at gate level (I8, I9), the walk operator, the
                                                     LCHS kernels, and the reversible fixed-point oracle arithmetic
    python -m analysis.costs encodings [--only ...]  uniform, R-NB and R-mix(NB,E1) master-equation encodings
    python -m analysis.costs baselines [--only ...]  amplitude estimation over coherent SSA, and the Lindblad embedding
    python -m analysis.costs comparison              every encoding against coherent SSA (Figure 3, Table S6)
    python -m analysis.costs split_design            the second (split) design of the diagonal oracle
    python -m analysis.costs observable_aware_ssa    coherent SSA with a step count matched to the observable

Models: toggle, schlogl, michaelis_menten, self_repressor, cascade1..cascade4, birth_death. Totals are Toffoli
counts at relative precision EPS and failure probability DELTA_FAIL. Weighted encodings are costed with three growth
factors: the certificate (cert, certified), the worst case W and the measured growth (meas), which are not certified.
The FSP solution is used only for W, the measured growth, the observable values and the Lindblad step count.
"""
import sys
import time
import warnings

import numpy as np

from .common import data, log_norm_weighted, num, observables, save
from qcme import baselines as BL
from qcme import certify, circuits, emulate, fsp, lchs, models, network, reference
from qcme import resources as R

warnings.filterwarnings("ignore", category=RuntimeWarning)
PART = "costs"
LEAK_TOL = 1e-6
EPS, DELTA_FAIL = 1e-2, 1e-2
EPS_SWEEP = (1e-1, 1e-2, 1e-3, 1e-4)
BENCHMARKS = ("toggle", "schlogl", "michaelis_menten", "self_repressor", "cascade2", "cascade3")
COST_MODELS = BENCHMARKS + ("cascade1", "cascade4", "birth_death")
WEIGHTED = {"R-mix(NB,E1)": "mix", "R-NB": "product"}
ENCS = {"uniform": "uniform", "R-NB": "product", "R-mix(NB,E1)": "mix"}
KERNELS = {"K1(beta=0.6)": lambda d: lchs.ACL(0.6), "K1(beta=0.7)": lambda d: lchs.ACL(0.7),
           "K1(beta=0.8)": lambda d: lchs.ACL(0.8), "K2(c=1)": lambda d: lchs.LowSomma(1.0, d),
           "K2(c=2)": lambda d: lchs.LowSomma(2.0, d)}
VARIANTS = ("single", "per_window")
FAMILIES = certify.FAMILIES
HEADLINE_GROWTH = "W"


def schedules(T):
    out = {f"uniform J={J}": certify.uniform_edges(T, J) for J in (5, 10, 20, 40, 80)}
    out["geometric J=20"] = certify.geometric_edges(T, 20, 2.0**-10)
    return out


def cost_settings():
    return {"leak_tol": LEAK_TOL, "eps": EPS, "delta_fail": DELTA_FAIL, "eps_sweep": EPS_SWEEP, "weighted": WEIGHTED,
            "kernels": list(KERNELS), "variants": VARIANTS, "families": FAMILIES, "headline_growth": HEADLINE_GROWTH,
            "eta": reference.ETA, "lchs_safety": lchs.SAFETY}


def model_by_name(name):
    if name.startswith("cascade"):
        return models.cascade(int(name[len("cascade"):]))
    return next(m for m in models.registry() if m.name == name)


class Setup:
    pass


def setup(name):
    m = model_by_name(name)
    s = Setup()
    s.name, s.model = name, m
    s.T = models.relaxation_time(m)[0]
    s.widths, s.leak = fsp.select_box(m, lambda sp_: reference.initial_law(m, sp_), s.T, LEAK_TOL)
    s.widths = tuple(int(w) for w in s.widths)
    s.space = m.space(s.widths)
    s.rx = m.reactions(s.widths)
    s.nus = [r.nu for r in s.rx]
    s.gen = network.generator(s.space, s.rx)
    s.p0 = reference.initial_law(m, s.space)
    refs = reference.extended_references(m, s.space, s.T, s.widths, gen=None, ranks=())
    s.refs = {k: refs[k] for k in WEIGHTED}
    s.uniform = reference.Uniform(len(s.space))
    s.st = R.ModelStructure.build(m, s.gen, s.widths)
    return s


def window_alphas(s, ref, edges, cw):
    """alpha_j = sum_k max_x b_k^(j)(x) + max_x (a_0(x) + c_j), and D_j max for the window boundaries."""
    X = s.space.states
    a = s.gen.a
    a0max = float(a.sum(axis=0).max())
    tg = [s.space.lookup(X + np.asarray(nu)) for nu in s.nus]
    alphas, lms = [], []
    for j in range(len(edges) - 1):
        lm = ref.log_mu(edges[j])
        lms.append(lm)
        tot = 0.0
        for k in range(len(s.nus)):
            ok = tg[k] >= 0
            if not np.any(ok):
                continue
            bk = a[k][ok] * np.exp(0.5 * (lm[ok] - lm[tg[k][ok]]))
            tot += float(bk.max()) if len(bk) else 0.0
        alphas.append(tot + a0max + cw[j])
    Dmax = [0.5 * float(np.ptp(lms[j] - lms[j + 1])) for j in range(len(lms) - 1)]
    return np.array(alphas), np.array(Dmax), lms


def select_certificate(s, ref):
    """Smallest K_T over the six schedules and two weight families (certificate quantities only)."""
    best, table = None, {}
    for sname, edges in schedules(s.T).items():
        cert = certify.window_certificate(s.gen, ref, edges)
        table[sname] = {f: cert.K[f] for f in FAMILIES}
        for f in FAMILIES:
            if best is None or cert.K[f] < best[0]:
                best = (cert.K[f], sname, f, cert, edges)
    return best, table


def prep_costs(s, enc, obs):
    """Toffolis (as a function of b) of the two preparations per Hadamard-test use: psi_0 and phi_f."""
    st = s.st
    n = len(s.space)
    if enc == "mix" or s.model.conservation is not None:
        return lambda b: 2 * R.generic_state_prep(n, b), "generic (alias over |S|)"

    def cost(b):
        psi = R.product_state_prep(st.bits, b)
        if enc == "uniform":  # phi = f / ||f||: uniform over the box (tail: over [theta, w_i) for species i)
            phi = 0
            for i, (w, nb) in enumerate(zip(st.widths, st.bits)):
                span = w - (obs["theta"] if obs["species"] == i else 0)
                phi += R.uniform_prep(span, b) + (R.cadd_const(nb) if obs["species"] == i else 0)
        else:
            phi = R.product_state_prep(st.bits, b)
        return psi + phi
    return cost, "product (rotation trees)"


# ----------------------------------------------------------------------------- circuit checks
def small_instances():
    return [("birth_death", (8,)), ("toggle", (4, 4)), ("cascade2", (4, 4)), ("self_repressor", (2, 2, 8)),
            ("michaelis_menten_small", None)]


def _small(name, w):
    m = models.michaelis_menten(E0=2, S0=3) if name == "michaelis_menten_small" else model_by_name(name)
    w = tuple(w or m.widths)
    sp_ = m.space(w)
    rx = m.reactions(w)
    gen = network.generator(sp_, rx)
    T = models.relaxation_time(m)[0]
    refs = reference.extended_references(m, sp_, T, w, gen=None, ranks=())
    return m, w, sp_, rx, gen, T, refs


def block_encoding_rows(diag="value"):
    rows = []
    for name, w in small_instances():
        m, w, sp_, rx, gen, T, refs = _small(name, w)
        nus = [r.nu for r in rx]
        for rname, ref in (("uniform", reference.Uniform(len(sp_))), ("R-NB", refs["R-NB"]),
                           ("R-mix(NB,E1)", refs["R-mix(NB,E1)"])):
            for t in (0.0, 0.5 * T):
                lm = ref.log_mu(t)
                c = certify.cw_sqrt_mu(certify.residual(gen.A, lm), gen.leak)
                if diag == "value":
                    inst = circuits.block_encoding_circuit(gen, nus, lm, c)
                else:
                    inst = circuits.block_encoding_circuit(gen, nus, lm, c, diag="split", reactions=rx)
                perm_ok, _ = inst.circuit.check_permutations()
                Bk = circuits.block_of(inst)
                sysidx = circuits.system_index(sp_.states, inst.sys_bits)
                Bt = certify.weighted_generator(gen.A, lm).toarray() - c * np.eye(len(sp_))
                out = np.setdiff1d(np.arange(Bk.shape[0]), sysidx)
                row = {"model": name, "widths": w, "reference": rname, "t": t, "qubits": inst.layout.n,
                       "alpha": inst.alpha, "I8": float(np.abs(Bk - inst.M / inst.alpha).max()),
                       "M_equals_Btilde_on_S": float(np.abs(inst.M[np.ix_(sysidx, sysidx)] - Bt).max()),
                       # I9: every transition from S that leaves S has exactly zero amplitude
                       "I9": bool(np.all(Bk[np.ix_(out, sysidx)] == 0)) if len(out) else True,
                       "permutations_ok": perm_ok,
                       "lognorm_M": float(np.linalg.eigvalsh(0.5 * (inst.M + inst.M.T)).max())}
                if diag == "value":
                    boundary = 0
                    for k, nu in enumerate(nus):
                        leave = sp_.lookup(sp_.states + np.asarray(nu)) < 0
                        boundary += int(np.sum(leave & (gen.a[k] > 0)))
                    row["boundary_transitions"] = boundary
                rows.append(row)
                print(f"  I8/I9 {name:22s} {rname:13s} t={t:.3g}: I8={row['I8']:.1e} I9={row['I9']}", flush=True)
    return rows


def walk_check():
    """The self-inverse select: V^2 = I, V Hermitian, top-left block (c_k B + conj(c_k) B^T)/Lam (toggle 4 x 4, R-NB)."""
    m = models.toggle()
    w = (4, 4)
    sp_ = m.space(w)
    rx = m.reactions(w)
    gen = network.generator(sp_, rx)
    nus = [r.nu for r in rx]
    lm = reference.extended_references(m, sp_, 3.0, w, ranks=())["R-NB"].log_mu(0.0)
    c = certify.cw_sqrt_mu(certify.residual(gen.A, lm), gen.leak)
    inst = circuits.block_encoding_circuit(gen, nus, lm, c)
    N = 1 << inst.layout.n
    U = np.zeros((N, N), dtype=complex)
    for i in range(N):
        e = np.zeros(N, dtype=complex)
        e[i] = 1
        U[:, i] = inst.circuit.apply(e)
    Nsys = 1 << inst.layout.regs["sys"][1]
    out = []
    for k in (0.0, 1.7, -12.5):
        ck = (-k + 1j) / 2
        Lam = inst.alpha * np.sqrt(1 + 30.0**2)
        V = circuits.hermitian_select(U, inst.layout.n - inst.layout.regs["sys"][1], ck, Lam, inst.alpha)
        blk = V[:Nsys, :Nsys]
        want = (ck * inst.M + np.conj(ck) * inst.M.T) / Lam
        out.append({"k": k, "block_err": float(np.abs(blk - want).max()),
                    "V2_err": float(np.abs(V @ V - np.eye(2 * N)).max()),
                    "herm_err": float(np.abs(V - V.conj().T).max())})
    ok = all(o["block_err"] <= 1e-10 and o["V2_err"] <= 1e-10 and o["herm_err"] <= 1e-12 for o in out)
    return {"rows": out, "pass": ok}


def kernel_validation():
    """Every kernel reproduces e^{-A} within its bound for random 6 x 6 accretive matrices and for time-ordered
    products of B~ windows with diagonal jumps on a small toggle box."""
    rng = np.random.default_rng(20260927)
    cases = []
    for trial in range(3):
        pieces = []
        for _ in range(2):
            X = rng.normal(size=(6, 6))
            Lm = X @ X.T / 6
            Hm = rng.normal(size=(6, 6))
            pieces.append(Lm + 1j * (Hm + Hm.T) / 2)
        pieces.insert(1, np.diag(rng.uniform(0, 3, 6)).astype(complex))
        cases.append((f"random{trial}", pieces))
    m = models.toggle()
    w = (8, 8)
    sp_ = m.space(w)
    gen = network.generator(sp_, m.reactions(w))
    ref = reference.extended_references(m, sp_, 1.0, w, ranks=())["R-NB"]
    edges = certify.uniform_edges(1.0, 3)
    cert = certify.window_certificate(gen, ref, edges, families=("sqrt_mu",))
    pieces = []
    for j in range(3):
        lm = ref.log_mu(edges[j])
        Bt = certify.weighted_generator(gen.A, lm).toarray() - cert.cw["sqrt_mu"][j] * np.eye(len(sp_))
        pieces.append(-Bt * (edges[j + 1] - edges[j]))
        if j < 2:
            D = cert.log_rho[j] - 0.5 * (lm - ref.log_mu(edges[j + 1]))
            pieces.append(np.diag(D).astype(complex))
    cases.append(("toggle8x8_RNB_J3", pieces))
    rows, ok = [], True
    for cname, pieces in cases:
        Lnorm = sum(np.linalg.norm(lchs.split(P)[0], 2) for P in pieces)
        W = lchs.exact_product(pieces)
        for kname, kf in KERNELS.items():
            for delta in (1e-4, 1e-8):
                disc = lchs.discretize(kf(delta / 3), Lnorm, delta)
                err = float(np.linalg.norm(lchs.emulate(disc, pieces) - W, 2))
                good = err <= disc.error
                ok &= good
                rows.append({"case": cname, "kernel": kname, "delta": delta, "Lnorm": Lnorm, "bound": disc.error,
                             "actual": err, "ok": good, "K": disc.K, "nodes": disc.M, "l1": disc.l1})
    return {"rows": rows, "pass": bool(ok)}


def reversible_check():
    """Fixed-point oracle arithmetic as a reversible program on the toggle 75 x 75, R-NB, reaction 0: log-factor
    tables are xor-loaded and added, the exponential is evaluated by a piecewise polynomial in fixed point, and
    everything is uncomputed. Checks that the workspace returns to zero, that the forward map is injective, and that
    the amplitude matches b_0/max b_0 within 2^{-b+3} for b = 24."""
    m = models.toggle()
    w = (75, 75)
    sp_ = m.space(w)
    gen = network.generator(sp_, m.reactions(w))
    lm = reference.extended_references(m, sp_, 3.0, w, ranks=())["R-NB"].log_mu(0.0)
    X = sp_.states
    tg = sp_.lookup(X + np.array([1, 0]))
    ok = tg >= 0
    b0 = np.zeros(len(X))
    b0[ok] = gen.a[0][ok] * np.exp(0.5 * (lm[ok] - lm[tg[ok]]))
    bmax = b0.max()
    nx = ny = 128
    la, lr = np.full(ny, -np.inf), np.full(nx, -np.inf)
    for y in range(w[1]):
        la[y] = np.log(gen.a[0][sp_.index((0, y))])
    for x in range(w[0] - 1):
        lr[x] = 0.5 * (lm[sp_.index((x, 0))] - lm[sp_.index((x + 1, 0))])
    b = 24
    frac = b
    Z = b * np.log(2)
    zc = np.log(bmax)
    ta = np.where(np.isfinite(la), np.round((la - zc / 2) * 2**frac), -(2**40)).astype(np.int64)
    tr = np.where(np.isfinite(lr), np.round((lr - zc / 2) * 2**frac), -(2**40)).astype(np.int64)
    fc = R.function_eval("exp", -Z, 0.0, b)
    edges = np.linspace(-Z, 0.0, fc.pieces + 1)
    nodes = np.cos(np.pi * (np.arange(fc.degree + 1) + 0.5) / (fc.degree + 1))
    coefs = []
    for p in range(fc.pieces):
        a_, c_ = edges[p], edges[p + 1]
        xs = 0.5 * (a_ + c_) + 0.5 * (c_ - a_) * nodes
        coefs.append(np.polynomial.polynomial.polyfit(xs - a_, np.exp(xs), fc.degree))
    width = 64
    mask = (1 << width) - 1
    max_err, restored, images = 0.0, True, set()
    for x in range(w[0]):
        for y in range(w[1]):
            if not ok[sp_.index((x, y))]:
                continue
            regs = {"x": x, "y": y, "la": 0, "lr": 0, "z": 0}
            prog = circuits.ReversibleProgram({"la": width, "lr": width, "z": width, "x": 7, "y": 7})
            prog.lookup_xor("la", "y", [int(v) & mask for v in ta])
            prog.lookup_xor("lr", "x", [int(v) & mask for v in tr])
            prog.add_into("z", "la")
            prog.add_into("z", "lr")
            fwd = prog.run(regs)
            images.add((fwd["x"], fwd["y"], fwd["z"]))
            zval = fwd["z"] - (1 << width) if fwd["z"] >> (width - 1) else fwd["z"]
            zf = min(max(zval / 2**frac, -Z), 0.0)
            pidx = min(int((zf + Z) / (Z / fc.pieces)), fc.pieces - 1)
            u = int(round((zf - edges[pidx]) * 2**b))
            acc = int(round(coefs[pidx][-1] * 2**b))
            for cf in coefs[pidx][-2::-1]:
                acc = (acc * u >> b) + int(round(cf * 2**b))
            max_err = max(max_err, abs(acc / 2**b - b0[sp_.index((x, y))] / bmax))
            restored &= prog.run(fwd, reverse=True) == regs
    good = restored and len(images) == int(ok.sum()) and max_err <= 2.0 ** (-b + 3)
    return {"b": b, "pieces": fc.pieces, "degree": fc.degree, "max_abs_error": max_err, "error_bound": 2.0 ** (-b + 3),
            "workspace_restored": bool(restored), "injective": len(images) == int(ok.sum()), "pass": bool(good)}


def stage_circuit_checks(only=None):
    rows = block_encoding_rows()
    I8_pass = all(r["I8"] <= 1e-10 and r["M_equals_Btilde_on_S"] <= 1e-12 and r["permutations_ok"] for r in rows)
    I9_pass = all(r["I9"] for r in rows)
    out = {"I8_I9": rows, "I8_pass": I8_pass, "I9_pass": I9_pass, "walk": walk_check(), "kernels": kernel_validation(),
           "reversible": reversible_check()}
    print(f"I8 {I8_pass}; I9 {I9_pass}; walk {out['walk']['pass']}; kernels {out['kernels']['pass']}; "
          f"reversible {out['reversible']['pass']}", flush=True)
    save(PART, "circuit_checks", cost_settings(), out)


# ----------------------------------------------------------------------------- master-equation encodings
def classical_ssa_samples(P, eps, delta_fail):
    from scipy.stats import norm
    z = norm.ppf(1 - delta_fail / 2)
    return float(np.ceil(z * z * (1 - P) / (eps * eps * P))) if P < 1 else 1.0


def readout_row(s, rt, growth, g, R_f, eps, info, kf, kname, var, prep):
    lg, include_s = growth[g]
    t0 = time.time()
    rc = R.weighted_readout(s.st, rt, float(np.exp(lg)) if lg < 700 else np.inf, R_f, eps, DELTA_FAIL,
                            info["alphas"], np.diff(info["edges"]), info["Dmax"], kf, var, prep,
                            include_s=include_s, label=f"{g}") if lg < 700 else None
    if rc is None:
        return {"growth": g, "kernel": kname, "variant": var, "eps": eps, "log10_gamma": lg / np.log(10),
                "toffoli_total": np.inf, "note": "Gamma beyond double precision"}
    c = rc.lchs
    return {"growth": g, "kernel": kname, "variant": var, "eps": eps, "log10_gamma": np.log10(rc.gamma),
            "eta": rc.eta, "M": rc.M, "n": rc.n, "N_U": rc.N_U, "toffoli_U": rc.toffoli_U,
            "toffoli_prep": rc.toffoli_prep, "toffoli_total": rc.toffoli_total, "qubits": rc.qubits,
            "s": c.s, "walk_steps": c.walk_steps, "nodes": c.nodes, "K_cut": c.K, "h": c.h, "b": c.b,
            "query": c.query, "breakdown": c.breakdown, "lchs_error": c.lchs_error, "q_max": int(max(c.q)),
            "alpha_max": float(np.max(info["alphas"])),
            "sum_alpha_dt": float(np.sum(info["alphas"] * np.diff(info["edges"]))), "seconds": time.time() - t0}


def run_costs(name):
    s = setup(name)
    print(f"-- {name}: T={s.T:.4g} widths={s.widths} |S|={len(s.space)} bits={s.st.bits}", flush=True)
    t_fsp = []
    for _ in range(5):
        t1 = time.perf_counter()
        pT = fsp.propagate(s.gen.A, s.p0, [0.0, s.T])[-1]
        t_fsp.append(time.perf_counter() - t1)
    obs = observables(s, pT)
    Ef = {k: float(o["f"] @ pT) for k, o in obs.items()}
    enc_info = {}
    cu = certify.window_certificate(s.gen, s.uniform, np.array([0.0, s.T]))
    fam_u = min(FAMILIES, key=lambda f: cu.K[f])
    lm_u = np.zeros(len(s.space))
    al_u, _, _ = window_alphas(s, s.uniform, np.array([0.0, s.T]), cu.cw[fam_u])
    W_u = emulate.horizon_worst_case(s.gen.A, lm_u, lm_u, s.T)
    enc_info["uniform"] = {"reftype": "uniform", "family": fam_u, "schedule": "single window", "K": cu.K[fam_u],
                           "edges": [0.0, s.T], "cw": cu.cw[fam_u], "alphas": al_u, "Dmax": np.zeros(0),
                           "log_psi0": float(np.log(np.linalg.norm(s.p0))), "log_W": float(np.log(W_u)),
                           "log_meas": float(np.log(np.linalg.norm(pT))), "lmT": lm_u}
    for rn, rt in WEIGHTED.items():
        ref = s.refs[rn]
        (K, sname, fam, cert, edges), table = select_certificate(s, ref)
        al, Dm, lms = window_alphas(s, ref, edges, cert.cw[fam])
        lm0, lmT = lms[0], lms[-1]  # mu_T = mu of the last window
        log_W = float(np.log(emulate.horizon_worst_case(s.gen.A, lm0, lmT, s.T))) \
            if np.ptp(lm0) < 1400 and np.ptp(lmT) < 1400 else None
        enc_info[rn] = {"reftype": rt, "family": fam, "schedule": sname, "K": K, "K_table": table, "edges": edges,
                        "cw": cert.cw[fam], "log_rho": cert.log_rho, "alphas": al, "Dmax": Dm,
                        "log_psi0": log_norm_weighted(s.p0, lm0), "log_W": log_W,
                        "log_meas": log_norm_weighted(pT, lmT), "lmT": lmT}
        print(f"   {rn:13s} K={K:.5g} ({fam}, {sname})", flush=True)
    results = {}
    for enc, info in enc_info.items():
        rt = info["reftype"]
        dts = np.diff(info["edges"])
        results[enc] = {}
        for oname, o in obs.items():
            R_f = float(np.linalg.norm(o["f"] * np.exp(0.5 * info["lmT"])) / abs(Ef[oname]))
            prep, prep_kind = prep_costs(s, rt, o)
            growth = {"cert": (info["log_psi0"] + 0.5 * info["K"], True)}
            if info["log_W"] is not None:
                growth["W"] = (info["log_psi0"] + info["log_W"], False)
            growth["meas"] = (info["log_meas"], False)
            rows = []
            head = HEADLINE_GROWTH if HEADLINE_GROWTH in growth else "cert"
            for kname, kf in KERNELS.items():
                for var in (("single",) if len(dts) == 1 else VARIANTS):
                    rows.append(readout_row(s, rt, growth, head, R_f, EPS, info, kf, kname, var, prep))
            best = min(rows, key=lambda r: r["toffoli_total"])
            for g in growth:
                if g != head:
                    rows.append(readout_row(s, rt, growth, g, R_f, EPS, info, KERNELS[best["kernel"]], best["kernel"],
                                            best["variant"], prep))
            sweep = [readout_row(s, rt, growth, head, R_f, e, info, KERNELS[best["kernel"]], best["kernel"],
                                 best["variant"], prep) for e in EPS_SWEEP]
            results[enc][oname] = {"R_f": R_f, "E_f": Ef[oname], "prep_kind": prep_kind, "rows": rows,
                                   "best_headline": {"kernel": best["kernel"], "variant": best["variant"],
                                                     "growth": head}, "eps_sweep": sweep,
                                   "theta": o["theta"], "species": o["species"]}
            print(f"   {enc:13s} {oname:4s} best {best['kernel']}/{best['variant']} {head}: "
                  f"{best['toffoli_total']:.3e} Toffolis", flush=True)
    summary = [{"label": f"{enc} [{row['growth']}]", "encoding": enc, "growth": row["growth"], "observable": oname,
                "toffoli": row["toffoli_total"], "qubits": row["qubits"]}
               for enc, per in results.items() for oname, r in per.items() for row in r["rows"]
               if row["kernel"] == r["best_headline"]["kernel"] and row["variant"] == r["best_headline"]["variant"]]
    return {"model": name, "level": s.model.level, "T": s.T, "widths": s.widths, "n_states": len(s.space),
            "leak": s.leak, "bits": s.st.bits, "structure": {"deps": s.st.deps, "supp": s.st.supp},
            "observables": {k: {"E_f": Ef[k], "theta": o["theta"], "species": o["species"]} for k, o in obs.items()},
            "fsp_wall_time_s": {"median": float(np.median(t_fsp)), "min": float(np.min(t_fsp)),
                                "max": float(np.max(t_fsp))},
            "classical_ssa_samples": {k: classical_ssa_samples(Ef[k], EPS, DELTA_FAIL) for k in obs},
            "encodings": {k: {kk: vv for kk, vv in v.items() if kk != "lmT"} for k, v in enc_info.items()},
            "costs": results, "summary": summary}


def stage_encodings(only=None):
    for name in (only or COST_MODELS):
        t0 = time.time()
        save(PART, name, {**cost_settings(), "model": name}, run_costs(name))
        print(f"== {name} [{time.time() - t0:.0f}s]", flush=True)


# ----------------------------------------------------------------------------- baselines and the comparison
def stage_baselines(only=None):
    for name in (only or COST_MODELS):
        s = setup(name)
        pT = fsp.propagate(s.gen.A, s.p0, [0.0, s.T])[-1]
        out = {"model": name, "T": s.T, "widths": s.widths, "rows": []}
        for oname, o in observables(s, pT).items():
            P = float(o["f"] @ pT)
            f_cost = 0 if o["species"] is None else 2 * R.cmp_const(s.st.bits[o["species"]])
            for method in ("uniformization", "direct"):
                c = BL.coherent_ssa_cost(s.st, s.gen, s.T, P, EPS, DELTA_FAIL, f_cost, method)
                out["rows"].append({"observable": oname, "kind": "ssa", "method": method, **c.__dict__})
            for order in (1, 2):
                N, err, diag = BL.lindblad_steps(s.gen, s.nus, order, s.T, s.p0, o["f"], P, EPS * P / 4)
                if N is None:
                    out["rows"].append({"observable": oname, "kind": "lindblad", "order": order,
                                        "note": diag["method"], "search": diag})
                    continue
                c = BL.lindblad_cost(s.st, s.gen, s.nus, order, s.T, N, P, EPS, DELTA_FAIL, f_cost, err)
                out["rows"].append({"observable": oname, "kind": "lindblad", **c.__dict__, "search": diag})
        save(PART, f"{name}_baselines", {**cost_settings(), "model": name}, out)


def stage_comparison(only=None):
    table = []
    for name in COST_MODELS:
        c, b = data(PART, name), data(PART, f"{name}_baselines")
        for oname in ("one", "tail"):
            ssa = next(r for r in b["rows"] if r["observable"] == oname and r.get("method") == "uniformization")
            lind = [r for r in b["rows"] if r["observable"] == oname and r["kind"] == "lindblad" and "total" in r]
            row = {"model": name, "observable": oname, "ssa_total": ssa["total"],
                   "lindblad_best_total": min((r["total"] for r in lind), default=None)}
            for enc in ("uniform",) + tuple(WEIGHTED):
                per = c["costs"][enc][oname]
                for g in ("W", "cert", "meas"):
                    rr = [r for r in per["rows"] if r["growth"] == g and r["kernel"] == per["best_headline"]["kernel"]
                          and r["variant"] == per["best_headline"]["variant"]]
                    if rr:
                        tot = float(rr[0]["toffoli_total"])
                        row[f"{enc}|{g}"] = tot
                        row[f"{enc}|{g}|ratio_to_ssa"] = tot / ssa["total"]
            if row["lindblad_best_total"] is not None:
                row["lindblad_over_ssa"] = row["lindblad_best_total"] / ssa["total"]
            table.append(row)
    verdict = {}
    for enc in tuple(WEIGHTED) + ("uniform",):
        for oname in ("one", "tail"):
            ratios = [r.get(f"{enc}|W|ratio_to_ssa") for r in table if r["model"] in BENCHMARKS and r["observable"] == oname]
            ok = None if any(x is None for x in ratios) else all(float(x) > 100 for x in ratios)
            verdict[f"{enc}|{oname}"] = {"all_ratios_above_100": ok,
                                          "min_ratio": None if not ratios or None in ratios else min(ratios)}
    lvs = {}
    for oname in ("one", "tail"):
        vals = [r.get("lindblad_over_ssa") for r in table if r["model"] in BENCHMARKS and r["observable"] == oname]
        lvs[oname] = {"ratios": vals, "within_factor_10": None if None in vals else all(0.1 <= v <= 10 for v in vals)}
    save(PART, "comparison", cost_settings(), {"table": table, "verdict": verdict, "lindblad_vs_ssa": lvs})


# ----------------------------------------------------------------------------- the split diagonal design
def headline(c, enc, obs, growth="W"):
    per = c["costs"][enc][obs]
    bh = per["best_headline"]
    return next((r for r in per["rows"] if r["growth"] == growth and r["kernel"] == bh["kernel"]
                 and r["variant"] == bh["variant"]), None)


def growth_inputs(info, g):
    """(gamma without s, include_s) for growth factor g of a stored encoding."""
    if g == "cert":
        return float(np.exp(info["log_psi0"] + 0.5 * num(info["K"]))), True
    if g == "W":
        return (None, False) if info["log_W"] is None else (float(np.exp(info["log_psi0"] + info["log_W"])), False)
    return float(np.exp(info["log_meas"])), False


def split_alphas(s, info):
    """alpha_j under the split diagonal, from the stored alpha_j of the single-oracle design."""
    st, a = s.st, s.gen.a
    uni, multi, const = R.diagonal_groups(st)
    a0max = float(a.sum(axis=0).max())
    gsum = sum(float(sum(a[k] for k in range(st.K) if st.deps[k] == {i}).max()) for i in uni)
    msum = sum(float(a[k].max()) for k in multi)
    C = float(sum(a[k][0] for k in const)) if const else 0.0
    out = R.needs_out_term(st)
    al, cw = np.asarray(info["alphas"], float), np.asarray(info["cw"], float)
    return al - (a0max + cw) + gsum + msum + np.abs(C + cw) + (np.abs(cw) + a0max if out else 0.0)


def stage_split_design(only=None):
    rows_i = block_encoding_rows(diag="split")
    inv_ok = all(r["I8"] <= 1e-10 and r["M_equals_Btilde_on_S"] <= 1e-12 and r["I9"] and r["permutations_ok"]
                 and r["lognorm_M"] <= 1e-9 for r in rows_i)
    cmp = {(r["model"], r["observable"]): r for r in data(PART, "comparison")["table"]}
    rows = []
    for name in COST_MODELS:
        s = setup(name)
        c = data(PART, name)
        for enc, rt in ENCS.items():
            info = c["encodings"][enc]
            als = split_alphas(s, info)
            dts = np.diff(np.asarray(info["edges"], float))
            for obs in ("one", "tail"):
                per = c["costs"][enc][obs]
                kname = per["best_headline"]["kernel"]
                prep, _ = prep_costs(s, rt, {"theta": per["theta"], "species": per["species"]})
                for g in ("W", "cert", "meas"):
                    gam, inc = growth_inputs(info, g)
                    if gam is None:
                        continue
                    rc = R.weighted_readout(s.st, rt, gam, per["R_f"], EPS, DELTA_FAIL, als, dts, info["Dmax"],
                                            KERNELS[kname], "single", prep, include_s=inc, diag="split")
                    dec = headline(c, enc, obs, g)
                    ssa = cmp[(name, obs)]["ssa_total"]
                    rows.append({"model": name, "encoding": enc, "observable": obs, "growth": g, "kernel": kname,
                                 "b": rc.lchs.b, "query_split": rc.lchs.query, "query_single": dec["query"],
                                 "walk_steps": rc.lchs.walk_steps, "N_U": rc.N_U, "M": rc.M, "n": rc.n,
                                 "qubits_formula": rc.qubits, "alpha_max_split": float(als.max()),
                                 "total_split": rc.toffoli_total, "total_single": num(dec["toffoli_total"]),
                                 "ratio_split_to_ssa": rc.toffoli_total / ssa,
                                 "ratio_best_to_ssa": min(rc.toffoli_total, num(dec["toffoli_total"])) / ssa})
    verdict = {}
    for enc in ENCS:
        for obs in ("one", "tail"):
            v = [r["ratio_best_to_ssa"] for r in rows if r["encoding"] == enc and r["observable"] == obs
                 and r["growth"] == "W" and r["model"] in BENCHMARKS]
            verdict[f"{enc}|{obs}"] = {"min_ratio_best_design": min(v), "all_ratios_above_100": all(x > 100 for x in v)}
    save(PART, "split_design", cost_settings(), {"invariants": {"rows": rows_i, "pass": inv_ok}, "rows": rows,
                                                  "cheaper_design": verdict})


# ----------------------------------------------------------------------------- observable-aware SSA
def truncated_law_N(gen, T, p0, f, target, bias, N_max=200000):
    """Smallest N with |E_N f - target| <= bias, E_N the law of the uniformized chain stopped after N jumps."""
    import scipy.sparse as sp
    from scipy.stats import poisson
    Lam = BL.uniformization_rate(gen)
    P = sp.identity(gen.A.shape[0], format="csr") + gen.A / Lam
    lamT = Lam * T
    v, acc, cdf = p0.copy(), 0.0, 0.0
    for N in range(0, N_max + 1):
        fv = float(f @ v)
        est = acc + (1.0 - cdf) * fv
        if abs(est - target) <= bias:
            return N, abs(est - target)
        pn = poisson.pmf(N, lamT)
        acc += pn * fv
        cdf += pn
        v = P @ v
    return None, None


def stage_observable_aware_ssa(only=None):
    rows = []
    for name in COST_MODELS:
        s = setup(name)
        pT = fsp.propagate(s.gen.A, s.p0, [0.0, s.T])[-1]
        b = data(PART, f"{name}_baselines")
        for oname, o in observables(s, pT).items():
            P = float(o["f"] @ pT)
            N, err = truncated_law_N(s.gen, s.T, s.p0, o["f"], P, EPS * P / 4)
            f_cost = 0 if o["species"] is None else 2 * R.cmp_const(s.st.bits[o["species"]])
            cst = BL.coherent_ssa_cost(s.st, s.gen, s.T, P, EPS, DELTA_FAIL, f_cost, "uniformization",
                                       N=max(N, 1)) if N is not None else None
            dflt = next(r for r in b["rows"] if r["observable"] == oname and r.get("method") == "uniformization")
            lind = [r["total"] for r in b["rows"] if r["observable"] == oname and r["kind"] == "lindblad" and "total" in r]
            rows.append({"model": name, "observable": oname, "P": P, "N_default": dflt["N_steps"], "N_aware": N,
                         "bias_at_N": err, "ssa_total_default": dflt["total"],
                         "ssa_total_aware": None if cst is None else cst.total,
                         "lindblad_best": min(lind) if lind else None,
                         "lindblad_over_ssa_aware": (min(lind) / cst.total) if (lind and cst) else None})
    lvs = {}
    for oname in ("one", "tail"):
        v = [r["lindblad_over_ssa_aware"] for r in rows if r["model"] in BENCHMARKS and r["observable"] == oname]
        lvs[oname] = {"ratios": v, "within_factor_10": None if None in v else all(0.1 <= x <= 10 for x in v)}
    save(PART, "observable_aware_ssa", cost_settings(), {"rows": rows, "lindblad_vs_ssa_aware": lvs})


# ----------------------------------------------------------------------------- per-use costs on larger cascades
SCALING_DS = tuple(range(1, 13))
SCALING_WIDTH = 15
SCALING_KERNEL = "K2(c=2)"
SCALING_SCHEDULE = "geometric J=20"


class CascadeStructure:
    """Every input of the per-use counts of the conversion cascade (width SCALING_WIDTH, T = 1, tail of the last
    species, R-NB reference, sqrt(mu) weight, the measured growth), computed from the network alone.

    On this cascade the R-NB reference and the law are products over species, every reaction touches at most two
    neighbouring species, and the CW bound max(r - l) is a maximum of a chain-structured sum, found by dynamic
    programming. So the window normalizations, the jump sizes, the readout of the tail and the measured growth are
    exact without the state space, whose size 15^d rules out the full pipeline beyond d = 5."""

    def __init__(self, d, width=SCALING_WIDTH):
        self.d, self.w = d, width
        self.m = models.cascade(d, width=width)
        self.k, self.c, self.g = (self.m.params[x] for x in ("k", "c", "g"))
        self.T = float(models.relaxation_time(self.m)[0])
        self.mom = reference.MomentTrajectory(self.m, self.T)
        self.x = np.arange(width, dtype=float)
        bits = R.species_bits((width,) * d)
        deps = [set()] + [{i} for i in range(d - 1)] + [{d - 1}]
        supp = [{0}] + [{i, i + 1} for i in range(d - 1)] + [{d - 1}]
        self.st = R.ModelStructure((width,) * d, bits, deps, supp, d + 1, None, width**d)
        self.Lambda = self.k + (d - 1) * self.c * (width - 1) + self.g * (width - 1)

    def marginals(self, t):
        """Per species (log mu_i, dlog mu_i) of R-NB at time t, on 0..width-1 (the branch logic of reference.Product)."""
        m, md, v, vd = self.mom.at(t)
        out = []
        for i in range(self.d):
            lam = max(m[i], reference.SAFETY_FLOOR)
            lamd = md[i] if m[i] > reference.SAFETY_FLOOR else 0.0
            if v[i] > lam:
                e = v[i] / lam - 1
                ed = (vd[i] * lam - v[i] * lamd) / lam**2
                out.append(reference._nb_terms(self.x, lam, e, lamd, ed))
            else:
                out.append((reference._log_poisson(self.x, lam), (self.x / lam - 1) * lamd))
        return out

    def cw_sqrt_mu(self, t):
        """(1/2) max_x (r - l)(x) by dynamic programming along the chain. The reference is held fixed on each window,
        so r = (A_S mu)/mu carries no time derivative (as in certify.window_certificate)."""
        L = self.marginals(t)
        d, w, x = self.d, self.w, self.x
        top = w - 1
        U = [np.zeros(w) for _ in range(d)]
        l0 = L[0][0]
        U[0][1:] += self.k * np.exp(l0[:-1] - l0[1:])  # inflow by birth
        U[0] -= self.k  # a0: birth
        U[0][top] -= self.k  # leak: birth at the edge
        ld = L[d - 1][0]
        U[d - 1][:-1] += self.g * (x[:-1] + 1) * np.exp(ld[1:] - ld[:-1])  # inflow by death of the last species
        U[d - 1] -= self.g * x  # a0: death
        V = []
        for i in range(d - 1):
            li, lj = L[i][0], L[i + 1][0]
            Vi = np.zeros((w, w))
            # inflow by conversion i from (x_i + 1, x_j - 1): x_i <= top - 1, x_j >= 1
            Vi[:-1, 1:] += self.c * (x[:-1, None] + 1) * np.exp((li[1:] - li[:-1])[:, None] + (lj[:-1] - lj[1:])[None, :])
            Vi -= self.c * x[:, None]  # a0: conversion i
            Vi[:, top] -= self.c * x  # leak: conversion i into x_j = top
            V.append(Vi)
        f = U[0].copy()
        for i in range(d - 1):
            f = U[i + 1] + np.max(f[:, None] + V[i], axis=0)
        return 0.5 * float(np.max(f))

    def bmax(self, t):
        """sum_k max_x b_k(x), b_k = a_k(x) exp((log mu(x) - log mu(x + nu_k))/2) over x with x + nu_k in S."""
        L = [l for l, _ in self.marginals(t)]
        x = self.x
        up = [np.exp(0.5 * (l[:-1] - l[1:])) for l in L]  # x -> x + 1, x <= top - 1
        down = [x[1:] * np.exp(0.5 * (l[1:] - l[:-1])) for l in L]  # x * ratio for x -> x - 1, x >= 1
        tot = self.k * float(up[0].max())
        for i in range(self.d - 1):
            tot += self.c * float(down[i].max()) * float(up[i + 1].max())
        tot += self.g * float(down[self.d - 1].max())
        return tot

    def ptp_half(self, t1, t2):
        a, b = self.marginals(t1), self.marginals(t2)
        return 0.5 * float(sum(np.ptp(la - lb) for (la, _), (lb, _) in zip(a, b)))

    def law(self):
        """Per-species Poisson marginals of p_T (product Poisson with the rate-equation means), on the box."""
        m = self.mom.at(self.T)[0]
        return [np.exp(reference._log_poisson(self.x, mi)) for mi in m]


def cascade_per_use(d, kernel=SCALING_KERNEL, eps=EPS):
    cs = CascadeStructure(d)
    edges = certify.geometric_edges(cs.T, 20)
    J = len(edges) - 1
    cw = np.array([cs.cw_sqrt_mu(edges[j]) for j in range(J)])
    a0max = cs.Lambda
    alphas = np.array([cs.bmax(edges[j]) + a0max + cw[j] for j in range(J)])
    Dmax = np.array([cs.ptp_half(edges[j], edges[j + 1]) for j in range(J - 1)])
    lmT = [l for l, _ in cs.marginals(edges[J - 1])]
    p = cs.law()
    s_idx = d - 1
    tail_marg = np.cumsum(p[s_idx][::-1])[::-1] * float(np.prod([q.sum() for q in p[:s_idx]]))
    theta = int(np.nonzero(tail_marg <= 0.05)[0][0])
    E_f = float(tail_marg[theta])
    fmu = float(np.prod([np.exp(l).sum() for l in lmT[:s_idx]])) * float(np.exp(lmT[s_idx][theta:]).sum())
    R_f = float(np.sqrt(fmu) / E_f)
    log_meas = 0.5 * float(sum(np.log(np.sum(q**2 * np.exp(-l))) for q, l in zip(p, lmT)))
    st = cs.st
    prep = lambda b: 2 * R.product_state_prep(st.bits, b)
    rc = R.weighted_readout(st, "product", float(np.exp(log_meas)), R_f, eps, DELTA_FAIL, alphas, np.diff(edges),
                            Dmax, KERNELS[kernel], "single", prep, include_s=False)
    ssa = BL.coherent_ssa_cost(st, None, cs.T, E_f, eps, DELTA_FAIL, 2 * R.cmp_const(st.bits[s_idx]),
                               "uniformization", Lam=cs.Lambda)
    return {"d": d, "T": cs.T, "Lambda": cs.Lambda, "theta": theta, "E_f": E_f, "R_f": R_f, "log_meas": log_meas,
            "alphas": alphas, "cw": cw, "Dmax": Dmax, "sum_alpha_dt": float(np.sum(alphas * np.diff(edges))),
            "toffoli_U": rc.toffoli_U, "walk_steps": rc.lchs.walk_steps, "query": rc.lchs.query, "b": rc.lchs.b,
            "s": rc.lchs.s, "K_cut": rc.lchs.K, "eta": rc.eta, "toffoli_total": rc.toffoli_total, "N_U": rc.N_U,
            "ssa_N": ssa.N_steps, "ssa_step": ssa.step, "ssa_per_A": ssa.per_A, "ssa_b": ssa.b, "ssa_total": ssa.total,
            "ssa_uses": ssa.uses}


def pipeline_per_use(d, kernel=SCALING_KERNEL, eps=EPS):
    """The same counts from the full pipeline (state space, generator, FSP), with the same schedule, weight and
    growth: the validation of CascadeStructure for d <= 4."""
    s = setup(f"cascade{d}")
    ref = s.refs["R-NB"]
    edges = certify.geometric_edges(s.T, 20)
    cert = certify.window_certificate(s.gen, ref, edges, families=("sqrt_mu",))
    al, Dm, lms = window_alphas(s, ref, edges, cert.cw["sqrt_mu"])
    pT = fsp.propagate(s.gen.A, s.p0, [0.0, s.T])[-1]
    o = observables(s, pT)["tail"]
    E_f = float(o["f"] @ pT)
    R_f = float(np.linalg.norm(o["f"] * np.exp(0.5 * lms[-1])) / E_f)
    log_meas = log_norm_weighted(pT, lms[-1])
    prep, _ = prep_costs(s, "product", o)
    rc = R.weighted_readout(s.st, "product", float(np.exp(log_meas)), R_f, eps, DELTA_FAIL, al, np.diff(edges), Dm,
                            KERNELS[kernel], "single", prep, include_s=False)
    ssa = BL.coherent_ssa_cost(s.st, s.gen, s.T, E_f, eps, DELTA_FAIL, 2 * R.cmp_const(s.st.bits[o["species"]]),
                               "uniformization")
    return {"d": d, "widths": s.widths, "leak": s.leak, "deps": [sorted(x) for x in s.st.deps],
            "supp": [sorted(x) for x in s.st.supp], "Lambda": ssa.Lambda, "theta": o["theta"], "E_f": E_f, "R_f": R_f,
            "log_meas": log_meas, "alphas": al, "cw": cert.cw["sqrt_mu"], "Dmax": Dm, "toffoli_U": rc.toffoli_U,
            "walk_steps": rc.lchs.walk_steps, "query": rc.lchs.query, "b": rc.lchs.b, "ssa_N": ssa.N_steps,
            "ssa_step": ssa.step, "ssa_per_A": ssa.per_A}


def stage_per_use_scaling(only=None):
    """Per-use costs of the cheapest encoding (R-NB) and of coherent SSA on the conversion cascade, d = 1..12, from
    the network structure (CascadeStructure), validated against the full pipeline for d = 1..4."""
    val = []
    for d in (1, 2, 3, 4):
        a, b = cascade_per_use(d), pipeline_per_use(d)
        chk = {"d": d, "widths": b["widths"], "leak": b["leak"],
               "structure_equal": [sorted(x) for x in CascadeStructure(d).st.deps] == b["deps"]
               and [sorted(x) for x in CascadeStructure(d).st.supp] == b["supp"],
               "max_rel_alpha": float(np.max(np.abs(a["alphas"] / b["alphas"] - 1))),
               "max_abs_cw": float(np.max(np.abs(a["cw"] - b["cw"]))),
               "max_abs_Dmax": float(np.max(np.abs(a["Dmax"] - b["Dmax"]))),
               "rel_R_f": abs(a["R_f"] / b["R_f"] - 1), "abs_log_meas": abs(a["log_meas"] - b["log_meas"]),
               "same_theta": a["theta"] == b["theta"], "same_Lambda": a["Lambda"] == b["Lambda"],
               "same_walk_steps": a["walk_steps"] == b["walk_steps"], "same_query": a["query"] == b["query"],
               "same_b": a["b"] == b["b"], "rel_toffoli_U": abs(a["toffoli_U"] / b["toffoli_U"] - 1),
               "same_ssa_per_A": a["ssa_per_A"] == b["ssa_per_A"]}
        chk["pass"] = bool(chk["structure_equal"] and chk["same_theta"] and chk["same_Lambda"] and chk["same_walk_steps"]
                           and chk["same_query"] and chk["same_b"] and chk["rel_toffoli_U"] <= 1e-6
                           and chk["same_ssa_per_A"])
        val.append(chk)
        print(f"   validation d={d}: pass={chk['pass']} alpha {chk['max_rel_alpha']:.1e} cw {chk['max_abs_cw']:.1e} "
              f"U {chk['rel_toffoli_U']:.1e}", flush=True)
    rows = []
    for d in SCALING_DS:
        t0 = time.time()
        r = cascade_per_use(d)
        r["per_use_ratio"] = float(np.log10(r["toffoli_U"] / r["ssa_per_A"]))
        r["steps_ratio"] = float(np.log10(r["walk_steps"] / r["ssa_N"]))
        r["per_step_ratio"] = r["per_use_ratio"] - r["steps_ratio"]
        r["seconds"] = time.time() - t0
        rows.append(r)
        print(f"-- d={d}: walk {r['walk_steps']} x {r['toffoli_U'] / r['walk_steps']:.0f}, SSA {r['ssa_N']} x "
              f"{r['ssa_per_A'] / r['ssa_N']:.0f}; per-use ratio 10^{r['per_use_ratio']:.3f}", flush=True)
    inc = np.diff([r["per_use_ratio"] for r in rows]).tolist()
    late = [r["per_use_ratio"] for r in rows if r["d"] >= 4]
    test = {"rise_d4_to_d12": late[-1] - late[0], "increments": inc, "increments_d5_to_d12": inc[3:],
            "mean_first_four": float(np.mean(inc[3:7])), "mean_last_four": float(np.mean(inc[7:11]))}
    test["supported"] = bool(test["rise_d4_to_d12"] < 0.5 and test["mean_last_four"] < test["mean_first_four"])
    # per-reaction (marginal) costs at the largest sizes: one more species adds one reaction
    q = [r["query"] for r in rows]
    st = [r["ssa_step"] for r in rows]
    test["query_increment_last"] = q[-1] - q[-2]
    test["ssa_step_increment_last"] = st[-1] - st[-2]
    save(PART, "per_use_scaling", {"width": SCALING_WIDTH, "kernel": SCALING_KERNEL, "schedule": SCALING_SCHEDULE,
                                   "weight": "sqrt_mu", "growth": "measured", "observable": "tail", "eps": EPS,
                                   "delta_fail": DELTA_FAIL}, {"validation": val, "rows": rows, "claim_test": test})


EPS_SECOND = 1e-3


def stage_eps_comparison(only=None):
    """The comparison on the tail probability at a second precision, EPS_SECOND: every encoding (growth W, headline
    kernel and variant) under both oracle designs, coherent SSA and the Lindblad embedding (both orders)."""
    rows = []
    for name in (only or COST_MODELS):
        t0 = time.time()
        s = setup(name)
        c = data(PART, name)
        pT = fsp.propagate(s.gen.A, s.p0, [0.0, s.T])[-1]
        o = observables(s, pT)["tail"]
        P = float(o["f"] @ pT)
        f_cost = 2 * R.cmp_const(s.st.bits[o["species"]])
        ssa = BL.coherent_ssa_cost(s.st, s.gen, s.T, P, EPS_SECOND, DELTA_FAIL, f_cost, "uniformization")
        lind = []
        for order in (1, 2):
            N, err, _ = BL.lindblad_steps(s.gen, s.nus, order, s.T, s.p0, o["f"], P, EPS_SECOND * P / 4)
            if N is not None:
                lind.append(BL.lindblad_cost(s.st, s.gen, s.nus, order, s.T, N, P, EPS_SECOND, DELTA_FAIL, f_cost,
                                             err).total)
        enc = {}
        for e, rt in ENCS.items():
            info = c["encodings"][e]
            per = c["costs"][e]["tail"]
            sweep = next((r for r in per["eps_sweep"] if abs(r["eps"] - EPS_SECOND) < 1e-15), None)
            single = None if sweep is None else num(sweep["toffoli_total"])
            gam, inc = growth_inputs(info, "W")
            split = None
            if gam is not None:
                prep, _ = prep_costs(s, rt, {"theta": per["theta"], "species": per["species"]})
                rc = R.weighted_readout(s.st, rt, gam, per["R_f"], EPS_SECOND, DELTA_FAIL, split_alphas(s, info),
                                        np.diff(np.asarray(info["edges"], float)), info["Dmax"],
                                        KERNELS[per["best_headline"]["kernel"]], "single", prep, include_s=inc,
                                        diag="split")
                split = rc.toffoli_total
            enc[e] = {"single": single, "split": split,
                      "best": min(x for x in (single, split) if x is not None)}
        cheapest = min(enc, key=lambda e: enc[e]["best"])
        row = {"model": name, "P": P, "ssa_total": ssa.total, "ssa_N": ssa.N_steps, "lindblad_best": min(lind),
               "encodings": enc, "cheapest_encoding": cheapest, "cheapest_total": enc[cheapest]["best"],
               "ratio_cheapest_to_ssa": enc[cheapest]["best"] / ssa.total, "lindblad_over_ssa": min(lind) / ssa.total,
               "seconds": time.time() - t0}
        rows.append(row)
        print(f"-- {name}: cheapest {cheapest} / SSA = {row['ratio_cheapest_to_ssa']:.3g}, "
              f"Lindblad / SSA = {row['lindblad_over_ssa']:.3g} [{row['seconds']:.0f}s]", flush=True)
    if only:
        return
    test = {"ssa_cheapest_everywhere": all(r["ratio_cheapest_to_ssa"] > 1 and r["lindblad_over_ssa"] > 1 for r in rows),
            "min_ratio_encoding": min(r["ratio_cheapest_to_ssa"] for r in rows),
            "min_ratio_lindblad": min(r["lindblad_over_ssa"] for r in rows),
            "max_ratio_lindblad": max(r["lindblad_over_ssa"] for r in rows)}
    save(PART, "eps_comparison", {**cost_settings(), "eps_second": EPS_SECOND}, {"rows": rows, "claim_test": test})


SSA_BATCH = 100_000
SSA_SEED = 20261125


def stage_ssa_time(only=None):
    """Wall-clock time of classical stochastic simulation for the tail probability on the nine counted networks: a
    vectorized NumPy direct-method SSA (one process, one core), one batch of SSA_BATCH trajectories from the initial
    law. The tail estimate is checked against the FSP value (within 4 standard errors), and the time for the number of
    trajectories of Table S20 is the measured time per trajectory times that number."""
    name = "ssa_time"
    prev = data(PART, name) or {"rows": []}
    rows = [r for r in prev["rows"] if not only or r["model"] not in only]
    done = {r["model"] for r in rows}
    for mname in (only or COST_MODELS):
        if mname in done:
            continue
        s = setup(mname)
        pT = fsp.propagate(s.gen.A, s.p0, [0.0, s.T])[-1]
        o = observables(s, pT)["tail"]
        P = float(o["f"] @ pT)
        need = classical_ssa_samples(P, EPS, DELTA_FAIL)
        rx = s.model.reactions(s.widths)
        nu = np.array([r.nu for r in rx], dtype=float)
        rng = np.random.default_rng(SSA_SEED)
        p0 = s.p0 / s.p0.sum()
        X = s.space.states[rng.choice(len(s.space), size=SSA_BATCH, p=p0)].astype(float)
        t = np.zeros(SSA_BATCH)
        events = np.zeros(SSA_BATCH)
        active = np.arange(SSA_BATCH)
        t0 = time.perf_counter()
        while active.size:
            Xa = X[active]
            a = np.column_stack([np.broadcast_to(np.asarray(r.prop(Xa), float), (len(Xa),)) for r in rx])
            a0 = a.sum(axis=1)
            live = a0 > 0
            tn = t[active] + rng.exponential(size=active.size) / np.where(live, a0, 1.0)
            go = live & (tn <= s.T)
            u = rng.random(active.size) * a0
            j = np.minimum((np.cumsum(a, axis=1) < u[:, None]).sum(axis=1), len(rx) - 1)
            idx = active[go]
            X[idx] += nu[j[go]]
            t[idx] = tn[go]
            events[idx] += 1
            active = idx
        secs = time.perf_counter() - t0
        est = float(np.mean(X[:, o["species"]] >= o["theta"]))
        se = float(np.sqrt(P * (1 - P) / SSA_BATCH))
        per = secs / SSA_BATCH
        rows.append({"model": mname, "T": s.T, "P_fsp": P, "P_ssa": est, "std_error": se,
                     "within_4_se": bool(abs(est - P) <= 4 * se), "batch": SSA_BATCH, "batch_seconds": secs,
                     "seconds_per_trajectory": per, "mean_events_per_trajectory": float(events.mean()),
                     "trajectories_needed": need, "seconds_for_needed": per * need})
        print(f"-- {mname}: P_fsp={P:.4f} P_ssa={est:.4f} (se {se:.1e}) {secs:.1f}s per batch, "
              f"{per * need:.1f}s for {need:.3g} trajectories", flush=True)
        save(PART, name, {"batch": SSA_BATCH, "seed": SSA_SEED, "eps": EPS, "delta_fail": DELTA_FAIL,
                          "method": "vectorized NumPy direct method, one process"}, {"rows": rows})


STAGES = {"ssa_time": stage_ssa_time, "eps_comparison": stage_eps_comparison, "per_use_scaling": stage_per_use_scaling, "circuit_checks": stage_circuit_checks, "encodings": stage_encodings, "baselines": stage_baselines,
          "comparison": stage_comparison, "split_design": stage_split_design,
          "observable_aware_ssa": stage_observable_aware_ssa}

if __name__ == "__main__":
    args = sys.argv[1:]
    only = args[args.index("--only") + 1].split(",") if "--only" in args else None
    for a in [a for a in args if a in STAGES]:
        STAGES[a](only=only)
