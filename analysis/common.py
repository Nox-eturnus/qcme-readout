"""Shared settings and helpers: result files, schedules, model setups and certificate tables.

Every result file is JSON with four fields: the name, the settings it was computed with, the software environment,
and the data. Results are grouped by task under results/<part>/.
"""
import json
import os
import pathlib
import platform
import sys
import time

import numpy as np
import scipy
from scipy.special import logsumexp

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qcme import certify, emulate, fsp, invariants as inv, models, network, reference  # noqa: E402

RESULTS = ROOT / "results"  # stored results (inputs of every stage)
# where stages write: results/ by default; run_all.py --quick points this elsewhere to compare with the stored files
OUT = pathlib.Path(os.environ.get("QCME_OUT", RESULTS))

# ----------------------------------------------------------------------------- settings
LEAK_TOL = 1e-6  # finite state projection: leaked probability at most this over the horizon
CEILING = 10_000_000  # largest box (states)
LARGE_BOX = 200_000  # above this many states (or LARGE_NNZ generator entries) the resolvent weight is applied
LARGE_NNZ = 1_000_000  # only to the best sqrt(mu) schedule
HUGE_BOX = 5_000_000  # above this size, or HUGE_WORK, only the sqrt(mu) weight is used and W is not iterated
HUGE_WORK = 1e10  # n * Lambda * T for one FSP solve to the horizon
STREAM_BOX = 1_000_000  # the exact reference streams p_t instead of caching every time
N_CONT = 129  # time points of the continuous estimate
SUB = 5
FAMILIES = certify.FAMILIES
TAIL_LEVEL = 0.05  # tail observable: smallest threshold with tail probability <= 5%
TAIL_GROW = 1.5  # tail term: every unbounded width grown by this factor


def schedules(T):
    """The six window schedules: uniform with J = 5, 10, 20, 40, 80, and geometric with J = 20."""
    out = {f"uniform J={J}": certify.uniform_edges(T, J) for J in (5, 10, 20, 40, 80)}
    out["geometric J=20"] = certify.geometric_edges(T, 20, 2.0**-10)
    return out


def settings():
    return {"leak_tol": LEAK_TOL, "ceiling": CEILING, "large_box": LARGE_BOX, "large_nnz": LARGE_NNZ,
            "huge_box": HUGE_BOX, "huge_work": HUGE_WORK, "stream_box": STREAM_BOX, "n_cont": N_CONT, "sub": SUB,
            "families": FAMILIES, "eta": reference.ETA, "mean_floor": reference.MEAN_FLOOR, "tail_level": TAIL_LEVEL,
            "tail_grow": TAIL_GROW, "schedules": schedules(1.0)}


# ----------------------------------------------------------------------------- result files
def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else str(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, set):
        return sorted(x)
    return x


def environment():
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def save(part, name, settings_, data, root=None):
    d = (root or OUT) / part
    d.mkdir(parents=True, exist_ok=True)
    out = {"name": name, "settings": jsonable(settings_), "environment": environment(), "data": jsonable(data)}
    (d / f"{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {part}/{name}.json", flush=True)


def load(part, name, root=None):
    f = (root or RESULTS) / part / f"{name}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def data(part, name, root=None):
    r = load(part, name, root)
    return None if r is None else r["data"]


def _decode(x):
    if isinstance(x, dict):
        return {k: _decode(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_decode(v) for v in x]
    if isinstance(x, str) and x in ("inf", "-inf", "nan"):
        return float(x)
    return x


def checkpoint_save(part, label, data_):
    d = OUT / part / "partial"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{label}.json").write_text(json.dumps(jsonable(data_), indent=1), encoding="utf-8")


def checkpoint_load(part, label):
    f = OUT / part / "partial" / f"{label}.json"
    return _decode(json.loads(f.read_text(encoding="utf-8"))) if f.exists() else None


def num(x):
    return float(x.replace("inf", "Infinity")) if isinstance(x, str) else float(x)


def log10(x):
    return float(x) / np.log(10)


def log_norm_weighted(p, lm):
    """log ||p / sqrt(mu)|| without overflow (p may contain zeros)."""
    with np.errstate(divide="ignore"):
        return 0.5 * float(logsumexp(2 * np.log(np.abs(p)) - lm))


# ----------------------------------------------------------------------------- model setups
def horizon(m):
    """One relaxation time, unless the box it needs would exceed the ceiling; then the source's horizon. The box
    at one relaxation time is estimated from the rate-equation means (width >= m + 10 sqrt(m) + 10)."""
    T_rel, _ = models.relaxation_time(m)
    if "T_source" not in m.params:
        return T_rel, {"rule": "one relaxation time", "T_relaxation": T_rel}
    mom = reference.MomentTrajectory(m, T_rel)
    mT = np.maximum(mom.at(T_rel)[0], 0.0)
    est = np.where(m.binary, 2, np.ceil(mT + 10 * np.sqrt(mT) + 10))
    n_est = float(np.prod(est))
    if n_est <= CEILING:
        return T_rel, {"rule": "one relaxation time", "T_relaxation": T_rel, "box_estimate": est, "n_estimate": n_est}
    return float(m.params["T_source"]), {"rule": "source horizon (the box at one relaxation time exceeds 10^7 states)",
                                         "T_relaxation": T_rel, "box_estimate": est, "n_estimate": n_est}


def build_setup(m, widths, T, refs="published"):
    space = m.space(widths)
    gen = network.generator(space, m.reactions(widths))
    p0 = reference.initial_law(m, space)
    rf = reference.published_references(m, space, T, widths) if refs == "published" else {}
    return inv.Setup(m, tuple(int(w) for w in widths), T, space, gen, p0, rf)


def capped_box(m, T, part="published_models"):
    """The box from fsp.select_box (leak <= LEAK_TOL). If its growth schedule leaves the ceiling first, the largest
    box of the same schedule within the ceiling, with its leak reported. A stored box record is reused."""
    rec = data(part, f"{m.name}_box")
    if rec and abs(rec["T"] - T) <= 1e-12 * T and tuple(rec["start_widths"]) == tuple(m.widths):
        return tuple(rec["widths"]), float(rec["leak"]), bool(rec["capped"])
    try:
        w, leak = fsp.select_box(m, lambda sp_: reference.initial_law(m, sp_), T, LEAK_TOL, max_states=CEILING)
        return tuple(int(v) for v in w), leak, False
    except RuntimeError:
        fixed = np.array(m.binary) | fsp._bounded_by_conservation(m)
        w = np.array(m.widths, dtype=int)
        while True:
            nxt = np.where(fixed, w, np.ceil(w * 1.25).astype(int))
            if len(m.space(tuple(nxt))) > CEILING:
                break
            w = nxt
        w = tuple(int(v) for v in w)
        space = m.space(w)
        gen = network.generator(space, m.reactions(w))
        pT = fsp.propagate(gen.A, reference.initial_law(m, space), [T])[-1]
        leak = 1.0 - fsp.retained_mass(pT)
        save(part, f"{m.name}_box", {**settings(), "model": m.name},
             {"T": T, "start_widths": list(m.widths), "leak_tol": LEAK_TOL, "ceiling": CEILING, "widths": list(w),
              "leak": leak, "capped": True, "trace": "box growth schedule of fsp.select_box; last box within 10^7 states"})
        return w, leak, True


def published_setup(m, refs="published"):
    T, hinfo = horizon(m)
    widths, leak, capped = capped_box(m, T)
    hinfo = {**hinfo, "box_capped_at_ceiling": capped}
    return build_setup(m, widths, T, refs), leak, hinfo


def published_model(name):
    return next(m for m in models.published() if m.name == name)


def observables(s, pT):
    """f = 1 and the tail indicator of the last count species at the smallest threshold whose tail probability
    is at most TAIL_LEVEL."""
    X = s.space.states
    i = [k for k, b in enumerate(s.model.binary) if not b][-1]
    marg = np.bincount(X[:, i], weights=pT, minlength=s.widths[i])
    tail = np.cumsum(marg[::-1])[::-1]
    theta = int(np.nonzero(tail <= TAIL_LEVEL)[0][0])
    return {"one": {"f": np.ones(len(X)), "species": None, "theta": None},
            "tail": {"f": (X[:, i] >= theta).astype(float), "species": i, "theta": theta}}


# ----------------------------------------------------------------------------- certificate tables
def is_large(s):
    return len(s.space) > LARGE_BOX or s.gen.A.nnz > LARGE_NNZ


def is_huge(s):
    n = len(s.space)
    lam = float(-s.gen.A.diagonal().min())
    return n > HUGE_BOX or n * lam * s.T > HUGE_WORK


def certificate_table(s, ref, sched, large):
    """Piecewise K_T per schedule and weight family. On large boxes the resolvent weight is evaluated only on the
    schedule with the smallest sqrt(mu) K_T; on very large boxes only the sqrt(mu) weight is used."""
    out = {}
    if is_huge(s):
        for sn, edges in sched.items():
            out[sn] = certify.window_certificate(s.gen, ref, edges, families=("sqrt_mu",))
        return out
    if not large:
        for sn, edges in sched.items():
            out[sn] = certify.window_certificate(s.gen, ref, edges)
        return out
    for sn, edges in sched.items():
        out[sn] = certify.window_certificate(s.gen, ref, edges, families=("sqrt_mu",))
    pick = min(out, key=lambda sn: out[sn].K["sqrt_mu"])
    out[pick] = certify.window_certificate(s.gen, ref, sched[pick])
    return out


def best_of(table):
    K, sn, f = min((c.K[f], sn, f) for sn, c in table.items() for f in c.K)
    return K, sn, f


def inf_if_nan(tab):
    """Where p_t is exactly 0 at two consecutive window edges, the jump is (-inf) - (-inf) = NaN; the CW of those
    windows is already +inf, so the valid bound is +inf."""
    for c in tab.values():
        for f in c.K:
            if np.isnan(c.K[f]):
                c.K[f] = np.inf
    return tab


def worst_case(s, lm0, lmT):
    """log W, the largest growth of the weighted state over the horizon (not iterated on very large boxes)."""
    if is_huge(s):
        return None
    if np.all(np.isfinite(lm0)) and np.all(np.isfinite(lmT)) and np.ptp(lm0) < 1400 and np.ptp(lmT) < 1400:
        return float(np.log(emulate.horizon_worst_case(s.gen.A, lm0, lmT, s.T)))
    return None
