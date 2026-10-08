"""Run the analysis.

    python run_all.py            figures, tables and the number check, from the stored results (about a minute)
    python run_all.py --quick    also recompute the faster calculations from scratch into outputs/recomputed/ and
                                 compare every number with the stored results (about fifteen minutes on a laptop)
    python run_all.py --full     recompute every result in order, writing to results/ (several days on a laptop;
                                 stages resume from checkpoints if interrupted)

The Qualtran cross-check (analysis.crosscheck_qualtran) and the rigorous kernel bounds (analysis.kernel_bounds) need
qualtran==0.7.0 and mpmath, and are run separately; see README.md.
"""
import json
import math
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
QUICK_OUT = ROOT / "outputs" / "recomputed"
SKIP_KEYS = {"seconds", "hours", "fsp_wall_time_s", "timestamp", "elapsed"}
RTOL, ATOL = 1e-9, 1e-12

QUICK = [  # (module, stage, only): the calculations that finish in seconds to minutes
    ("readout", "cascade", ["1", "2", "3"]), ("readout", "swap_test", None), ("readout", "concept", None),
    ("readout", "published", ["schlogl", "toggle_hegland", "toggle_kuntz", "feedback_b1", "feedback_b10",
                              "feedback_b100", "toggle_tang", "autoreg_tang_1", "autoreg_tang_2", "autoreg_tang_3"]),
    ("published_models", "references", ["schlogl", "feedback_b1", "autoreg_tang_1", "autoreg_tang_2", "autoreg_tang_3"]),
    ("published_models", "exact", ["schlogl", "feedback_b1", "autoreg_tang_1", "autoreg_tang_2", "autoreg_tang_3"]),
    ("published_models", "exact_log", ["schlogl", "feedback_b1", "autoreg_tang_1", "autoreg_tang_2", "autoreg_tang_3"]),
]  # the cost stages take 15 minutes or more per model; the test suite checks their formulas and circuits
FULL = [
    ("readout", "cascade", None), ("readout", "lambda_phage", None), ("readout", "cascade_summary", None),
    ("readout", "swap_test", None), ("readout", "concept", None),
    ("certificates", "toggle", None), ("certificates", "toggle_best", None), ("certificates", "toggle_exact", None),
    ("published_models", "references", None), ("published_models", "exact", None),
    ("published_models", "exact_log", None), ("published_models", "exact_log_uniformization", None),
    ("published_models", "overflow_bounds", None), ("published_models", "worst_case", None),
    ("published_models", "resolvent", None), ("published_models", "summary", None),
    ("readout", "published", None),
    ("costs", "circuit_checks", None), ("costs", "encodings", None), ("costs", "baselines", None),
    ("costs", "comparison", None), ("costs", "split_design", None), ("costs", "observable_aware_ssa", None),
]


def run(stages):
    import importlib
    for mod, stage, only in stages:
        t0 = time.time()
        m = importlib.import_module(f"analysis.{mod}")
        print(f"== {mod}.{stage}" + (f" ({', '.join(only)})" if only else ""), flush=True)
        fn = m.STAGES[stage]
        fn(only=only) if "only" in fn.__code__.co_varnames else fn()
        print(f"== {mod}.{stage} done [{time.time() - t0:.0f}s]", flush=True)


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k not in SKIP_KEYS:
                yield from leaves(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}/{i}")
    elif isinstance(x, bool) or x is None:
        yield path, x
    elif isinstance(x, (int, float)):
        yield path, float(x)
    elif isinstance(x, str) and x in ("inf", "-inf", "nan"):
        yield path, float(x)


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        if math.isinf(a) or math.isinf(b):
            return a == b
        return abs(a - b) <= ATOL + RTOL * max(abs(a), abs(b))
    return a == b


def compare():
    """Every number in each recomputed file against the stored file of the same name."""
    rows, ok_all = [], True
    for f in sorted(QUICK_OUT.rglob("*.json")):
        if "partial" in f.parts:
            continue
        rel = f.relative_to(QUICK_OUT)
        stored = ROOT / "results" / rel
        if not stored.exists():
            continue
        new = dict(leaves(json.loads(f.read_text(encoding="utf-8"))["data"]))
        old = dict(leaves(json.loads(stored.read_text(encoding="utf-8"))["data"]))
        shared = sorted(set(new) & set(old))
        bad = [p for p in shared if not same(new[p], old[p])]
        ok = not bad and len(shared) > 0
        ok_all &= ok
        rows.append(f"| {rel.as_posix()} | {len(shared)} | {len(bad)} | {'yes' if ok else '**NO**'} |")
        print(f"  {rel.as_posix():45s} {len(shared):6d} numbers, {len(bad)} differ" + (f" (first: {bad[0]})" if bad else ""))
    out = ROOT / "outputs" / "recomputation_check.md"
    out.write_text("# Recomputed results compared with the stored results\n\n"
                   f"Relative tolerance {RTOL:g}. Timings are not compared.\n\n"
                   "| File | Numbers compared | Differ | Agrees |\n| --- | --- | --- | --- |\n" + "\n".join(rows) + "\n",
                   encoding="utf-8")
    return ok_all


def main(argv):
    if "--quick" in argv:
        os.environ["QCME_OUT"] = str(QUICK_OUT)
    sys.path.insert(0, str(ROOT))
    if "--quick" in argv:
        run(QUICK)
        print("== comparing recomputed and stored results", flush=True)
        ok = compare()
        print("recomputation agrees with the stored results" if ok else "RECOMPUTATION DIFFERS (outputs/recomputation_check.md)")
    elif "--full" in argv:
        run(FULL)
        ok = True
    else:
        ok = True
    from analysis import check_numbers, figures, tables
    figures.main()
    tables.main()
    bad = check_numbers.main()
    return 0 if ok and not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
