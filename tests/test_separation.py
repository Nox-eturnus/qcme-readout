"""The certificate code never imports the FSP solution.

`certify`, `reference` and `lowrank` (and everything they import inside qcme) must not reach `fsp`,
`emulate` or `invariants`, so a certificate cannot be computed from the CME solution.
"""
import ast
import pathlib

QCME = pathlib.Path(__file__).resolve().parents[1] / "qcme"
FORBIDDEN = {"fsp", "emulate", "invariants"}


def qcme_imports(module):
    tree = ast.parse((QCME / f"{module}.py").read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1:  # from .x import y  /  from . import x
                if node.module:
                    found.add(node.module.split(".")[0])
                else:
                    found.update(a.name for a in node.names)
            elif node.module and node.module.startswith("qcme"):
                parts = node.module.split(".")
                found.update([parts[1]] if len(parts) > 1 else [a.name for a in node.names])
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("qcme."):
                    found.add(a.name.split(".")[1])
    return found


def closure(module):
    seen, todo = set(), [module]
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        todo.extend(qcme_imports(m))
    return seen


def test_certificate_code_cannot_see_the_fsp_solution():
    for module in ("certify", "reference", "hermite", "lowrank"):
        reached = closure(module)
        assert not (reached & FORBIDDEN), f"{module} reaches {reached & FORBIDDEN}"


def test_the_checker_detects_a_forbidden_import():
    assert "fsp" in closure("invariants")  # sanity: the walker does follow imports


def test_cost_code_cannot_see_the_fsp_solution():
    """circuits, resources, lchs and baselines never import fsp; the FSP-based growth factors (Gamma_W,
    Gamma_meas) are computed only in the analysis scripts."""
    for module in ("circuits", "resources", "lchs", "baselines"):
        reached = closure(module)
        assert not (reached & FORBIDDEN), f"{module} reaches {reached & FORBIDDEN}"
