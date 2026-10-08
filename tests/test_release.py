"""Release consistency: one version everywhere, and every figure and number check runs from the stored results."""
import pathlib
import tomllib

import qcme

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_version_is_consistent():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert meta["project"]["version"] == qcme.__version__
    assert f"version: {qcme.__version__}" in (ROOT / "CITATION.cff").read_text(encoding="utf-8")


def test_requirements_pin_the_dependencies():
    pins = dict(line.strip().split("==") for line in (ROOT / "requirements.txt").read_text().splitlines() if "==" in line)
    assert {"numpy", "scipy", "matplotlib", "pytest"} <= set(pins)


def test_every_quoted_number_agrees_with_the_stored_results():
    from analysis import check_numbers
    assert check_numbers.main() == 0
