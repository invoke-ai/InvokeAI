from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_pins = _load_module(REPO_ROOT / "scripts" / "check_pins.py", "check_pins")


@pytest.fixture
def repo_copy(tmp_path: Path) -> Path:
    """A throwaway directory holding copies of the real pins.json and pyproject.toml."""
    for name in ("pins.json", "pyproject.toml"):
        shutil.copyfile(REPO_ROOT / name, tmp_path / name)
    return tmp_path


def _read_pins(repo: Path) -> dict:
    return json.loads((repo / "pins.json").read_text())


def _write_pins(repo: Path, pins: dict) -> None:
    (repo / "pins.json").write_text(json.dumps(pins, indent=2))


def _set_requires_python(repo: Path, specifier: str) -> None:
    path = repo / "pyproject.toml"
    updated, count = re.subn(r"(?m)^requires-python = .*$", f'requires-python = "{specifier}"', path.read_text())
    assert count == 1, "expected exactly one top-level requires-python line in pyproject.toml"
    path.write_text(updated)


# Every (platform, backend) the checker must insist on, derived from the checker's own matrix so
# the two cannot drift apart.
REQUIRED_ENTRIES = [
    (platform, backend)
    for platform, backends in sorted(check_pins.REQUIRED_BACKENDS.items())
    for backend in sorted(backends)
]


def test_required_matrix_covers_cpu_cuda_and_rocm():
    """Guard against the matrix being narrowed to the point where it checks nothing."""
    assert ("linux", "rocm") in REQUIRED_ENTRIES
    assert ("linux", "cuda") in REQUIRED_ENTRIES
    assert ("linux", "cpu") in REQUIRED_ENTRIES
    assert ("win32", "cuda") in REQUIRED_ENTRIES
    assert ("win32", "cpu") in REQUIRED_ENTRIES
    # macOS uses MPS, which has no dedicated torch index.
    assert check_pins.REQUIRED_BACKENDS["darwin"] == set()


def test_repo_pins_are_consistent(repo_copy: Path):
    """The checked-in pins.json and pyproject.toml agree."""
    assert check_pins.main(repo_copy) == 0


@pytest.mark.parametrize(("platform", "backend"), REQUIRED_ENTRIES)
def test_missing_entry_fails(repo_copy: Path, platform: str, backend: str):
    """Removing any required entry must fail, not silently pass."""
    pins = _read_pins(repo_copy)
    del pins["torchIndexUrl"][platform][backend]
    _write_pins(repo_copy, pins)

    assert check_pins.main(repo_copy) == 1


@pytest.mark.parametrize("platform", sorted(check_pins.REQUIRED_BACKENDS))
def test_missing_platform_section_fails(repo_copy: Path, platform: str):
    """Removing a whole platform section must fail, even for platforms with no required backends."""
    pins = _read_pins(repo_copy)
    del pins["torchIndexUrl"][platform]
    _write_pins(repo_copy, pins)

    assert check_pins.main(repo_copy) == 1


@pytest.mark.parametrize(("platform", "backend"), REQUIRED_ENTRIES)
def test_stale_url_fails(repo_copy: Path, platform: str, backend: str):
    """The original drift case: an entry that exists but points at the wrong index."""
    pins = _read_pins(repo_copy)
    pins["torchIndexUrl"][platform][backend] = "https://download.pytorch.org/whl/stale"
    _write_pins(repo_copy, pins)

    assert check_pins.main(repo_copy) == 1


def test_rocm_url_regression_is_reported(repo_copy: Path, capsys: pytest.CaptureFixture[str]):
    """Reverting the ROCm index to 6.3 (issue #9328) is caught and named."""
    pins = _read_pins(repo_copy)
    pins["torchIndexUrl"]["linux"]["rocm"] = "https://download.pytorch.org/whl/rocm6.3"
    _write_pins(repo_copy, pins)

    assert check_pins.main(repo_copy) == 1
    assert "torchIndexUrl.linux.rocm" in capsys.readouterr().err


def test_unsupported_backend_fails(repo_copy: Path):
    """ROCm on Windows has no wheels; pinning an index for it must be rejected."""
    pins = _read_pins(repo_copy)
    pins["torchIndexUrl"]["win32"]["rocm"] = "https://download.pytorch.org/whl/rocm7.1"
    _write_pins(repo_copy, pins)

    assert check_pins.main(repo_copy) == 1


def test_unknown_platform_fails(repo_copy: Path):
    pins = _read_pins(repo_copy)
    pins["torchIndexUrl"]["freebsd"] = {"cpu": "https://download.pytorch.org/whl/cpu"}
    _write_pins(repo_copy, pins)

    assert check_pins.main(repo_copy) == 1


def test_missing_pyproject_index_fails(repo_copy: Path):
    """Dropping a [[tool.uv.index]] entry that pins.json references must fail."""
    pyproject = (repo_copy / "pyproject.toml").read_text()
    pyproject = pyproject.replace('name = "torch-rocm"', 'name = "torch-rocm-renamed"')
    (repo_copy / "pyproject.toml").write_text(pyproject)

    assert check_pins.main(repo_copy) == 1


def test_missing_python_fails(repo_copy: Path, capsys: pytest.CaptureFixture[str]):
    """The launcher builds the venv from this field; it cannot be absent."""
    pins = _read_pins(repo_copy)
    del pins["python"]
    _write_pins(repo_copy, pins)

    assert check_pins.main(repo_copy) == 1
    assert "'python'" in capsys.readouterr().err


def test_requires_python_moving_off_the_pin_fails(repo_copy: Path, capsys: pytest.CaptureFixture[str]):
    """Bumping requires-python without bumping pins.json: the launcher would build a python
    3.12 venv, then install metadata demanding 3.13, and the install fails at the last step."""
    _set_requires_python(repo_copy, ">=3.13,<3.14")

    assert check_pins.main(repo_copy) == 1
    # Not just "requires-python" - that word also appears in the advice footer printed on
    # every failure, so it would be satisfied by any unrelated error.
    assert "pins.json python is '3.12'" in capsys.readouterr().err


@pytest.mark.parametrize("pinned", ["3.10", "3.13", "3.9", "4.0"])
def test_python_outside_requires_python_fails(repo_copy: Path, pinned: str):
    """The other direction: pins.json moved to a version the metadata rejects."""
    pins = _read_pins(repo_copy)
    pins["python"] = pinned
    _write_pins(repo_copy, pins)

    assert check_pins.main(repo_copy) == 1


@pytest.mark.parametrize(
    "pinned",
    [
        "3.12.x",
        "",
        "python3.12",
        "3.12\n",
        "3",  # `uv venv --python 3` is not a version the launcher should be pinning
        "3.12.0.0.0.0",  # satisfies every specifier; uv finds no such interpreter
        "3.12.99999999",  # ditto
        "3.12.9999",  # ditto, and short enough to look plausible
        "3.12.0",  # patch pins are rejected outright, even valid-looking ones
        "3.12.7",  # ditto: uv already resolves "3.12" to the newest patch
        "03.12",  # normalizes to 3.12 here but reaches the launcher verbatim
        3.12,
        ["3.12"],
        {"major": 3, "minor": 12},
    ],
)
def test_non_version_python_fails(repo_copy: Path, pinned: object):
    pins = _read_pins(repo_copy)
    pins["python"] = pinned
    _write_pins(repo_copy, pins)

    assert check_pins.main(repo_copy) == 1


@pytest.mark.parametrize("pinned", ["3.12.7", "3.12.9999"])
def test_patch_pin_is_rejected_for_being_a_patch_pin(repo_copy: Path, capsys: pytest.CaptureFixture[str], pinned: str):
    """Assert the reason, not just the exit code. The classifier check rejects these too, so
    without this a loosened version pattern would leave the suite green while telling the user
    to add a classifier for '3.12.7' - advice that makes no sense."""
    pins = _read_pins(repo_copy)
    pins["python"] = pinned
    _write_pins(repo_copy, pins)

    assert check_pins.main(repo_copy) == 1
    stderr = capsys.readouterr().err
    assert "a patch component is deliberately not accepted" in stderr
    # Not the bare word "classifiers" - that appears in the advice footer printed on every failure.
    assert "classifiers declare support only for" not in stderr


def _drop_python_classifiers(repo: Path) -> None:
    path = repo / "pyproject.toml"
    updated, count = re.subn(r"(?m)^\s*'Programming Language :: Python :: [0-9]+\.[0-9]+',$\n", "", path.read_text())
    assert count, "expected at least one 'Programming Language :: Python :: X.Y' classifier"
    path.write_text(updated)


def test_python_not_in_classifiers_fails(repo_copy: Path, capsys: pytest.CaptureFixture[str]):
    """requires-python allows 3.11, but the project only ships classifiers for 3.12. The pin
    has to name a version this project actually claims to support."""
    pins = _read_pins(repo_copy)
    pins["python"] = "3.11"
    _write_pins(repo_copy, pins)

    assert check_pins.main(repo_copy) == 1
    assert "classifiers declare support only for" in capsys.readouterr().err


def test_unreal_version_inside_an_open_ended_requires_python_fails(repo_copy: Path):
    """The hole the classifier check exists to close: with no upper bound in requires-python,
    a version that no interpreter has would otherwise satisfy every clause."""
    _set_requires_python(repo_copy, ">=3.11")
    pins = _read_pins(repo_copy)
    pins["python"] = "3.99"
    _write_pins(repo_copy, pins)

    assert check_pins.main(repo_copy) == 1


def test_missing_python_classifiers_fails(repo_copy: Path, capsys: pytest.CaptureFixture[str]):
    """Deleting the classifiers must not silently downgrade the pin check to requires-python only."""
    _drop_python_classifiers(repo_copy)

    assert check_pins.main(repo_copy) == 1
    assert "no 'Programming Language :: Python :: X.Y' classifier" in capsys.readouterr().err


@pytest.mark.parametrize(
    "classifier",
    [
        "Programming Language :: Python :: 3 :: Only",  # no major.minor at all
        "Programming Language :: Python :: 3",  # ditto
        "Programming Language :: Python :: 3.12 :: Only",  # not a real classifier; declares nothing
        "Programming Language :: Python :: 3.1.4",  # ditto
        "Programming Language :: Python :: 3.12 ",  # trailing space
        "Programming Language :: Python :: Implementation :: CPython",
    ],
)
def test_classifier_is_not_read_as_a_version(repo_copy: Path, classifier: str):
    """Only a whole classifier naming exactly one major.minor version counts. Anything that
    merely contains one must not be read as a declaration of support for it."""
    _drop_python_classifiers(repo_copy)
    pyproject = tomllib.loads((repo_copy / "pyproject.toml").read_text())
    pyproject["project"]["classifiers"].append(classifier)

    assert check_pins._classifier_versions(pyproject) == []


def test_duplicate_classifiers_are_reported_once(repo_copy: Path):
    pyproject = tomllib.loads((repo_copy / "pyproject.toml").read_text())
    pyproject["project"]["classifiers"].append("Programming Language :: Python :: 3.12")

    assert check_pins._classifier_versions(pyproject) == ["3.12"]


@pytest.mark.parametrize("literal", ["3", "true", '"3.12"', "{}", "[3.12]", "[[]]", "[]"])
def test_malformed_classifiers_do_not_raise(repo_copy: Path, literal: str):
    """A non-list, or a list of non-strings, must be reported rather than blowing up - the
    same tolerance requires-python already gets."""
    path = repo_copy / "pyproject.toml"
    path.write_text(re.sub(r"(?ms)^classifiers = \[.*?^\]$", f"classifiers = {literal}", path.read_text()))
    assert "classifiers = [\n" not in (repo_copy / "pyproject.toml").read_text()

    assert check_pins.main(repo_copy) == 1


def test_missing_project_table_does_not_raise(repo_copy: Path):
    """No [project] table at all: both the requires-python and the classifier lookups have to
    survive it, because a raise here would skip the torch index checks entirely."""
    path = repo_copy / "pyproject.toml"
    pyproject = tomllib.loads(path.read_text())
    del pyproject["project"]
    assert check_pins._classifier_versions(pyproject) == []
    assert check_pins._requires_python_clauses(pyproject)[0] == []

    errors = check_pins.check_python(_read_pins(repo_copy), pyproject)
    assert errors  # reported, not raised


def test_absurdly_long_classifier_version_does_not_mask_url_drift(repo_copy: Path, capsys: pytest.CaptureFixture[str]):
    """int() refuses to parse a string of more than 4300 digits. An unbounded classifier
    pattern would let one reach _parse_version and raise straight out of check_python,
    taking the torchIndexUrl checks with it."""
    path = repo_copy / "pyproject.toml"
    absurd = "Programming Language :: Python :: " + "9" * 4400 + ".0"
    path.write_text(path.read_text().replace("classifiers = [\n", f"classifiers = [\n  '{absurd}',\n", 1))
    pins = _read_pins(repo_copy)
    pins["torchIndexUrl"]["linux"]["rocm"] = "https://download.pytorch.org/whl/rocm6.3"
    _write_pins(repo_copy, pins)

    assert check_pins.main(repo_copy) == 1
    assert "torchIndexUrl.linux.rocm" in capsys.readouterr().err


def test_dynamic_classifiers_are_reported_as_such(repo_copy: Path, capsys: pytest.CaptureFixture[str]):
    """PEP 621 lets the build backend supply classifiers. The checker can't read them, and must
    say so rather than claiming none are declared."""
    _drop_python_classifiers(repo_copy)
    path = repo_copy / "pyproject.toml"
    path.write_text(re.sub(r'(?m)^dynamic = \["version"\]$', 'dynamic = ["version", "classifiers"]', path.read_text()))

    assert check_pins.main(repo_copy) == 1
    assert "project.dynamic" in capsys.readouterr().err


@pytest.mark.parametrize("value", [None, "3.12", ["3.12"], 5])
def test_non_object_pins_does_not_raise(repo_copy: Path, value: object):
    """The top level of pins.json is indexed into everywhere; a wrong shape must be reported."""
    (repo_copy / "pins.json").write_text(json.dumps(value))

    assert check_pins.main(repo_copy) == 1


def test_unevaluatable_requires_python_fails(repo_copy: Path):
    """A specifier form the checker cannot reason about must fail loudly rather than pass."""
    _set_requires_python(repo_copy, "~=3.12")

    assert check_pins.main(repo_copy) == 1


def test_missing_requires_python_fails(repo_copy: Path, capsys: pytest.CaptureFixture[str]):
    """requires-python is optional in PEP 621, so its absence must be reported, not raise."""
    path = repo_copy / "pyproject.toml"
    path.write_text(re.sub(r"(?m)^requires-python = .*$\n", "", path.read_text()))

    assert check_pins.main(repo_copy) == 1
    assert "no project.requires-python" in capsys.readouterr().err


@pytest.mark.parametrize("literal", ["3.12", "3", "true", '["<3.13"]', '""'])
def test_non_string_requires_python_fails(repo_copy: Path, literal: str):
    """An unquoted version is an easy TOML typo; it must not crash the checker."""
    path = repo_copy / "pyproject.toml"
    path.write_text(re.sub(r"(?m)^requires-python = .*$", f"requires-python = {literal}", path.read_text()))

    assert check_pins.main(repo_copy) == 1


def test_trailing_comma_in_requires_python_is_accepted(repo_copy: Path):
    """A trailing comma is legal and must not be reported as an unevaluatable clause."""
    _set_requires_python(repo_copy, ">=3.11, <3.13,")

    assert check_pins.main(repo_copy) == 0


def test_python_check_failure_does_not_mask_url_drift(repo_copy: Path, capsys: pytest.CaptureFixture[str]):
    """The python check runs first; a problem there must not stop the torch index checks -
    otherwise a bad requires-python would hide the very drift this script exists to catch."""
    path = repo_copy / "pyproject.toml"
    path.write_text(re.sub(r"(?m)^requires-python = .*$\n", "", path.read_text()))
    pins = _read_pins(repo_copy)
    pins["torchIndexUrl"]["linux"]["rocm"] = "https://download.pytorch.org/whl/rocm6.3"
    _write_pins(repo_copy, pins)

    assert check_pins.main(repo_copy) == 1
    stderr = capsys.readouterr().err
    assert "no project.requires-python" in stderr
    assert "torchIndexUrl.linux.rocm" in stderr


@pytest.mark.parametrize("platform", sorted(check_pins.REQUIRED_BACKENDS))
def test_null_platform_section_fails(repo_copy: Path, platform: str):
    """A null section is not the same as an absent one: `.get()` returns None for both,
    so a present-but-null platform used to pass silently."""
    pins = _read_pins(repo_copy)
    pins["torchIndexUrl"][platform] = None
    _write_pins(repo_copy, pins)

    assert check_pins.main(repo_copy) == 1


@pytest.mark.parametrize("value", [None, "https://download.pytorch.org/whl/cpu", ["cpu"], 5])
def test_non_object_platform_section_fails(repo_copy: Path, value: object):
    pins = _read_pins(repo_copy)
    pins["torchIndexUrl"]["linux"] = value
    _write_pins(repo_copy, pins)

    assert check_pins.main(repo_copy) == 1


@pytest.mark.parametrize("value", [None, "https://download.pytorch.org/whl/cpu", [], 5])
def test_non_object_torch_index_url_fails(repo_copy: Path, value: object):
    pins = _read_pins(repo_copy)
    pins["torchIndexUrl"] = value
    _write_pins(repo_copy, pins)

    assert check_pins.main(repo_copy) == 1


@pytest.mark.parametrize(
    ("version", "operator", "bound", "expected"),
    [
        # Zero-padding: "3.12" and "3.12.0" are the same version.
        ("3.12", ">=", "3.12.0", True),
        ("3.12", "<", "3.12.0", False),
        ("3.12", "==", "3.12.0", True),
        ("3.12.0", ">=", "3.12", True),
        # The repo's own bounds.
        ("3.12", ">=", "3.11", True),
        ("3.12", "<", "3.13", True),
        ("3.13", "<", "3.13", False),
        ("3.10", ">=", "3.11", False),
        # Component-wise, not lexicographic - "3.9" is not above "3.10".
        ("3.9", "<", "3.10", True),
        ("3.9", ">", "3.10", False),
        ("3.12.7", ">", "3.12", True),
        # Each operator needs at least one row where its answer differs from every other
        # operator's, or a deleted branch can inherit a neighbour's semantics unnoticed.
        ("3.12", "!=", "3.13", True),
        ("3.12", "!=", "3.11", True),
        ("3.12", "!=", "3.12", False),
        ("3.12", "<=", "3.12", True),
        ("3.11", "<=", "3.12", True),
        ("3.13", "<=", "3.12", False),
        ("3.11", "==", "3.12", False),
        ("3.12", ">=", "3.13", False),
    ],
)
def test_version_comparison(version: str, operator: str, bound: str, expected: bool):
    assert (
        check_pins._satisfies(check_pins._parse_version(version), operator, check_pins._parse_version(bound))
        is expected
    )


@pytest.mark.parametrize("operator", ["~=", "===", "", "foo"])
def test_unknown_operator_raises(operator: str):
    """An operator with no branch must not quietly fall through to some other comparison."""
    with pytest.raises(ValueError):
        check_pins._satisfies((3, 12), operator, (3, 13))


def test_script_runs_from_any_working_directory(tmp_path: Path):
    """The repo root comes from the script's own path, so cwd is irrelevant - which is what the
    module docstring promises, and what CI relies on."""
    script = REPO_ROOT / "scripts" / "check_pins.py"
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "consistent" in result.stdout


def test_every_supported_operator_is_covered():
    """Guard against an operator being added to the clause pattern but never exercised."""
    covered = {operator for _, operator, _, _ in test_version_comparison.pytestmark[0].args[1]}
    assert covered == {"==", "!=", ">=", "<=", ">", "<"}
