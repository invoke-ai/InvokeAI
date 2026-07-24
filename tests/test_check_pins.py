from __future__ import annotations

import importlib.util
import json
import shutil
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
