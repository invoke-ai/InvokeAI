"""scripts/win_arm64_wheels/plan_release_upload.py must never let a publish replace an asset the lockfile pins."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "win_arm64_wheels" / "plan_release_upload.py"

spec = importlib.util.spec_from_file_location("plan_release_upload", SCRIPT)
assert spec is not None and spec.loader is not None
plan_release_upload = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plan_release_upload
spec.loader.exec_module(plan_release_upload)

WHEEL = "blake3-1.0.9-cp312-cp312-win_arm64.whl"


def _write(tmp_path: Path, name: str, content: bytes) -> tuple[Path, str]:
    path = tmp_path / name
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def _run(tmp_path: Path, assets: list[dict], *files: Path, capsys) -> tuple[int, list[str], str]:
    release = tmp_path / "release.json"
    release.write_text(json.dumps({"assets": assets}))
    code = plan_release_upload.main(["plan", str(release), *map(str, files)])
    out, err = capsys.readouterr()
    return code, out.split(), err


def test_a_new_release_uploads_everything(tmp_path, capsys) -> None:
    wheel, _ = _write(tmp_path, WHEEL, b"wheel")
    index, _ = _write(tmp_path, "index.html", b"<html>")

    code, upload, _ = _run(tmp_path, [], wheel, index, capsys=capsys)

    assert code == 0
    assert sorted(upload) == sorted([str(wheel), str(index)])


def test_identical_published_assets_are_skipped_so_an_interrupted_upload_can_finish(tmp_path, capsys) -> None:
    wheel, wheel_sha = _write(tmp_path, WHEEL, b"wheel")
    index, _ = _write(tmp_path, "index.html", b"<html>")

    code, upload, err = _run(tmp_path, [{"name": WHEEL, "digest": f"sha256:{wheel_sha}"}], wheel, index, capsys=capsys)

    assert code == 0
    assert upload == [str(index)]
    assert WHEEL in err


@pytest.mark.parametrize("name", [WHEEL, "index.html"])
def test_a_published_asset_with_different_contents_fails_and_nothing_is_uploaded(tmp_path, capsys, name) -> None:
    wheel, _ = _write(tmp_path, WHEEL, b"rebuilt wheel")
    index, _ = _write(tmp_path, "index.html", b"<html> rebuilt")

    code, upload, err = _run(tmp_path, [{"name": name, "digest": "sha256:" + "0" * 64}], wheel, index, capsys=capsys)

    assert code == 1
    assert upload == []
    assert name in err


def test_a_published_asset_without_a_digest_is_refused(tmp_path, capsys) -> None:
    wheel, _ = _write(tmp_path, WHEEL, b"wheel")

    code, upload, err = _run(tmp_path, [{"name": WHEEL, "digest": None}], wheel, capsys=capsys)

    assert code == 1
    assert upload == []
    assert "without a digest" in err
