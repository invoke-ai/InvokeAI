"""Regression tests for SafetyChecker's lazy init under concurrency.

`_load_safety_checker` is reached from ImageNSFWBlurInvocation on every session worker
thread. Its check-then-act on two class attributes used to let several workers run the
whole init body at once, which duplicated the 1.2GB download and construction, wrote
save_pretrained() into one directory from several threads, and — because the first
thread's mkdir + feature-extractor save makes model_path exist before the weights land —
let a second thread take the "already downloaded" branch, fail on the missing weights,
and leave safety_checker as None. has_nsfw_concept() reports that as "not NSFW", so the
image ships unblurred.
"""

import threading
from typing import Any

import pytest

import invokeai.backend.image_util.safety_checker as safety_checker_module
from invokeai.backend.image_util.safety_checker import CHECKER_PATH, SafetyChecker


class _FakeSaveable:
    """Stands in for the feature extractor / safety checker; records its saves."""

    def __init__(self, saves: list[Any]) -> None:
        self._saves = saves

    def save_pretrained(self, path: Any) -> None:
        self._saves.append(path)


@pytest.fixture
def reset_checker():
    """SafetyChecker caches on class attributes, so restore them around each test."""
    original = (SafetyChecker.feature_extractor, SafetyChecker.safety_checker)
    SafetyChecker.feature_extractor = None
    SafetyChecker.safety_checker = None
    yield
    SafetyChecker.feature_extractor, SafetyChecker.safety_checker = original


def test_concurrent_lazy_init_downloads_and_constructs_exactly_once(
    tmp_path, monkeypatch: pytest.MonkeyPatch, reset_checker
) -> None:
    downloads: list[str] = []
    constructions: list[Any] = []
    saves: list[Any] = []
    barrier = threading.Barrier(8)

    download_dir = tmp_path / "hf_snapshot"
    download_dir.mkdir()

    class _FakeConfig:
        models_path = tmp_path / "models"

    def fake_snapshot_download(repo: str) -> str:
        downloads.append(repo)
        # Widen the window a real network transfer would occupy.
        threading.Event().wait(0.05)
        return str(download_dir)

    def fake_image_processor_from_pretrained(path: Any) -> _FakeSaveable:
        return _FakeSaveable(saves)

    def fake_checker_from_pretrained(path: Any) -> _FakeSaveable:
        constructions.append(path)
        threading.Event().wait(0.05)
        return _FakeSaveable(saves)

    monkeypatch.setattr(safety_checker_module, "get_config", lambda: _FakeConfig())
    monkeypatch.setattr(safety_checker_module, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        safety_checker_module.AutoImageProcessor,
        "from_pretrained",
        staticmethod(fake_image_processor_from_pretrained),
    )
    monkeypatch.setattr(
        safety_checker_module.StableDiffusionSafetyChecker,
        "from_pretrained",
        staticmethod(fake_checker_from_pretrained),
    )

    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            SafetyChecker._load_safety_checker()
        except BaseException as e:  # noqa: BLE001 - surface anything the thread hit
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not [t for t in threads if t.is_alive()], "lazy init deadlocked"
    assert not errors, f"worker threads raised: {errors}"

    # The point of the fix: eight concurrent callers do the work once, not eight times.
    assert len(downloads) == 1, f"expected a single download, got {len(downloads)}"
    assert len(constructions) == 1, f"expected a single construction, got {len(constructions)}"

    # And every caller sees a usable checker — none observes the half-initialized state
    # that would make has_nsfw_concept() silently return False.
    assert SafetyChecker.safety_checker is not None
    assert SafetyChecker.feature_extractor is not None


def test_failed_download_leaves_no_directory_to_poison_the_next_attempt(
    tmp_path, monkeypatch: pytest.MonkeyPatch, reset_checker
) -> None:
    """A failed download must not create model_path.

    If it does, every later call takes the `model_path.exists()` branch, fails on the
    absent weights, and the download is never retried for the life of the install.
    """

    class _FakeConfig:
        models_path = tmp_path / "models"

    def exploding_snapshot_download(repo: str) -> str:
        raise RuntimeError("network is down")

    monkeypatch.setattr(safety_checker_module, "get_config", lambda: _FakeConfig())
    monkeypatch.setattr(safety_checker_module, "snapshot_download", exploding_snapshot_download)

    # Swallowed and logged, per the existing contract.
    SafetyChecker._load_safety_checker()

    assert SafetyChecker.safety_checker is None
    assert not (_FakeConfig.models_path / CHECKER_PATH).exists(), (
        "a failed download left model_path behind; the next attempt will take the "
        "'already downloaded' branch and never retry"
    )


def test_failed_construction_after_a_good_download_does_not_poison_the_install(
    tmp_path, monkeypatch: pytest.MonkeyPatch, reset_checker
) -> None:
    """The download succeeding is not enough — the *save* must be all-or-nothing.

    `save_pretrained()` creates its own destination, so persisting the feature extractor
    before the checker is built would make model_path exist with no weights in it. Every
    later call would then take the `model_path.exists()` branch, fail on the missing
    weights, swallow the error, and `has_nsfw_concept()` would report the absent checker
    as "not NSFW" forever, without ever retrying the download.
    """
    downloads: list[str] = []
    saves: list[Any] = []

    download_dir = tmp_path / "hf_snapshot"
    download_dir.mkdir()

    class _FakeConfig:
        models_path = tmp_path / "models"

    def fake_snapshot_download(repo: str) -> str:
        downloads.append(repo)
        return str(download_dir)

    monkeypatch.setattr(safety_checker_module, "get_config", lambda: _FakeConfig())
    monkeypatch.setattr(safety_checker_module, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        safety_checker_module.AutoImageProcessor,
        "from_pretrained",
        staticmethod(lambda path: _FakeSaveable(saves)),
    )

    def exploding_checker_from_pretrained(path: Any):
        raise RuntimeError("truncated safetensors header")

    monkeypatch.setattr(
        safety_checker_module.StableDiffusionSafetyChecker,
        "from_pretrained",
        staticmethod(exploding_checker_from_pretrained),
    )

    SafetyChecker._load_safety_checker()

    assert SafetyChecker.safety_checker is None
    model_path = _FakeConfig.models_path / CHECKER_PATH
    assert not model_path.exists(), (
        "a failed construction left a weightless model_path behind; the checker is now "
        "permanently disabled and the download will never be retried"
    )
    # No staging directory should survive either.
    parent = model_path.parent
    leftovers = list(parent.iterdir()) if parent.exists() else []
    assert not leftovers, f"staging directory leaked: {leftovers}"

    # And the next attempt must genuinely retry rather than trusting a poisoned dir.
    SafetyChecker._load_safety_checker()
    assert len(downloads) == 2, f"expected the second attempt to re-download, got {downloads}"
