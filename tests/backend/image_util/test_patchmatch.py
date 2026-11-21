"""Regression tests for the PatchMatch load-failure guidance.

When pypatchmatch's C extension fails to build, the `patchmatch` package logs an
installation-help URL from its own logger (`patchmatch.patch_match`). The URL in
pypatchmatch 1.0.2 is hard-coded to a documentation page that moved, so a failed
build pointed users at a 404 (invoke-ai/InvokeAI#9775). Loading must not surface
that stale URL and must print the current installation help location instead,
while keeping the compile error detail pypatchmatch reports.
"""

import logging
import sys
import types

import pytest

import invokeai.backend.image_util.infill_methods.patchmatch as patchmatch_module
from invokeai.backend.image_util.infill_methods.patchmatch import PatchMatch
from invokeai.backend.util.logging import InvokeAILogger

STALE_URL = "https://invoke-ai.github.io/InvokeAI/installation/060_INSTALL_PATCHMATCH/"
CURRENT_URL = "https://invoke.ai/configuration/patchmatch/#installation"
UPSTREAM_LOGGER_NAME = "patchmatch.patch_match"


class _Collector(logging.Handler):
    """Records every message emitted through the loggers a failed load touches."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class _FakePatchMatchModule:
    """Stands in for pypatchmatch's `patch_match` module.

    The real module compiles its C extension at import and, on failure, logs the
    compile error and its installation-help URL through its own logger before the
    caller can observe `patchmatch_available`. The fake replays that output the
    first time the flag is read, which is the first statement after the import.
    """

    def __init__(self, available: bool) -> None:
        self._available = available
        self._reported = False

    @property
    def patchmatch_available(self) -> bool:
        if not self._available and not self._reported:
            self._reported = True
            upstream = logging.getLogger(UPSTREAM_LOGGER_NAME)
            upstream.error(
                "patchmatch failed to load or compile (Command 'make clean && make' returned non-zero exit status 2.)."
            )
            upstream.info(f"Refer to {STALE_URL} for installation instructions.")
        return self._available


@pytest.fixture
def load_failure(monkeypatch: pytest.MonkeyPatch):
    """Route loading through a pypatchmatch whose build failed."""
    yield from _patched_load(monkeypatch, available=False)


@pytest.fixture
def load_success(monkeypatch: pytest.MonkeyPatch):
    """Route loading through a pypatchmatch that built successfully."""
    yield from _patched_load(monkeypatch, available=True)


def _patched_load(monkeypatch: pytest.MonkeyPatch, available: bool):
    fake_pkg = types.ModuleType("patchmatch")
    fake_pkg.patch_match = _FakePatchMatchModule(available)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "patchmatch", fake_pkg)
    monkeypatch.setattr(patchmatch_module, "get_config", lambda: types.SimpleNamespace(patchmatch=True))
    monkeypatch.setattr(PatchMatch, "tried_load", False)
    monkeypatch.setattr(PatchMatch, "patch_match", None)

    upstream_logger = logging.getLogger(UPSTREAM_LOGGER_NAME)
    invokeai_logger = InvokeAILogger.get_logger()
    saved_levels = (upstream_logger.level, invokeai_logger.level)
    saved_filters = list(upstream_logger.filters)
    upstream_logger.setLevel(logging.INFO)
    collector = _Collector()
    upstream_logger.addHandler(collector)
    invokeai_logger.addHandler(collector)
    yield collector
    upstream_logger.removeHandler(collector)
    invokeai_logger.removeHandler(collector)
    upstream_logger.filters[:] = saved_filters
    upstream_logger.setLevel(saved_levels[0])
    invokeai_logger.setLevel(saved_levels[1])


def test_build_failure_guidance_points_at_current_documentation(load_failure: _Collector) -> None:
    PatchMatch.patchmatch_available()

    joined = "\n".join(load_failure.messages)
    assert STALE_URL not in joined
    assert CURRENT_URL in joined
    # The compile error detail must survive; only the stale guidance is replaced.
    assert "patchmatch failed to load or compile" in joined
    assert "Patchmatch not loaded (nonfatal)" in joined


def test_successful_load_prints_no_installation_guidance(load_success: _Collector) -> None:
    assert PatchMatch.patchmatch_available()

    joined = "\n".join(load_success.messages)
    assert "Patchmatch initialized" in joined
    assert STALE_URL not in joined
    assert "Refer to" not in joined
