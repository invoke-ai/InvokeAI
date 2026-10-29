from types import SimpleNamespace

import pytest

from invokeai.backend.working_memory import user_fit


@pytest.fixture(autouse=True)
def _no_real_paged_vram_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every completed session asks Windows how much of the GPU memory is paged out. On a Windows ROCm machine that
    would be a real PDH query per session; the tests that need an answer script their own."""
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.paged_bytes", lambda device: None
    )


@pytest.fixture(autouse=True)
def _no_user_fit_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """A started calibration service registers the process-global provider every estimate asks; one a test leaves
    running must not price the estimates of later tests on this worker."""
    monkeypatch.setattr(user_fit, "_provider", None)


@pytest.fixture(autouse=True)
def _no_full_collection_per_claimed_item(monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker runs a full collection before each claimed item. It walks the whole heap: most of a second in a
    fresh test process, seconds late in an xdist session. None of these tests check memory."""
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.gc", SimpleNamespace(collect=lambda: 0)
    )
