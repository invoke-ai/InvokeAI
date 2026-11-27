import pytest

from invokeai.backend.working_memory import user_fit


@pytest.fixture(autouse=True)
def _no_user_fit_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """A started calibration service registers the process-global provider every estimate asks; one a test leaves
    running must not price the estimates of later tests on this worker."""
    monkeypatch.setattr(user_fit, "_provider", None)
