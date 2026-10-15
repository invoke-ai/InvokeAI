import pytest


@pytest.fixture(autouse=True)
def _no_real_paged_vram_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every completed session asks Windows how much of the GPU memory is paged out. On a Windows ROCm machine that
    would be a real PDH query per session; the tests that need an answer script their own."""
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.paged_bytes", lambda device: None
    )
