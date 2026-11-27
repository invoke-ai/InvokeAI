"""A bitsandbytes that is installed but fails to load must be reported as such, not as "please install it"."""

from unittest.mock import MagicMock

import pytest

from invokeai.backend.model_manager.configs.t5_encoder import T5Encoder_BnBLLMint8_Config
from invokeai.backend.model_manager.load.model_loaders import flux
from invokeai.backend.quantization import bnb_import


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)


@pytest.fixture
def fresh_state(monkeypatch) -> _RecordingLogger:
    recorder = _RecordingLogger()
    monkeypatch.setattr(bnb_import, "_failure", None)
    monkeypatch.setattr(bnb_import, "logger", recorder)
    return recorder


def test_a_missing_bitsandbytes_is_quiet_and_asks_for_an_install(fresh_state: _RecordingLogger) -> None:
    missing = ModuleNotFoundError("No module named 'bitsandbytes'", name="bitsandbytes")
    bnb_import.record_bnb_import_failure(missing)

    assert fresh_state.warnings == []
    assert "Please install bitsandbytes" in bnb_import.bnb_unavailable_message()


def test_a_broken_bitsandbytes_is_logged_once_and_named_at_load_time(fresh_state: _RecordingLogger) -> None:
    broken = OSError("libbitsandbytes_cuda134.dll: cannot open shared object file")
    # Every optional-import guard records the same failure.
    for error in (broken, broken, RuntimeError("second guard")):
        bnb_import.record_bnb_import_failure(error)

    assert len(fresh_state.warnings) == 1
    assert "libbitsandbytes_cuda134.dll" in fresh_state.warnings[0]
    message = bnb_import.bnb_unavailable_message()
    assert "libbitsandbytes_cuda134.dll" in message
    assert "Please install" not in message


def test_a_missing_dependency_of_bitsandbytes_is_a_failure_not_an_absence(fresh_state: _RecordingLogger) -> None:
    bnb_import.record_bnb_import_failure(ModuleNotFoundError("No module named 'scipy'", name="scipy"))

    assert len(fresh_state.warnings) == 1
    assert "scipy" in bnb_import.bnb_unavailable_message()


def test_the_bnb_loader_raises_the_recorded_error(fresh_state: _RecordingLogger, monkeypatch) -> None:
    monkeypatch.setattr(flux, "bnb_available", False)
    bnb_import.record_bnb_import_failure(OSError("libbitsandbytes_cuda134.dll not found"))

    with pytest.raises(ImportError, match="libbitsandbytes_cuda134.dll"):
        flux.BnbQuantizedLlmInt8bCheckpointModel._load_model(MagicMock(), T5Encoder_BnBLLMint8_Config.model_construct())
