"""Regression tests for PiD decoder checkpoint identification.

InvokeAI's build_pid_net constructs only the legacy 512-dim PidNet. NVIDIA's v1.5 decoders use
lq_hidden_dim=1024 (plus PiT injection, scalar gates, ...), which cannot be loaded into it. The config
must reject such a checkpoint at identification time instead of accepting it and crashing with a size
mismatch deep inside the decode.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.pid_decoder import (
    PiDDecoder_Checkpoint_FLUX_Config,
    _lq_hidden_dim_from_state_dict,
)

_OVERRIDE_FIELDS: dict[str, object] = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/pid.pth",
    "file_size": 1000,
    "name": "pid",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
    "base": "flux",
}


class _FakeShapeTensor:
    def __init__(self, *shape: int) -> None:
        self.shape = shape


def _pid_state_dict(lq_hidden_dim: int, latent_channels: int = 16) -> dict[str, object]:
    # A minimal PiD-looking state dict: the lq_proj marker + the diagnostic input Conv whose dim-0 is the
    # lq_hidden_dim and dim-1 is the latent channel count.
    return {"net.lq_proj.latent_proj.0.weight": _FakeShapeTensor(lq_hidden_dim, latent_channels, 3, 3)}


def test_lq_hidden_dim_read_from_state_dict() -> None:
    assert _lq_hidden_dim_from_state_dict(_pid_state_dict(512)) == 512
    assert _lq_hidden_dim_from_state_dict(_pid_state_dict(1024)) == 1024
    assert _lq_hidden_dim_from_state_dict({}) is None


def _mock_mod(root: Path, state_dict: dict[str, object]) -> MagicMock:
    path = root / "model_ema_bf16.pth"
    path.write_bytes(b"x")
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = state_dict
    return mod


def test_v1_5_checkpoint_is_rejected_at_identification() -> None:
    """A 1024-dim (v1.5) checkpoint must be rejected, not accepted and crashed on later."""
    with TemporaryDirectory() as tmpdir:
        mod = _mock_mod(Path(tmpdir), _pid_state_dict(1024))
        with pytest.raises(NotAMatchError, match="lq_proj hidden dim 1024"):
            PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def test_legacy_512_checkpoint_is_accepted() -> None:
    """A legacy 512-dim FLUX checkpoint (16 latent channels) still identifies successfully."""
    with TemporaryDirectory() as tmpdir:
        mod = _mock_mod(Path(tmpdir), _pid_state_dict(512, latent_channels=16))
        config = PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
        assert config.type.value == "pid_decoder"
        assert config.base.value == "flux"
