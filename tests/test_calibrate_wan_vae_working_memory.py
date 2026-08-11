from contextlib import nullcontext
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import MagicMock

import torch


def _load_calibration_script():
    path = Path(__file__).parents[1] / "scripts" / "calibrate_wan_vae_working_memory.py"
    spec = spec_from_file_location("calibrate_wan_vae_working_memory", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_vae_accepts_single_wan_safetensors_checkpoint(tmp_path, monkeypatch):
    script = _load_calibration_script()
    checkpoint = tmp_path / "wan.safetensors"
    checkpoint.touch()
    state_dict = {"decoder.conv_in.weight": torch.zeros(2, 48, 1, 1, 1)}
    fake_vae = MagicMock()
    fake_autoencoder = MagicMock(return_value=fake_vae)

    monkeypatch.setattr(script, "AutoencoderKLWan", fake_autoencoder)
    monkeypatch.setattr(script, "_wan_vae_init_kwargs_for", lambda latent_channels: {"z_dim": latent_channels})
    monkeypatch.setattr("safetensors.torch.load_file", lambda path, device: state_dict)
    monkeypatch.setattr("accelerate.init_empty_weights", lambda: nullcontext())

    result = script._load_vae(checkpoint, torch.float16)

    assert result is fake_vae
    fake_autoencoder.assert_called_once_with(z_dim=48)
    fake_vae.load_state_dict.assert_called_once_with(state_dict, strict=True, assign=True)
    fake_vae.eval.assert_called_once_with()


def test_load_vae_accepts_diffusers_directory(tmp_path, monkeypatch):
    script = _load_calibration_script()
    directory = tmp_path / "vae"
    directory.mkdir()
    fake_vae = MagicMock()
    monkeypatch.setattr(script.AutoencoderKLWan, "from_pretrained", MagicMock(return_value=fake_vae))

    result = script._load_vae(directory, torch.bfloat16)

    assert result is fake_vae
    script.AutoencoderKLWan.from_pretrained.assert_called_once_with(
        directory, local_files_only=True, torch_dtype=torch.bfloat16
    )
    fake_vae.eval.assert_called_once_with()
