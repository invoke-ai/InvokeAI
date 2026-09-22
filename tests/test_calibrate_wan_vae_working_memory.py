from contextlib import nullcontext
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
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


def test_measure_tiling_uses_full_decode_and_tile_estimate(monkeypatch):
    script = _load_calibration_script()
    parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
    fake_vae = MagicMock()
    fake_vae.config = SimpleNamespace(scale_factor_temporal=4, scale_factor_spatial=8, z_dim=16)
    fake_vae.parameters.side_effect = lambda: iter([parameter])
    fake_vae.tile_sample_min_height = 256
    fake_vae.tile_sample_min_width = 256
    fake_vae.decode.return_value = (torch.zeros(1, 3, 4, 64, 64),)

    monkeypatch.setattr(script.torch, "randn", lambda *args, **kwargs: torch.zeros(*args, dtype=kwargs["dtype"]))
    monkeypatch.setattr(script.torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(script.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(script.torch.cuda, "reset_peak_memory_stats", lambda *args, **kwargs: None)
    monkeypatch.setattr(script.torch.cuda, "memory_reserved", lambda device: 100)
    monkeypatch.setattr(script.torch.cuda, "max_memory_reserved", lambda device: 200)
    monkeypatch.setattr(script.torch.cuda, "memory_allocated", lambda device: 50)
    monkeypatch.setattr(script.torch.cuda, "max_memory_allocated", lambda device: 150)
    monkeypatch.setattr(script.torch.cuda, "get_device_name", lambda device: "test-device")
    estimate = MagicMock(return_value=123)
    monkeypatch.setattr(script, "estimate_vae_working_memory_wan", estimate)
    monkeypatch.setattr(script, "iter_wan_vae_decode_chunks", MagicMock(side_effect=AssertionError))

    result = script._measure(fake_vae, 512, 512, 81, streaming=True, tiling=True, tile_size=128)

    assert result["streaming"] is False
    assert result["tiling"] is True
    assert result["tile_size"] == 128
    fake_vae.enable_tiling.assert_called_once_with(tile_sample_min_height=128, tile_sample_min_width=128)
    fake_vae.disable_tiling.assert_called_once_with()
    fake_vae.decode.assert_called_once()
    estimate.assert_called_once_with(
        operation="decode",
        vae=fake_vae,
        pixel_height=512,
        pixel_width=512,
        pixel_frames=81,
        tile_size=128,
        streaming=False,
    )


def test_measure_tiling_implied_constant_uses_tiled_area(monkeypatch):
    script = _load_calibration_script()
    parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
    fake_vae = MagicMock()
    fake_vae.config = SimpleNamespace(scale_factor_temporal=4, scale_factor_spatial=8, z_dim=16)
    fake_vae.parameters.side_effect = lambda: iter([parameter])
    fake_vae.decode.return_value = (torch.zeros(1, 3, 81, 64, 64),)

    tile_size = 128
    constant = 4321.0
    element_size = parameter.element_size()
    pixel_height = pixel_width = 512
    pixel_frames = 81
    clip_bytes = 2 * 3 * pixel_frames * pixel_height * pixel_width * element_size
    measured_delta = int(tile_size**2 * element_size * constant * 1.25 + clip_bytes)

    monkeypatch.setattr(script.torch, "randn", lambda *args, **kwargs: torch.zeros(*args, dtype=kwargs["dtype"]))
    monkeypatch.setattr(script.torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(script.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(script.torch.cuda, "reset_peak_memory_stats", lambda *args, **kwargs: None)
    monkeypatch.setattr(script.torch.cuda, "memory_reserved", lambda device: 100)
    monkeypatch.setattr(script.torch.cuda, "max_memory_reserved", lambda device: measured_delta + 100 + 12345)
    monkeypatch.setattr(script.torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(script.torch.cuda, "max_memory_allocated", lambda device: measured_delta)
    monkeypatch.setattr(script.torch.cuda, "get_device_name", lambda device: "test-device")
    monkeypatch.setattr(script, "estimate_vae_working_memory_wan", lambda **kwargs: measured_delta)

    result = script._measure(
        fake_vae,
        pixel_height,
        pixel_width,
        pixel_frames,
        streaming=True,
        tiling=True,
        tile_size=tile_size,
    )

    assert result["measured_allocated_delta_bytes"] == measured_delta
    assert result["measured_reserved_delta_bytes"] == measured_delta + 12345
    assert result["implied_scaling_constant"] == pytest.approx(constant, abs=0.01)
