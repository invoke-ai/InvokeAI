"""What the Z-Image ControlNet loader does with a quantized checkpoint.

It handles no quantization side channel at all, and it is the worst-shaped of the loaders that do
not: it builds the adapter under `init_empty_weights` and then calls
`load_state_dict(sd, assign=True, strict=False)`. `assign=True` means the checkpoint's tensors
*become* the parameters, so an fp8 weight stays fp8 -- there is no cast anywhere to widen it -- and
`strict=False` sends the orphaned `weight_scale` to `unexpected_keys`, where `log_unexpected_keys`
reports it at DEBUG and drops it. The result is an adapter whose every quantized weight is off by
`1/weight_scale` -- orders of magnitude -- and nothing says so at the default log level.

No quantized Z-Image ControlNet is published today (checked 2026-09-18: `alibaba-pai`'s ten builds
are all dense, and `Comfy-Org/z_image_turbo` ships none), so refusing costs nothing that works.
"""

from pathlib import Path

import pytest
import safetensors.torch
import torch

from invokeai.backend.model_manager.configs.controlnet import ControlNet_Checkpoint_ZImage_Config
from invokeai.backend.model_manager.load.model_loaders import z_image
from invokeai.backend.model_manager.load.model_loaders.z_image import ZImageControlCheckpointModel
from tests.fixtures.loader_seams import Seam, prepare
from tests.fixtures.quantized_payloads import comfy_quant_marker, quantize_convrot, quantize_scaled_fp8

SEAM = Seam(
    loader=ZImageControlCheckpointModel,
    module=z_image,
    entry="_load_control_adapter",
    # The loader imports `load_file` inside the method, so the name it resolves is the package's.
    load_file_host=safetensors.torch,
    casts_fp8_storage=False,
)


def _dense_adapter() -> dict[str, torch.Tensor]:
    """The smallest state dict the loader's geometry probe accepts: one control block, two refiner
    layers, and the embedder it reads `control_in_dim` off."""
    return {
        "control_layers.0.weight": torch.ones(8, 8),
        "control_noise_refiner.0.weight": torch.ones(8, 8),
        "control_noise_refiner.1.weight": torch.ones(8, 8),
        "control_all_x_embedder.2-1.weight": torch.ones(3840, 64),
    }


def _driver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state_dict: dict[str, torch.Tensor]):
    checkpoint = tmp_path / "z_image_controlnet.safetensors"
    checkpoint.touch()
    config = ControlNet_Checkpoint_ZImage_Config.model_construct(path=str(checkpoint), name="z-image-control")
    return prepare(SEAM, monkeypatch, state_dict=state_dict), config


def test_a_scaled_fp8_checkpoint_is_refused_rather_than_loaded_unscaled(monkeypatch, tmp_path) -> None:
    """`assign=True` would make the raw e4m3 codes the adapter's weights, and the scale that says
    what they mean is dropped as an unexpected key."""
    payload = quantize_scaled_fp8(torch.randn(8, 8))
    state_dict = _dense_adapter()
    state_dict["control_layers.0.weight"] = payload.codes
    state_dict["control_layers.0.weight_scale"] = payload.scale
    run, config = _driver(monkeypatch, tmp_path, state_dict)

    with pytest.raises(ValueError, match="quantization side channel"):
        run.load(config)


def test_an_int8_checkpoint_is_refused_by_its_marker(monkeypatch, tmp_path) -> None:
    """int8 is not even a dtype the adapter could run: `assign=True` would install an int8 tensor
    where a float parameter belongs, and the un-rotation the scheme needs has no home here."""
    codes, scale, _ = quantize_convrot(torch.randn(8, 256))
    state_dict = _dense_adapter()
    state_dict["control_layers.0.weight"] = codes
    state_dict["control_layers.0.weight_scale"] = scale
    state_dict["control_layers.0.comfy_quant"] = comfy_quant_marker({"format": "int8_tensorwise", "convrot": True})
    run, config = _driver(monkeypatch, tmp_path, state_dict)

    with pytest.raises(ValueError, match="quantization side channel"):
        run.load(config)


def test_a_dense_checkpoint_still_loads(monkeypatch, tmp_path) -> None:
    """The other half: the guard must not cost the builds that exist. A refusal that fires on every
    checkpoint is not a guard, it is an outage."""
    run, config = _driver(monkeypatch, tmp_path, _dense_adapter())

    model = run.load(config)

    assert model.__class__.__name__ == "ZImageControlAdapter"
