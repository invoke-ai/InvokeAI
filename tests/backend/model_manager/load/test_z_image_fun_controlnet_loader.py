"""What the Z-Image Fun-ControlNet loader makes of a checkpoint: the layout it reads off the tensors, the dtype it casts
to, and the quantized files it refuses rather than load unscaled.

It builds under `init_empty_weights` and assigns the file's tensors, so an fp8 weight whose scale was dropped would
become the parameter as it is, off by `1/weight_scale`. No quantized Z-Image ControlNet is published (checked
2026-09-18: alibaba-pai's builds are all dense bf16, and `Comfy-Org/z_image_turbo` ships none), so refusing costs
nothing that works.
"""

from pathlib import Path

import pytest
import safetensors.torch
import torch

from invokeai.backend.fun_controlnet.z_image import (
    Z_IMAGE_FUN_CONTROL_2_0,
    Z_IMAGE_FUN_CONTROL_2_1,
    Z_IMAGE_FUN_CONTROL_2_1_LITE,
    ZImageFunControl,
)
from invokeai.backend.model_manager.configs.controlnet import ControlNet_Checkpoint_ZImage_Config
from invokeai.backend.model_manager.load.model_loaders import z_image
from invokeai.backend.model_manager.load.model_loaders.z_image import ZImageFunControlNetLoader
from tests.fixtures.loader_seams import Seam, prepare
from tests.fixtures.quantized_payloads import comfy_quant_marker, quantize_convrot, quantize_scaled_fp8

# One 128-wide head: the narrowest control the loader's head rule accepts.
HIDDEN = 128
QUANTIZED_LAYER = "control_layers.0.attention.to_q"

SEAM = Seam(
    loader=ZImageFunControlNetLoader,
    module=z_image,
    entry="_load_model",
    # The loader imports `load_file` inside the method, so the name it resolves is the package's.
    load_file_host=safetensors.torch,
    compute_dtype=torch.bfloat16,
    # The loader takes its dtype from the device and never reads `self._torch_dtype`.
    patches_device=True,
    sets_torch_dtype=False,
    casts_fp8_storage=False,
)


def _checkpoint(variant=Z_IMAGE_FUN_CONTROL_2_1_LITE) -> dict[str, torch.Tensor]:
    """A float32 file with the variant's real layout; the loader has to cast it."""
    torch.manual_seed(0)
    control = ZImageFunControl(variant, hidden_size=HIDDEN, num_attention_heads=1)
    return {name: tensor.detach().clone() for name, tensor in control.state_dict().items()}


def _driver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state_dict: dict[str, torch.Tensor]):
    checkpoint = tmp_path / "z_image_controlnet.safetensors"
    checkpoint.touch()
    config = ControlNet_Checkpoint_ZImage_Config.model_construct(path=str(checkpoint), name="z-image-control")
    return prepare(SEAM, monkeypatch, state_dict=state_dict), config


def test_a_dense_checkpoint_loads_as_its_layout_in_the_compute_dtype(monkeypatch, tmp_path) -> None:
    run, config = _driver(monkeypatch, tmp_path, _checkpoint())

    model = run.load(config)

    assert isinstance(model, ZImageFunControl)
    assert model.variant == Z_IMAGE_FUN_CONTROL_2_1_LITE
    assert (model.hidden_size, model.control_in_dim, model.control_layers[0].attention.heads) == (HIDDEN, 132, 1)
    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}
    assert not model.training


def test_the_2_0_layout_loads_without_its_untrained_refiner(monkeypatch, tmp_path) -> None:
    """The 2.0 file has the 2.1 keys, with the refiner projections all zero; its refiner chain runs over the main
    control blocks, so the refiner blocks are dropped rather than kept resident."""
    state_dict = {
        name: torch.zeros_like(tensor) if name.startswith("control_noise_refiner.") and "_proj." in name else tensor
        for name, tensor in _checkpoint(Z_IMAGE_FUN_CONTROL_2_1).items()
    }
    run, config = _driver(monkeypatch, tmp_path, state_dict)

    model = run.load(config)

    assert model.variant == Z_IMAGE_FUN_CONTROL_2_0
    assert len(model.control_noise_refiner) == 0
    assert not any(name.startswith("control_noise_refiner.") for name in model.state_dict())


def test_a_scaled_fp8_checkpoint_is_refused_rather_than_loaded_unscaled(monkeypatch, tmp_path) -> None:
    payload = quantize_scaled_fp8(torch.randn(HIDDEN, HIDDEN))
    state_dict = _checkpoint()
    state_dict[f"{QUANTIZED_LAYER}.weight"] = payload.codes
    state_dict[f"{QUANTIZED_LAYER}.weight_scale"] = payload.scale
    run, config = _driver(monkeypatch, tmp_path, state_dict)

    with pytest.raises(ValueError, match="quantization side channel"):
        run.load(config)


def test_an_int8_checkpoint_is_refused_by_its_marker(monkeypatch, tmp_path) -> None:
    """int8 is not even a dtype the control could run: the tensor would be assigned where a float parameter belongs,
    and the un-rotation the scheme needs has no home here."""
    codes, scale, _ = quantize_convrot(torch.randn(HIDDEN, 256))
    state_dict = _checkpoint()
    state_dict[f"{QUANTIZED_LAYER}.weight"] = codes
    state_dict[f"{QUANTIZED_LAYER}.weight_scale"] = scale
    state_dict[f"{QUANTIZED_LAYER}.comfy_quant"] = comfy_quant_marker({"format": "int8_tensorwise", "convrot": True})
    run, config = _driver(monkeypatch, tmp_path, state_dict)

    with pytest.raises(ValueError, match="quantization side channel"):
        run.load(config)


def test_a_layout_that_was_never_published_is_refused_by_name(monkeypatch, tmp_path) -> None:
    state_dict = {name: tensor for name, tensor in _checkpoint().items() if not name.startswith("control_layers.2.")}
    run, config = _driver(monkeypatch, tmp_path, state_dict)

    with pytest.raises(ValueError, match="2 control blocks"):
        run.load(config)
