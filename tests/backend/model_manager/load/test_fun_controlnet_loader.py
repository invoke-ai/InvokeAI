"""What the Fun-ControlNet loaders (Qwen-Image, FLUX.2 [dev]) make of a checkpoint: the geometry they read off the
tensors, the dtype they cast to, and the quantized files they refuse rather than load unscaled.

Like the Z-Image ControlNet loader it builds under `init_empty_weights` and assigns the file's tensors, so an fp8
weight with its scale dropped would become the parameter as it is (`test_z_image_fun_controlnet_loader.py`). No
quantized build of this adapter is published (alibaba-pai ships two dense bf16 files), so refusing costs nothing.
"""

from pathlib import Path

import pytest
import torch

from invokeai.backend.fun_controlnet.flux2 import FLUX2_FUN_CONTROL_LAYERS, Flux2FunControlBranch
from invokeai.backend.fun_controlnet.qwen_image import QWEN_IMAGE_FUN_CONTROL_LAYERS, QwenImageFunControlBranch
from invokeai.backend.model_manager.configs.controlnet import (
    ControlNet_Checkpoint_Flux2_Config,
    ControlNet_Checkpoint_QwenImage_Config,
)
from invokeai.backend.model_manager.load.model_loaders import fun_controlnet
from invokeai.backend.model_manager.load.model_loaders.fun_controlnet import (
    Flux2FunControlNetLoader,
    QwenImageFunControlNetLoader,
)
from tests.fixtures.loader_seams import Seam, prepare
from tests.fixtures.quantized_payloads import comfy_quant_marker, quantize_convrot, quantize_scaled_fp8

# One 128-wide head: the narrowest branch the loader's head rule accepts, with the real five-block layout.
HIDDEN = 128
QUANTIZED_LAYER = "control_blocks.0.attn.to_q"

SEAM = Seam(
    loader=QwenImageFunControlNetLoader,
    module=fun_controlnet,
    entry="_load_model",
    compute_dtype=torch.bfloat16,
    # The loader takes its dtype from the device and never reads `self._torch_dtype`.
    patches_device=True,
    sets_torch_dtype=False,
    casts_fp8_storage=False,
)


def _checkpoint() -> dict[str, torch.Tensor]:
    """A float32 file with the real layout; the loader has to cast it."""
    torch.manual_seed(0)
    branch = QwenImageFunControlBranch(hidden_size=HIDDEN, num_attention_heads=1, attention_head_dim=128)
    return {name: tensor.detach().clone() for name, tensor in branch.state_dict().items()}


def _driver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state_dict: dict[str, torch.Tensor]):
    checkpoint = tmp_path / "qwen_image_fun_controlnet.safetensors"
    checkpoint.touch()
    config = ControlNet_Checkpoint_QwenImage_Config.model_construct(path=str(checkpoint), name="qwen-image-control")
    return prepare(SEAM, monkeypatch, state_dict=state_dict), config


def test_a_dense_checkpoint_loads_with_its_geometry_in_the_compute_dtype(monkeypatch, tmp_path) -> None:
    run, config = _driver(monkeypatch, tmp_path, _checkpoint())

    model = run.load(config)

    assert isinstance(model, QwenImageFunControlBranch)
    assert model.control_layers == QWEN_IMAGE_FUN_CONTROL_LAYERS
    assert (model.hidden_size, model.control_in_dim, model.control_blocks[0].attn.heads) == (HIDDEN, 132, 1)
    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}
    assert not model.training


def test_a_scaled_fp8_checkpoint_is_refused_rather_than_loaded_unscaled(monkeypatch, tmp_path) -> None:
    payload = quantize_scaled_fp8(torch.randn(HIDDEN, HIDDEN))
    state_dict = _checkpoint()
    state_dict[f"{QUANTIZED_LAYER}.weight"] = payload.codes
    state_dict[f"{QUANTIZED_LAYER}.weight_scale"] = payload.scale
    run, config = _driver(monkeypatch, tmp_path, state_dict)

    with pytest.raises(ValueError, match="quantization side channel"):
        run.load(config)


def test_an_int8_checkpoint_is_refused_by_its_marker(monkeypatch, tmp_path) -> None:
    codes, scale, _ = quantize_convrot(torch.randn(HIDDEN, 256))
    state_dict = _checkpoint()
    state_dict[f"{QUANTIZED_LAYER}.weight"] = codes
    state_dict[f"{QUANTIZED_LAYER}.weight_scale"] = scale
    state_dict[f"{QUANTIZED_LAYER}.comfy_quant"] = comfy_quant_marker({"format": "int8_tensorwise", "convrot": True})
    run, config = _driver(monkeypatch, tmp_path, state_dict)

    with pytest.raises(ValueError, match="quantization side channel"):
        run.load(config)


@pytest.mark.parametrize(
    ("rewrite", "message"),
    [
        (lambda sd: {k: v for k, v in sd.items() if not k.startswith("control_blocks.4.")}, "4 control blocks"),
        (lambda sd: {**sd, "control_img_in.weight": torch.zeros(HIDDEN, 64)}, "takes 64"),
    ],
    ids=["four-blocks", "sixteen-channel-context"],
)
def test_another_recipe_is_refused_by_name(monkeypatch, tmp_path, rewrite, message) -> None:
    run, config = _driver(monkeypatch, tmp_path, rewrite(_checkpoint()))

    with pytest.raises(ValueError, match=message):
        run.load(config)


# --- FLUX.2 [dev] ---

FLUX2_SEAM = Seam(
    loader=Flux2FunControlNetLoader,
    module=fun_controlnet,
    entry="_load_model",
    compute_dtype=torch.bfloat16,
    # The loader casts to bfloat16, the dtype the FLUX.2 denoise runs in, whatever the device would pick.
    patches_device=False,
    sets_torch_dtype=False,
    casts_fp8_storage=False,
)
FLUX2_QUANTIZED_LAYER = "control_transformer_blocks.0.attn.to_q"


def _flux2_checkpoint() -> dict[str, torch.Tensor]:
    """A float32 file with the real four-block layout, under the names alibaba-pai's file uses."""
    torch.manual_seed(0)
    branch = Flux2FunControlBranch(hidden_size=HIDDEN, num_attention_heads=1, attention_head_dim=128)
    return {
        name.replace("control_blocks.", "control_transformer_blocks.", 1): tensor.detach().clone()
        for name, tensor in branch.state_dict().items()
    }


def _flux2_driver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state_dict: dict[str, torch.Tensor]):
    checkpoint = tmp_path / "flux2_fun_controlnet.safetensors"
    checkpoint.touch()
    config = ControlNet_Checkpoint_Flux2_Config.model_construct(path=str(checkpoint), name="flux2-control")
    return prepare(FLUX2_SEAM, monkeypatch, state_dict=state_dict), config


def test_a_dense_flux2_checkpoint_loads_with_its_geometry_in_bfloat16(monkeypatch, tmp_path) -> None:
    run, config = _flux2_driver(monkeypatch, tmp_path, _flux2_checkpoint())

    model = run.load(config)

    assert isinstance(model, Flux2FunControlBranch)
    assert model.control_layers == FLUX2_FUN_CONTROL_LAYERS
    first = model.control_blocks[0]
    assert (model.hidden_size, model.control_in_dim, first.attn.heads, first.mlp_hidden_dim) == (HIDDEN, 260, 1, 384)
    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}
    assert not model.training


def test_a_scaled_fp8_flux2_checkpoint_is_refused_rather_than_loaded_unscaled(monkeypatch, tmp_path) -> None:
    payload = quantize_scaled_fp8(torch.randn(HIDDEN, HIDDEN))
    state_dict = _flux2_checkpoint()
    state_dict[f"{FLUX2_QUANTIZED_LAYER}.weight"] = payload.codes
    state_dict[f"{FLUX2_QUANTIZED_LAYER}.weight_scale"] = payload.scale
    run, config = _flux2_driver(monkeypatch, tmp_path, state_dict)

    with pytest.raises(ValueError, match="quantization side channel"):
        run.load(config)


def test_the_qwen_image_recipe_is_refused_by_the_flux2_loader(monkeypatch, tmp_path) -> None:
    run, config = _flux2_driver(monkeypatch, tmp_path, _checkpoint())

    with pytest.raises(ValueError, match="takes 132"):
        run.load(config)
