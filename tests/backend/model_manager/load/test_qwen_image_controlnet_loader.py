"""What the Qwen-Image ControlNet loader makes of a checkpoint: which design it builds (the one the record names), the
geometry it reads off the tensors, the dtype it casts to, and the files it refuses rather than loads wrongly.

Both designs build under `init_empty_weights` and assign the file's tensors, so an fp8 weight with its scale dropped
would become the parameter as it is (`test_z_image_fun_controlnet_loader.py`). No quantized build of either is
published (alibaba-pai and Comfy-Org ship dense bf16 files), so refusing costs nothing.
"""

from pathlib import Path

import pytest
import safetensors.torch
import torch
from diffusers.models.controlnets.controlnet_qwenimage import QwenImageControlNetModel

from invokeai.backend.fun_controlnet.qwen_image import QWEN_IMAGE_FUN_CONTROL_LAYERS, QwenImageFunControlBranch
from invokeai.backend.model_manager.configs.controlnet import ControlNet_Checkpoint_QwenImage_Config
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_QwenImage_Config
from invokeai.backend.model_manager.load.model_loaders import fun_controlnet, qwen_image, qwen_image_controlnet
from invokeai.backend.model_manager.load.model_loaders.qwen_image import QwenImageCheckpointModel
from invokeai.backend.model_manager.load.model_loaders.qwen_image_controlnet import (
    QWEN_IMAGE_AXES_DIMS_ROPE,
    QwenImageControlNetLoader,
)
from tests.backend.model_manager.load.state_dicts.qwen_image_instantx_controlnet_keys import state_dict_keys
from tests.fixtures.loader_seams import Seam, prepare
from tests.fixtures.quantized_payloads import comfy_quant_marker, quantize_convrot, quantize_scaled_fp8

# One 128-wide head: the narrowest ControlNet the loaders' head rule accepts, with the real five-block layouts.
HIDDEN = 128
QUANTIZED_LAYER = "control_blocks.0.attn.to_q"
INSTANTX_BLOCKS = 5
INSTANTX_TEXT_DIM = 24

SEAM = Seam(
    loader=QwenImageControlNetLoader,
    module=qwen_image_controlnet,
    entry="_load_model",
    # The file is read by the dense-checkpoint reader the ControlNet loaders share.
    load_file_host=fun_controlnet,
    compute_dtype=torch.bfloat16,
    # The loader takes its dtype from the device and never reads `self._torch_dtype`.
    patches_device=True,
    sets_torch_dtype=False,
    casts_fp8_storage=False,
)


def _checkpoint() -> dict[str, torch.Tensor]:
    """A float32 Fun-ControlNet file with the real layout; the loader has to cast it."""
    torch.manual_seed(0)
    branch = QwenImageFunControlBranch(hidden_size=HIDDEN, num_attention_heads=1, attention_head_dim=128)
    return {name: tensor.detach().clone() for name, tensor in branch.state_dict().items()}


def _instantx_checkpoint() -> dict[str, torch.Tensor]:
    """A float32 InstantX file: diffusers' model at one 128-wide head, with every tensor random so a load that
    dropped or swapped one shows."""
    torch.manual_seed(1)
    model = QwenImageControlNetModel(
        num_layers=INSTANTX_BLOCKS,
        attention_head_dim=128,
        num_attention_heads=1,
        joint_attention_dim=INSTANTX_TEXT_DIM,
        axes_dims_rope=QWEN_IMAGE_AXES_DIMS_ROPE,
    )
    return {name: torch.randn_like(tensor) for name, tensor in model.state_dict().items()}


def _driver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state_dict: dict[str, torch.Tensor], **record):
    checkpoint = tmp_path / "qwen_image_controlnet.safetensors"
    checkpoint.touch()
    config = ControlNet_Checkpoint_QwenImage_Config.model_construct(
        path=str(checkpoint), name="qwen-image-control", **record
    )
    return prepare(SEAM, monkeypatch, state_dict=state_dict), config


def test_a_record_without_a_design_loads_the_fun_branch_with_its_geometry_in_the_compute_dtype(
    monkeypatch, tmp_path
) -> None:
    run, config = _driver(monkeypatch, tmp_path, _checkpoint())

    model = run.load(config)

    assert isinstance(model, QwenImageFunControlBranch)
    assert model.control_layers == QWEN_IMAGE_FUN_CONTROL_LAYERS
    assert (model.hidden_size, model.control_in_dim, model.control_blocks[0].attn.heads) == (HIDDEN, 132, 1)
    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}
    assert not model.training


def test_an_instantx_record_loads_every_tensor_into_the_controlnet_its_keys_describe(monkeypatch, tmp_path) -> None:
    checkpoint = _instantx_checkpoint()
    run, config = _driver(monkeypatch, tmp_path, checkpoint, control_architecture="instantx_union")

    model = run.load(config)

    assert isinstance(model, QwenImageControlNetModel)
    assert (len(model.transformer_blocks), len(model.controlnet_blocks)) == (INSTANTX_BLOCKS, INSTANTX_BLOCKS)
    assert (model.inner_dim, model.transformer_blocks[0].attn.heads) == (HIDDEN, 1)
    assert (model.config.joint_attention_dim, model.config.in_channels) == (INSTANTX_TEXT_DIM, 64)
    assert model.config.extra_condition_channels == 0
    assert not model.training
    loaded = model.state_dict()
    assert loaded.keys() == checkpoint.keys()
    for name, tensor in checkpoint.items():
        assert torch.equal(loaded[name], tensor.to(torch.bfloat16)), name


def test_the_released_instantx_header_loads_strictly_into_the_controlnet_built_from_it(monkeypatch, tmp_path) -> None:
    """The captured header widened to the file's five blocks (its 181 tensors), on the meta device so nothing of the
    3.5 GB is allocated: every name and shape lands, and the geometry is Qwen-Image's."""
    full = {
        name.replace(".0.", f".{index}.", 1): spec
        for name, spec in state_dict_keys.items()
        for index in (
            range(INSTANTX_BLOCKS) if name.startswith(("transformer_blocks.0.", "controlnet_blocks.0.")) else [0]
        )
    }
    assert len(full) == 181
    with torch.device("meta"):
        checkpoint = {name: torch.empty(shape, dtype=torch.bfloat16) for name, (shape, _dtype) in full.items()}
    run, config = _driver(monkeypatch, tmp_path, checkpoint, control_architecture="instantx_union")

    model = run.load(config)

    assert isinstance(model, QwenImageControlNetModel)
    assert {name: list(tensor.shape) for name, tensor in model.state_dict().items()} == {
        name: shape for name, (shape, _dtype) in full.items()
    }
    assert (model.config.num_layers, model.config.num_attention_heads, model.config.joint_attention_dim) == (
        5,
        24,
        3584,
    )


def test_the_record_decides_the_design_and_a_wrong_one_fails_by_name(monkeypatch, tmp_path) -> None:
    run, config = _driver(monkeypatch, tmp_path, _instantx_checkpoint(), control_architecture="fun_union")
    with pytest.raises(ValueError, match="control_img_in.weight"):
        run.load(config)

    run, config = _driver(monkeypatch, tmp_path, _checkpoint(), control_architecture="instantx_union")
    with pytest.raises(ValueError, match="Not an InstantX Qwen Image ControlNet"):
        run.load(config)


def test_the_instantx_inpainting_layout_is_refused_by_name(monkeypatch, tmp_path) -> None:
    checkpoint = _instantx_checkpoint()
    checkpoint["controlnet_x_embedder.weight"] = torch.zeros(HIDDEN, 68)
    run, config = _driver(monkeypatch, tmp_path, checkpoint, control_architecture="instantx_union")

    with pytest.raises(ValueError, match="inpainting layout"):
        run.load(config)


@pytest.mark.parametrize(
    ("rewrite", "message"),
    [
        (lambda sd: {k: v for k, v in sd.items() if not k.startswith("transformer_blocks.4.")}, "missing"),
        (lambda sd: {**sd, "controlnet_blocks.5.weight": torch.zeros(HIDDEN, HIDDEN)}, "missing"),
        (lambda sd: {**sd, "pos_embed_input.weight": torch.zeros(HIDDEN, 64)}, "unexpected"),
        (lambda sd: {**sd, "transformer_blocks.0.attn.norm_q.weight": torch.zeros(64)}, "attention heads"),
    ],
    ids=["fewer-blocks-than-projections", "more-projections-than-blocks", "stray-key", "other-head-width"],
)
def test_an_instantx_checkpoint_that_is_not_the_layout_is_refused(monkeypatch, tmp_path, rewrite, message) -> None:
    run, config = _driver(monkeypatch, tmp_path, rewrite(_instantx_checkpoint()), control_architecture="instantx_union")

    with pytest.raises(ValueError, match=message):
        run.load(config)


@pytest.mark.parametrize(
    ("checkpoint", "layer", "architecture"),
    [
        (_checkpoint, QUANTIZED_LAYER, "fun_union"),
        (_instantx_checkpoint, "transformer_blocks.0.attn.to_q", "instantx_union"),
    ],
    ids=["fun", "instantx"],
)
def test_a_scaled_fp8_checkpoint_is_refused_rather_than_loaded_unscaled(
    monkeypatch, tmp_path, checkpoint, layer, architecture
) -> None:
    payload = quantize_scaled_fp8(torch.randn(HIDDEN, HIDDEN))
    state_dict = checkpoint()
    state_dict[f"{layer}.weight"] = payload.codes
    state_dict[f"{layer}.weight_scale"] = payload.scale
    run, config = _driver(monkeypatch, tmp_path, state_dict, control_architecture=architecture)

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
def test_another_fun_recipe_is_refused_by_name(monkeypatch, tmp_path, rewrite, message) -> None:
    run, config = _driver(monkeypatch, tmp_path, rewrite(_checkpoint()))

    with pytest.raises(ValueError, match=message):
        run.load(config)


def test_an_instantx_file_an_earlier_version_recorded_as_a_main_model_is_refused_with_the_fix(
    monkeypatch, tmp_path
) -> None:
    """Earlier versions identified the InstantX union as a Qwen Image main model, and such records outlive the fix
    to identification: loading one as a transformer says what it is and how to repair the record, before any room
    is reserved for a transformer that cannot be built."""
    with torch.device("meta"):
        checkpoint = {
            name: torch.empty(shape, dtype=torch.bfloat16) for name, (shape, _dtype) in state_dict_keys.items()
        }
    path = tmp_path / "Qwen-Image-InstantX-ControlNet-Union.safetensors"
    path.touch()
    seam = Seam(
        loader=QwenImageCheckpointModel,
        module=qwen_image,
        entry="_load_from_singlefile",
        load_file_host=safetensors.torch,
        compute_dtype=torch.bfloat16,
        patches_device=True,
        sets_torch_dtype=False,
        casts_fp8_storage=False,
    )
    run = prepare(seam, monkeypatch, state_dict=checkpoint, metadata=None)

    with pytest.raises(ValueError, match="is a Qwen Image ControlNet, not a main model.*Re-identify"):
        run.load(Main_Checkpoint_QwenImage_Config.model_construct(path=str(path), name="qwen-instantx"))

    assert run.reserved == []
