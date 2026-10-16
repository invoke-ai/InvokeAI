"""Qwen-Image-2.1 LoRAs onto the diffusers transformer: every layout, and ComfyUI's fused gate_up split in two."""

import accelerate
import pytest
import torch

from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.patches.layers.full_layer import FullLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.lora_conversions.qwen_image_2_1_lora_conversion_utils import (
    QWEN_IMAGE_21_LORA_NORM_PREFIX as NORM_PREFIX,
)
from invokeai.backend.patches.lora_conversions.qwen_image_2_1_lora_conversion_utils import (
    QWEN_IMAGE_21_LORA_TRANSFORMER_PREFIX as PREFIX,
)
from invokeai.backend.patches.lora_conversions.qwen_image_2_1_lora_conversion_utils import (
    lora_model_from_qwen_image_21_state_dict as convert,
)

RANK = 4
BLOCK_MODULES = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_up", "img_mlp.out")
TOP_MODULES = (
    "img_in",
    "proj_out",
    "norm_out.linear",
    "modulation.1",
    "txt_in.in_layer",
    "txt_in.out_layer",
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
)


@pytest.fixture(scope="module")
def transformer():
    from diffusers import QwenImage21Transformer2DModel

    with accelerate.init_empty_weights():
        # The class defaults are the released configuration; two blocks are enough to address.
        return QwenImage21Transformer2DModel(num_layers=2)


def _pair(module_key: str, out_features: int, in_features: int, *, a="lora_A", b="lora_B") -> dict:
    generator = torch.Generator().manual_seed(len(module_key))
    return {
        f"{module_key}.{a}.weight": torch.randn(RANK, in_features, generator=generator),
        f"{module_key}.{b}.weight": torch.randn(out_features, RANK, generator=generator),
    }


def _shape(transformer, module: str) -> tuple[int, int]:
    """The weight shape a LoRA on `module` updates; the fused gate_up is gate_layer and proj stacked."""
    if module.endswith("gate_up"):
        out, inp = transformer.get_submodule(module.replace("gate_up", "gate_layer")).weight.shape
        return 2 * out, inp
    return tuple(transformer.get_submodule(module).weight.shape)


def _kohya(module: str) -> str:
    return "lora_unet_" + module.replace(".", "_")


@pytest.mark.parametrize("layout", ["comfy", "kohya"])
def test_every_module_lands_on_a_linear_of_its_shape(transformer, layout: str) -> None:
    modules = [f"transformer_blocks.{b}.{m}" for b in range(2) for m in BLOCK_MODULES] + list(TOP_MODULES)
    sd: dict = {}
    for module in modules:
        name = f"diffusion_model.{module}" if layout == "comfy" else _kohya(module)
        lora = ("lora_A", "lora_B") if layout == "comfy" else ("lora_down", "lora_up")
        sd |= _pair(name, *_shape(transformer, module), a=lora[0], b=lora[1])

    layers = convert(sd).layers
    # Every module, gate_up as two.
    assert len(layers) == len(modules) + 2
    for key, layer in layers.items():
        target = transformer.get_submodule(key.removeprefix(PREFIX))
        assert isinstance(target, torch.nn.Linear), key
        assert (layer.up.shape[0], layer.down.shape[1]) == tuple(target.weight.shape), key


def _delta(layer: LoRALayer) -> torch.Tensor:
    return layer.up @ layer.down * layer.scale()


def test_gate_up_splits_into_gate_layer_then_proj_exactly() -> None:
    # The split does not depend on the size: 8 output rows, two halves of 4.
    sd = _pair("diffusion_model.transformer_blocks.0.img_mlp.gate_up", 8, 6)
    sd["diffusion_model.transformer_blocks.0.img_mlp.gate_up.alpha"] = torch.tensor(2.0)
    up = sd["diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight"]
    down = sd["diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight"]

    layers = convert(sd).layers
    gate = _delta(layers[f"{PREFIX}transformer_blocks.0.img_mlp.gate_layer"])
    proj = _delta(layers[f"{PREFIX}transformer_blocks.0.img_mlp.proj"])
    torch.testing.assert_close(torch.cat([gate, proj]), up @ down * (2.0 / RANK))


@pytest.mark.parametrize("magnitude_key", ["lora_magnitude_vector.weight", "lora_magnitude_vector", "magnitude"])
def test_an_output_dim_dora_on_gate_up_splits_its_magnitude_with_the_rows(magnitude_key: str) -> None:
    # PEFT writes the magnitude with or without `.weight`, ai-toolkit as `.magnitude`: all one per output row.
    module = "transformer.transformer_blocks.0.img_mlp.gate_up"
    magnitude = torch.arange(8.0)
    sd = _pair(module, 8, 6) | {f"{module}.{magnitude_key}": magnitude}

    layers = convert(sd).layers
    gate, proj = (layers[f"{PREFIX}transformer_blocks.0.img_mlp.{half}"] for half in ("gate_layer", "proj"))
    assert isinstance(gate, DoRALayer) and isinstance(proj, DoRALayer)
    assert gate.magnitude_is_out_dim and proj.magnitude_is_out_dim
    assert torch.equal(gate.dora_scale, magnitude[:4]) and torch.equal(proj.dora_scale, magnitude[4:])


@pytest.mark.parametrize("layout", ["comfy", "kohya"])
def test_norm_diffs_land_on_their_norm_weights_apart_from_the_linears(transformer, layout: str) -> None:
    # As the official Turbo LoRA carries them: a full diff on every block's q/k norm and on the text norm.
    norms = ["transformer_blocks.1.attn.norm_q", "transformer_blocks.1.attn.norm_k", "txt_in.text_norm"]
    name = (lambda m: f"diffusion_model.{m}") if layout == "comfy" else _kohya
    sd = {f"{name(m)}.diff": torch.randn(transformer.get_submodule(m).weight.shape) for m in norms}
    sd |= _pair(name("transformer_blocks.1.attn.to_q"), 4096, 4096, a="lora_down", b="lora_up")

    layers = convert(sd).layers
    assert set(layers) == {f"{NORM_PREFIX}{m}" for m in norms} | {f"{PREFIX}transformer_blocks.1.attn.to_q"}
    for module in norms:
        layer = layers[f"{NORM_PREFIX}{module}"]
        assert isinstance(layer, FullLayer)
        assert layer.weight.shape == transformer.get_submodule(module).weight.shape


def test_a_full_layer_on_gate_up_splits_by_rows() -> None:
    diff = torch.randn(8, 6)
    layers = convert({"diffusion_model.transformer_blocks.0.img_mlp.gate_up.diff": diff}).layers
    gate, proj = (layers[f"{PREFIX}transformer_blocks.0.img_mlp.{half}"] for half in ("gate_layer", "proj"))
    assert isinstance(gate, FullLayer) and isinstance(proj, FullLayer)
    assert torch.equal(torch.cat([gate.weight, proj.weight]), diff)


@pytest.mark.parametrize(
    "prefix", ["diffusion_model.", "transformer.", "base_model.model.transformer.", "base_model.model."]
)
def test_every_dotted_prefix_names_the_same_layer(prefix: str) -> None:
    sd = _pair(f"{prefix}transformer_blocks.3.attn.to_out.0", 6, 6)
    assert list(convert(sd).layers) == [f"{PREFIX}transformer_blocks.3.attn.to_out.0"]


def test_a_layers_own_alpha_scales_it_and_a_missing_one_means_its_rank() -> None:
    sd = _pair("diffusion_model.transformer_blocks.0.attn.to_q", 6, 6)
    sd |= _pair("diffusion_model.transformer_blocks.0.attn.to_k", 6, 6)
    sd["diffusion_model.transformer_blocks.0.attn.to_q.alpha"] = torch.tensor(2.0)
    layers = convert(sd).layers
    assert layers[f"{PREFIX}transformer_blocks.0.attn.to_q"].scale() == 2.0 / RANK
    assert layers[f"{PREFIX}transformer_blocks.0.attn.to_k"].scale() == 1.0


@pytest.mark.parametrize(
    "gate_up_layer",
    [
        {"lokr_w1": torch.zeros(4, 4), "lokr_w2": torch.zeros(2, 6)},
        # LyCORIS's input-dim magnitude normalizes across both halves at once.
        {"lora_A.weight": torch.zeros(RANK, 6), "lora_B.weight": torch.zeros(8, RANK), "dora_scale": torch.ones(1, 6)},
    ],
    ids=["lokr", "lycoris-dora"],
)
def test_a_gate_up_layer_that_cannot_be_split_is_refused(gate_up_layer: dict) -> None:
    sd = {f"diffusion_model.transformer_blocks.0.img_mlp.gate_up.{k}": v for k, v in gate_up_layer.items()}
    with pytest.raises(ValueError, match="gate_up"):
        convert(sd)


def test_a_low_rank_layer_on_a_norm_is_refused() -> None:
    # A norm has no matrix to factor; only a full diff of its weight applies.
    with pytest.raises(ValueError, match="norm_q"):
        convert(_pair("diffusion_model.transformer_blocks.0.attn.norm_q", 128, 128))


def test_a_lora_with_layers_this_model_lacks_is_refused_whole() -> None:
    # A FLUX.2 double block's text stream, beside image attention Qwen-Image-2.1 has: applying only the part that
    # fits would pass a partial LoRA off as a whole one.
    sd = _pair("transformer.transformer_blocks.0.attn.to_q", 6, 6) | _pair(
        "transformer.transformer_blocks.0.attn.add_q_proj", 6, 6
    )
    with pytest.raises(ValueError, match="add_q_proj"):
        convert(sd)
