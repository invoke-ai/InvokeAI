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


_TO_Q = "diffusion_model.transformer_blocks.0.attn.to_q"


def _update(layer, weight: torch.Tensor) -> torch.Tensor:
    """The weight update a layer adds to `weight` at strength 1, as the patcher asks it for one."""
    return layer.get_parameters({"weight": weight}, weight=1.0)["weight"]


@pytest.mark.parametrize("kind", ["lokr", "lokr-factored", "lokr-factored-w1", "loha", "full"])
def test_a_layer_of_any_kind_converts_to_the_update_it_defines(kind: str) -> None:
    # On a 6x6 projection. LoKR is kron(w1, w2); LoHA the elementwise product of two low-rank products; a full layer
    # the update itself. An alpha scales by alpha / rank, the rank being the inner size of a factored operand.
    generator = torch.Generator().manual_seed(0)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    w1, w2, w2_a, w2_b = randn(2, 2), randn(3, 3), randn(3, 1), randn(1, 3)
    w1_a, w1_b = randn(2, 1), randn(1, 2)
    h1_a, h1_b, h2_a, h2_b = randn(6, 2), randn(2, 6), randn(6, 2), randn(2, 6)
    diff = randn(6, 6)
    values, expected = {
        "lokr": ({"lokr_w1": w1, "lokr_w2": w2}, torch.kron(w1, w2)),
        "lokr-factored": (
            {"lokr_w1": w1, "lokr_w2_a": w2_a, "lokr_w2_b": w2_b, "alpha": torch.tensor(0.5)},
            torch.kron(w1, w2_a @ w2_b) * 0.5,
        ),
        "lokr-factored-w1": (
            {"lokr_w1_a": w1_a, "lokr_w1_b": w1_b, "lokr_w2": w2, "alpha": torch.tensor(0.5)},
            torch.kron(w1_a @ w1_b, w2) * 0.5,
        ),
        "loha": (
            {"hada_w1_a": h1_a, "hada_w1_b": h1_b, "hada_w2_a": h2_a, "hada_w2_b": h2_b, "alpha": torch.tensor(1.0)},
            (h1_a @ h1_b) * (h2_a @ h2_b) * 0.5,
        ),
        "full": ({"diff": diff}, diff),
    }[kind]

    layers = convert({f"{_TO_Q}.{name}": value for name, value in values.items()}).layers
    assert list(layers) == [f"{PREFIX}transformer_blocks.0.attn.to_q"]
    torch.testing.assert_close(_update(layers[f"{PREFIX}transformer_blocks.0.attn.to_q"], torch.zeros(6, 6)), expected)


_INCOMPLETE_OR_MIXED = {
    "alpha-only": ["alpha"],
    "magnitude-only": ["lora_magnitude_vector.weight"],
    "half-pair": ["lora_A.weight"],
    "both-magnitudes": ["lora_A.weight", "lora_B.weight", "dora_scale", "lora_magnitude_vector.weight"],
    "lokr-half-operand": ["lokr_w1", "lokr_w2_a"],
    "lokr-operand-twice": ["lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2"],
    # Tucker cores, which LyCORIS writes for convolution kernels only.
    "lokr-tucker-core": ["lokr_w1", "lokr_w2_a", "lokr_w2_b", "lokr_t2"],
    "loha-tucker-cores": ["hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "hada_t1", "hada_t2"],
    "loha-three-of-four": ["hada_w1_a", "hada_w1_b", "hada_w2_a"],
    "diff-plus-pair": ["diff", "lora_A.weight", "lora_B.weight"],
    # A full layer has no rank to scale by: the alpha would be ignored.
    "diff-with-alpha": ["diff", "alpha"],
}


@pytest.mark.parametrize("params", _INCOMPLETE_OR_MIXED.values(), ids=_INCOMPLETE_OR_MIXED.keys())
def test_a_module_that_is_not_exactly_one_layer_is_refused(params: list[str]) -> None:
    # Beside a complete layer on to_k, so it is the to_q module that is refused, not the file for being empty.
    sd = _pair("diffusion_model.transformer_blocks.0.attn.to_k", 6, 6)
    sd |= {f"{_TO_Q}.{param}": torch.ones(()) for param in params}
    with pytest.raises(ValueError, match=r"attn\.to_q"):
        convert(sd)


@pytest.mark.parametrize(
    "spellings",
    [
        _pair(_TO_Q, 6, 6) | _pair("transformer.transformer_blocks.0.attn.to_q", 6, 6),
        _pair(_TO_Q, 6, 6) | _pair("lora_unet_transformer_blocks_0_attn_to_q", 6, 6, a="lora_down", b="lora_up"),
        _pair(_TO_Q, 6, 6) | {f"{_TO_Q}.lora_down.weight": torch.zeros(RANK, 6)},
        _pair(_TO_Q, 6, 6)
        | {f"{_TO_Q}.lora_magnitude_vector.weight": torch.ones(6), f"{_TO_Q}.magnitude": torch.ones(6)},
    ],
    ids=["two-prefixes", "kohya-and-dotted", "lora_A-and-lora_down", "two-magnitude-spellings"],
)
def test_a_value_given_twice_is_refused(spellings: dict) -> None:
    # Which of the two would be applied would come down to the order of the file's keys.
    with pytest.raises(ValueError, match="to_q"):
        convert(spellings)


@pytest.mark.parametrize("half", ["gate_layer", "proj"])
def test_a_gate_up_beside_the_half_it_is_split_into_is_refused(half: str) -> None:
    # The split would overwrite the layer the file names directly.
    gate_up = _pair("diffusion_model.transformer_blocks.0.img_mlp.gate_up", 8, 6)
    with pytest.raises(ValueError, match="gate_up"):
        convert(gate_up | _pair(f"diffusion_model.transformer_blocks.0.img_mlp.{half}", 4, 6))
    # Another block's halves are not in the way.
    layers = convert(gate_up | _pair(f"diffusion_model.transformer_blocks.1.img_mlp.{half}", 4, 6)).layers
    assert set(layers) == {
        f"{PREFIX}transformer_blocks.0.img_mlp.gate_layer",
        f"{PREFIX}transformer_blocks.0.img_mlp.proj",
        f"{PREFIX}transformer_blocks.1.img_mlp.{half}",
    }


@pytest.mark.parametrize(
    "gate_up_layer",
    [_pair("m", 7, 6), {"m.diff": torch.zeros(7, 6)}, _pair("m", 8, 6) | {"m.magnitude": torch.ones(7)}],
    ids=["low-rank", "full", "magnitude"],
)
def test_a_gate_up_layer_with_an_odd_number_of_rows_is_refused(gate_up_layer: dict) -> None:
    # gate_layer and proj are the two halves of its rows; 7 has none.
    module = "diffusion_model.transformer_blocks.0.img_mlp.gate_up"
    with pytest.raises(ValueError, match="gate_up"):
        convert({f"{module}.{key.removeprefix('m.')}": value for key, value in gate_up_layer.items()})


@pytest.mark.parametrize("block", ["32", "40", "99", "01", "00"])
@pytest.mark.parametrize("layout", ["dotted", "kohya"])
def test_a_block_the_transformer_does_not_have_is_refused(block: str, layout: str) -> None:
    # Its 32 blocks, numbered without leading zeros: `01` would be block 1 under a second name, and a gate_up in
    # block `0` would not collide with the gate_layer of a block `00`.
    if layout == "dotted":
        sd = _pair(f"diffusion_model.transformer_blocks.{block}.attn.to_q", 6, 6)
    else:
        sd = _pair(f"lora_unet_transformer_blocks_{block}_attn_to_q", 6, 6, a="lora_down", b="lora_up")
    with pytest.raises(ValueError, match=f"transformer_blocks[._]{block}[._]"):
        convert(sd)


def test_the_last_block_is_the_thirty_second() -> None:
    sd = _pair("diffusion_model.transformer_blocks.31.attn.to_q", 6, 6)
    assert list(convert(sd).layers) == [f"{PREFIX}transformer_blocks.31.attn.to_q"]
