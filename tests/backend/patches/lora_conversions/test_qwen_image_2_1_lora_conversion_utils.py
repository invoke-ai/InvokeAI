"""Qwen-Image-2.1 LoRAs onto the diffusers transformer: every layout, and ComfyUI's fused gate_up split in two.

The expected weight shapes are the transformer's own, built from diffusers' `QwenImage21Transformer2DModel` at its
defaults, which are the released configuration: hidden size 4096, MLP 12288, 64 latent channels.
"""

import re

import accelerate
import pytest
import torch

from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.patches.layers.full_layer import FullLayer
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
# Every Linear a LoRA may name: the block's, diffusers' two MLP input halves, ComfyUI's fused gate_up, the top level's.
LINEARS = (
    *(f"transformer_blocks.0.{m}" for m in (*BLOCK_MODULES, "img_mlp.gate_layer", "img_mlp.proj")),
    *TOP_MODULES,
)
KINDS = ("low-rank", "peft-dora", "lycoris-dora", "lokr", "lokr-factored", "loha", "full")
# The kinds the fused gate_up splits by its rows; the others mix them.
ROW_SPLITTABLE = ("low-rank", "peft-dora", "full")


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


def _z(*shape: int) -> torch.Tensor:
    """Zeros in one element of storage: these tests read shapes, not values."""
    return torch.zeros(()).expand(shape)


def _layer(kind: str, out_features: int, in_features: int) -> dict[str, torch.Tensor]:
    """One layer of the kind, updating an `out_features x in_features` weight, by parameter name."""
    low_rank = {"lora_A.weight": _z(RANK, in_features), "lora_B.weight": _z(out_features, RANK)}
    # LoKR's update is kron(w1, w2): with a 4x4 w1, w2 is a quarter of each side.
    w2_out, w2_in = out_features // 4, in_features // 4
    return {
        "low-rank": low_rank,
        "peft-dora": low_rank | {"lora_magnitude_vector.weight": _z(out_features)},
        "lycoris-dora": low_rank | {"dora_scale": _z(1, in_features)},
        "lokr": {"lokr_w1": _z(4, 4), "lokr_w2": _z(w2_out, w2_in)},
        "lokr-factored": {"lokr_w1": _z(4, 4), "lokr_w2_a": _z(w2_out, RANK), "lokr_w2_b": _z(RANK, w2_in)},
        "loha": {
            name: _z(out_features, RANK) if name.endswith("_a") else _z(RANK, in_features)
            for name in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")
        },
        "full": {"diff": _z(out_features, in_features)},
    }[kind]


def _named(module_key: str, layer: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {f"{module_key}.{name}": value for name, value in layer.items()}


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


@pytest.mark.parametrize("kind", KINDS)
def test_a_layer_of_any_kind_converts_on_every_module_of_its_shape(transformer, kind: str) -> None:
    mlp_input = ("img_mlp.gate_up",) if kind in ROW_SPLITTABLE else ("img_mlp.gate_layer", "img_mlp.proj")
    block = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", *mlp_input, "img_mlp.out")
    modules = [f"transformer_blocks.0.{m}" for m in block] + list(TOP_MODULES)
    sd: dict = {}
    for module in modules:
        sd |= _named(f"diffusion_model.{module}", _layer(kind, *_shape(transformer, module)))

    targets = {m for m in modules if not m.endswith("gate_up")}
    if "img_mlp.gate_up" in mlp_input:
        targets |= {"transformer_blocks.0.img_mlp.gate_layer", "transformer_blocks.0.img_mlp.proj"}
    assert set(convert(sd).layers) == {PREFIX + target for target in targets}


def _widened(shape: tuple[int, int], side: str) -> tuple[int, int]:
    out_features, in_features = shape
    return (2 * out_features, in_features) if side == "out" else (out_features, 2 * in_features)


# A layer that fits, in a block of its own, beside the one under test: the refusal names the module that does not.
_FITTING = _named("diffusion_model.transformer_blocks.1.attn.to_k", _layer("low-rank", 4096, 4096))


@pytest.mark.parametrize("side", ["out", "in"])
@pytest.mark.parametrize("module", LINEARS)
def test_a_layer_wider_than_its_module_is_refused(transformer, module: str, side: str) -> None:
    sd = _FITTING | _named(
        f"diffusion_model.{module}", _layer("low-rank", *_widened(_shape(transformer, module), side))
    )
    with pytest.raises(ValueError, match=re.escape(f"diffusion_model.{module}.")):
        convert(sd)


@pytest.mark.parametrize("side", ["out", "in"])
@pytest.mark.parametrize("kind", KINDS)
def test_a_layer_of_any_kind_on_another_shape_is_refused(kind: str, side: str) -> None:
    # On `proj_out`, 64 x 4096 here. Widened on the output side it is FLUX.2 Klein 9B's, with 128 rows: the names
    # match, and a LoRA on it would fail only at generation.
    sd = _FITTING | _named("diffusion_model.proj_out", _layer(kind, *_widened((64, 4096), side)))
    with pytest.raises(ValueError, match=r"diffusion_model\.proj_out\."):
        convert(sd)


@pytest.mark.parametrize(
    ("magnitude", "fits"),
    [
        ({"lora_magnitude_vector.weight": _z(4096)}, True),
        ({"magnitude": _z(4096, 1)}, True),
        ({"dora_scale": _z(4096, 1)}, True),
        ({"dora_scale": _z(1, 64)}, True),
        ({"dora_scale": _z(64)}, True),
        ({"lora_magnitude_vector.weight": _z(64)}, False),
        ({"dora_scale": _z(1, 4096)}, False),
        ({"dora_scale": _z(64, 1)}, False),
        # A 1-D `dora_scale` has no axis to tell its orientation by, and is read per column.
        ({"dora_scale": _z(4096)}, False),
    ],
    ids=[
        "peft-per-row",
        "ai-toolkit-per-row",
        "lycoris-per-row",
        "lycoris-per-column",
        "lycoris-1d-per-column",
        "peft-of-input-length",
        "lycoris-row-of-output-length",
        "lycoris-column-of-input-length",
        "lycoris-1d-of-output-length",
    ],
)
def test_a_dora_magnitude_must_index_its_weights_rows_or_columns(magnitude: dict, fits: bool) -> None:
    # On img_in, 4096 rows of 64 columns, so the two orientations differ. PEFT's and ai-toolkit's magnitude has an
    # entry per row; LyCORIS's `dora_scale` one per row as a column, or per column as a row or a vector.
    sd = _named("diffusion_model.img_in", _layer("low-rank", 4096, 64) | magnitude)
    if not fits:
        with pytest.raises(ValueError, match="img_in"):
            convert(sd)
        return
    layer = convert(sd).layers[f"{PREFIX}img_in"]
    assert isinstance(layer, DoRALayer)
    # What the check accepts, the layer applies: an update of the weight's own shape, no broadcast error.
    weight = torch.ones(4096, 64)
    assert layer.get_parameters({"weight": weight}, weight=1.0)["weight"].shape == weight.shape


@pytest.mark.parametrize(
    "layer",
    [
        {"lora_A.weight": _z(8, 4096), "lora_B.weight": _z(4096, RANK)},
        {"lora_A.weight": _z(RANK, 4096, 1, 1), "lora_B.weight": _z(4096, RANK, 1, 1)},
        {
            "hada_w1_a": _z(4096, RANK),
            "hada_w1_b": _z(RANK, 4096),
            "hada_w2_a": _z(4096, RANK),
            "hada_w2_b": _z(RANK, 2048),
        },
        {"lokr_w1": _z(4, 4), "lokr_w2_a": _z(1024, 8), "lokr_w2_b": _z(RANK, 1024)},
        {"lokr_w1": _z(4, 4, 1, 1), "lokr_w2": _z(1024, 1024)},
    ],
    ids=["low-rank-ranks-differ", "convolution-shaped", "loha-products-differ", "lokr-ranks-differ", "lokr-4d-operand"],
)
def test_a_layer_whose_factors_do_not_compose_is_refused(layer: dict) -> None:
    # On attn.to_q, 4096 x 4096: each of these has the module's sizes somewhere, but makes no 4096 x 4096 update.
    with pytest.raises(ValueError, match="attn.to_q"):
        convert(_FITTING | _named("diffusion_model.transformer_blocks.0.attn.to_q", layer))


_GATE_UP = "diffusion_model.transformer_blocks.0.img_mlp.gate_up"


def test_gate_up_splits_into_gate_layer_then_proj_exactly() -> None:
    # 24576 output rows: gate_layer's 12288, then proj's.
    sd = _pair(_GATE_UP, 24576, 4096) | {f"{_GATE_UP}.alpha": torch.tensor(2.0)}
    up, down = sd[f"{_GATE_UP}.lora_B.weight"], sd[f"{_GATE_UP}.lora_A.weight"]

    layers = convert(sd).layers
    gate = layers[f"{PREFIX}transformer_blocks.0.img_mlp.gate_layer"]
    proj = layers[f"{PREFIX}transformer_blocks.0.img_mlp.proj"]
    assert torch.equal(gate.up, up[:12288]) and torch.equal(proj.up, up[12288:])
    assert torch.equal(gate.down, down) and torch.equal(proj.down, down)
    assert gate.scale() == proj.scale() == 2.0 / RANK


@pytest.mark.parametrize("magnitude_key", ["lora_magnitude_vector.weight", "lora_magnitude_vector", "magnitude"])
def test_an_output_dim_dora_on_gate_up_splits_its_magnitude_with_the_rows(magnitude_key: str) -> None:
    # PEFT writes the magnitude with or without `.weight`, ai-toolkit as `.magnitude`: all one per output row.
    module = "transformer.transformer_blocks.0.img_mlp.gate_up"
    magnitude = torch.arange(24576.0)
    sd = _pair(module, 24576, 4096) | {f"{module}.{magnitude_key}": magnitude}

    layers = convert(sd).layers
    gate, proj = (layers[f"{PREFIX}transformer_blocks.0.img_mlp.{half}"] for half in ("gate_layer", "proj"))
    assert isinstance(gate, DoRALayer) and isinstance(proj, DoRALayer)
    assert gate.magnitude_is_out_dim and proj.magnitude_is_out_dim
    assert torch.equal(gate.dora_scale, magnitude[:12288]) and torch.equal(proj.dora_scale, magnitude[12288:])


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


@pytest.mark.parametrize("norm", ["transformer_blocks.0.attn.norm_q", "txt_in.text_norm"])
def test_a_norm_diff_of_another_width_is_refused(transformer, norm: str) -> None:
    (width,) = transformer.get_submodule(norm).weight.shape
    with pytest.raises(ValueError, match=re.escape(norm)):
        convert(_FITTING | {f"diffusion_model.{norm}.diff": _z(2 * width)})


def test_a_full_layer_on_gate_up_splits_by_rows() -> None:
    # Each row holds its own index, so the halves show which rows they took.
    diff = torch.arange(24576.0).unsqueeze(1).expand(24576, 4096)
    layers = convert({f"{_GATE_UP}.diff": diff}).layers
    gate, proj = (layers[f"{PREFIX}transformer_blocks.0.img_mlp.{half}"] for half in ("gate_layer", "proj"))
    assert isinstance(gate, FullLayer) and isinstance(proj, FullLayer)
    assert gate.weight.shape == proj.weight.shape == (12288, 4096)
    assert torch.equal(gate.weight[:, 0], diff[:12288, 0]) and torch.equal(proj.weight[:, 0], diff[12288:, 0])


@pytest.mark.parametrize(
    "prefix", ["diffusion_model.", "transformer.", "base_model.model.transformer.", "base_model.model."]
)
def test_every_dotted_prefix_names_the_same_layer(prefix: str) -> None:
    sd = _pair(f"{prefix}transformer_blocks.3.attn.to_out.0", 4096, 4096)
    assert list(convert(sd).layers) == [f"{PREFIX}transformer_blocks.3.attn.to_out.0"]


_TO_Q = "diffusion_model.transformer_blocks.0.attn.to_q"


def test_a_layers_own_alpha_scales_it_and_a_missing_one_means_its_rank() -> None:
    sd = _pair(_TO_Q, 4096, 4096) | _pair("diffusion_model.transformer_blocks.0.attn.to_k", 4096, 4096)
    sd[f"{_TO_Q}.alpha"] = torch.tensor(2.0)
    layers = convert(sd).layers
    assert layers[f"{PREFIX}transformer_blocks.0.attn.to_q"].scale() == 2.0 / RANK
    assert layers[f"{PREFIX}transformer_blocks.0.attn.to_k"].scale() == 1.0


@pytest.mark.parametrize(
    "gate_up_layer",
    [
        {"lokr_w1": _z(4, 4), "lokr_w2": _z(6144, 1024)},
        # LyCORIS's input-dim magnitude normalizes across both halves at once.
        _layer("lycoris-dora", 24576, 4096),
    ],
    ids=["lokr", "lycoris-dora"],
)
def test_a_gate_up_layer_that_cannot_be_split_is_refused(gate_up_layer: dict) -> None:
    with pytest.raises(ValueError, match="gate_up"):
        convert(_named(_GATE_UP, gate_up_layer))


def test_a_low_rank_layer_on_a_norm_is_refused() -> None:
    # A norm has no matrix to factor; only a full diff of its weight applies.
    with pytest.raises(ValueError, match="norm_q"):
        convert(_pair("diffusion_model.transformer_blocks.0.attn.norm_q", 128, 128))


def test_a_lora_with_layers_this_model_lacks_is_refused_whole() -> None:
    # A FLUX.2 double block's text stream, beside image attention Qwen-Image-2.1 has: applying only the part that
    # fits would pass a partial LoRA off as a whole one.
    sd = _pair("transformer.transformer_blocks.0.attn.to_q", 4096, 4096) | _pair(
        "transformer.transformer_blocks.0.attn.add_q_proj", 4096, 4096
    )
    with pytest.raises(ValueError, match="add_q_proj"):
        convert(sd)


def _update(layer, weight: torch.Tensor) -> torch.Tensor:
    """The weight update a layer adds to `weight` at strength 1, as the patcher asks it for one."""
    return layer.get_parameters({"weight": weight}, weight=1.0)["weight"]


@pytest.mark.parametrize("kind", ["lokr", "lokr-factored", "lokr-factored-w1", "loha", "full"])
def test_a_layer_of_any_kind_converts_to_the_update_it_defines(kind: str) -> None:
    # On img_in, 4096 x 64, the narrowest Linear. LoKR is kron(w1, w2); LoHA the elementwise product of two low-rank
    # products; a full layer the update itself. An alpha scales by alpha / rank, the rank being the inner size of a
    # factored operand.
    generator = torch.Generator().manual_seed(0)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    w1, w2, w2_a, w2_b = randn(2, 2), randn(2048, 32), randn(2048, 1), randn(1, 32)
    w1_a, w1_b = randn(2, 1), randn(1, 2)
    h1_a, h1_b, h2_a, h2_b = randn(4096, 2), randn(2, 64), randn(4096, 2), randn(2, 64)
    diff = randn(4096, 64)
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

    layers = convert(_named("diffusion_model.img_in", values)).layers
    assert list(layers) == [f"{PREFIX}img_in"]
    torch.testing.assert_close(_update(layers[f"{PREFIX}img_in"], torch.zeros(4096, 64)), expected)


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
    sd = _pair("diffusion_model.transformer_blocks.0.attn.to_k", 4096, 4096)
    sd |= {f"{_TO_Q}.{param}": torch.ones(()) for param in params}
    with pytest.raises(ValueError, match=r"attn\.to_q"):
        convert(sd)


@pytest.mark.parametrize(
    "spellings",
    [
        _pair(_TO_Q, 4096, 4096) | _pair("transformer.transformer_blocks.0.attn.to_q", 4096, 4096),
        _pair(_TO_Q, 4096, 4096)
        | _pair("lora_unet_transformer_blocks_0_attn_to_q", 4096, 4096, a="lora_down", b="lora_up"),
        _pair(_TO_Q, 4096, 4096) | {f"{_TO_Q}.lora_down.weight": torch.zeros(RANK, 4096)},
        _pair(_TO_Q, 4096, 4096)
        | {f"{_TO_Q}.lora_magnitude_vector.weight": torch.ones(4096), f"{_TO_Q}.magnitude": torch.ones(4096)},
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
    gate_up = _pair(_GATE_UP, 24576, 4096)
    with pytest.raises(ValueError, match="gate_up"):
        convert(gate_up | _pair(f"diffusion_model.transformer_blocks.0.img_mlp.{half}", 12288, 4096))
    # Another block's halves are not in the way.
    layers = convert(gate_up | _pair(f"diffusion_model.transformer_blocks.1.img_mlp.{half}", 12288, 4096)).layers
    assert set(layers) == {
        f"{PREFIX}transformer_blocks.0.img_mlp.gate_layer",
        f"{PREFIX}transformer_blocks.0.img_mlp.proj",
        f"{PREFIX}transformer_blocks.1.img_mlp.{half}",
    }


@pytest.mark.parametrize(
    "gate_up_layer",
    [
        _layer("low-rank", 24575, 4096),
        _layer("full", 24575, 4096),
        # Even, but one half's rows: the split would leave gate_layer and proj 6144 rows each.
        _layer("low-rank", 12288, 4096),
        _layer("low-rank", 24576, 4096) | {"magnitude": _z(12288)},
    ],
    ids=["odd-low-rank", "odd-full", "one-halfs-rows", "magnitude-of-one-half"],
)
def test_a_gate_up_layer_without_both_halves_rows_is_refused(gate_up_layer: dict) -> None:
    # gate_layer and proj are the two halves of its 24576 rows.
    with pytest.raises(ValueError, match="gate_up"):
        convert(_named(_GATE_UP, gate_up_layer))


@pytest.mark.parametrize("block", ["32", "40", "99", "01", "00"])
@pytest.mark.parametrize("layout", ["dotted", "kohya"])
def test_a_block_the_transformer_does_not_have_is_refused(block: str, layout: str) -> None:
    # Its 32 blocks, numbered without leading zeros: `01` would be block 1 under a second name, and a gate_up in
    # block `0` would not collide with the gate_layer of a block `00`.
    if layout == "dotted":
        sd = _pair(f"diffusion_model.transformer_blocks.{block}.attn.to_q", 4096, 4096)
    else:
        sd = _pair(f"lora_unet_transformer_blocks_{block}_attn_to_q", 4096, 4096, a="lora_down", b="lora_up")
    with pytest.raises(ValueError, match=f"transformer_blocks[._]{block}[._]"):
        convert(sd)


def test_the_last_block_is_the_thirty_second() -> None:
    sd = _pair("diffusion_model.transformer_blocks.31.attn.to_q", 4096, 4096)
    assert list(convert(sd).layers) == [f"{PREFIX}transformer_blocks.31.attn.to_q"]
