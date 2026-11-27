"""FLUX.3 LoRAs onto diffusers' `Flux3Transformer2DModel`.

Each published key layout converts to a patch that changes the transformer as merging the LoRA into its weights by
hand does, `W + weight * (alpha / rank) * B @ A` in float32, with the halves of the update swapped on BFL's output
modulation, whose rows diffusers holds as [scale, shift] where BFL writes [shift, scale]. The transformer is the tiny one
of `flux3_fixture`, every weight random: diffusers zero-initializes the output and modulation layers, where a mis-wired
update would pass unnoticed. The conversion maps names only, so a LoRA sized to the tiny transformer converts as a
released one does; the captured headers of published LoRAs then land on Linears of the released geometry.

On the CPU the patcher runs every layer as a sidecar unless told otherwise; the tests force the direct path where they
say so.
"""

import copy
import json
import logging
import re
from pathlib import Path
from unittest.mock import MagicMock

import accelerate
import gguf
import pytest
import torch
from diffusers import Flux3Transformer2DModel
from gguf.quants import quantize
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_Flux3_Config, LoRA_LyCORIS_FLUX_Config
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders.lora import LoRALoader
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.patches.layers.full_layer import FullLayer
from invokeai.backend.patches.layers.loha_layer import LoHALayer
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.layers.swapped_halves_layer_patch import SwappedHalvesLayerPatch
from invokeai.backend.patches.lora_conversions import flux3_lora_constants as constants
from invokeai.backend.patches.lora_conversions.flux3_lora_constants import FLUX3_LORA_TRANSFORMER_PREFIX as PREFIX
from invokeai.backend.patches.lora_conversions.flux3_lora_conversion_utils import lora_model_from_flux3_state_dict
from invokeai.backend.patches.lora_conversions.peft_adapter_utils import normalize_peft_adapter_names
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear
from tests.backend.flux3.flux3_fixture import forward, tiny_flux3_model
from tests.backend.patches.lora_conversions.lora_state_dicts import (
    flux3_lora_aitoolkit_format,
    flux3_lora_comfy_alpha_format,
    flux3_lora_diffusers_format,
    flux3_lora_peft_adapter_format,
)

RANK = 4
# A LoRA's own alpha, in the layout that carries one: half its rank, so a scale of 1 would show.
ALPHA = 2.0
PATCH_WEIGHT = 0.8
LORA_NAME = "Turbo"


def convert(state_dict: dict[str, torch.Tensor], metadata: dict[str, str] | None = None):
    return lora_model_from_flux3_state_dict(state_dict, name=LORA_NAME, metadata=metadata)


# BFL's name of every module kind, written out from the released checkpoint's layout (`flux3_klein_keys`) rather than
# derived from the conversion under test: the diffusers stack and leaf, and BFL's for them.
_NATIVE_STACKS = {
    "mode_blocks.image": "content_mode_blocks.image",
    "mode_blocks.image_cond": "content_mode_blocks.image_cond",
    "txt_mode_blocks": "txt_mode_blocks",
    "single_transformer_blocks": "single_blocks",
}
_NATIVE_LEAVES = {
    "attn.to_q": "q_proj",
    "attn.to_k": "k_proj",
    "attn.to_v": "v_proj",
    "attn.to_out.0": "attn_out",
    "ff.linear_in": "mlp_in",
    "ff.linear_out": "mlp_out",
}
_NATIVE_TOP = {
    "x_embedders.image": "emb_in.image",
    "x_embedders.image_cond": "emb_in.image_cond",
    "context_embedder": "txt_in",
    "timestep_embedder.linear_1": "time_in.in_layer",
    "timestep_embedder.linear_2": "time_in.out_layer",
    "target_timestep_embedder.linear_1": "ref_time_in.in_layer",
    "target_timestep_embedder.linear_2": "ref_time_in.out_layer",
    **{
        f"{stage}.{mode}.linear": f"{stage}.{mode}.lin"
        for stage in ("early_stream_modulations", "single_stream_modulations")
        for mode in ("image", "image_cond", "txt")
    },
    "norm_out.linear": "final_layer.image.adaLN_modulation.1",
    "proj_out": "final_layer.image.linear",
}
_OUTPUT_MODULATION = "norm_out.linear"


def _native(path: str) -> str:
    for stack, native_stack in _NATIVE_STACKS.items():
        if path.startswith(f"{stack}."):
            index, leaf = path[len(stack) + 1 :].split(".", 1)
            return f"{native_stack}.{index}.{_NATIVE_LEAVES[leaf]}"
    return _NATIVE_TOP[path]


def _linears(model: torch.nn.Module) -> dict[str, torch.nn.Linear]:
    return {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}


def _swap_halves(t: torch.Tensor) -> torch.Tensor:
    first, second = t.chunk(2, dim=0)
    return torch.cat([second, first])


# Each layout's keys for one module's low-rank pair, from its diffusers path. The alpha-carrying one is ComfyUI's form.
_LAYOUTS = {
    "diffusers": lambda path, a, b: {f"transformer.{path}.lora_A.weight": a, f"transformer.{path}.lora_B.weight": b},
    "native": lambda path, a, b: {
        f"diffusion_model.{_native(path)}.lora_A.weight": a,
        f"diffusion_model.{_native(path)}.lora_B.weight": b,
    },
    "native-alpha": lambda path, a, b: {
        f"diffusion_model.{_native(path)}.lora_down.weight": a,
        f"diffusion_model.{_native(path)}.lora_up.weight": b,
        f"diffusion_model.{_native(path)}.alpha": torch.tensor(ALPHA),
    },
    "peft-adapter": lambda path, a, b: {f"{path}.lora_A.default.weight": a, f"{path}.lora_B.default.weight": b},
}


def _lora_on_every_linear(model: torch.nn.Module, layout: str) -> tuple[dict[str, torch.Tensor], dict[str, tuple]]:
    """A LoRA on every Linear of `model` in `layout`, and each module's (A, B) pair."""
    generator = torch.Generator().manual_seed(7)
    pairs = {
        path: (
            torch.randn(RANK, linear.in_features, generator=generator) * 0.05,
            torch.randn(linear.out_features, RANK, generator=generator) * 0.05,
        )
        for path, linear in _linears(model).items()
    }
    state_dict: dict[str, torch.Tensor] = {}
    for path, (a, b) in pairs.items():
        state_dict |= _LAYOUTS[layout](path, a, b)
    return state_dict, pairs


def _merged(model: torch.nn.Module, pairs: dict[str, tuple], layout: str) -> torch.nn.Module:
    """`model` with the LoRA merged into its weights by hand."""
    reference = copy.deepcopy(model)
    scale = ALPHA / RANK if layout == "native-alpha" else 1.0
    with torch.no_grad():
        for path, (a, b) in pairs.items():
            update = b @ a
            if path == _OUTPUT_MODULATION and layout.startswith("native"):
                update = _swap_halves(update)
            reference.get_submodule(path).weight += PATCH_WEIGHT * scale * update
    return reference


@pytest.mark.parametrize("direct", [True, False], ids=["direct", "sidecar"])
@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_a_lora_changes_the_transformer_as_merging_it_by_hand_does(layout: str, direct: bool) -> None:
    model = tiny_flux3_model(flow_map=True)
    state_dict, pairs = _lora_on_every_linear(model, layout)
    reference = _merged(model, pairs, layout)
    # As the model cache holds it, so the patcher can wrap a layer in a sidecar.
    apply_custom_layers_to_model(model)
    # As identification and the loader read the file: PEFT's adapter name stripped.
    patch = convert(normalize_peft_adapter_names(state_dict))
    unpatched = forward(model, flow_map=True)

    assert len(patch.layers) == len(_linears(model)) == len(pairs)
    with LayerPatcher.apply_smart_model_patches(
        model=model, patches=[(patch, PATCH_WEIGHT)], prefix=PREFIX, dtype=torch.float32, force_direct_patching=direct
    ):
        patched = forward(model, flow_map=True)

    expected = forward(reference, flow_map=True)
    assert not torch.allclose(expected, unpatched, atol=1e-3)
    torch.testing.assert_close(patched, expected, rtol=1e-4, atol=1e-5)
    # Taken back out again.
    torch.testing.assert_close(forward(model, flow_map=True), unpatched, rtol=0, atol=0)


def _int8(linear: torch.nn.Linear) -> tuple[Int8ConvrotLinear, torch.Tensor]:
    """`linear` as an int8 Linear, and the weight it dequantizes to."""
    weight = linear.weight.detach()
    scale = weight.abs().amax(dim=1, keepdim=True) / 127
    codes = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
    return Int8ConvrotLinear(codes, scale, convrot=False), codes.float() * scale


def _gguf_q8_0(linear: torch.nn.Linear) -> torch.Tensor:
    """Store `linear`'s weight packed as GGUF Q8_0, as the GGUF loader leaves it; returns the weight it unpacks to."""
    weight = linear.weight.detach().numpy()
    raw = quantize(weight, gguf.GGMLQuantizationType.Q8_0)
    packed = GGMLTensor(torch.from_numpy(raw), gguf.GGMLQuantizationType.Q8_0, torch.Size(weight.shape), torch.float32)
    linear.weight = torch.nn.Parameter(packed, requires_grad=False)
    return packed.get_dequantized_tensor()


def test_a_lora_on_quantized_layers_runs_beside_them_as_a_sidecar() -> None:
    """An int8 Linear keeps its weight in buffers and a GGUF one packed, so the denoise forces the sidecar path
    (`requires_sidecar_patching`). The output modulation's swapped halves apply there too, against a weight the int8
    Linear hands over by shape only."""
    model = tiny_flux3_model(flow_map=True)
    state_dict, pairs = _lora_on_every_linear(model, "native")
    dequantized: dict[str, torch.Tensor] = {}
    for path in ("single_transformer_blocks.0.attn.to_q", _OUTPUT_MODULATION):
        parent, name = path.rsplit(".", 1)
        int8, dequantized[path] = _int8(model.get_submodule(path))
        setattr(model.get_submodule(parent), name, int8)
    gguf_path = "mode_blocks.image.0.ff.linear_in"
    dequantized[gguf_path] = _gguf_q8_0(model.get_submodule(gguf_path))
    # The reference holds the weights the quantized layers dequantize to, plus the LoRA.
    plain = tiny_flux3_model(flow_map=True)
    with torch.no_grad():
        for path, weight in dequantized.items():
            plain.get_submodule(path).weight.copy_(weight)
    reference = _merged(plain, pairs, "native")
    apply_custom_layers_to_model(model)

    with LayerPatcher.apply_smart_model_patches(
        model=model,
        patches=[(convert(state_dict), PATCH_WEIGHT)],
        prefix=PREFIX,
        dtype=torch.float32,
        force_sidecar_patching=True,
    ):
        patched = forward(model, flow_map=True)

    torch.testing.assert_close(patched, forward(reference, flow_map=True), rtol=1e-4, atol=1e-5)


def _z(*shape: int) -> torch.Tensor:
    """Zeros in one element of storage: these tests read shapes, not values."""
    return torch.zeros(()).expand(shape)


_TO_Q = "single_transformer_blocks.0.attn.to_q"
# Every kind on one 256 x 256 Linear of the tiny transformer: LoKR's update is kron(w1, w2), a 4 x 4 w1 and w2 a quarter
# of each side, factored here.
_KINDS = {
    "low-rank": ({"lora_A.weight": _z(RANK, 256), "lora_B.weight": _z(256, RANK)}, LoRALayer),
    "peft-dora": (
        {"lora_A.weight": _z(RANK, 256), "lora_B.weight": _z(256, RANK), "lora_magnitude_vector.weight": _z(256)},
        DoRALayer,
    ),
    "lycoris-dora": (
        {"lora_down.weight": _z(RANK, 256), "lora_up.weight": _z(256, RANK), "dora_scale": _z(1, 256)},
        DoRALayer,
    ),
    "lokr": ({"lokr_w1": _z(4, 4), "lokr_w2_a": _z(64, RANK), "lokr_w2_b": _z(RANK, 64), "alpha": _z()}, LoKRLayer),
    "loha": (
        {
            "hada_w1_a": _z(256, RANK),
            "hada_w1_b": _z(RANK, 256),
            "hada_w2_a": _z(256, RANK),
            "hada_w2_b": _z(RANK, 256),
        },
        LoHALayer,
    ),
    "full": ({"diff": _z(256, 256)}, FullLayer),
}


@pytest.mark.parametrize("kind", list(_KINDS))
def test_each_layer_kind_converts_to_its_layer(kind: str) -> None:
    values, layer_class = _KINDS[kind]
    patch = convert({f"diffusion_model.single_blocks.0.q_proj.{name}": value for name, value in values.items()})
    layer = patch.layers[PREFIX + _TO_Q]
    assert type(layer) is layer_class
    if kind.endswith("dora"):
        # PEFT's magnitude is one per output row; LyCORIS's (1, in) one per input column.
        assert layer.magnitude_is_out_dim is (kind == "peft-dora")
    # On BFL's output modulation the same layer comes wrapped, to swap its halves.
    wrapped = convert({f"final_layer.image.adaLN_modulation.1.{name}": value for name, value in values.items()})
    swapped = wrapped.layers[PREFIX + _OUTPUT_MODULATION]
    assert isinstance(swapped, SwappedHalvesLayerPatch) and type(swapped.layer) is layer_class


def _valid() -> dict[str, torch.Tensor]:
    return {"transformer.proj_out.lora_A.weight": _z(RANK, 256), "transformer.proj_out.lora_B.weight": _z(32, RANK)}


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        ({"diffusion_model.ref_time_in.in_layer.set_weight": _z(256, 32)}, constants.DISTILLATION_EXTRACT),
        ({"ref_time_in.out_layer.weight": _z(256, 256)}, constants.NOT_A_LORA_PARAMETER),
        # Three dimensions fit no parameter of the transformer; only PDD's head bank is skipped (below).
        ({"diffusion_model.final_layer.image.linear.weight": _z(8, 128, 3072)}, constants.NOT_A_LORA_PARAMETER),
        ({"proj_out.weight": _z(8, 32, 256)}, constants.NOT_A_LORA_PARAMETER),
        ({"diffusion_model.single_blocks.0.norm.key_norm.diff": _z(128)}, constants.NO_FLUX3_LAYER),
        (
            {
                "diffusion_model.single_blocks.0.qkv.lora_A.weight": _z(RANK, 256),
                "diffusion_model.single_blocks.0.qkv.lora_B.weight": _z(768, RANK),
            },
            constants.NO_FLUX3_LAYER,
        ),
        (
            {
                "single_transformer_blocks.28.attn.to_q.lora_A.weight": _z(RANK, 256),
                "single_transformer_blocks.28.attn.to_q.lora_B.weight": _z(256, RANK),
            },
            constants.NO_FLUX3_LAYER,
        ),
        ({"text_encoder.model.layers.0.self_attn.q_proj.lora_A.weight": _z(RANK, 256)}, constants.NO_FLUX3_LAYER),
        ({"transformer.x_embedders.image.lora_A.weight": _z(RANK, 32)}, constants.NOT_ONE_LAYER),
        ({"transformer.proj_out.alpha": _z(), "diffusion_model.proj_out.alpha": _z()}, constants.NAMED_TWICE),
        (
            {
                "norm_out.linear.lora_A.weight": _z(RANK, 256),
                "final_layer.image.adaLN_modulation.1.lora_B.weight": _z(512, RANK),
            },
            constants.NAMED_TWICE,
        ),
        (
            {"transformer.context_embedder.lokr_w1": _z(4, 4), "transformer.context_embedder.lokr_t2": _z(2, 2)},
            constants.NOT_ONE_LAYER,
        ),
    ],
    ids=[
        "set-weight",
        "full-weight",
        "other-3d-tensor",
        "head-bank-of-another-shape",
        "norm-diff",
        "fused-qkv",
        "beyond-the-depth",
        "text-encoder",
        "half-a-pair",
        "two-prefixes",
        "two-names",
        "tucker-core",
    ],
)
def test_a_lora_that_does_not_apply_in_full_is_refused(extra: dict[str, torch.Tensor], reason: str) -> None:
    """Applying the rest would make a partial LoRA that passes for a whole one."""
    with pytest.raises(ValueError, match=re.escape(reason)):
        convert(_valid() | extra)


def test_a_pdd_lora_runs_without_its_head_bank_and_says_so_once(caplog: pytest.LogCaptureFixture) -> None:
    """PDD few-step LoRAs carry one output head per step of PDD's trajectory, (steps, 128, 3072), which PDD's own
    sampler fuses per jump in place of `proj_out`: NoiseToNova's PEFT export as `proj_out.weight`, its ComfyUI export
    as `pdd.heads`. On GPU, Klein Base with the LoRA alone samples sharp in 8 steps where it is blurry without it."""
    heads = {"proj_out.weight": _z(128, 128, 3072), "pdd.heads": _z(128, 128, 3072)}

    with caplog.at_level(logging.WARNING):
        patch = convert(_valid() | heads)

    assert set(patch.layers) == {PREFIX + "proj_out"}
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert LORA_NAME in warnings[0] and all(key in warnings[0] for key in heads) and "few-step LoRA" in warnings[0]


_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32}


def _header(keys: dict[str, tuple[list[int], str]]) -> dict[str, torch.Tensor]:
    """A captured header as tensors on the meta device, but for the 0-D alphas, which a layer reads a value of."""
    return {
        key: torch.zeros(()) if not shape else torch.empty(shape, dtype=_DTYPES[dtype], device="meta")
        for key, (shape, dtype) in keys.items()
    }


@pytest.fixture(scope="module")
def released_transformer() -> Flux3Transformer2DModel:
    """The released geometry (`transformer/config.json` of both checkpoints), one block per stack as the captured
    headers keep block 0 of each, on the meta device."""
    with accelerate.init_empty_weights():
        return Flux3Transformer2DModel(
            in_channels=128,
            cond_in_channels=128,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=20480,
            timestep_channels=256,
            mlp_ratio=3.0,
            flow_map=True,
        )


@pytest.mark.parametrize(
    ("fixture", "pdd_heads"),
    [
        (flux3_lora_diffusers_format, set()),
        (flux3_lora_aitoolkit_format, set()),
        (flux3_lora_comfy_alpha_format, set()),
        (flux3_lora_peft_adapter_format, {"proj_out.weight"}),
    ],
    ids=["diffusers", "ai-toolkit", "comfy-alpha", "peft-adapter"],
)
def test_a_published_lora_lands_on_linears_of_its_shape(fixture, pdd_heads: set[str], released_transformer) -> None:
    state_dict = normalize_peft_adapter_names(_header(fixture.state_dict_keys))
    keys = constants.read_flux3_lora_keys(state_dict)
    patch = convert(state_dict)

    assert set(keys.pdd_heads) == pdd_heads
    # Every other tensor is part of a layer: two matrices each, and an alpha where the file has them.
    alphas = sum(key.endswith(".alpha") for key in state_dict)
    assert 2 * len(patch.layers) + alphas == len(state_dict) - len(pdd_heads)
    for key, layer in patch.layers.items():
        linear = released_transformer.get_submodule(key.removeprefix(PREFIX))
        assert isinstance(linear, torch.nn.Linear), key
        if isinstance(layer, SwappedHalvesLayerPatch):
            layer = layer.layer
        assert isinstance(layer, LoRALayer)
        assert (layer.up.shape[0], layer.down.shape[1]) == tuple(linear.weight.shape), key
        assert layer.up.shape[1] == layer.down.shape[0], key


def _header_with(**fields) -> dict[str, str]:
    """A safetensors header holding diffusers' PEFT config of the transformer, `transformer.`-prefixed as it saves it."""
    return {"lora_adapter_metadata": json.dumps({f"transformer.{name}": value for name, value in fields.items()})}


def _unprefixed_header(**fields) -> dict[str, str]:
    """The transformer's PEFT config as diffusers' `save_lora_adapter` saves a model's own adapter: unprefixed."""
    return {"lora_adapter_metadata": json.dumps(fields)}


_A, _B = torch.randn(RANK, 32, generator=torch.Generator().manual_seed(3)), torch.randn(16, RANK)


@pytest.mark.parametrize(
    ("metadata", "file_alpha", "scale"),
    [
        (None, None, 1.0),
        (_header_with(r=RANK, lora_alpha=2), None, 2 / RANK),
        # PEFT's defaults for the fields a header lacks: lora_alpha 8.
        (_header_with(r=RANK), None, 8 / RANK),
        (_header_with(r=RANK, lora_alpha=8, use_rslora=True), None, 8 / RANK**0.5),
        # A header without the transformer's config describes no transformer layer.
        ({"lora_adapter_metadata": json.dumps({"text_encoder.r": RANK, "text_encoder.lora_alpha": 2})}, None, 1.0),
        # The file's own alpha wins over the header's.
        (_header_with(r=RANK, lora_alpha=16), 2.0, 2 / RANK),
        # 8 / 4.
        (_unprefixed_header(r=4, lora_alpha=8), None, 2.0),
        # 8 / sqrt(4).
        (_unprefixed_header(r=4, lora_alpha=8, use_rslora=True), None, 4.0),
        # A header with both is read by its prefixed fields, a pipeline's: 2 / 4, not 8 / 4.
        (
            {
                "lora_adapter_metadata": json.dumps(
                    {"transformer.r": 4, "transformer.lora_alpha": 2, "r": 4, "lora_alpha": 8}
                )
            },
            None,
            0.5,
        ),
    ],
    ids=[
        "no-header",
        "lora-alpha",
        "peft-defaults",
        "rslora",
        "other-component",
        "file-alpha-wins",
        "unprefixed",
        "unprefixed-rslora",
        "prefixed-wins",
    ],
)
def test_a_diffusers_header_scales_the_lora_as_peft_does(metadata, file_alpha: float | None, scale: float) -> None:
    values = {"transformer.proj_out.lora_A.weight": _A, "transformer.proj_out.lora_B.weight": _B}
    if file_alpha is not None:
        values["transformer.proj_out.alpha"] = torch.tensor(file_alpha)

    layer = convert(values, metadata).layers[PREFIX + "proj_out"]

    update = layer.get_parameters({"weight": torch.zeros(16, 32)}, weight=1.0)["weight"]
    torch.testing.assert_close(update, scale * _B @ _A)


@pytest.mark.parametrize(
    "metadata",
    [
        _header_with(r=RANK, lora_alpha=8, alpha_pattern={"proj_out": 4}),
        _header_with(r=RANK, lora_alpha=8, rank_pattern={"proj_out": 8}),
        _header_with(r=RANK, alpha=8, peft_type="LOHA"),
        _unprefixed_header(r=RANK, alpha=8, peft_type="LOKR"),
        _header_with(r=0, lora_alpha=8),
        {"lora_adapter_metadata": "{not json"},
    ],
    ids=["alpha-pattern", "rank-pattern", "loha-config", "unprefixed-lokr-config", "rank-zero", "not-json"],
)
def test_a_header_this_cannot_scale_as_peft_does_is_refused(metadata: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="lora_adapter_metadata"):
        convert({"transformer.proj_out.lora_A.weight": _A, "transformer.proj_out.lora_B.weight": _B}, metadata)


@pytest.mark.parametrize("header", ["[4, 8]", '"r=4"', "4", "null"], ids=["list", "string", "number", "null"])
def test_a_header_that_is_no_json_object_is_refused_with_its_reason(header: str) -> None:
    """diffusers saves an object of `<component>.<field>` entries. Any other JSON value is refused with a reason, which
    identification shows at install, as for the other malformed headers."""
    values = {"transformer.proj_out.lora_A.weight": _A, "transformer.proj_out.lora_B.weight": _B}

    with pytest.raises(ValueError, match="lora_adapter_metadata header is not a JSON object"):
        convert(values, {"lora_adapter_metadata": header})


def _loader(models_path: Path, config) -> LoRALoader:
    loader = LoRALoader(app_config=MagicMock(models_path=models_path), logger=MagicMock(), ram_cache=MagicMock())
    loader._get_model_path(config)
    return loader


def _released_shape_lora(path: Path, metadata: dict[str, str] | None = None) -> Path:
    """A rank-1 LoRA of the released shapes, in raw PEFT keys with PEFT's adapter name."""
    save_file(
        {
            "single_transformer_blocks.0.attn.to_q.lora_A.default.weight": torch.zeros(1, 3072),
            "single_transformer_blocks.0.attn.to_q.lora_B.default.weight": torch.zeros(3072, 1),
            "x_embedders.image.lora_A.default.weight": torch.zeros(1, 128),
            "x_embedders.image.lora_B.default.weight": torch.zeros(3072, 1),
        },
        path,
        metadata=metadata,
    )
    return path


def test_a_peft_file_installs_and_loads_as_a_flux3_lora_scaled_by_its_header(tmp_path: Path) -> None:
    """Through identification and the LoRA loader, as an install and a generation reach a file: PEFT's adapter name
    is stripped on both paths, a released-shape LoRA is a FLUX.3 one, and the loader hands the converter the header."""
    path = _released_shape_lora(tmp_path / "flux3_lora.safetensors", _header_with(r=1, lora_alpha=2))

    config = ModelConfigFactory.from_model_on_disk(path, allow_unknown=False).config
    assert isinstance(config, LoRA_LyCORIS_Flux3_Config)
    patch = _loader(tmp_path, config)._load_model(config)

    assert set(patch.layers) == {PREFIX + "single_transformer_blocks.0.attn.to_q", PREFIX + "x_embedders.image"}
    assert patch.layers[PREFIX + "x_embedders.image"].scale() == 2.0


def test_a_flux3_lora_an_earlier_version_registered_as_flux1_asks_to_be_reidentified(tmp_path: Path) -> None:
    """Without FLUX.3 LoRA support, identification filed ai-toolkit FLUX.3 LoRAs as FLUX.1 ones, whose conversion maps
    their layers onto nothing or the wrong modules."""
    path = _released_shape_lora(tmp_path / "flux3_lora.safetensors")
    stale = LoRA_LyCORIS_FLUX_Config.model_construct(path=str(path), name="Stale", key="stale")

    with pytest.raises(ValueError, match="Re-identify model"):
        _loader(tmp_path, stale)._load_model(stale)
