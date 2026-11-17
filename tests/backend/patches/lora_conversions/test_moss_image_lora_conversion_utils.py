"""MOSS-Image LoRAs onto `MOSSImageTransformer`.

Each published key layout converts to a patch that changes the transformer as merging the LoRA into its weights by
hand does, `W + weight * scale * delta` in float32 -- merged onto the fused weight and cut into thirds where a layout
fuses a block's `proj.qkv`. The transformer is the tiny one of `moss_image_fixtures`, every weight random. The
conversion maps names and rows only, so a LoRA sized to the tiny transformer converts as a released one does; the
captured headers of published LoRAs are held to the released geometry in `test_moss_image_lora_constants.py`.

On the CPU the patcher runs every layer as a sidecar unless told otherwise; the tests force the direct path where they
say so.
"""

import copy
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_MossImage_Config, LoRA_LyCORIS_ZImage_Config
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_MossImage_Config, Main_GGUF_MossImage_Config
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders import moss_image as loader_module
from invokeai.backend.model_manager.load.model_loaders.lora import LoRALoader
from invokeai.backend.model_manager.load.model_loaders.moss_image import MossImageCheckpointModel, MossImageGGUFModel
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.moss_image.pipeline_moss_image import MOSSImageTransformer
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.layers.merged_layer_patch import MergedLayerPatch
from invokeai.backend.patches.lora_conversions import moss_image_lora_constants as constants
from invokeai.backend.patches.lora_conversions.moss_image_lora_constants import (
    MOSS_IMAGE_LORA_TRANSFORMER_PREFIX as PREFIX,
)
from invokeai.backend.patches.lora_conversions.moss_image_lora_conversion_utils import (
    lora_model_from_moss_image_state_dict,
)
from invokeai.backend.patches.lora_conversions.peft_adapter_utils import normalize_peft_adapter_names
from tests.backend.model_manager.load.moss_image_fixtures import (
    TINY_GEOMETRY,
    forward,
    int8_convrot_state_dict,
    tiny_reference,
    with_state_dict,
    write_moss_image_gguf,
)

RANK = 4
# A LoRA's own alpha, in the layouts that carry one: half its rank, so a scale of 1 would show.
ALPHA = 2.0
PATCH_WEIGHT = 0.8
DIM = TINY_GEOMETRY["dim"]


def convert(state_dict: dict[str, torch.Tensor], metadata: dict[str, str] | None = None):
    return lora_model_from_moss_image_state_dict(state_dict, name="test", metadata=metadata)


def _linears(model: torch.nn.Module) -> dict[str, torch.nn.Linear]:
    return {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * 0.05


def _block_diagonal(parts: list[torch.Tensor]) -> torch.Tensor:
    """Each part in its own rows and columns: KohakuMoss's fused `lora_up`, which keeps q, k and v apart."""
    return torch.block_diag(*parts)


# Each layout's keys for one module's low-rank pair from its path. `scale` is what the layout's reader applies.
_LAYOUTS: dict[str, Callable[[str, torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]] = {
    "ai-toolkit": lambda path, a, b: {
        f"diffusion_model.{path}.lora_A.weight": a,
        f"diffusion_model.{path}.lora_B.weight": b,
    },
    "kohya": lambda path, a, b: {
        f"lora_unet_{path.replace('.', '_')}.lora_down.weight": a,
        f"lora_unet_{path.replace('.', '_')}.lora_up.weight": b,
        f"lora_unet_{path.replace('.', '_')}.alpha": torch.tensor(ALPHA),
    },
    "comfy": lambda path, a, b: {
        f"diffusion_model.{path}.lora_down.weight": a,
        f"diffusion_model.{path}.lora_up.weight": b,
        f"diffusion_model.{path}.alpha": torch.tensor(ALPHA),
    },
    "diffusers": lambda path, a, b: {f"transformer.{path}.lora_A.weight": a, f"transformer.{path}.lora_B.weight": b},
    "bare": lambda path, a, b: {f"{path}.lora_A.weight": a, f"{path}.lora_B.weight": b},
    # DiffSynth's adapter name, which `ModelOnDisk` strips before anything reads the keys.
    "peft-adapter": lambda path, a, b: {f"{path}.lora_A.default.weight": a, f"{path}.lora_B.default.weight": b},
}
# (layout, fuse each block's q, k, v into proj.qkv, the header ComfyUI never reads but diffusers writes)
_CASES = {
    "ai-toolkit": ("ai-toolkit", False, None),
    "ai-toolkit-fused": ("ai-toolkit", True, None),
    "kohya": ("kohya", False, None),
    "kohya-fused": ("kohya", True, None),
    "comfy-fused": ("comfy", True, None),
    "diffusers-header": (
        "diffusers",
        False,
        {"lora_adapter_metadata": '{"transformer.r": 4, "transformer.lora_alpha": 2}'},
    ),
    "bare": ("bare", False, None),
    "diffsynth": ("peft-adapter", False, None),
}
_HAS_ALPHA = {"kohya", "comfy"}


def _lora_on_every_linear(model: torch.nn.Module, layout: str, fused: bool) -> tuple[dict[str, torch.Tensor], dict]:
    """A LoRA on every Linear of `model` in `layout`, and each module's (A, B) pair. A fused layout writes q, k and v of
    a block as one rank-3r `proj.qkv` layer whose `lora_up` is block-diagonal, which is what applies each pair to its
    own rows."""
    pairs = {
        path: (_randn(RANK, linear.in_features, seed=i), _randn(linear.out_features, RANK, seed=1000 + i))
        for i, (path, linear) in enumerate(_linears(model).items())
    }
    state_dict: dict[str, torch.Tensor] = {}
    written = dict(pairs)
    if fused:
        for path in list(pairs):
            if path.endswith(".proj.q"):
                block = path.removesuffix(".q")
                parts = [written.pop(f"{block}.{name}") for name in "qkv"]
                written[f"{block}.qkv"] = (
                    torch.cat([a for a, _ in parts]),
                    _block_diagonal([b for _, b in parts]),
                )
    for path, (a, b) in written.items():
        state_dict |= _LAYOUTS[layout](path, a, b)
    return state_dict, pairs


def _scale(layout: str, fused: bool, header: bool, path: str) -> float:
    """What a layout's reader scales the layer on `path` by: its alpha over its rank (3r where a fused layer holds q, k
    and v), PEFT's scale from a diffusers header, or 1 for a file that says nothing."""
    if layout in _HAS_ALPHA:
        return ALPHA / (3 * RANK if fused and path.rsplit(".", 1)[-1] in ("q", "k", "v") else RANK)
    return ALPHA / RANK if header else 1.0


def _merged(model: torch.nn.Module, pairs: dict, scale_of: Callable[[str], float]) -> torch.nn.Module:
    """`model` with each pair merged into its weights by hand."""
    reference = copy.deepcopy(model)
    with torch.no_grad():
        for path, (a, b) in pairs.items():
            reference.get_submodule(path).weight += PATCH_WEIGHT * scale_of(path) * (b @ a)
    return reference


@pytest.mark.parametrize("direct", [True, False], ids=["direct", "sidecar"])
@pytest.mark.parametrize("case", list(_CASES))
def test_a_lora_changes_the_transformer_as_merging_it_by_hand_does(case: str, direct: bool) -> None:
    layout, fused, metadata = _CASES[case]
    model = tiny_reference()
    state_dict, pairs = _lora_on_every_linear(model, layout, fused)
    reference = _merged(model, pairs, lambda path: _scale(layout, fused, metadata is not None, path))
    # As the model cache holds it, so the patcher can wrap a layer in a sidecar.
    apply_custom_layers_to_model(model)
    patch = convert(normalize_peft_adapter_names(state_dict), metadata)
    unpatched = forward(model)

    assert set(patch.layers) == {PREFIX + path for path in pairs}
    with LayerPatcher.apply_smart_model_patches(
        model=model, patches=[(patch, PATCH_WEIGHT)], prefix=PREFIX, dtype=torch.float32, force_direct_patching=direct
    ):
        patched = forward(model)

    expected = forward(reference)
    assert not torch.allclose(expected, unpatched, atol=1e-3)
    torch.testing.assert_close(patched, expected, rtol=1e-4, atol=1e-5)
    # Taken back out again.
    torch.testing.assert_close(forward(model), unpatched, rtol=0, atol=0)


# Every kind of layer on a fused `proj.qkv` of the tiny transformer (192 x 64), by what LyCORIS, PEFT and ai-toolkit
# write. Each gives its values and the update to the fused weight, from the definitions of the kinds and not from the
# layer classes: the delta a reader applies, scaled, before any magnitude.
_FUSED_ROWS = 3 * DIM


def _lora_values() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    a, b = _randn(RANK, DIM, seed=1), _randn(_FUSED_ROWS, RANK, seed=2)
    return {"lora_down.weight": a, "lora_up.weight": b, "alpha": torch.tensor(ALPHA)}, b @ a * (ALPHA / RANK)


def _block_diagonal_values() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    a, b = _randn(3 * RANK, DIM, seed=3), _block_diagonal([_randn(DIM, RANK, seed=4 + i) for i in range(3)])
    return {"lora_down.weight": a, "lora_up.weight": b, "alpha": torch.tensor(ALPHA)}, b @ a * (ALPHA / (3 * RANK))


def _diff_values() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    diff = _randn(_FUSED_ROWS, DIM, seed=5)
    return {"diff": diff}, diff


def _loha_values() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    w1a, w1b = _randn(_FUSED_ROWS, RANK, seed=6), _randn(RANK, DIM, seed=7)
    w2a, w2b = _randn(_FUSED_ROWS, RANK, seed=8), _randn(RANK, DIM, seed=9)
    values = {"hada_w1_a": w1a, "hada_w1_b": w1b, "hada_w2_a": w2a, "hada_w2_b": w2b, "alpha": torch.tensor(ALPHA)}
    return values, (w1a @ w1b) * (w2a @ w2b) * (ALPHA / RANK)


def _lokr_values(w1_shape: tuple[int, int], w2_shape: tuple[int, int]) -> Callable[[], tuple[dict, torch.Tensor]]:
    def values() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        w1, w2 = _randn(*w1_shape, seed=10), _randn(*w2_shape, seed=11)
        # Whole operands have no rank for the alpha to divide by.
        return {"lokr_w1": w1, "lokr_w2": w2, "alpha": torch.tensor(ALPHA)}, torch.kron(w1, w2)

    return values


def _factored_lokr_values() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    w1a, w1b = _randn(8, RANK, seed=12), _randn(RANK, 8, seed=13)
    w2a, w2b = _randn(24, 2, seed=14), _randn(2, 8, seed=15)
    values = {"lokr_w1_a": w1a, "lokr_w1_b": w1b, "lokr_w2_a": w2a, "lokr_w2_b": w2b, "alpha": torch.tensor(ALPHA)}
    # The layer's rank is its first factored operand's.
    return values, torch.kron(w1a @ w1b, w2a @ w2b) * (ALPHA / RANK)


# Factors 8 and 16 split no third of the 192 rows into whole blocks of `lokr_w2`'s rows (24 and 12); 6 does (32 rows
# per block, two to a third); a single row of `lokr_w1` is cut through by every third.
_KINDS: dict[str, Callable[[], tuple[dict[str, torch.Tensor], torch.Tensor]]] = {
    "low-rank": _lora_values,
    "block-diagonal": _block_diagonal_values,
    "diff": _diff_values,
    "loha": _loha_values,
    "lokr-factor-8": _lokr_values((8, 8), (24, 8)),
    "lokr-factor-16": _lokr_values((16, 16), (12, 4)),
    "lokr-thirds-of-whole-blocks": _lokr_values((6, 8), (32, 8)),
    "lokr-one-block": _lokr_values((1, 2), (192, 32)),
    "lokr-factored": _factored_lokr_values,
}


def _dora(magnitude: torch.Tensor, weight: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """The weight DoRA makes of `weight + delta`: each row scaled to its magnitude."""
    updated = weight + delta
    return updated * (magnitude.reshape(-1, 1) / updated.norm(dim=1, keepdim=True))


def _fused_expected(kind: str, model: MOSSImageTransformer) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """The values of one fused layer on block 0's attention, and the three weights q, k and v hold once merged: the
    update on the fused weight, then cut into thirds."""
    block = model.layers[0].proj
    fused = torch.cat([block[name].weight.detach() for name in "qkv"])
    if kind.startswith("dora"):
        values, delta = _lora_values()
        magnitude = fused.norm(dim=1) * (1 + torch.linspace(0.1, 0.5, _FUSED_ROWS))
        if kind == "dora-scale":
            values["dora_scale"] = magnitude.reshape(-1, 1)
        else:
            values["dora_magnitude"] = magnitude
        # The layer's update at weight 1 is the DoRA weight less the original, scaled by the patch weight.
        return values, fused + PATCH_WEIGHT * (_dora(magnitude, fused, delta) - fused)
    values, delta = _KINDS[kind]()
    return values, fused + PATCH_WEIGHT * delta


def _check_fused(model: MOSSImageTransformer, patch, expected: torch.Tensor, *, direct: bool) -> None:
    """Block 0's q, k and v hold the thirds of `expected`: by their weights where the patch is merged into them, by
    what the Linears compute where it runs beside them."""
    apply_custom_layers_to_model(model)
    x = _randn(5, DIM, seed=20)
    with LayerPatcher.apply_smart_model_patches(
        model=model, patches=[(patch, PATCH_WEIGHT)], prefix=PREFIX, dtype=torch.float32, force_direct_patching=direct
    ):
        for name, third in zip("qkv", expected.chunk(3), strict=True):
            linear = model.layers[0].proj[name]
            if direct:
                torch.testing.assert_close(linear.weight, third, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(linear(x), x @ third.T, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("direct", [True, False], ids=["direct", "sidecar"])
@pytest.mark.parametrize("kind", [*_KINDS, "dora-scale", "dora-magnitude"])
def test_a_fused_qkv_layer_applies_as_its_update_cut_into_q_k_and_v(kind: str, direct: bool) -> None:
    model = tiny_reference()
    values, expected = _fused_expected(kind, model)
    spelled = {key.replace("dora_magnitude", "lora_magnitude_vector.weight"): value for key, value in values.items()}
    patch = convert({f"diffusion_model.layers.0.proj.qkv.{key}": value for key, value in spelled.items()})

    assert set(patch.layers) == {f"{PREFIX}layers.0.proj.{name}" for name in "qkv"}
    _check_fused(model, patch, expected, direct=direct)


@pytest.mark.parametrize(
    ("kind", "layer_class"),
    [
        ("low-rank", LoRALayer),
        ("block-diagonal", LoRALayer),
        ("lokr-factor-8", MergedLayerPatch),
        ("lokr-factor-16", MergedLayerPatch),
        ("lokr-thirds-of-whole-blocks", LoKRLayer),
        ("lokr-one-block", LoKRLayer),
        ("lokr-factored", MergedLayerPatch),
        ("dora-scale", DoRALayer),
    ],
)
def test_a_fused_layer_keeps_the_cheapest_layer_class_that_is_exact(kind: str, layer_class: type) -> None:
    """A low-rank layer keeps its sidecar fast path, a LoKr that cuts through blocks of `lokr_w2` is merged from the
    pieces, and one whose thirds are whole blocks needs no merge."""
    values, _ = _fused_expected(kind, tiny_reference())
    patch = convert({f"diffusion_model.layers.0.proj.qkv.{key}": value for key, value in values.items()})

    assert {type(layer) for layer in patch.layers.values()} == {layer_class}


def test_a_fused_qkv_dora_with_an_input_axis_magnitude_is_not_read_as_rows() -> None:
    values, _ = _fused_expected("dora-scale", tiny_reference())
    values["dora_scale"] = values["dora_scale"].reshape(1, -1)[:, :DIM]

    refused = constants.unsupported_keys(
        {f"diffusion_model.layers.0.proj.qkv.{key}": value for key, value in values.items()}
    )

    assert set(refused.values()) == {constants.WRONG_SHAPE}


def _split_lora(values_seed: int) -> dict[str, dict[str, torch.Tensor]]:
    """q, k and v as split `lora_A`/`lora_B` layers with an `alpha` each, and each layer's update."""
    out = {}
    for i, name in enumerate("qkv"):
        out[name] = {
            "lora_A.weight": _randn(RANK, DIM, seed=values_seed + i),
            "lora_B.weight": _randn(DIM, RANK, seed=values_seed + 10 + i),
            "alpha": torch.tensor(ALPHA),
        }
    return out


@pytest.mark.parametrize("direct", [True, False], ids=["direct", "sidecar"])
def test_ai_toolkits_dora_keeps_its_magnitude_on_the_fused_name_alone(direct: bool) -> None:
    """The split q, k and v LoRA layers carry no magnitude; the one `proj.qkv.dora_scale` of the three is theirs."""
    model = tiny_reference()
    fused = torch.cat([model.layers[0].proj[name].weight.detach() for name in "qkv"])
    layers = _split_lora(30)
    magnitude = fused.norm(dim=1, keepdim=True) * 1.2
    state_dict = {
        f"diffusion_model.layers.0.proj.{name}.{key}": value
        for name, values in layers.items()
        for key, value in values.items()
    }
    state_dict["diffusion_model.layers.0.proj.qkv.dora_scale"] = magnitude
    delta = torch.cat([values["lora_B.weight"] @ values["lora_A.weight"] for values in layers.values()]) * (
        ALPHA / RANK
    )
    expected = fused + PATCH_WEIGHT * (_dora(magnitude, fused, delta) - fused)

    patch = convert(state_dict)

    assert {type(layer) for layer in patch.layers.values()} == {DoRALayer}
    _check_fused(model, patch, expected, direct=direct)


@pytest.mark.parametrize("direct", [True, False], ids=["direct", "sidecar"])
def test_ai_toolkits_lokr_takes_the_alpha_its_split_layers_left_behind(direct: bool) -> None:
    """ai-toolkit writes the LoKr on the fused `proj.qkv` without an alpha, and copies one alpha onto the lone `q`, `k`
    and `v` of the layers it split."""
    model = tiny_reference()
    fused = torch.cat([model.layers[0].proj[name].weight.detach() for name in "qkv"])
    w1a, w1b = _randn(8, RANK, seed=40), _randn(RANK, 8, seed=41)
    w2 = _randn(24, 8, seed=42)
    state_dict = {
        "diffusion_model.layers.0.proj.qkv.lokr_w1_a": w1a,
        "diffusion_model.layers.0.proj.qkv.lokr_w1_b": w1b,
        "diffusion_model.layers.0.proj.qkv.lokr_w2": w2,
        **{f"diffusion_model.layers.0.proj.{name}.alpha": torch.tensor(ALPHA) for name in "qkv"},
    }
    expected = fused + PATCH_WEIGHT * torch.kron(w1a @ w1b, w2) * (ALPHA / RANK)

    patch = convert(state_dict)

    assert set(patch.layers) == {f"{PREFIX}layers.0.proj.{name}" for name in "qkv"}
    _check_fused(model, patch, expected, direct=direct)


@pytest.mark.parametrize("direct", [True, False], ids=["direct", "sidecar"])
def test_a_fused_lokr_with_a_factored_operand_takes_the_alpha_of_its_left_behind_ones_as_its_scale(
    direct: bool,
) -> None:
    """ai-toolkit trains a LoKr with a factored operand at alpha / rank (here 16 / 8), and clones its one alpha onto q, k
    and v of the split layers. ComfyUI would apply 1; the training scale is what the LoRA was made for."""
    model = tiny_reference()
    fused = torch.cat([model.layers[0].proj[name].weight.detach() for name in "qkv"])
    w1, w2a, w2b = _randn(8, 8, seed=60), _randn(24, 8, seed=61), _randn(8, 8, seed=62)
    state_dict = {
        "diffusion_model.layers.0.proj.qkv.lokr_w1": w1,
        "diffusion_model.layers.0.proj.qkv.lokr_w2_a": w2a,
        "diffusion_model.layers.0.proj.qkv.lokr_w2_b": w2b,
        **{f"diffusion_model.layers.0.proj.{name}.alpha": torch.tensor(16.0) for name in "qkv"},
    }
    expected = fused + PATCH_WEIGHT * torch.kron(w1, w2a @ w2b) * (16 / 8)

    _check_fused(model, convert(state_dict), expected, direct=direct)


def test_alphas_left_behind_that_disagree_give_the_lokr_the_first_and_warn(caplog: pytest.LogCaptureFixture) -> None:
    """ai-toolkit clones one alpha, so differing ones are not its doing, and identification cannot compare values."""
    w1, w2a, w2b = _randn(8, 8, seed=63), _randn(24, 8, seed=64), _randn(8, 8, seed=65)
    state_dict = {
        "diffusion_model.layers.0.proj.qkv.lokr_w1": w1,
        "diffusion_model.layers.0.proj.qkv.lokr_w2_a": w2a,
        "diffusion_model.layers.0.proj.qkv.lokr_w2_b": w2b,
        "diffusion_model.layers.0.proj.q.alpha": torch.tensor(16.0),
        "diffusion_model.layers.0.proj.k.alpha": torch.tensor(2.0),
        "diffusion_model.layers.0.proj.v.alpha": torch.tensor(2.0),
    }

    with caplog.at_level(logging.WARNING):
        patch = convert(state_dict)

    third = patch.layers[f"{PREFIX}layers.0.proj.q"].get_parameters({"weight": torch.zeros(DIM, DIM)}, weight=1.0)
    torch.testing.assert_close(third["weight"], torch.kron(w1, w2a @ w2b)[:DIM] * (16 / 8))
    assert any("layers.0.proj.qkv" in r.getMessage() and "[16.0, 2.0, 2.0]" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("metadata", "file_alpha", "scale"),
    [
        (None, None, 1.0),
        ({"lora_adapter_metadata": '{"transformer.r": 4, "transformer.lora_alpha": 2}'}, None, 0.5),
        # The file's own alpha wins over the header's.
        ({"lora_adapter_metadata": '{"transformer.r": 4, "transformer.lora_alpha": 16}'}, 1.0, 0.25),
        # What a header without PEFT's config says (ComfyUI's `alpha`/`rank` header entries) is not read.
        ({"alpha": "32", "rank": "4"}, None, 1.0),
    ],
    ids=["rank", "header", "file-alpha-wins", "other-header"],
)
def test_a_layers_scale_is_its_alpha_then_the_peft_header_then_one(
    metadata: dict[str, str] | None, file_alpha: float | None, scale: float
) -> None:
    a, b = _randn(RANK, DIM, seed=50), _randn(DIM, RANK, seed=51)
    values = {"transformer.layers.0.proj.q.lora_A.weight": a, "transformer.layers.0.proj.q.lora_B.weight": b}
    if file_alpha is not None:
        values["transformer.layers.0.proj.q.alpha"] = torch.tensor(file_alpha)

    layer = convert(values, metadata).layers[PREFIX + "layers.0.proj.q"]

    update = layer.get_parameters({"weight": torch.zeros(DIM, DIM)}, weight=1.0)["weight"]
    torch.testing.assert_close(update, scale * b @ a)


def test_a_header_that_asks_for_a_scaling_this_cannot_apply_is_refused_naming_moss_image() -> None:
    values = {
        "transformer.layers.0.proj.q.lora_A.weight": _randn(RANK, DIM),
        "transformer.layers.0.proj.q.lora_B.weight": _randn(DIM, RANK),
    }
    header = {
        "lora_adapter_metadata": '{"transformer.r": 4, "transformer.lora_alpha": 8, "transformer.alpha_pattern": {"q": 2}}'
    }

    with pytest.raises(ValueError, match="This MOSS-Image LoRA sets its rank or alpha per module"):
        convert(values, header)


def _z(*shape: int) -> torch.Tensor:
    """Zeros in one element of storage: these tests read shapes, not values."""
    return torch.zeros(()).expand(shape)


_Q = "diffusion_model.layers.0.proj.q"


def _valid() -> dict[str, torch.Tensor]:
    return {f"{_Q}.lora_A.weight": _z(RANK, DIM), f"{_Q}.lora_B.weight": _z(DIM, RANK)}


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        ({"diffusion_model.layers.0.norm1.diff": _z(DIM)}, constants.NORM_WEIGHT),
        ({"diffusion_model.final_norm_video.diff": _z(DIM)}, constants.NORM_WEIGHT),
        ({"diffusion_model.layers.0.proj.o.diff_b": _z(DIM)}, constants.NOT_A_LORA_PARAMETER),
        ({"diffusion_model.layers.0.norm1.w_norm": _z(DIM)}, constants.NOT_A_LORA_PARAMETER),
        ({"diffusion_model.video_embedder.set_weight": _z(DIM, 96)}, constants.NOT_A_LORA_PARAMETER),
        ({"diffusion_model.layers.0.proj.o.weight": _z(DIM, DIM)}, constants.NOT_A_LORA_PARAMETER),
        ({"diffusion_model.layers.42.proj.o.diff": _z(DIM, DIM)}, constants.NO_MOSS_IMAGE_LAYER),
        ({"diffusion_model.layers.01.proj.o.diff": _z(DIM, DIM)}, constants.NO_MOSS_IMAGE_LAYER),
        ({"diffusion_model.text_embedder.blocks.2.proj.diff": _z(DIM, DIM)}, constants.NO_MOSS_IMAGE_LAYER),
        (
            {"lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": _z(RANK, DIM)},
            constants.NO_MOSS_IMAGE_LAYER,
        ),
        ({"diffusion_model.time_sign_embed.lora_A.weight": _z(RANK, DIM)}, constants.NO_MOSS_IMAGE_LAYER),
        ({"diffusion_model.layers.0.proj.o.lora_A.weight": _z(RANK, DIM)}, constants.NOT_ONE_LAYER),
        ({"diffusion_model.layers.0.proj.o.alpha": _z()}, constants.NOT_ONE_LAYER),
        ({"diffusion_model.layers.0.proj.o.dora_scale": _z(DIM, 1)}, constants.NOT_ONE_LAYER),
        (
            {"diffusion_model.layers.0.proj.o.lokr_w1": _z(4, 4), "diffusion_model.layers.0.proj.o.lokr_t2": _z(2, 2)},
            constants.NOT_ONE_LAYER,
        ),
        ({"transformer.layers.0.proj.q.lora_A.weight": _z(RANK, DIM)}, constants.NAMED_TWICE),
        (
            {f"{_Q}.lora_down.weight": _z(RANK, DIM), f"{_Q}.lora_up.weight": _z(DIM, RANK)},
            constants.NAMED_TWICE,
        ),
        (
            {"diffusion_model.layers.0.proj.qkv.lora_A.weight": _z(RANK, DIM)}
            | {"diffusion_model.layers.0.proj.qkv.lora_B.weight": _z(3 * DIM, RANK)},
            constants.FUSED_AND_SPLIT,
        ),
    ],
    ids=[
        "norm-diff",
        "final-norm-diff",
        "bias-diff",
        "w-norm",
        "set-weight",
        "full-weight",
        "block-beyond-the-depth",
        "leading-zero",
        "text-block-beyond-the-depth",
        "text-encoder",
        "simpletuner-module",
        "half-a-pair",
        "lone-alpha",
        "lone-magnitude",
        "tucker-core",
        "two-prefixes",
        "two-spellings",
        "fused-and-split",
    ],
)
def test_a_lora_that_does_not_apply_in_full_is_refused(extra: dict[str, torch.Tensor], reason: str) -> None:
    """Applying the rest would make a partial LoRA that passes for a whole one."""
    with pytest.raises(ValueError, match=re.escape(reason)):
        convert(_valid() | extra)


def test_a_lone_alpha_beside_a_whole_lokr_is_not_the_alpha_of_a_layer_with_its_own() -> None:
    state_dict = {
        "diffusion_model.layers.0.proj.qkv.lokr_w1": _z(8, 8),
        "diffusion_model.layers.0.proj.qkv.lokr_w2": _z(24, 8),
        "diffusion_model.layers.0.proj.qkv.alpha": _z(),
        "diffusion_model.layers.0.proj.q.alpha": _z(),
    }

    with pytest.raises(ValueError, match=re.escape(constants.NOT_ONE_LAYER)):
        convert(state_dict)


def _loader(cls: type, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(loader_module.TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(
        loader_module.TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _device: torch.float32)
    )
    loader = object.__new__(cls)
    loader._ram_cache = MagicMock()
    loader._logger = logging.getLogger(__name__)
    loader._torch_device = torch.device("cpu")
    loader._torch_dtype = torch.float32
    return loader


def _int8_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[MOSSImageTransformer, MOSSImageTransformer]:
    stored, meant = int8_convrot_state_dict(tiny_reference())
    path = tmp_path / "moss_image_base_int8_convrot.safetensors"
    save_file(stored, str(path))
    loader = _loader(MossImageCheckpointModel, monkeypatch)
    config = Main_Checkpoint_MossImage_Config.model_construct(path=str(path), name="moss")
    return loader._load_model(config, SubModelType.Transformer), with_state_dict(meant)


def _gguf_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[MOSSImageTransformer, MOSSImageTransformer]:
    path = tmp_path / "moss.gguf"
    meant = write_moss_image_gguf(path, tiny_reference(), fused=False, architecture="moss_image")
    loader = _loader(MossImageGGUFModel, monkeypatch)
    config = Main_GGUF_MossImage_Config.model_construct(path=str(path), name="moss")
    return loader._load_model(config, SubModelType.Transformer), with_state_dict(meant)


@pytest.mark.parametrize("build", [_int8_model, _gguf_model], ids=["int8-convrot", "gguf"])
@pytest.mark.parametrize("case", ["ai-toolkit", "comfy-fused"])
def test_a_lora_on_quantized_layers_runs_beside_them_as_a_sidecar(
    build: Callable, case: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An int8 Linear keeps its weight in buffers and a GGUF one packed, so the denoise forces the sidecar path
    (`requires_sidecar_patching`). The layer on the fused `proj.qkv` of the int8 file lands on `proj.q`, `proj.k` and
    `proj.v`, the first of which is dense there and the others int8, and the reference is the dense model holding the
    weights the file decodes to, plus the LoRA merged by hand."""
    model, plain = build(monkeypatch, tmp_path)
    layout, fused, metadata = _CASES[case]
    state_dict, pairs = _lora_on_every_linear(plain, layout, fused)
    reference = _merged(plain, pairs, lambda path: _scale(layout, fused, metadata is not None, path))
    apply_custom_layers_to_model(model)

    with LayerPatcher.apply_smart_model_patches(
        model=model,
        patches=[(convert(state_dict, metadata), PATCH_WEIGHT)],
        prefix=PREFIX,
        dtype=torch.float32,
        force_sidecar_patching=True,
    ):
        patched = forward(model)

    torch.testing.assert_close(patched, forward(reference), rtol=1e-3, atol=1e-3)


def _loader_for(models_path: Path, config) -> LoRALoader:
    loader = LoRALoader(app_config=MagicMock(models_path=models_path), logger=MagicMock(), ram_cache=MagicMock())
    loader._get_model_path(config)
    return loader


def _released_shape_lora(path: Path, metadata: dict[str, str] | None = None) -> Path:
    """A rank-1 LoRA of the released shapes on the fused `proj.qkv` and `video_embedder`, in ComfyUI's names with
    PEFT's adapter name, as DiffSynth writes it."""
    save_file(
        {
            "layers.0.proj.qkv.lora_A.default.weight": torch.zeros(1, 3072),
            "layers.0.proj.qkv.lora_B.default.weight": torch.zeros(9216, 1),
            "video_embedder.lora_A.default.weight": torch.zeros(1, 128),
            "video_embedder.lora_B.default.weight": torch.zeros(3072, 1),
        },
        path,
        metadata=metadata,
    )
    return path


def test_a_peft_file_installs_and_loads_as_a_moss_image_lora_scaled_by_its_header(tmp_path: Path) -> None:
    """Through identification and the LoRA loader, as an install and a generation reach a file: PEFT's adapter name is
    stripped on both paths, a released-shape LoRA is a MOSS-Image one, the fused layer lands on q, k and v, and the
    loader hands the converter the header."""
    header = {"lora_adapter_metadata": '{"transformer.r": 1, "transformer.lora_alpha": 2}'}
    path = _released_shape_lora(tmp_path / "moss_lora.safetensors", header)

    config = ModelConfigFactory.from_model_on_disk(path, allow_unknown=False).config
    assert isinstance(config, LoRA_LyCORIS_MossImage_Config)
    patch = _loader_for(tmp_path, config)._load_model(config)

    assert set(patch.layers) == {f"{PREFIX}layers.0.proj.{name}" for name in "qkv"} | {PREFIX + "video_embedder"}
    assert patch.layers[PREFIX + "video_embedder"].scale() == 2.0


def test_a_moss_image_lora_an_earlier_version_registered_as_z_image_asks_to_be_reidentified(tmp_path: Path) -> None:
    """Without MOSS-Image LoRA support, identification filed `diffusion_model.layers.N` LoRAs as Z-Image ones, whose
    conversion maps their layers onto nothing."""
    path = _released_shape_lora(tmp_path / "moss_lora.safetensors")
    stale = LoRA_LyCORIS_ZImage_Config.model_construct(path=str(path), name="Stale", key="stale")

    with pytest.raises(ValueError, match="Re-identify model"):
        _loader_for(tmp_path, stale)._load_model(stale)
