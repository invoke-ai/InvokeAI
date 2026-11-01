"""What the FLUX.3 LoRA config accepts at the released transformer's shapes, and what it refuses.

Every layer kind the conversion builds is accepted in each spelling a real producer writes, and refused when it does
not update a weight of its module's shape: a layer of another shape would install, then be skipped by the patcher or
fail in it. The geometry the config holds layers to is diffusers' own module list for the released configuration, not
a table restated here. State dicts are read as identification reads a file: on the meta device, PEFT's adapter names
stripped.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import accelerate
import pytest
import torch
from diffusers import Flux3Transformer2DModel

from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_Flux3_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.patches.lora_conversions import flux3_lora_constants as constants
from invokeai.backend.patches.lora_conversions.peft_adapter_utils import normalize_peft_adapter_names

R = 4
# Rows and columns differ, so a factor read along the wrong axis shows.
MODULE = "single_transformer_blocks.0.ff.linear_in"
OUT, IN = 18432, 3072


def _m(*shape: int) -> torch.Tensor:
    return torch.empty(shape, device="meta")


def _identify(state_dict: dict[str, torch.Tensor], tmp_path: Path, metadata: dict[str, str] | None = None):
    """The config identification builds for a file holding `state_dict`, or the error it refuses it with."""
    path = tmp_path / "lora.safetensors"
    path.touch()
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = normalize_peft_adapter_names(state_dict)
    mod.metadata.return_value = metadata or {}
    fields = {
        "hash": "h",
        "path": str(path),
        "file_size": 0,
        "name": "lora",
        "source": str(path),
        "source_type": "path",
    }
    try:
        return LoRA_LyCORIS_Flux3_Config.from_model_on_disk(mod, fields)
    except (NotAMatchError, InvalidMatchError) as error:
        return error


def _layer(values: dict[str, tuple[int, ...]], prefix: str = "transformer.") -> dict[str, torch.Tensor]:
    return {f"{prefix}{MODULE}.{name}": _m(*shape) for name, shape in values.items()}


_LOW_RANK = {
    "lora_A/lora_B": {"lora_A.weight": (R, IN), "lora_B.weight": (OUT, R)},
    "lora_down/lora_up": {"lora_down.weight": (R, IN), "lora_up.weight": (OUT, R)},
}
# Each producer's spelling of a DoRA magnitude, and the shapes it is written in.
_PEFT_MAGNITUDES = {
    # PEFT's `get_peft_model_state_dict`, which drops the `.weight`.
    "lora_magnitude_vector": [(OUT,), (OUT, 1)],
    # A PEFT model's own state dict.
    "lora_magnitude_vector.weight": [(OUT,), (OUT, 1)],
    # ai-toolkit.
    "magnitude": [(OUT,), (OUT, 1)],
}
# LyCORIS's `dora_scale`, per input column by default, per output row with `wd_on_out`.
_DORA_SCALES = [(1, IN), (IN,), (OUT, 1)]
_LOKR_W1 = {"whole": {"lokr_w1": (4, 8)}, "factored": {"lokr_w1_a": (4, R), "lokr_w1_b": (R, 8)}}
_LOKR_W2 = {
    "whole": {"lokr_w2": (OUT // 4, IN // 8)},
    "factored": {"lokr_w2_a": (OUT // 4, R), "lokr_w2_b": (R, IN // 8)},
}
_LOHA = {"hada_w1_a": (OUT, R), "hada_w1_b": (R, IN), "hada_w2_a": (OUT, R), "hada_w2_b": (R, IN)}


def _accepted_layers() -> dict[str, dict[str, tuple[int, ...]]]:
    ranked: dict[str, dict[str, tuple[int, ...]]] = dict(_LOW_RANK)
    for name, shapes in _PEFT_MAGNITUDES.items():
        for shape in shapes:
            ranked[f"dora-{name}-{len(shape)}d"] = _LOW_RANK["lora_A/lora_B"] | {name: shape}
    for shape in _DORA_SCALES:
        ranked[f"dora_scale-{'x'.join(map(str, shape))}"] = _LOW_RANK["lora_down/lora_up"] | {"dora_scale": shape}
    for (w1_form, w1), (w2_form, w2) in ((a, b) for a in _LOKR_W1.items() for b in _LOKR_W2.items()):
        ranked[f"lokr-w1-{w1_form}-w2-{w2_form}"] = w1 | w2
    ranked["loha"] = _LOHA
    with_alpha = {f"{name}+alpha": values | {"alpha": ()} for name, values in ranked.items()}
    # A full diff has no rank for an alpha to scale.
    return ranked | with_alpha | {"full": {"diff": (OUT, IN)}}


_ACCEPTED = _accepted_layers()


@pytest.mark.parametrize("kind", list(_ACCEPTED))
def test_every_layer_kind_is_accepted_at_the_released_shape(kind: str, tmp_path: Path) -> None:
    result = _identify(_layer(_ACCEPTED[kind]), tmp_path)

    assert isinstance(result, LoRA_LyCORIS_Flux3_Config), result
    assert result.base is BaseModelType.Flux3


# One layer of each kind whose update, or magnitude, misses the module's shape.
_MISSHAPEN = {
    "low-rank": _LOW_RANK["lora_A/lora_B"] | {"lora_B.weight": (OUT // 2, R)},
    "low-rank-whose-ranks-differ": _LOW_RANK["lora_A/lora_B"] | {"lora_B.weight": (OUT, R + 1)},
    "peft-dora-on-the-input-axis": _LOW_RANK["lora_A/lora_B"] | {"lora_magnitude_vector": (IN,)},
    "lycoris-dora-on-neither-axis": _LOW_RANK["lora_down/lora_up"] | {"dora_scale": (1, OUT)},
    "lokr": _LOKR_W1["whole"] | {"lokr_w2": (OUT // 4, IN // 16)},
    "loha-whose-products-differ": _LOHA | {"hada_w2_a": (OUT // 2, R)},
    "transposed-full": {"diff": (IN, OUT)},
}


@pytest.mark.parametrize("kind", list(_MISSHAPEN))
def test_a_layer_of_another_shape_is_refused(kind: str, tmp_path: Path) -> None:
    result = _identify(_layer(_MISSHAPEN[kind]), tmp_path)

    assert isinstance(result, InvalidMatchError)
    assert constants.WRONG_SHAPE in str(result) and MODULE in str(result)


def test_a_raw_peft_dora_file_is_accepted(tmp_path: Path) -> None:
    """A PEFT model's state dict names a DoRA magnitude like the pair, adapter name and all; identification strips it
    from all three, or the magnitude would be no LoRA parameter."""
    state_dict = {
        f"{MODULE}.lora_A.default.weight": _m(R, IN),
        f"{MODULE}.lora_B.default.weight": _m(OUT, R),
        f"{MODULE}.lora_magnitude_vector.default.weight": _m(OUT),
    }

    assert isinstance(_identify(state_dict, tmp_path), LoRA_LyCORIS_Flux3_Config)


def test_a_peft_model_s_saved_adapter_is_accepted_under_its_prefix(tmp_path: Path) -> None:
    """A PEFT model's `save_pretrained` keys its adapter `base_model.model.` + module path."""
    result = _identify(_layer(_LOW_RANK["lora_A/lora_B"], prefix="base_model.model."), tmp_path)

    assert isinstance(result, LoRA_LyCORIS_Flux3_Config), result


@pytest.fixture(scope="module")
def released_linears() -> dict[str, tuple[int, int]]:
    """Every Linear of the released transformer (`transformer/config.json` of both checkpoints), at its full depth."""
    with accelerate.init_empty_weights():
        model = Flux3Transformer2DModel(
            in_channels=128,
            cond_in_channels=128,
            num_layers=5,
            num_single_layers=28,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=20480,
            timestep_channels=256,
            mlp_ratio=3.0,
            flow_map=True,
        )
    return {
        name: tuple(module.weight.shape)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }


def test_a_lora_on_every_linear_of_the_released_transformer_is_accepted(released_linears, tmp_path: Path) -> None:
    state_dict: dict[str, torch.Tensor] = {}
    for module, (out_features, in_features) in released_linears.items():
        state_dict[f"transformer.{module}.lora_A.weight"] = _m(1, in_features)
        state_dict[f"transformer.{module}.lora_B.weight"] = _m(out_features, 1)

    assert isinstance(_identify(state_dict, tmp_path), LoRA_LyCORIS_Flux3_Config)


@pytest.mark.parametrize(
    ("stack", "depth"),
    [
        ("mode_blocks.image", 5),
        ("mode_blocks.image_cond", 5),
        ("txt_mode_blocks", 5),
        ("single_transformer_blocks", 28),
    ],
)
def test_a_block_past_the_released_depth_is_refused(stack: str, depth: int, released_linears, tmp_path: Path) -> None:
    assert f"{stack}.{depth - 1}.attn.to_q" in released_linears and f"{stack}.{depth}.attn.to_q" not in released_linears
    beyond = f"transformer.{stack}.{depth}.attn.to_q"
    state_dict = _layer(_LOW_RANK["lora_A/lora_B"]) | {
        f"{beyond}.lora_A.weight": _m(R, 3072),
        f"{beyond}.lora_B.weight": _m(3072, R),
    }

    result = _identify(state_dict, tmp_path)

    assert isinstance(result, InvalidMatchError)
    assert constants.NO_FLUX3_LAYER in str(result) and f"{stack}.{depth}." in str(result)


def _header(**fields: Any) -> dict[str, str]:
    return {"lora_adapter_metadata": json.dumps({f"transformer.{name}": value for name, value in fields.items()})}


@pytest.mark.parametrize(
    "metadata",
    [_header(r=R, lora_alpha=8, alpha_pattern={MODULE: 16}), _header(r=R, lora_alpha=8, rank_pattern={MODULE: 8})],
    ids=["alpha-pattern", "rank-pattern"],
)
def test_a_header_that_scales_modules_apart_is_refused(metadata: dict[str, str], tmp_path: Path) -> None:
    """PEFT would scale its modules apart; applying one scale to all would apply it wrongly."""
    result = _identify(_layer(_LOW_RANK["lora_A/lora_B"]), tmp_path, metadata)

    assert isinstance(result, InvalidMatchError)
    assert "rank_pattern or alpha_pattern" in str(result)
