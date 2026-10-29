"""Ideogram 4 LoRA identification, including mutual exclusivity with every other LoRA family.

``Config_Base.CONFIG_CLASSES`` is a set, so which config claims a file must not depend on probe order: each LoRA
family has to be exclusive at the per-config level. Ideogram 4 is the case that used to go wrong -- its blocks sit
under ``layers.N.attention`` / ``feed_forward.w1..3`` behind ``diffusion_model.`` or ``transformer.``, exactly like
Z-Image's, so the Z-Image probe claimed them. Names cannot tell the two apart (``attention.qkv`` is Z-Image's original
name too); the patched widths can: 4608 / 13824 / 12288 against Z-Image's 3840.
"""

import atexit
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_Ideogram4_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.patches.lora_conversions.ideogram4_lora_constants import IDEOGRAM4_LORA_KEY_PREFIXES
from tests.backend.patches.lora_conversions.lora_state_dicts import (
    anima_lora_kohya_format,
    anima_lora_lokr_format,
    anima_lora_peft_format,
    flux_dora_onetrainer_format,
    flux_lokr_bfl_format,
    flux_lora_aitoolkit_format,
    flux_lora_diffusers_base_model_format,
    flux_lora_diffusers_format,
    flux_lora_kohya_format,
    ideogram4_lokr_aitoolkit_format,
    ideogram4_lora_aitoolkit_format,
    ideogram4_lora_aitoolkit_unconditional_format,
    ideogram4_lora_diffusers_format,
    krea2_lora_kohya_format,
    qwen_image_lora_diffusers_format,
    qwen_image_lora_kohya_format,
    z_image_lora_aitoolkit_format,
)

_TMP_DIR = TemporaryDirectory()
atexit.register(_TMP_DIR.cleanup)


def _shape(value) -> list[int]:
    """Fixtures record either a bare shape or `(shape, dtype)`."""
    return list(value[0]) if isinstance(value, tuple) else list(value)


def _state_dict(keys: dict) -> dict[str, torch.Tensor]:
    """The layout with a single element of storage per tensor: probes read names and shapes, never values."""
    return {key: torch.zeros((), dtype=torch.float32).expand(_shape(value)) for key, value in keys.items()}


def _mod(sd: dict[str, torch.Tensor], name: str) -> tuple[MagicMock, dict]:
    path = Path(_TMP_DIR.name) / f"{name}.safetensors"
    path.touch()
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = sd
    overrides = {
        "hash": "test-hash",
        "path": str(path),
        "file_size": 0,
        "name": name,
        "source": str(path),
        "source_type": "path",
    }
    return mod, overrides


def _accepting_lora_configs(sd: dict[str, torch.Tensor], name: str) -> list[str]:
    """Every LoRA config class that claims ``sd``."""
    accepting: list[str] = []
    for cls in list(Config_Base.CONFIG_CLASSES):
        if getattr(cls.model_fields.get("type", None), "default", None) != ModelType.LoRA:
            continue
        mod, overrides = _mod(sd, name)
        try:
            cls.from_model_on_disk(mod, overrides)
        except Exception:
            # Identification records any failure as "no match"; only a returned config claims the file.
            continue
        accepting.append(cls.__name__)
    return accepting


def _reprefixed(keys: dict, prefix: str) -> dict:
    return {prefix + key.removeprefix("diffusion_model."): value for key, value in keys.items()}


_IDEOGRAM4_LAYOUTS = {
    "ai-toolkit": ideogram4_lora_aitoolkit_format.state_dict_keys,
    "ai-toolkit-lokr": ideogram4_lokr_aitoolkit_format.state_dict_keys,
    "ai-toolkit-unconditional": ideogram4_lora_aitoolkit_unconditional_format.state_dict_keys,
    "diffusers": ideogram4_lora_diffusers_format.state_dict_keys,
    # The other container prefixes the conversion strips, over the same modules.
    **{
        f"prefix={prefix or '<none>'}": _reprefixed(ideogram4_lora_aitoolkit_format.state_dict_keys, prefix)
        for prefix in (*IDEOGRAM4_LORA_KEY_PREFIXES, "")
        if prefix != "diffusion_model."
    },
}


@pytest.mark.parametrize("layout", list(_IDEOGRAM4_LAYOUTS))
def test_an_ideogram4_lora_is_claimed_by_ideogram4_alone(layout: str) -> None:
    sd = _state_dict(_IDEOGRAM4_LAYOUTS[layout])

    assert _accepting_lora_configs(sd, "ideogram4") == ["LoRA_LyCORIS_Ideogram4_Config"]
    mod, overrides = _mod(sd, "ideogram4")
    assert LoRA_LyCORIS_Ideogram4_Config.from_model_on_disk(mod, overrides).base is BaseModelType.Ideogram4


_OTHER_FAMILIES = {
    "anima-kohya": anima_lora_kohya_format,
    "anima-lokr": anima_lora_lokr_format,
    "anima-peft": anima_lora_peft_format,
    "flux-aitoolkit": flux_lora_aitoolkit_format,
    "flux-diffusers": flux_lora_diffusers_format,
    "flux-diffusers-base-model": flux_lora_diffusers_base_model_format,
    "flux-dora-onetrainer": flux_dora_onetrainer_format,
    "flux-kohya": flux_lora_kohya_format,
    "flux-lokr-bfl": flux_lokr_bfl_format,
    "krea2-kohya": krea2_lora_kohya_format,
    "qwen-image-diffusers": qwen_image_lora_diffusers_format,
    "qwen-image-kohya": qwen_image_lora_kohya_format,
    "z-image-aitoolkit": z_image_lora_aitoolkit_format,
}


@pytest.mark.parametrize("family", list(_OTHER_FAMILIES))
def test_the_ideogram4_probe_leaves_other_families_alone(family: str) -> None:
    mod, overrides = _mod(_state_dict(_OTHER_FAMILIES[family].state_dict_keys), family)

    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Ideogram4_Config.from_model_on_disk(mod, overrides)


def test_a_z_image_lora_is_still_claimed_by_z_image_alone() -> None:
    sd = _state_dict(z_image_lora_aitoolkit_format.state_dict_keys)

    assert _accepting_lora_configs(sd, "z-image") == ["LoRA_LyCORIS_ZImage_Config"]


def test_a_z_image_lora_on_its_original_fused_names_is_still_z_image() -> None:
    """Z-Image's original checkpoint names its attention `attention.qkv` / `attention.out` too. Only the width
    (3840) keeps such a LoRA away from Ideogram 4."""
    keys: dict[str, list[int]] = {}
    for module, (rows, cols) in {"attention.qkv": (11520, 3840), "feed_forward.w1": (10240, 3840)}.items():
        keys[f"diffusion_model.layers.0.{module}.lora_A.weight"] = [16, cols]
        keys[f"diffusion_model.layers.0.{module}.lora_B.weight"] = [rows, 16]
    sd = _state_dict(keys)
    mod, overrides = _mod(sd, "z-image-fused")

    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Ideogram4_Config.from_model_on_disk(mod, overrides)
    assert "LoRA_LyCORIS_Ideogram4_Config" not in _accepting_lora_configs(sd, "z-image-fused")


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"diffusion_model.layers.0.attention.o.lora_magnitude_vector.weight": [4608]}, "lora_magnitude_vector"),
        (
            {
                "text_encoders.layers.0.mlp.gate_proj.lora_A.weight": [16, 4096],
                "text_encoders.layers.0.mlp.gate_proj.lora_B.weight": [12288, 16],
            },
            "text_encoders",
        ),
    ],
    ids=["dora", "stray-layer"],
)
def test_an_ideogram4_lora_the_conversion_cannot_apply_is_refused_at_install(extra: dict, match: str) -> None:
    """Recognised but unusable: the file must fail at install rather than half-way into (or silently during) a
    generation -- and not fall back to "unknown"."""
    keys = {**ideogram4_lora_aitoolkit_format.state_dict_keys, **extra}
    sd = _state_dict(keys)
    mod, overrides = _mod(sd, "ideogram4-refused")

    with pytest.raises(InvalidMatchError, match=match):
        LoRA_LyCORIS_Ideogram4_Config.from_model_on_disk(mod, overrides)
    assert _accepting_lora_configs(sd, "ideogram4-refused") == []


def test_an_ideogram4_transformer_checkpoint_is_not_taken_for_a_lora() -> None:
    sd = _state_dict(
        {
            "layers.0.attention.qkv.weight": [13824, 4608],
            "layers.0.attention.o.weight": [4608, 4608],
            "layers.0.feed_forward.w1.weight": [12288, 4608],
        }
    )
    mod, overrides = _mod(sd, "ideogram4-checkpoint")

    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Ideogram4_Config.from_model_on_disk(mod, overrides)


def test_a_lora_on_the_block_modulation_alone_is_still_ideogram4() -> None:
    """The per-block `adaln_modulation` is 512 -> 18432 (4 x 4608): neither side is the hidden width."""
    keys = {
        "diffusion_model.layers.0.adaln_modulation.lora_A.weight": [16, 512],
        "diffusion_model.layers.0.adaln_modulation.lora_B.weight": [18432, 16],
    }

    assert _accepting_lora_configs(_state_dict(keys), "ideogram4-adaln") == ["LoRA_LyCORIS_Ideogram4_Config"]
