"""Identification of SimpleTuner's LyCORIS flat-key LoRAs (`lycoris_` prefix) — issue #10174.

SimpleTuner's LyCORIS output names each trained module by flattening the model's own diffusers module
path with underscores behind a `lycoris_` prefix (`lycoris_single_transformer_blocks_0_attn_to_k`), or
behind `lycoris__` (the wrapper's separator plus the name). The probes look for container prefixes
(`transformer.`, `lora_unet_`) or for the dotted diffusers layout, so none of this matches and such
files install as unknown on every base. Each family's probe now resolves the flattened names against
that family's module list — the same move the SD 3 probe already made for its own flat spelling
(`sd3_lokr_simpletuner_format`).

The fixtures are the real key layouts of the Hugging Face files named in issue #10174, captured from
their safetensors headers. Which config claims a file must not depend on probe order (`Config_Base.
CONFIG_CLASSES` is a set), so every layout is checked against every LoRA config.
"""

import atexit
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_Wan_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, WanLoRAVariantType
from tests.backend.patches.lora_conversions.lora_state_dicts import (
    flux1_lycoris_flat_lokr_double_underscore_format,
    flux1_lycoris_flat_lokr_format,
    flux2_lycoris_flat_lokr_format,
    sd3_lokr_simpletuner_format,
    sdxl_lycoris_flat_lokr_format,
    wan_lycoris_flat_lokr_format,
    z_image_lycoris_flat_lokr_format,
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
    """Every LoRA config class that claims ``sd`` as a single file."""
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


_FLAT_LAYOUTS = {
    "flux1-lokr": (flux1_lycoris_flat_lokr_format, "LoRA_LyCORIS_FLUX_Config", BaseModelType.Flux),
    "flux1-lokr-double-underscore": (
        flux1_lycoris_flat_lokr_double_underscore_format,
        "LoRA_LyCORIS_FLUX_Config",
        BaseModelType.Flux,
    ),
    "flux2-klein-lokr": (flux2_lycoris_flat_lokr_format, "LoRA_LyCORIS_Flux2_Config", BaseModelType.Flux2),
    "sdxl-lokr": (sdxl_lycoris_flat_lokr_format, "LoRA_LyCORIS_SDXL_Config", BaseModelType.StableDiffusionXL),
    "z-image-lokr": (z_image_lycoris_flat_lokr_format, "LoRA_LyCORIS_ZImage_Config", BaseModelType.ZImage),
    "wan-lokr": (wan_lycoris_flat_lokr_format, "LoRA_LyCORIS_Wan_Config", BaseModelType.Wan),
}


@pytest.mark.parametrize("layout", list(_FLAT_LAYOUTS))
def test_a_flat_lycoris_lora_is_claimed_by_its_family_alone(layout: str) -> None:
    fixture, expected_config, _base = _FLAT_LAYOUTS[layout]
    sd = _state_dict(fixture.state_dict_keys)

    assert _accepting_lora_configs(sd, layout) == [expected_config]


@pytest.mark.parametrize("layout", list(_FLAT_LAYOUTS))
def test_a_flat_lycoris_lora_records_its_base(layout: str) -> None:
    fixture, expected_config, base = _FLAT_LAYOUTS[layout]
    mod, overrides = _mod(_state_dict(fixture.state_dict_keys), layout)

    for cls in list(Config_Base.CONFIG_CLASSES):
        if cls.__name__ == expected_config:
            break
    else:
        raise AssertionError(f"{expected_config} is not registered")
    assert cls.from_model_on_disk(mod, overrides).base is base


def test_the_wan_flat_layout_records_its_variant_from_lokr_widths() -> None:
    """The inner dimension LoKR factors multiply to tells the Wan family apart the way a low-rank pair does."""
    mod, overrides = _mod(_state_dict(wan_lycoris_flat_lokr_format.state_dict_keys), "wan-lokr")

    assert LoRA_LyCORIS_Wan_Config.from_model_on_disk(mod, overrides).variant is WanLoRAVariantType.A14B


def test_the_sd3_flat_layout_is_still_sd3() -> None:
    """The one flat `lycoris_` layout identified before this change must not move."""
    sd = _state_dict(sd3_lokr_simpletuner_format.state_dict_keys)

    assert _accepting_lora_configs(sd, "sd3-simpletuner-lokr") == ["LoRA_LyCORIS_SD3_Config"]


def test_a_flat_lycoris_file_that_resolves_in_no_family_stays_unidentified() -> None:
    """The flattening alone is no evidence of any base: names no family's module list holds claim nothing."""
    keys = {
        "lycoris_totally_unseen_module.alpha": ([], "BF16"),
        "lycoris_totally_unseen_module.lokr_w1": ([8, 8], "BF16"),
        "lycoris_totally_unseen_module.lokr_w2": ([384, 384], "BF16"),
    }
    assert _accepting_lora_configs(_state_dict(keys), "unseen") == []
