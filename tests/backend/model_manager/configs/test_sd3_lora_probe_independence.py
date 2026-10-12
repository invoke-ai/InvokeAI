"""SD 3 LoRA identification, including mutual exclusivity with every other LoRA family.

``Config_Base.CONFIG_CLASSES`` is a set, so which config claims a file must not depend on probe order: each LoRA
family has to be exclusive at the per-config level. SD 3 is the case that used to go wrong: its diffusers LoRAs name
the same attention modules as Qwen-Image's (`add_q_proj`, `to_add_out`), so the Qwen-Image probe claimed them, and a
kohya file that trains both CLIP encoders read as SDXL. Widths tell them apart: 1536 / 2432 against Qwen-Image's 3072.
"""

import atexit
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.lora import LoRA_Diffusers_SD3_Config, LoRA_LyCORIS_SD3_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import ModelType, SD3VariantType
from tests.backend.patches.lora_conversions.lora_state_dicts import (
    anima_lora_kohya_format,
    anima_lora_kohya_with_te_format,
    anima_lora_lokr_format,
    anima_lora_peft_format,
    flux_control_lora_format,
    flux_dora_onetrainer_format,
    flux_lokr_bfl_format,
    flux_lora_aitoolkit_format,
    flux_lora_diffusers_base_model_format,
    flux_lora_diffusers_format,
    flux_lora_diffusers_no_proj_mlp_format,
    flux_lora_kohya_format,
    flux_lora_kohya_with_te1_format,
    flux_lora_xlabs_format,
    krea2_lora_kohya_format,
    qwen_image_lora_diffusers_format,
    qwen_image_lora_kohya_format,
    sd3_lokr_simpletuner_format,
    sd3_lora_diffusers_format,
    sd3_lora_diffusers_full_format,
    sd3_lora_diffusers_te_format,
    sd3_lora_kohya_format,
    sd3_lora_peft_format,
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


def _widened(keys: dict, width: int) -> dict[str, list[int]]:
    """The same layout at another transformer width: every 1536 becomes `width`."""
    return {key: [width if dim == 1536 else dim for dim in _shape(value)] for key, value in keys.items()}


_SD3_LAYOUTS = {
    "diffusers": (sd3_lora_diffusers_format.state_dict_keys, SD3VariantType.Medium),
    "peft": (sd3_lora_peft_format.state_dict_keys, SD3VariantType.Medium),
    "diffusers-full": (sd3_lora_diffusers_full_format.state_dict_keys, SD3VariantType.Medium),
    "diffusers-te": (sd3_lora_diffusers_te_format.state_dict_keys, SD3VariantType.Medium),
    "simpletuner-lokr": (sd3_lokr_simpletuner_format.state_dict_keys, SD3VariantType.Medium),
    "kohya": (sd3_lora_kohya_format.state_dict_keys, SD3VariantType.Medium),
    "diffusers-large": (_widened(sd3_lora_diffusers_format.state_dict_keys, 2432), SD3VariantType.Large),
    "kohya-large": (_widened(sd3_lora_kohya_format.state_dict_keys, 2432), SD3VariantType.Large),
}


@pytest.mark.parametrize("layout", list(_SD3_LAYOUTS))
def test_an_sd3_lora_is_claimed_by_sd3_alone_with_its_size(layout: str) -> None:
    keys, variant = _SD3_LAYOUTS[layout]
    sd = _state_dict(keys)

    assert _accepting_lora_configs(sd, "sd3") == ["LoRA_LyCORIS_SD3_Config"]
    mod, overrides = _mod(sd, "sd3")
    assert LoRA_LyCORIS_SD3_Config.from_model_on_disk(mod, overrides).variant is variant


_OTHER_FAMILIES = {
    "anima-kohya": anima_lora_kohya_format,
    "anima-kohya-te": anima_lora_kohya_with_te_format,
    "anima-lokr": anima_lora_lokr_format,
    "anima-peft": anima_lora_peft_format,
    "flux-aitoolkit": flux_lora_aitoolkit_format,
    "flux-diffusers": flux_lora_diffusers_format,
    "flux-control": flux_control_lora_format,
    "flux-diffusers-base-model": flux_lora_diffusers_base_model_format,
    "flux-diffusers-no-proj-mlp": flux_lora_diffusers_no_proj_mlp_format,
    "flux-xlabs": flux_lora_xlabs_format,
    "flux-dora-onetrainer": flux_dora_onetrainer_format,
    "flux-kohya": flux_lora_kohya_format,
    "flux-kohya-te1": flux_lora_kohya_with_te1_format,
    "flux-lokr-bfl": flux_lokr_bfl_format,
    "krea2-kohya": krea2_lora_kohya_format,
    "qwen-image-diffusers": qwen_image_lora_diffusers_format,
    "qwen-image-kohya": qwen_image_lora_kohya_format,
}


@pytest.mark.parametrize("family", list(_OTHER_FAMILIES))
def test_the_sd3_probe_leaves_other_families_alone(family: str) -> None:
    mod, overrides = _mod(_state_dict(_OTHER_FAMILIES[family].state_dict_keys), family)

    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_SD3_Config.from_model_on_disk(mod, overrides)


@pytest.mark.parametrize("family", ["qwen-image-diffusers", "qwen-image-kohya"])
def test_a_qwen_image_lora_is_still_claimed_by_qwen_image_alone(family: str) -> None:
    sd = _state_dict(_OTHER_FAMILIES[family].state_dict_keys)

    assert _accepting_lora_configs(sd, family) == ["LoRA_LyCORIS_QwenImage_Config"]


@pytest.mark.parametrize("spelling", ["diffusers", "kohya"])
def test_a_cogview4_lora_is_no_longer_taken_for_qwen_image(spelling: str) -> None:
    """CogView4 keeps its blocks under `transformer_blocks.N.attn1`; it has no LoRA support, so nothing claims it."""
    keys: dict[str, list[int]] = {}
    for proj in ("to_q", "to_k", "to_v", "to_out.0"):
        layer = (
            f"transformer.transformer_blocks.0.attn1.{proj}"
            if spelling == "diffusers"
            else f"lora_unet_transformer_blocks_0_attn1_{proj.replace('.', '_')}"
        )
        down, up = ("lora_A", "lora_B") if spelling == "diffusers" else ("lora_down", "lora_up")
        keys[f"{layer}.{down}.weight"] = [16, 4096]
        keys[f"{layer}.{up}.weight"] = [4096, 16]

    assert _accepting_lora_configs(_state_dict(keys), f"cogview4-{spelling}") == []


def _clip_layers(prefix_l: str, prefix_g: str) -> dict[str, list[int]]:
    """kohya CLIP-L and CLIP-G attention layers: what `lora_token_vector_length` adds up to SDXL's 2048.

    In this order: it reads the two `lora_down` widths as it meets them, and a `lora_up` in between resets them.
    """
    return {
        f"{prefix_l}_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight": [4, 768],
        f"{prefix_g}_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight": [4, 1280],
        f"{prefix_l}_text_model_encoder_layers_0_self_attn_k_proj.lora_up.weight": [768, 4],
        f"{prefix_g}_text_model_encoder_layers_0_self_attn_k_proj.lora_up.weight": [1280, 4],
    }


def test_a_kohya_sd3_lora_that_trains_both_clips_is_not_taken_for_sdxl() -> None:
    keys = {**_clip_layers("lora_te1", "lora_te2"), **sd3_lora_kohya_format.state_dict_keys}

    assert _accepting_lora_configs(_state_dict(keys), "sd3-te") == ["LoRA_LyCORIS_SD3_Config"]


def test_an_sdxl_lora_that_trains_both_clips_is_still_sdxl() -> None:
    keys = {
        **_clip_layers("lora_te1", "lora_te2"),
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight": [4, 2048],
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight": [640, 4],
    }

    assert _accepting_lora_configs(_state_dict(keys), "sdxl-te") == ["LoRA_LyCORIS_SDXL_Config"]


def test_a_dora_sd3_lora_is_refused_at_install() -> None:
    """Recognised but unusable: the file must fail at install rather than during a generation -- and not fall back
    to "unknown"."""
    keys = dict(sd3_lora_diffusers_format.state_dict_keys)
    keys["transformer.transformer_blocks.0.attn.to_q.lora_magnitude_vector.weight"] = [1536]
    sd = _state_dict(keys)
    mod, overrides = _mod(sd, "sd3-dora")

    with pytest.raises(InvalidMatchError, match="lora_magnitude_vector"):
        LoRA_LyCORIS_SD3_Config.from_model_on_disk(mod, overrides)
    assert _accepting_lora_configs(sd, "sd3-dora") == []


def _write_dreambooth_folder(name: str, keys: dict) -> Path:
    folder = Path(_TMP_DIR.name) / name
    (folder / "checkpoint-500").mkdir(parents=True)
    save_file(
        {key: torch.zeros(_shape(value)) for key, value in keys.items()}, folder / "pytorch_lora_weights.safetensors"
    )
    # The intermediate checkpoints DreamBooth leaves next to the final weights are not what the folder installs. This
    # one is written at Large's width, so reading it instead would show in the variant.
    large = {key: torch.zeros(shape) for key, shape in _widened(keys, 2432).items()}
    save_file(large, folder / "checkpoint-500" / "pytorch_lora_weights.safetensors")
    return folder


def test_a_dreambooth_folder_is_an_sd3_diffusers_lora() -> None:
    """The DreamBooth script's folders are the most common SD 3 LoRA on the Hub, and were "unknown" until now."""
    folder = _write_dreambooth_folder("sd3-dreambooth", sd3_lora_diffusers_te_format.state_dict_keys)
    overrides = {
        "hash": "x",
        "path": str(folder),
        "file_size": 0,
        "name": "x",
        "source": str(folder),
        "source_type": "path",
    }

    accepting = []
    for cls in list(Config_Base.CONFIG_CLASSES):
        if getattr(cls.model_fields.get("type", None), "default", None) != ModelType.LoRA:
            continue
        try:
            cls.from_model_on_disk(ModelOnDisk(folder), dict(overrides))
        except Exception:
            continue
        accepting.append(cls.__name__)

    assert accepting == ["LoRA_Diffusers_SD3_Config"]
    assert LoRA_Diffusers_SD3_Config.from_model_on_disk(ModelOnDisk(folder), dict(overrides)).variant is (
        SD3VariantType.Medium
    )


def test_onetrainers_flat_names_are_not_taken_for_sd3() -> None:
    """`lora_transformer_` + flattened diffusers names is how OneTrainer writes FLUX LoRAs; at SD 3's width the SD 3
    probe would claim the same file, and a tie between two LoRA configs is decided by set order. No published SD 3
    LoRA was found in that spelling, so SD 3 leaves it alone."""
    keys: dict[str, list[int]] = {}
    for proj in ("to_q", "to_k", "to_v"):
        keys[f"lora_transformer_transformer_blocks_0_attn_{proj}.lora_down.weight"] = [4, 1536]
        keys[f"lora_transformer_transformer_blocks_0_attn_{proj}.lora_up.weight"] = [1536, 4]

    accepting = _accepting_lora_configs(_state_dict(keys), "onetrainer-flat")

    assert "LoRA_LyCORIS_SD3_Config" not in accepting
    assert len(accepting) <= 1
