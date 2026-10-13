"""ERNIE-Image LoRA identification, including mutual exclusivity with every other LoRA family.

``Config_Base.CONFIG_CLASSES`` is a set, so which config claims a file must not depend on probe order: each LoRA
family has to be exclusive at the per-config level. ERNIE-Image is the case that used to go wrong -- its LoRAs keep
the transformer blocks under ``layers.N`` behind ``diffusion_model.`` / ``transformer.``, exactly like Z-Image's,
so the Z-Image probe claimed them. What tells the two apart is ERNIE's block naming (``self_attention``,
``mlp.gate_proj``), against Z-Image's ``attention`` / ``feed_forward``.
"""

import atexit
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_ErnieImage_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from tests.backend.patches.lora_conversions.lora_state_dicts import (
    anima_lora_kohya_format,
    anima_lora_lokr_format,
    anima_lora_peft_format,
    ernie_image_lora_aitoolkit_format,
    ernie_image_lora_peft_format,
    flux_dora_onetrainer_format,
    flux_lokr_bfl_format,
    flux_lora_aitoolkit_format,
    flux_lora_diffusers_base_model_format,
    flux_lora_diffusers_format,
    flux_lora_kohya_format,
    krea2_lora_kohya_format,
    qwen_image_lora_diffusers_format,
    qwen_image_lora_kohya_format,
    z_image_lora_aitoolkit_format,
)

_TMP_DIR = TemporaryDirectory()
atexit.register(_TMP_DIR.cleanup)


def _state_dict(keys: dict[str, list[int]]) -> dict[str, torch.Tensor]:
    """The layout with a single element of storage per tensor: probes read names and shapes, never values."""
    return {key: torch.zeros((), dtype=torch.float32).expand(shape) for key, shape in keys.items()}


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
    for cls in Config_Base.CONFIG_CLASSES:
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


def _reprefixed(keys: dict[str, list[int]], prefix: str) -> dict[str, list[int]]:
    return {prefix + key.removeprefix("diffusion_model."): shape for key, shape in keys.items()}


def _lokr(keys: dict[str, list[int]]) -> dict[str, list[int]]:
    """The same modules as LoKR layers (ai-toolkit's LoKR spelling: `lokr_w1` [8, 8] and a full `lokr_w2`)."""
    out: dict[str, list[int]] = {}
    for key, shape in keys.items():
        if key.endswith(".lora_A.weight"):
            module = key.removesuffix(".lora_A.weight")
            rows = keys[f"{module}.lora_B.weight"][0]
            out[f"{module}.lokr_w1"] = [8, 8]
            out[f"{module}.lokr_w2"] = [rows // 8, shape[1] // 8]
            out[f"{module}.alpha"] = []
    return out


_ERNIE_LAYOUTS = {
    "ai-toolkit": ernie_image_lora_aitoolkit_format.state_dict_keys,
    "peft-top-level": ernie_image_lora_peft_format.state_dict_keys,
    # The other container prefixes the conversion strips, over the same modules.
    **{
        f"prefix={prefix or '<none>'}": _reprefixed(ernie_image_lora_aitoolkit_format.state_dict_keys, prefix)
        for prefix in (
            "model.diffusion_model.",
            "transformer.",
            "base_model.model.",
            "base_model.model.transformer.",
            "",
        )
    },
    "lokr": _lokr(ernie_image_lora_aitoolkit_format.state_dict_keys),
}


@pytest.mark.parametrize("layout", list(_ERNIE_LAYOUTS))
def test_an_ernie_image_lora_is_claimed_by_ernie_image_alone(layout: str) -> None:
    sd = _state_dict(_ERNIE_LAYOUTS[layout])

    assert _accepting_lora_configs(sd, "ernie") == ["LoRA_LyCORIS_ErnieImage_Config"]
    mod, overrides = _mod(sd, "ernie")
    assert LoRA_LyCORIS_ErnieImage_Config.from_model_on_disk(mod, overrides).base is BaseModelType.ErnieImage


_OTHER_FAMILIES = {
    "anima-kohya": anima_lora_kohya_format,
    "anima-lokr": anima_lora_lokr_format,
    "anima-peft": anima_lora_peft_format,
    "flux-aitoolkit": flux_lora_aitoolkit_format,
    "flux-diffusers": flux_lora_diffusers_format,
    "flux-diffusers-base-model": flux_lora_diffusers_base_model_format,
    "flux-dora-onetrainer": flux_dora_onetrainer_format,
    "flux-lokr-bfl": flux_lokr_bfl_format,
    "flux-kohya": flux_lora_kohya_format,
    "krea2-kohya": krea2_lora_kohya_format,
    "qwen-image-diffusers": qwen_image_lora_diffusers_format,
    "qwen-image-kohya": qwen_image_lora_kohya_format,
    "z-image-aitoolkit": z_image_lora_aitoolkit_format,
}


@pytest.mark.parametrize("family", list(_OTHER_FAMILIES))
def test_the_ernie_image_probe_leaves_other_families_alone(family: str) -> None:
    mod, overrides = _mod(_state_dict(_OTHER_FAMILIES[family].state_dict_keys), family)

    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_ErnieImage_Config.from_model_on_disk(mod, overrides)


def test_a_z_image_lora_is_still_claimed_by_z_image_alone() -> None:
    sd = _state_dict(z_image_lora_aitoolkit_format.state_dict_keys)

    assert _accepting_lora_configs(sd, "z-image") == ["LoRA_LyCORIS_ZImage_Config"]


def test_a_z_image_lora_with_text_encoder_layers_is_still_claimed_by_z_image_alone() -> None:
    """Qwen3's MLP is named like ERNIE-Image's (`mlp.gate_proj`), and Z-Image LoRAs may train the encoder under
    `text_encoder.` (the Z-Image conversion maps that prefix). No published file found does, which is why this
    one is built from the real transformer layout plus encoder layers in that documented spelling."""
    keys = dict(z_image_lora_aitoolkit_format.state_dict_keys)
    for proj, (rows, cols) in {"gate_proj": (9728, 2560), "up_proj": (9728, 2560), "down_proj": (2560, 9728)}.items():
        keys[f"text_encoder.layers.0.mlp.{proj}.lora_A.weight"] = [16, cols]
        keys[f"text_encoder.layers.0.mlp.{proj}.lora_B.weight"] = [rows, 16]

    assert _accepting_lora_configs(_state_dict(keys), "z-image-te") == ["LoRA_LyCORIS_ZImage_Config"]


def test_an_llm_lora_on_the_shared_mlp_names_is_not_taken_for_ernie_image() -> None:
    """A PEFT LoRA over a Qwen3-style LLM trains `mlp.gate_proj` / `up_proj` under the same `base_model.model.`
    prefix ERNIE-Image's conversion strips. Those names only confirm a file the ERNIE-only names identified."""
    keys: dict[str, list[int]] = {}
    for proj, (rows, cols) in {"gate_proj": (9728, 2560), "up_proj": (9728, 2560), "down_proj": (2560, 9728)}.items():
        keys[f"base_model.model.layers.0.mlp.{proj}.lora_A.weight"] = [16, cols]
        keys[f"base_model.model.layers.0.mlp.{proj}.lora_B.weight"] = [rows, 16]
    mod, overrides = _mod(_state_dict(keys), "llm")

    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_ErnieImage_Config.from_model_on_disk(mod, overrides)


def test_an_ernie_image_lora_with_layers_on_no_transformer_module_is_refused_at_install() -> None:
    """The generation would skip those layers with only a log warning, so the LoRA would silently do less."""
    keys = dict(ernie_image_lora_aitoolkit_format.state_dict_keys)
    keys["text_encoders.layers.0.mlp.gate_proj.lora_A.weight"] = [16, 4096]
    keys["text_encoders.layers.0.mlp.gate_proj.lora_B.weight"] = [12288, 16]
    mod, overrides = _mod(_state_dict(keys), "ernie-stray")

    with pytest.raises(InvalidMatchError, match="text_encoders"):
        LoRA_LyCORIS_ErnieImage_Config.from_model_on_disk(mod, overrides)


def test_a_dora_ernie_image_lora_is_refused_at_install() -> None:
    """Recognised but unusable: the sidecar wrappers of the quantized builds cannot feed DoRA the real weight, so
    the file must fail at install rather than half-way into a generation -- and not fall back to "unknown"."""
    keys = dict(ernie_image_lora_aitoolkit_format.state_dict_keys)
    keys["diffusion_model.layers.0.self_attention.to_q.lora_magnitude_vector.weight"] = [4096]
    sd = _state_dict(keys)
    mod, overrides = _mod(sd, "ernie-dora")

    with pytest.raises(InvalidMatchError, match="lora_magnitude_vector"):
        LoRA_LyCORIS_ErnieImage_Config.from_model_on_disk(mod, overrides)
    assert _accepting_lora_configs(sd, "ernie-dora") == []


def test_an_ernie_image_transformer_checkpoint_is_not_taken_for_a_lora() -> None:
    """Same module names, but only weights and biases -- no refusal message about LoRA tensors either."""
    sd = _state_dict(
        {
            "layers.0.self_attention.to_q.weight": [4096, 4096],
            "layers.0.mlp.gate_proj.weight": [12288, 4096],
            "layers.0.mlp.linear_fc2.weight": [4096, 12288],
        }
    )
    mod, overrides = _mod(sd, "ernie-checkpoint")

    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_ErnieImage_Config.from_model_on_disk(mod, overrides)
