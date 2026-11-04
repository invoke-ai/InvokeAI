"""ERNIE-Image LoRA identification, including mutual exclusivity with every other LoRA family.

``Config_Base.CONFIG_CLASSES`` is a set, so which config claims a file must not depend on probe order: each LoRA
family has to be exclusive at the per-config level. ERNIE-Image is the case that used to go wrong -- its LoRAs keep
the transformer blocks under ``layers.N`` behind ``diffusion_model.`` / ``transformer.``, exactly like Z-Image's,
so the Z-Image probe claimed them. What tells the two apart is ERNIE's block naming (``self_attention``,
``mlp.linear_fc2``) and top-level modules, against Z-Image's ``attention`` / ``feed_forward``.
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


_BASE = ernie_image_lora_aitoolkit_format.state_dict_keys
_Q = "diffusion_model.layers.5.self_attention.to_q"
_K = "diffusion_model.layers.5.self_attention.to_k"


def _with(extra: dict[str, list[int]], base: dict[str, list[int]] = _BASE) -> dict[str, list[int]]:
    return {**base, **extra}


_UNAPPLICABLE_FILES = {
    "lora_A without lora_B": ({f"{_Q}.lora_A.weight": [4, 4096]}, "low-rank pair"),
    "orphan alpha": ({f"{_K}.alpha": []}, "incomplete or inconsistent"),
    "lora_mid next to a valid pair": (
        {"diffusion_model.layers.0.self_attention.to_q.lora_mid.weight": [32, 32, 1, 1]},
        "cannot be applied",
    ),
    "low-rank and LoHA keys on one module": (
        {
            f"{_Q}.lora_A.weight": [32, 4096],
            f"{_Q}.lora_B.weight": [4096, 32],
            f"{_Q}.hada_w1_a": [4096, 4],
            f"{_Q}.hada_w1_b": [4, 4096],
        },
        "mixes layer types",
    ),
    "hada_w1_a only": ({f"{_Q}.hada_w1_a": [4096, 4]}, "incomplete or inconsistent"),
    "lokr_w1 only": ({f"{_Q}.lokr_w1": [8, 8]}, "incomplete or inconsistent"),
    "lokr_w2 only": ({f"{_Q}.lokr_w2": [512, 512]}, "incomplete or inconsistent"),
    "block beyond the model's 36": (
        {
            "diffusion_model.layers.99.self_attention.to_q.lora_A.weight": [32, 4096],
            "diffusion_model.layers.99.self_attention.to_q.lora_B.weight": [4096, 32],
        },
        "beyond the model's 36",
    ),
    "low-rank and only the second LoHA half on one module": (
        {
            f"{_Q}.lora_A.weight": [32, 4096],
            f"{_Q}.lora_B.weight": [4096, 32],
            f"{_Q}.hada_w2_a": [4096, 4],
            f"{_Q}.hada_w2_b": [4, 4096],
        },
        "mixes layer types",
    ),
    "wrong input width": (
        {f"{_Q}.lora_A.weight": [32, 3072], f"{_Q}.lora_B.weight": [4096, 32]},
        "shape",
    ),
    "wrong output width": (
        {f"{_Q}.lora_A.weight": [32, 4096], f"{_Q}.lora_B.weight": [12288, 32]},
        "shape",
    ),
    "LoKR whose Kronecker product is too small": (
        {f"{_Q}.lokr_w1": [8, 8], f"{_Q}.lokr_w2": [256, 256]},
        "shape",
    ),
    "the Conv2d patch embedding": (
        {
            "diffusion_model.x_embedder.proj.lora_A.weight": [32, 128],
            "diffusion_model.x_embedder.proj.lora_B.weight": [4096, 32],
        },
        "x_embedder.proj",
    ),
}


@pytest.mark.parametrize("case", list(_UNAPPLICABLE_FILES))
def test_an_ernie_image_lora_that_conversion_or_patching_would_fail_on_is_refused_at_install(case: str) -> None:
    """Each of these used to identify as ERNIE-Image and then fail (or drop a layer with a log line) at generation."""
    extra, message = _UNAPPLICABLE_FILES[case]
    sd = _state_dict(_with(extra))
    mod, overrides = _mod(sd, "ernie-defect")

    with pytest.raises(InvalidMatchError, match=message):
        LoRA_LyCORIS_ErnieImage_Config.from_model_on_disk(mod, overrides)
    assert _accepting_lora_configs(sd, "ernie-defect") == []


def test_complete_loha_and_lokr_layers_of_the_right_shape_are_accepted() -> None:
    keys = _with(
        {
            f"{_Q}.hada_w1_a": [4096, 4],
            f"{_Q}.hada_w1_b": [4, 4096],
            f"{_Q}.hada_w2_a": [4096, 4],
            f"{_Q}.hada_w2_b": [4, 4096],
            f"{_K}.lokr_w1_a": [8, 2],
            f"{_K}.lokr_w1_b": [2, 8],
            f"{_K}.lokr_w2": [512, 512],
        }
    )

    assert _accepting_lora_configs(_state_dict(keys), "ernie-complete") == ["LoRA_LyCORIS_ErnieImage_Config"]


def test_a_file_touching_only_the_mlp_and_the_modulation_is_an_ernie_image_lora_not_a_z_image_lora() -> None:
    """No `self_attention` or `linear_fc2` key to identify it: the top-level `adaLN_modulation.1` does."""
    keys = {
        "transformer.layers.0.mlp.gate_proj.lora_A.weight": [16, 4096],
        "transformer.layers.0.mlp.gate_proj.lora_B.weight": [12288, 16],
        "transformer.adaLN_modulation.1.lora_A.weight": [16, 4096],
        "transformer.adaLN_modulation.1.lora_B.weight": [24576, 16],
    }

    assert _accepting_lora_configs(_state_dict(keys), "ernie-mlp-only") == ["LoRA_LyCORIS_ErnieImage_Config"]


def test_the_weight_shape_table_matches_the_released_transformer() -> None:
    """The refusal compares updates against this table, so it has to be the model's own geometry."""
    from diffusers import ErnieImageTransformer2DModel

    from invokeai.backend.model_manager.load.model_loaders.ernie_image import ERNIE_IMAGE_TRANSFORMER_CONFIG
    from invokeai.backend.patches.lora_conversions.ernie_image_lora_constants import (
        ERNIE_IMAGE_NUM_LAYERS,
        ernie_image_weight_shape,
    )

    with torch.device("meta"):
        model = ErnieImageTransformer2DModel(**ERNIE_IMAGE_TRANSFORMER_CONFIG)

    assert ERNIE_IMAGE_NUM_LAYERS == len(model.layers)
    linears = {name: m for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    assert {name: tuple(m.weight.shape) for name, m in linears.items()} == {
        name: ernie_image_weight_shape(name) for name in linears
    }
    # The one Conv2d is deliberately outside the table (and the module regex): no sidecar pricing exists for it.
    assert ernie_image_weight_shape("x_embedder.proj") is None
