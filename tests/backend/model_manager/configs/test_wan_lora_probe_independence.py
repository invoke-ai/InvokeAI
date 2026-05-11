"""Regression tests for Wan vs Anima LoRA probe mutual exclusivity.

InvokeAI's ``Config_Base.CONFIG_CLASSES`` is a ``set``, so iteration order is
non-deterministic across Python process restarts. The probe MUST therefore be
mutually exclusive at the per-config level — first-match-wins is not safe to
rely on.

The historic bug these tests guard against: Anima's probe accepted anything
with the ``cross_attn`` or ``self_attn`` substring, which collides with Wan's
native LoRA key layout (``diffusion_model.blocks.X.cross_attn.q.lora_down.weight``).
A Wan native LoRA — including lightx2v's Lightning distillations — would
randomly identify as ``BaseModelType.Anima`` depending on dict hash order.

The fix tightened Anima's probe to require Cosmos-DiT-exclusive markers
(``mlp``, ``adaln_modulation``, or attention with the ``_proj`` suffix).

Each test below feeds a fixed state dict shape to BOTH the Wan and Anima
probes individually and asserts at most one accepts — order-independent.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.lora import (
    LoRA_LyCORIS_Anima_Config,
    LoRA_LyCORIS_Wan_Config,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType


def _t(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.zeros(shape)


def _make_mod(path: Path, sd: dict) -> MagicMock:
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = sd
    return mod


def _overrides(p: Path, name: str) -> dict:
    return {
        "hash": "test-hash",
        "path": str(p),
        "file_size": 0,
        "name": name,
        "source": str(p),
        "source_type": "path",
    }


def _probe(cls, path: Path, sd: dict, name: str):
    """Try a probe; return (accepted: bool, instance_or_exc)."""
    try:
        return True, cls.from_model_on_disk(_make_mod(path, sd), _overrides(path, name))
    except NotAMatchError as e:
        return False, e


def _i2v_lightning_v1_keys() -> dict:
    """Realistic key shape from lightx2v's I2V-A14B Lightning V1 — the actual
    LoRA that triggered the bug. Native upstream Wan naming with
    ``diffusion_model.`` prefix, no ``_proj`` suffix on attention."""
    sd: dict[str, torch.Tensor] = {}
    for block in range(3):
        for sub in ("self_attn", "cross_attn"):
            for proj in ("q", "k", "v", "o"):
                base = f"diffusion_model.blocks.{block}.{sub}.{proj}"
                sd[f"{base}.lora_down.weight"] = _t((64, 5120))
                sd[f"{base}.lora_up.weight"] = _t((5120, 64))
                sd[f"{base}.alpha"] = torch.tensor(8.0)
        for ffn_idx in (0, 2):
            base = f"diffusion_model.blocks.{block}.ffn.{ffn_idx}"
            sd[f"{base}.lora_down.weight"] = _t((64, 5120))
            sd[f"{base}.lora_up.weight"] = _t((5120, 64))
            sd[f"{base}.alpha"] = torch.tensor(8.0)
    return sd


def _t2v_lightning_v2_keys() -> dict:
    """Same layout as I2V Lightning — both lightx2v releases use native Wan
    keys with ``diffusion_model.`` prefix. The T2V version had been working
    (after a manual factory reorder), but only by luck of dict-hash order."""
    return _i2v_lightning_v1_keys()  # structurally identical to I2V V1


def _wan_kohya_keys() -> dict:
    """Hypothetical Kohya-format Wan LoRA — same native naming, underscore
    separators. Lightning hasn't shipped in this format, but other community
    LoRAs do."""
    sd: dict[str, torch.Tensor] = {}
    for block in range(2):
        for sub in ("self_attn", "cross_attn"):
            for proj in ("q", "k", "v", "o"):
                base = f"lora_unet_blocks_{block}_{sub}_{proj}"
                sd[f"{base}.lora_down.weight"] = _t((64, 5120))
                sd[f"{base}.lora_up.weight"] = _t((5120, 64))
    return sd


def _wan_diffusers_peft_keys() -> dict:
    """Wan diffusers-style LoRA: ``transformer.blocks.X.attn1.to_q.lora_A.weight``
    etc. Distinct enough from Anima that even the loose probes wouldn't collide,
    but covered here for completeness."""
    sd: dict[str, torch.Tensor] = {}
    for block in range(2):
        for attn in ("attn1", "attn2"):
            for to in ("to_q", "to_k", "to_v"):
                base = f"transformer.blocks.{block}.{attn}.{to}"
                sd[f"{base}.lora_A.weight"] = _t((64, 5120))
                sd[f"{base}.lora_B.weight"] = _t((5120, 64))
        sd[f"transformer.blocks.{block}.ffn.net.0.proj.lora_A.weight"] = _t((64, 5120))
        sd[f"transformer.blocks.{block}.ffn.net.0.proj.lora_B.weight"] = _t((13824, 64))
    return sd


def _anima_peft_keys() -> dict:
    """Realistic Anima Cosmos-DiT LoRA: ``q_proj``/``k_proj`` attention naming
    plus ``mlp`` and ``adaln_modulation`` modules. Wan has none of these."""
    sd: dict[str, torch.Tensor] = {}
    for block in range(2):
        for sub in ("self_attn", "cross_attn"):
            for proj in ("q_proj", "k_proj", "v_proj", "output_proj"):
                base = f"transformer.blocks.{block}.{sub}.{proj}"
                sd[f"{base}.lora_A.weight"] = _t((64, 4096))
                sd[f"{base}.lora_B.weight"] = _t((4096, 64))
        sd[f"transformer.blocks.{block}.mlp.layer1.lora_A.weight"] = _t((64, 4096))
        sd[f"transformer.blocks.{block}.mlp.layer1.lora_B.weight"] = _t((4096, 64))
        sd[f"transformer.blocks.{block}.adaln_modulation.linear.lora_A.weight"] = _t((64, 4096))
        sd[f"transformer.blocks.{block}.adaln_modulation.linear.lora_B.weight"] = _t((4096, 64))
    return sd


def _anima_kohya_keys() -> dict:
    """Same Anima content in Kohya format."""
    sd: dict[str, torch.Tensor] = {}
    for block in range(2):
        for sub in ("self_attn", "cross_attn"):
            for proj in ("q_proj", "k_proj", "v_proj", "output_proj"):
                base = f"lora_unet_blocks_{block}_{sub}_{proj}"
                sd[f"{base}.lora_down.weight"] = _t((64, 4096))
                sd[f"{base}.lora_up.weight"] = _t((4096, 64))
        sd[f"lora_unet_blocks_{block}_mlp_layer1.lora_down.weight"] = _t((64, 4096))
        sd[f"lora_unet_blocks_{block}_mlp_layer1.lora_up.weight"] = _t((4096, 64))
    return sd


# ---------------------------------------------------------------------------
# Mutual-exclusivity assertions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, sd_builder",
    [
        ("i2v_lightning_v1", _i2v_lightning_v1_keys),
        ("t2v_lightning_v2", _t2v_lightning_v2_keys),
        ("wan_kohya_native", _wan_kohya_keys),
        ("wan_diffusers_peft", _wan_diffusers_peft_keys),
    ],
)
def test_wan_loras_only_match_wan(label: str, sd_builder) -> None:
    """Wan probe accepts; Anima probe rejects. Independent of factory order."""
    sd = sd_builder()
    with TemporaryDirectory() as tmp:
        f = Path(tmp) / f"{label}.safetensors"
        f.touch()

        wan_ok, wan_result = _probe(LoRA_LyCORIS_Wan_Config, f, sd, label)
        anima_ok, anima_result = _probe(LoRA_LyCORIS_Anima_Config, f, sd, label)

    assert wan_ok, f"Wan probe must accept {label}; got {wan_result}"
    assert wan_result.base == BaseModelType.Wan
    assert not anima_ok, (
        f"Anima probe must reject {label} so probing is order-independent. "
        f"Instead it accepted: {anima_result}"
    )


@pytest.mark.parametrize(
    "label, sd_builder",
    [
        ("anima_peft", _anima_peft_keys),
        ("anima_kohya", _anima_kohya_keys),
    ],
)
def test_anima_loras_only_match_anima(label: str, sd_builder) -> None:
    """Anima probe accepts; Wan probe rejects. Independent of factory order."""
    sd = sd_builder()
    with TemporaryDirectory() as tmp:
        f = Path(tmp) / f"{label}.safetensors"
        f.touch()

        wan_ok, wan_result = _probe(LoRA_LyCORIS_Wan_Config, f, sd, label)
        anima_ok, anima_result = _probe(LoRA_LyCORIS_Anima_Config, f, sd, label)

    assert anima_ok, f"Anima probe must accept {label}; got {anima_result}"
    assert anima_result.base == BaseModelType.Anima
    assert not wan_ok, (
        f"Wan probe must reject {label} so probing is order-independent. "
        f"Instead it accepted: {wan_result}"
    )


# ---------------------------------------------------------------------------
# Belt-and-suspenders: confirm CONFIG_CLASSES doesn't ALSO produce a match for
# any unrelated LoRA config. This is the test that would have caught the
# original bug regardless of which LoRA configs are registered in the future.
# ---------------------------------------------------------------------------


def test_at_most_one_lora_config_matches_wan_lightning() -> None:
    """Run every LoRA config in the factory against an I2V Lightning state
    dict. Only one should accept. If a future LoRA config (a hypothetical
    new model with cross_attn naming) starts matching too, this test fires
    so we can tighten that probe rather than relying on factory ordering."""
    from invokeai.backend.model_manager.configs.base import Config_Base
    from invokeai.backend.model_manager.taxonomy import ModelType

    sd = _i2v_lightning_v1_keys()
    with TemporaryDirectory() as tmp:
        f = Path(tmp) / "wan_lightning.safetensors"
        f.touch()
        mod = _make_mod(f, sd)
        overrides = _overrides(f, "wan_lightning")

        accepting: list[str] = []
        for cls in Config_Base.CONFIG_CLASSES:
            # Only LoRA configs are at risk of collision with each other; skip
            # the rest. (Main models can also probe-accept-then-reject on type
            # mismatch, but they're disambiguated by ``matches_sort_key``.)
            if getattr(cls.model_fields.get("type", None), "default", None) != ModelType.LoRA:
                continue
            try:
                cls.from_model_on_disk(mod, dict(overrides))
                accepting.append(cls.__name__)
            except (NotAMatchError, Exception):
                continue

    assert accepting == ["LoRA_LyCORIS_Wan_Config"], (
        f"Exactly one LoRA config must accept a Wan Lightning LoRA; got {accepting}. "
        "If a new LoRA config starts matching here, tighten its probe to be "
        "mutually exclusive with Wan rather than relying on factory ordering."
    )


def test_at_most_one_lora_config_matches_anima_peft() -> None:
    """Same exclusivity guarantee for the Anima side."""
    from invokeai.backend.model_manager.configs.base import Config_Base
    from invokeai.backend.model_manager.taxonomy import ModelType

    sd = _anima_peft_keys()
    with TemporaryDirectory() as tmp:
        f = Path(tmp) / "anima_peft.safetensors"
        f.touch()
        mod = _make_mod(f, sd)
        overrides = _overrides(f, "anima_peft")

        accepting: list[str] = []
        for cls in Config_Base.CONFIG_CLASSES:
            if getattr(cls.model_fields.get("type", None), "default", None) != ModelType.LoRA:
                continue
            try:
                cls.from_model_on_disk(mod, dict(overrides))
                accepting.append(cls.__name__)
            except (NotAMatchError, Exception):
                continue

    assert accepting == ["LoRA_LyCORIS_Anima_Config"], (
        f"Exactly one LoRA config must accept an Anima LoRA; got {accepting}."
    )
