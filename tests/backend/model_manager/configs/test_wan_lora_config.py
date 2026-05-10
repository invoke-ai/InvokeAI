"""Tests for the Wan LoRA probe (LoRA_LyCORIS_Wan_Config).

These tests cover detection across the three formats Wan LoRAs ship in:

- **Diffusers PEFT**, with or without a ``transformer.`` prefix
- **Native upstream PEFT** with ``diffusion_model.`` prefix (ComfyUI-trained)
- **Kohya** ``lora_unet_blocks_N_<submodule>`` with both diffusers and native
  attention naming

And the anti-pattern guards that prevent false positives on:

- Anima (Cosmos DiT — ``cross_attn_q_proj`` / ``mlp`` / ``adaln_modulation``)
- QwenImage (``transformer_blocks.``)
- Flux (``double_blocks`` / ``single_blocks`` / ``single_transformer_blocks``)
- Z-Image (``diffusion_model.layers.``)
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_Wan_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat
from invokeai.backend.patches.lora_conversions.wan_lora_constants import (
    has_non_wan_architecture_keys,
    has_wan_kohya_keys,
    has_wan_peft_keys,
)


def _make_mod(path: Path, sd: dict) -> MagicMock:
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = sd
    return mod


def _overrides(model_path: Path, name: str) -> dict:
    return {
        "hash": "test-hash",
        "path": str(model_path),
        "file_size": 0,
        "name": name,
        "source": str(model_path),
        "source_type": "path",
    }


def _t(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.zeros(shape)


class TestDiffusersPEFTPositives:
    def test_attn1_to_q(self):
        keys = ["transformer.blocks.0.attn1.to_q.lora_A.weight"]
        assert has_wan_peft_keys(keys) is True
        assert has_non_wan_architecture_keys(keys) is False

    def test_attn2_to_k(self):
        keys = ["blocks.0.attn2.to_k.lora_A.weight"]
        assert has_wan_peft_keys(keys) is True
        assert has_non_wan_architecture_keys(keys) is False

    def test_ffn_net(self):
        keys = ["transformer.blocks.0.ffn.net.0.proj.lora_A.weight"]
        assert has_wan_peft_keys(keys) is True
        assert has_non_wan_architecture_keys(keys) is False

    def test_base_model_peft_prefix(self):
        keys = ["base_model.model.transformer.blocks.0.attn1.to_q.lora_A.weight"]
        assert has_wan_peft_keys(keys) is True
        assert has_non_wan_architecture_keys(keys) is False


class TestNativePEFTPositives:
    def test_self_attn_q(self):
        keys = ["diffusion_model.blocks.0.self_attn.q.lora_A.weight"]
        assert has_wan_peft_keys(keys) is True
        assert has_non_wan_architecture_keys(keys) is False

    def test_cross_attn_k(self):
        keys = ["diffusion_model.blocks.0.cross_attn.k.lora_A.weight"]
        assert has_wan_peft_keys(keys) is True
        assert has_non_wan_architecture_keys(keys) is False

    def test_cross_attn_o(self):
        keys = ["transformer.blocks.0.cross_attn.o.lora_A.weight"]
        assert has_wan_peft_keys(keys) is True
        assert has_non_wan_architecture_keys(keys) is False

    def test_ffn_native(self):
        keys = ["diffusion_model.blocks.0.ffn.0.lora_A.weight"]
        assert has_wan_peft_keys(keys) is True
        assert has_non_wan_architecture_keys(keys) is False


class TestKohyaPositives:
    def test_kohya_diffusers_attn1_to_q(self):
        keys = ["lora_unet_blocks_0_attn1_to_q.lora_down.weight"]
        assert has_wan_kohya_keys(keys) is True
        assert has_non_wan_architecture_keys(keys) is False

    def test_kohya_diffusers_attn2_to_out(self):
        keys = ["lora_unet_blocks_0_attn2_to_out_0.lora_down.weight"]
        assert has_wan_kohya_keys(keys) is True
        assert has_non_wan_architecture_keys(keys) is False

    def test_kohya_native_self_attn_q(self):
        keys = ["lora_unet_blocks_0_self_attn_q.lora_down.weight"]
        assert has_wan_kohya_keys(keys) is True
        assert has_non_wan_architecture_keys(keys) is False

    def test_kohya_native_cross_attn_v(self):
        keys = ["lora_unet_blocks_5_cross_attn_v.lora_down.weight"]
        assert has_wan_kohya_keys(keys) is True
        assert has_non_wan_architecture_keys(keys) is False

    def test_kohya_native_ffn_0(self):
        keys = ["lora_unet_blocks_0_ffn_0.lora_down.weight"]
        assert has_wan_kohya_keys(keys) is True
        assert has_non_wan_architecture_keys(keys) is False


class TestArchitectureGuards:
    """Anti-pattern checks: non-Wan architectures must be flagged so the
    probe rejects them even if a wan-ish substring matches."""

    @pytest.mark.parametrize(
        "label, keys",
        [
            ("anima_kohya_q_proj",
             ["lora_unet_blocks_0_cross_attn_q_proj.lora_down.weight"]),
            ("anima_peft_mlp",
             ["transformer.blocks.0.mlp.layer1.lora_A.weight"]),
            ("anima_peft_adaln",
             ["transformer.blocks.0.adaln_modulation.linear.lora_A.weight"]),
            ("anima_peft_self_attn_q_proj",
             ["transformer.blocks.0.self_attn.q_proj.lora_A.weight"]),
            ("qwen_image",
             ["transformer_blocks.0.attn.to_q.lora_A.weight"]),
            ("flux_kohya_double",
             ["lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight"]),
            ("flux_kohya_single",
             ["lora_unet_single_blocks_0_linear1.lora_down.weight"]),
            ("flux_diffusers_single_transformer",
             ["transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight"]),
            ("z_image",
             ["diffusion_model.layers.0.attn.to_q.lora_A.weight"]),
        ],
    )
    def test_non_wan_archs_are_flagged(self, label: str, keys: list[str]):
        assert has_non_wan_architecture_keys(keys) is True


class TestProbeAcceptance:
    """End-to-end probe behavior — Wan LoRA must be accepted, non-Wan rejected."""

    def _wan_diffusers_sd(self) -> dict:
        return {
            "transformer.blocks.0.attn1.to_q.lora_A.weight": _t((128, 5120)),
            "transformer.blocks.0.attn1.to_q.lora_B.weight": _t((5120, 128)),
            "transformer.blocks.0.ffn.net.0.proj.lora_A.weight": _t((128, 5120)),
            "transformer.blocks.0.ffn.net.0.proj.lora_B.weight": _t((13824, 128)),
        }

    def _wan_native_sd(self) -> dict:
        return {
            "diffusion_model.blocks.0.self_attn.q.lora_A.weight": _t((128, 5120)),
            "diffusion_model.blocks.0.self_attn.q.lora_B.weight": _t((5120, 128)),
        }

    def _wan_kohya_sd(self) -> dict:
        return {
            "lora_unet_blocks_0_attn1_to_q.lora_down.weight": _t((128, 5120)),
            "lora_unet_blocks_0_attn1_to_q.lora_up.weight": _t((5120, 128)),
        }

    def test_accepts_diffusers_wan(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "my-wan-lora.safetensors"
            f.touch()
            cfg = LoRA_LyCORIS_Wan_Config.from_model_on_disk(
                _make_mod(f, self._wan_diffusers_sd()),
                _overrides(f, "wan-lora"),
            )
            assert cfg.base == BaseModelType.Wan
            assert cfg.format == ModelFormat.LyCORIS
            assert cfg.expert is None

    def test_accepts_native_wan(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "wan-style-lora.safetensors"
            f.touch()
            cfg = LoRA_LyCORIS_Wan_Config.from_model_on_disk(
                _make_mod(f, self._wan_native_sd()),
                _overrides(f, "wan-native"),
            )
            assert cfg.base == BaseModelType.Wan

    def test_accepts_kohya_wan(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "wan-kohya.safetensors"
            f.touch()
            cfg = LoRA_LyCORIS_Wan_Config.from_model_on_disk(
                _make_mod(f, self._wan_kohya_sd()),
                _overrides(f, "wan-kohya"),
            )
            assert cfg.base == BaseModelType.Wan

    def test_filename_marks_high_noise_expert(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "stylize-high_noise.safetensors"
            f.touch()
            cfg = LoRA_LyCORIS_Wan_Config.from_model_on_disk(
                _make_mod(f, self._wan_diffusers_sd()),
                _overrides(f, "high-noise lora"),
            )
            assert cfg.expert == "high"

    def test_filename_marks_low_noise_expert(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "fine-detail-LowNoise.safetensors"
            f.touch()
            cfg = LoRA_LyCORIS_Wan_Config.from_model_on_disk(
                _make_mod(f, self._wan_diffusers_sd()),
                _overrides(f, "low-noise lora"),
            )
            assert cfg.expert == "low"

    def test_explicit_expert_override_wins(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "ambiguous-name.safetensors"
            f.touch()
            overrides = _overrides(f, "override")
            overrides["expert"] = "low"
            cfg = LoRA_LyCORIS_Wan_Config.from_model_on_disk(
                _make_mod(f, self._wan_diffusers_sd()),
                overrides,
            )
            assert cfg.expert == "low"

    def test_expert_none_for_untagged_filename(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "my-lora.safetensors"
            f.touch()
            cfg = LoRA_LyCORIS_Wan_Config.from_model_on_disk(
                _make_mod(f, self._wan_diffusers_sd()),
                _overrides(f, "untagged"),
            )
            assert cfg.expert is None

    def test_rejects_anima_lora(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "anima.safetensors"
            f.touch()
            sd = {
                "transformer.blocks.0.cross_attn.q_proj.lora_A.weight": _t((128, 4096)),
                "transformer.blocks.0.mlp.layer1.lora_A.weight": _t((128, 4096)),
            }
            with pytest.raises(NotAMatchError, match="Wan LoRA"):
                LoRA_LyCORIS_Wan_Config.from_model_on_disk(_make_mod(f, sd), _overrides(f, "anima"))

    def test_rejects_qwen_image_lora(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "qwen.safetensors"
            f.touch()
            sd = {"transformer_blocks.0.attn.to_q.lora_A.weight": _t((128, 4096))}
            with pytest.raises(NotAMatchError, match="Wan LoRA"):
                LoRA_LyCORIS_Wan_Config.from_model_on_disk(_make_mod(f, sd), _overrides(f, "qwen"))

    def test_rejects_flux_lora(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "flux.safetensors"
            f.touch()
            sd = {"lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": _t((128, 3072))}
            with pytest.raises(NotAMatchError, match="Wan LoRA"):
                LoRA_LyCORIS_Wan_Config.from_model_on_disk(_make_mod(f, sd), _overrides(f, "flux"))


class TestFactoryOrdering:
    """Regression: native-PEFT Wan LoRAs share the ``cross_attn``/``self_attn``
    substring with Anima/Cosmos DiT. Anima's probe matches on the bare substring
    (it doesn't require Anima's ``_proj`` suffix or ``mlp``/``adaln_modulation``),
    so a Wan LoRA would be mis-tagged as Anima unless Wan's probe runs first
    in the AnyModelConfig union — or unless Anima's probe gets tightened.

    This test pins the order by importing the union and asserting Wan appears
    before Anima in the LyCORIS section.
    """

    def test_wan_appears_before_anima_in_lora_union(self):
        from typing import get_args

        from invokeai.backend.model_manager.configs.factory import AnyModelConfig
        from invokeai.backend.model_manager.configs.lora import (
            LoRA_LyCORIS_Anima_Config,
            LoRA_LyCORIS_Wan_Config,
        )

        # AnyModelConfig is an Annotated[Union[...], Discriminator(...)] — the
        # first arg of get_args is the Union itself.
        union_type = get_args(AnyModelConfig)[0]
        union_members = get_args(union_type)

        def _index_of(cls) -> int:
            for i, m in enumerate(union_members):
                # Each member is Annotated[ConfigClass, Tag(...)]; first get_args is the class.
                if get_args(m)[0] is cls:
                    return i
            raise AssertionError(f"{cls.__name__} not in union")

        wan_idx = _index_of(LoRA_LyCORIS_Wan_Config)
        anima_idx = _index_of(LoRA_LyCORIS_Anima_Config)
        assert wan_idx < anima_idx, (
            f"LoRA_LyCORIS_Wan_Config must come before LoRA_LyCORIS_Anima_Config in "
            f"the AnyModelConfig union (Wan at {wan_idx}, Anima at {anima_idx}). "
            "Otherwise Anima's cross_attn/self_attn substring match will steal Wan LoRAs."
        )

    def test_anima_would_have_matched_a_wan_native_lora(self):
        """Sanity check: confirm that Anima's probe DOES match a Wan native LoRA
        if asked directly. This is why ordering matters — Wan must run first."""
        from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_Anima_Config

        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "wan_native_lora.safetensors"
            f.touch()
            # Realistic Wan native PEFT keys: this is what lightx2v's Lightning
            # LoRAs and most ComfyUI-trained Wan LoRAs look like.
            sd = {
                "diffusion_model.blocks.0.self_attn.q.lora_A.weight": _t((128, 5120)),
                "diffusion_model.blocks.0.self_attn.q.lora_B.weight": _t((5120, 128)),
                "diffusion_model.blocks.0.cross_attn.k.lora_A.weight": _t((128, 5120)),
                "diffusion_model.blocks.0.cross_attn.k.lora_B.weight": _t((5120, 128)),
            }
            # Anima's probe (today) erroneously accepts these. If this assertion
            # ever flips, Anima's probe got tightened and the Wan-first ordering
            # constraint is no longer required (but it's still safe to keep).
            cfg = LoRA_LyCORIS_Anima_Config.from_model_on_disk(_make_mod(f, sd), _overrides(f, "anima-false-positive"))
            assert cfg.base == BaseModelType.Anima  # NB: a false positive; protected against by ordering
