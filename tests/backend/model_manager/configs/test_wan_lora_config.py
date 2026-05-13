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
            ("anima_kohya_q_proj", ["lora_unet_blocks_0_cross_attn_q_proj.lora_down.weight"]),
            ("anima_peft_mlp", ["transformer.blocks.0.mlp.layer1.lora_A.weight"]),
            ("anima_peft_adaln", ["transformer.blocks.0.adaln_modulation.linear.lora_A.weight"]),
            ("anima_peft_self_attn_q_proj", ["transformer.blocks.0.self_attn.q_proj.lora_A.weight"]),
            ("qwen_image", ["transformer_blocks.0.attn.to_q.lora_A.weight"]),
            ("flux_kohya_double", ["lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight"]),
            ("flux_kohya_single", ["lora_unet_single_blocks_0_linear1.lora_down.weight"]),
            ("flux_diffusers_single_transformer", ["transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight"]),
            ("z_image", ["diffusion_model.layers.0.attn.to_q.lora_A.weight"]),
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

    def _wan_ti2v5b_sd(self) -> dict:
        """A TI2V-5B LoRA — inner_dim 3072, not 5120."""
        return {
            "transformer.blocks.0.attn1.to_q.lora_A.weight": _t((64, 3072)),
            "transformer.blocks.0.attn1.to_q.lora_B.weight": _t((3072, 64)),
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
            assert cfg.variant == "a14b"  # 5120-dim state dict

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

    def test_variant_detected_as_5b_when_inner_dim_3072(self):
        """TI2V-5B LoRAs have inner_dim 3072. Detector must classify them as
        '5b' so the FE filter doesn't route them to an A14B main and crash."""
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "ti2v5b-lora.safetensors"
            f.touch()
            cfg = LoRA_LyCORIS_Wan_Config.from_model_on_disk(
                _make_mod(f, self._wan_ti2v5b_sd()),
                _overrides(f, "ti2v5b"),
            )
            assert cfg.base == BaseModelType.Wan
            assert cfg.variant == "5b"

    def test_variant_none_when_unrecognised_inner_dim(self):
        """A future Wan family or a LoRA touching only ffn at non-attn dims
        should map to variant=None rather than mis-classify."""
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "future-wan.safetensors"
            f.touch()
            # Only an ffn LoRA — no attn weight to read inner_dim from.
            # Also a non-5120, non-3072 dim that would otherwise mis-classify.
            sd = {
                "transformer.blocks.0.ffn.net.0.proj.lora_A.weight": _t((128, 4096)),
                "transformer.blocks.0.ffn.net.0.proj.lora_B.weight": _t((11008, 128)),
            }
            cfg = LoRA_LyCORIS_Wan_Config.from_model_on_disk(_make_mod(f, sd), _overrides(f, "future"))
            assert cfg.variant is None

    def test_explicit_variant_override_wins(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "manual.safetensors"
            f.touch()
            overrides = _overrides(f, "manual")
            overrides["variant"] = "5b"
            # State dict is 5120-dim (auto-detect would say "a14b") but the
            # explicit override should stick.
            cfg = LoRA_LyCORIS_Wan_Config.from_model_on_disk(
                _make_mod(f, self._wan_diffusers_sd()),
                overrides,
            )
            assert cfg.variant == "5b"

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


class TestProbeMutualExclusivity:
    """Regression: Anima's probe must REJECT Wan-native LoRA keys, so probing
    is correct regardless of which config the factory iterates first.

    ``Config_Base.CONFIG_CLASSES`` is a ``set``, so iteration order is
    non-deterministic across Python process restarts. Probes therefore need
    to be mutually exclusive at the per-config level — see also
    ``test_wan_lora_probe_independence.py`` for the broader cross-architecture
    coverage."""

    def test_anima_rejects_wan_native_lora(self):
        """Wan native LoRAs (``diffusion_model.blocks.X.self_attn.q.lora_*``)
        used to false-positive on Anima's probe because Anima accepted any
        ``cross_attn``/``self_attn`` substring. Anima now requires
        Cosmos-DiT-exclusive markers (``mlp``, ``adaln_modulation``, or the
        ``_proj`` attention suffix), so a Wan LoRA — which has none of those —
        is correctly rejected."""
        from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_Anima_Config

        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "wan_native_lora.safetensors"
            f.touch()
            # Realistic Wan native PEFT keys — what lightx2v's Lightning
            # distillations and most ComfyUI-trained Wan LoRAs look like.
            sd = {
                "diffusion_model.blocks.0.self_attn.q.lora_A.weight": _t((128, 5120)),
                "diffusion_model.blocks.0.self_attn.q.lora_B.weight": _t((5120, 128)),
                "diffusion_model.blocks.0.cross_attn.k.lora_A.weight": _t((128, 5120)),
                "diffusion_model.blocks.0.cross_attn.k.lora_B.weight": _t((5120, 128)),
            }
            with pytest.raises(NotAMatchError, match="Anima LoRA"):
                LoRA_LyCORIS_Anima_Config.from_model_on_disk(_make_mod(f, sd), _overrides(f, "wan-native-lora"))

    def test_wan_rejects_anima_lora(self):
        """Mirror direction: a real Anima LoRA must not be matched by Wan.
        Wan's anti-patterns already cover ``_proj`` suffix, ``mlp``, and
        ``adaln_modulation``."""
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "anima_lora.safetensors"
            f.touch()
            sd = {
                "transformer.blocks.0.self_attn.q_proj.lora_A.weight": _t((128, 4096)),
                "transformer.blocks.0.self_attn.q_proj.lora_B.weight": _t((4096, 128)),
                "transformer.blocks.0.mlp.layer1.lora_A.weight": _t((128, 4096)),
                "transformer.blocks.0.mlp.layer1.lora_B.weight": _t((4096, 128)),
            }
            with pytest.raises(NotAMatchError, match="Wan LoRA"):
                LoRA_LyCORIS_Wan_Config.from_model_on_disk(_make_mod(f, sd), _overrides(f, "anima-lora"))
