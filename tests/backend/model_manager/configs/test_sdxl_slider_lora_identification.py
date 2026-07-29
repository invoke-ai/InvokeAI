"""Tests for identifying UNet-only SDXL LoRAs (e.g. self-attention "slider" LoRAs).

Some SDXL LoRAs only patch the UNet and contain no cross-attention (`attn2`) or
text-encoder (`lora_te*`) keys. `lora_token_vector_length()` reads the base's
context dimension from exactly those keys, so for such LoRAs it returns `None`
and the base can't be inferred that way.

`_state_dict_looks_like_sdxl_unet_lora()` recovers SDXL identification from the
UNet block structure alone: SDXL's UNet has a deep transformer stack (up to 10
transformer blocks) in its lower-resolution attention blocks, so
`transformer_blocks` indices reach >= 2, whereas SD1.x/SD2.x only ever have a
single transformer block (index 0) per attention.

kohya `sd-scripts` emits two block-naming conventions and both must be covered:
Stability-AI names (`lora_unet_input_blocks_8_1_...`, `middle_block_1`,
`output_blocks_0_1`) and diffusers names (`lora_unet_down_blocks_2_attentions_1_...`,
`mid_block_attentions_0`). `convert_sdxl_keys_to_diffusers_format()` accepts both.

The heuristic is deliberately anchored on the `lora_unet_` prefix: that is exactly the
key set the SDXL loader can convert. Keys outside it (diffusers/PEFT `unet.…`) must NOT
be identified as SDXL — see `test_diffusers_peft_format_not_matched`.

Real-world example: https://civitai.com/models/1105685 ("Dramatic Lighting
Slider"), a Kohya-format LoRA whose 840 keys are all `lora_unet_..._attn1_...`.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.lora import (
    LoRA_Diffusers_SD1_Config,
    LoRA_Diffusers_SDXL_Config,
    LoRA_LyCORIS_SD1_Config,
    LoRA_LyCORIS_SD2_Config,
    LoRA_LyCORIS_SDXL_Config,
    _state_dict_looks_like_sdxl_unet_lora,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.patches.lora_conversions.sdxl_lora_conversion_utils import (
    convert_sdxl_keys_to_diffusers_format,
)

_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/slider.safetensors",
    "file_size": 1000,
    "name": "test-slider",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}

# The deep (10-transformer-block) attention blocks of the SDXL UNet, in both block-naming
# conventions kohya `sd-scripts` emits.
_SDXL_DEEP_BLOCKS_STABILITY = ("input_blocks_8_1", "middle_block_1", "output_blocks_0_1")
_SDXL_DEEP_BLOCKS_DIFFUSERS = ("down_blocks_2_attentions_1", "mid_block_attentions_0", "up_blocks_0_attentions_2")


def _self_attention_only_keys(groups: tuple[str, ...], transformer_blocks: int = 10) -> dict[str | int, None]:
    """Build a minimal UNet self-attention-only (slider) state dict.

    Mirrors the real "Dramatic Lighting Slider" LoRA: only `attn1` keys, no `attn2` and no
    `lora_te*`. Values are unused by the key-only heuristics under test.
    """
    sd: dict[str | int, None] = {}
    for tb in range(transformer_blocks):
        for group in groups:
            for proj in ("to_q", "to_k", "to_v", "to_out_0"):
                base = f"lora_unet_{group}_transformer_blocks_{tb}_attn1_{proj}"
                sd[f"{base}.lora_down.weight"] = None
                sd[f"{base}.lora_up.weight"] = None
                sd[f"{base}.alpha"] = None
    return sd


def _sdxl_stability_slider_keys() -> dict[str | int, None]:
    return _self_attention_only_keys(_SDXL_DEEP_BLOCKS_STABILITY)


def _sdxl_diffusers_slider_keys() -> dict[str | int, None]:
    return _self_attention_only_keys(_SDXL_DEEP_BLOCKS_DIFFUSERS)


def _diffusers_peft_keys() -> dict[str | int, None]:
    """The standard HF-hub `pytorch_lora_weights.safetensors` key shape."""
    return {
        f"unet.down_blocks.2.attentions.1.transformer_blocks.{tb}.attn1.{proj}.{ab}.weight": None
        for tb in (0, 9)
        for proj in ("to_q", "to_k", "to_v")
        for ab in ("lora_A", "lora_B")
    }


class TestSdxlUnetLoraDetection:
    """Unit tests for the key-only structural heuristic."""

    def test_stability_named_slider(self):
        # kohya sd-scripts' Stability-AI block naming — what the reported LoRA uses.
        assert _state_dict_looks_like_sdxl_unet_lora(_sdxl_stability_slider_keys()) is True

    def test_diffusers_named_slider(self):
        assert _state_dict_looks_like_sdxl_unet_lora(_sdxl_diffusers_slider_keys()) is True

    @pytest.mark.parametrize(
        "key",
        [
            "lora_unet_input_blocks_8_1_transformer_blocks_9_attn1_to_q.lora_down.weight",
            "lora_unet_middle_block_1_transformer_blocks_9_attn1_to_q.lora_down.weight",
            "lora_unet_output_blocks_0_1_transformer_blocks_9_attn1_to_q.lora_down.weight",
            "lora_unet_down_blocks_2_attentions_1_transformer_blocks_9_attn1_to_q.lora_down.weight",
            "lora_unet_mid_block_attentions_0_transformer_blocks_9_attn1_to_q.lora_down.weight",
            "lora_unet_up_blocks_0_attentions_2_transformer_blocks_9_attn1_to_q.lora_down.weight",
        ],
    )
    def test_each_deep_block_name_matches(self, key: str):
        assert _state_dict_looks_like_sdxl_unet_lora({key: None}) is True

    @pytest.mark.parametrize(
        "state_dict_factory", [_sdxl_stability_slider_keys, _sdxl_diffusers_slider_keys], ids=["stability", "diffusers"]
    )
    def test_matched_keys_are_convertible_by_the_loader(self, state_dict_factory):
        # The heuristic must only claim key sets the SDXL loader can actually convert,
        # otherwise identification succeeds and generation fails with an opaque ValueError.
        state_dict = state_dict_factory()
        assert _state_dict_looks_like_sdxl_unet_lora(state_dict) is True
        convert_sdxl_keys_to_diffusers_format(state_dict)  # must not raise

    def test_diffusers_peft_format_not_matched(self):
        # `unet.….lora_A.weight` also returns None from lora_token_vector_length(), so it
        # reaches this fallback. It must NOT match: convert_sdxl_keys_to_diffusers_format()
        # rejects any prefix other than lora_unet_/lora_te1_/lora_te2_, so identifying it as
        # SDXL would turn an inert Unknown model into a mid-generation crash.
        sd = _diffusers_peft_keys()
        assert _state_dict_looks_like_sdxl_unet_lora(sd) is False
        with pytest.raises(ValueError, match="Unrecognized SDXL LoRA key prefix"):
            convert_sdxl_keys_to_diffusers_format(sd)

    def test_omi_source_keys_not_matched(self):
        # OMI SDXL LoRAs use `unet.input_blocks.…` and are handled by LoRA_OMI_SDXL_Config.
        sd = {"unet.input_blocks.8.1.transformer_blocks.9.attn1.to_q.lora_A.weight": None}
        assert _state_dict_looks_like_sdxl_unet_lora(sd) is False

    @pytest.mark.parametrize(
        "groups", [_SDXL_DEEP_BLOCKS_STABILITY, _SDXL_DEEP_BLOCKS_DIFFUSERS], ids=["stability", "diffusers"]
    )
    def test_shallow_transformer_stack_not_matched(self, groups: tuple[str, ...]):
        # SD1.x/SD2.x only ever have transformer_blocks_0 per attention, in either naming.
        assert _state_dict_looks_like_sdxl_unet_lora(_self_attention_only_keys(groups, transformer_blocks=1)) is False

    def test_sdxl_confined_to_shallow_blocks_is_a_known_miss(self):
        # Documented limitation: SDXL's `down_blocks_1` / `up_blocks_1` attentions have only
        # 2 transformer blocks (indices 0-1), which is indistinguishable from SD1/SD2 by
        # block structure alone. Such a LoRA stays unidentified.
        sd = _self_attention_only_keys(("down_blocks_1_attentions_0",), transformer_blocks=2)
        assert _state_dict_looks_like_sdxl_unet_lora(sd) is False

    def test_qwen_image_lora_not_matched(self):
        # Qwen uses `transformer_blocks.N` with high N but no UNet block name.
        sd = {
            "lora_unet_transformer_blocks_20_attn_to_k.lora_down.weight": None,
            "transformer.transformer_blocks.39.attn.to_q.lora_A.weight": None,
        }
        assert _state_dict_looks_like_sdxl_unet_lora(sd) is False

    def test_flux_lora_not_matched(self):
        sd = {"lora_unet_double_blocks_5_img_attn_proj.lora_down.weight": None}
        assert _state_dict_looks_like_sdxl_unet_lora(sd) is False

    def test_empty_state_dict(self):
        assert _state_dict_looks_like_sdxl_unet_lora({}) is False

    def test_non_string_keys_ignored(self):
        # GGUF-style integer keys must not crash the heuristic.
        assert _state_dict_looks_like_sdxl_unet_lora({0: None, 1: None}) is False


class TestSdxlSliderLoraLyCORISIdentification:
    """End-to-end identification through the LyCORIS (single-file) config classes."""

    def _make_mock_mod(self, state_dict: dict[str | int, None]) -> MagicMock:
        mod = MagicMock()
        mod.path = Path(_REQUIRED_FIELDS["path"])
        mod.load_state_dict.return_value = state_dict
        mod.metadata.return_value = {}
        return mod

    @pytest.mark.parametrize(
        "state_dict_factory", [_sdxl_stability_slider_keys, _sdxl_diffusers_slider_keys], ids=["stability", "diffusers"]
    )
    @patch("invokeai.backend.model_manager.configs.lora._get_flux_lora_format", return_value=None)
    @patch("invokeai.backend.model_manager.configs.lora.has_cosmos_dit_peft_keys", return_value=False)
    @patch("invokeai.backend.model_manager.configs.lora.has_cosmos_dit_kohya_keys", return_value=False)
    @patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
    def test_sdxl_config_matches(self, _rif, _hck, _hcp, _flux, state_dict_factory):
        mod = self._make_mock_mod(state_dict_factory())
        config = LoRA_LyCORIS_SDXL_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
        assert config.base is BaseModelType.StableDiffusionXL

    @pytest.mark.parametrize("cls", [LoRA_LyCORIS_SD1_Config, LoRA_LyCORIS_SD2_Config])
    @patch("invokeai.backend.model_manager.configs.lora._get_flux_lora_format", return_value=None)
    @patch("invokeai.backend.model_manager.configs.lora.has_cosmos_dit_peft_keys", return_value=False)
    @patch("invokeai.backend.model_manager.configs.lora.has_cosmos_dit_kohya_keys", return_value=False)
    @patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
    def test_sd1_sd2_configs_reject(self, _rif, _hck, _hcp, _flux, cls):
        mod = self._make_mock_mod(_sdxl_stability_slider_keys())
        # Match the message so this asserts rejection *because the base resolved to SDXL*,
        # rather than passing on an unrelated rejection (e.g. the LoRA-heuristic gate).
        with pytest.raises(NotAMatchError, match=r"base is BaseModelType\.StableDiffusionXL, not"):
            cls.from_model_on_disk(mod, {**_REQUIRED_FIELDS})


class TestSdxlSliderLoraDiffusersFolderIdentification:
    """End-to-end identification through the Diffusers-folder config classes.

    This is the path where a false positive is most damaging: the folder path never runs
    `_validate_looks_like_lora`, so an over-broad heuristic silently promotes every
    otherwise-unidentified diffusers LoRA folder to SDXL.
    """

    def _make_mock_mod(self, tmp_path: Path, state_dict: dict[str | int, None]) -> MagicMock:
        (tmp_path / "pytorch_lora_weights.safetensors").touch()
        mod = MagicMock()
        mod.path = tmp_path
        mod.load_state_dict.return_value = state_dict
        mod.metadata.return_value = {}
        return mod

    @patch("invokeai.backend.model_manager.configs.lora._get_flux_lora_format", return_value=None)
    @patch("invokeai.backend.model_manager.configs.lora.raise_if_not_dir")
    def test_diffusers_peft_slider_is_not_claimed_as_sdxl(self, _rid, _flux, tmp_path: Path):
        # Regression guard for the false positive: these keys must stay unidentified rather
        # than install as SDXL and crash later in convert_sdxl_keys_to_diffusers_format().
        mod = self._make_mock_mod(tmp_path, _diffusers_peft_keys())
        with pytest.raises(NotAMatchError, match="unrecognized token vector length"):
            LoRA_Diffusers_SDXL_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "path": str(tmp_path)})

    @patch("invokeai.backend.model_manager.configs.lora._get_flux_lora_format", return_value=None)
    @patch("invokeai.backend.model_manager.configs.lora.raise_if_not_dir")
    def test_kohya_slider_keys_in_a_folder_match_sdxl(self, _rid, _flux, tmp_path: Path):
        mod = self._make_mock_mod(tmp_path, _sdxl_stability_slider_keys())
        config = LoRA_Diffusers_SDXL_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "path": str(tmp_path)})
        assert config.base is BaseModelType.StableDiffusionXL

    @patch("invokeai.backend.model_manager.configs.lora._get_flux_lora_format", return_value=None)
    @patch("invokeai.backend.model_manager.configs.lora.raise_if_not_dir")
    def test_diffusers_sd1_config_rejects_slider_keys(self, _rid, _flux, tmp_path: Path):
        mod = self._make_mock_mod(tmp_path, _sdxl_stability_slider_keys())
        with pytest.raises(NotAMatchError, match=r"base is BaseModelType\.StableDiffusionXL, not"):
            LoRA_Diffusers_SD1_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "path": str(tmp_path)})
