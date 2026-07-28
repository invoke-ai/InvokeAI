from unittest.mock import MagicMock, patch

import pytest

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_Krea2_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType

_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/krea2-lora.safetensors",
    "file_size": 1000,
    "name": "krea2-lora",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def _ambiguous_transformer_only_lora() -> MagicMock:
    mod = MagicMock()
    mod.load_state_dict.return_value = {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": object(),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": object(),
    }
    return mod


def _ambiguous_text_encoder_only_lora() -> MagicMock:
    mod = MagicMock()
    mod.load_state_dict.return_value = {
        "text_encoder.language_model.layers.0.self_attn.q_proj.lora_A.weight": object(),
        "text_encoder.language_model.layers.0.self_attn.q_proj.lora_B.weight": object(),
    }
    return mod


def _diffusion_model_transformer_only_lora() -> MagicMock:
    mod = MagicMock()
    mod.load_state_dict.return_value = {
        "diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": object(),
        "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight": object(),
    }
    return mod


@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_explicit_krea2_override_accepts_ambiguous_transformer_only_lora(_raise_if_not_file) -> None:
    config = LoRA_LyCORIS_Krea2_Config.from_model_on_disk(
        _ambiguous_transformer_only_lora(), {**_REQUIRED_FIELDS, "base": BaseModelType.Krea2}
    )

    assert config.base is BaseModelType.Krea2


@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_automatic_probe_rejects_ambiguous_transformer_only_lora(_raise_if_not_file) -> None:
    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Krea2_Config.from_model_on_disk(_ambiguous_transformer_only_lora(), {**_REQUIRED_FIELDS})


@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_explicit_krea2_override_rejects_incomplete_lora_pair(_raise_if_not_file) -> None:
    mod = MagicMock()
    mod.load_state_dict.return_value = {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": object(),
    }

    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Krea2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "base": BaseModelType.Krea2})


@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_explicit_krea2_override_accepts_text_encoder_only_lora(_raise_if_not_file) -> None:
    config = LoRA_LyCORIS_Krea2_Config.from_model_on_disk(
        _ambiguous_text_encoder_only_lora(), {**_REQUIRED_FIELDS, "base": BaseModelType.Krea2}
    )

    assert config.base is BaseModelType.Krea2


@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_automatic_probe_rejects_ambiguous_text_encoder_only_lora(_raise_if_not_file) -> None:
    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Krea2_Config.from_model_on_disk(_ambiguous_text_encoder_only_lora(), {**_REQUIRED_FIELDS})


@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_explicit_krea2_override_accepts_diffusion_model_transformer_only_lora(_raise_if_not_file) -> None:
    # The converter supports the `diffusion_model.` transformer layout (lora_model_from_krea2_state_dict), so a
    # transformer-only LoRA using it must install under an explicit Krea-2 override (regression: review 4791964047).
    config = LoRA_LyCORIS_Krea2_Config.from_model_on_disk(
        _diffusion_model_transformer_only_lora(), {**_REQUIRED_FIELDS, "base": BaseModelType.Krea2}
    )

    assert config.base is BaseModelType.Krea2


@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_explicit_krea2_override_accepts_multiple_complete_pairs(_raise_if_not_file) -> None:
    # Several fully-paired layers (both lora_A/B and lora_down/up styles) must install cleanly.
    mod = MagicMock()
    mod.load_state_dict.return_value = {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": object(),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": object(),
        "transformer.transformer_blocks.0.attn.to_k.lora_down.weight": object(),
        "transformer.transformer_blocks.0.attn.to_k.lora_up.weight": object(),
        "transformer.transformer_blocks.1.attn.to_v.lora_A.weight": object(),
        "transformer.transformer_blocks.1.attn.to_v.lora_B.weight": object(),
    }
    config = LoRA_LyCORIS_Krea2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "base": BaseModelType.Krea2})
    assert config.base is BaseModelType.Krea2


# One complete pair plus a dangling half of each kind - every mixed case must be rejected, because an
# orphaned half installs here but crashes later during LoRA conversion (review 4800904928).
_ORPHAN_HALVES = [
    "transformer.transformer_blocks.0.attn.to_k.lora_A.weight",  # missing lora_B
    "transformer.transformer_blocks.0.attn.to_k.lora_B.weight",  # missing lora_A
    "transformer.transformer_blocks.0.attn.to_k.lora_down.weight",  # missing lora_up
    "transformer.transformer_blocks.0.attn.to_k.lora_up.weight",  # missing lora_down
]


@pytest.mark.parametrize("orphan_key", _ORPHAN_HALVES)
@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_explicit_krea2_override_rejects_complete_pair_plus_orphan(_raise_if_not_file, orphan_key: str) -> None:
    mod = MagicMock()
    mod.load_state_dict.return_value = {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": object(),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": object(),
        orphan_key: object(),
    }
    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Krea2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "base": BaseModelType.Krea2})


@pytest.mark.parametrize("orphan_key", _ORPHAN_HALVES)
@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_automatic_probe_rejects_complete_pair_plus_orphan(_raise_if_not_file, orphan_key: str) -> None:
    # The automatic (no-override) path also validates completeness. text_fusion.* makes it look like Krea-2.
    mod = MagicMock()
    mod.load_state_dict.return_value = {
        "transformer.text_fusion.0.lora_A.weight": object(),
        "transformer.text_fusion.0.lora_B.weight": object(),
        orphan_key: object(),
    }
    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Krea2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})


@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_explicit_krea2_override_rejects_incomplete_diffusion_model_lora_pair(_raise_if_not_file) -> None:
    mod = MagicMock()
    mod.load_state_dict.return_value = {
        "diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": object(),
    }

    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Krea2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "base": BaseModelType.Krea2})


@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_explicit_krea2_override_rejects_orphan_outside_approved_prefixes(_raise_if_not_file) -> None:
    # A complete pair under an approved prefix plus a dangling half elsewhere (text_fusion, which is NOT in
    # the approved-prefix list) must still be rejected - the orphan crashes conversion (review 4802322488).
    mod = MagicMock()
    mod.load_state_dict.return_value = {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": object(),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": object(),
        "transformer.text_fusion.0.lora_A.weight": object(),  # orphan: no matching lora_B
    }
    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Krea2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "base": BaseModelType.Krea2})


@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_automatic_probe_rejects_dora_scale_without_weight_pair(_raise_if_not_file) -> None:
    # dora_scale alone is not a loadable LoRA - there are no A/B (or down/up) weights to apply. text_fusion.*
    # makes it look like Krea-2, but it must be rejected for lacking a complete pair (review 4802322488).
    mod = MagicMock()
    mod.load_state_dict.return_value = {
        "transformer.text_fusion.0.dora_scale": object(),
    }
    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Krea2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})


@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_automatic_probe_accepts_complete_pair_with_dora_scale(_raise_if_not_file) -> None:
    # A complete A/B pair accompanied by dora_scale is a valid DoRA LoRA and must be accepted.
    mod = MagicMock()
    mod.load_state_dict.return_value = {
        "transformer.text_fusion.0.lora_A.weight": object(),
        "transformer.text_fusion.0.lora_B.weight": object(),
        "transformer.text_fusion.0.dora_scale": object(),
    }
    config = LoRA_LyCORIS_Krea2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
    assert config.base is BaseModelType.Krea2


@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_native_comfyui_krea2_lora_is_identified_as_krea2(_raise_if_not_file) -> None:
    # A native (ComfyUI) Krea-2 slider LoRA uses `blocks.N.attn.wq/gate` + `mlp` (no diffusers `text_fusion`
    # key). It must auto-identify as Krea-2 via the native gated-attention signature, not fall through to
    # Anima (whose strict detector previously false-matched the native `blocks.N.mlp.*`).
    mod = MagicMock()
    mod.load_state_dict.return_value = {
        "diffusion_model.blocks.0.attn.wq.lora_A.weight": object(),
        "diffusion_model.blocks.0.attn.wq.lora_B.weight": object(),
        "diffusion_model.blocks.0.attn.gate.lora_A.weight": object(),
        "diffusion_model.blocks.0.attn.gate.lora_B.weight": object(),
        "diffusion_model.blocks.0.mlp.down.lora_A.weight": object(),
        "diffusion_model.blocks.0.mlp.down.lora_B.weight": object(),
    }
    config = LoRA_LyCORIS_Krea2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
    assert config.base is BaseModelType.Krea2
