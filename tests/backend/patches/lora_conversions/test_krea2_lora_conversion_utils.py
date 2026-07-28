import pytest
import torch

from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.lora_conversions.krea2_lora_constants import KREA2_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.lora_conversions.krea2_lora_conversion_utils import lora_model_from_krea2_state_dict


def test_peft_layer_preserves_explicit_alpha() -> None:
    state_dict = {
        "transformer.text_fusion.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.text_fusion.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        "transformer.text_fusion.0.attn.to_q.alpha": torch.tensor(1.0),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}text_fusion.0.attn.to_q"]
    assert isinstance(layer, LoRALayer)
    assert layer._alpha == 1.0


def test_peft_dora_layer_preserves_magnitude_and_alpha() -> None:
    dora_scale = torch.full((4, 1), 2.0)
    state_dict = {
        "transformer.text_fusion.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.text_fusion.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        "transformer.text_fusion.0.attn.to_q.dora_scale": dora_scale,
        "transformer.text_fusion.0.attn.to_q.alpha": torch.tensor(1.0),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}text_fusion.0.attn.to_q"]
    assert isinstance(layer, DoRALayer)
    assert layer._alpha == 1.0
    assert torch.equal(layer.dora_scale, dora_scale)


def test_peft_layer_without_explicit_alpha_uses_rank_default() -> None:
    state_dict = {
        "transformer.text_fusion.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.text_fusion.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}text_fusion.0.attn.to_q"]
    assert isinstance(layer, LoRALayer)
    assert layer._alpha is None


def test_incomplete_peft_pair_raises_descriptive_error() -> None:
    # A layer with lora_A but no matching lora_B is malformed. It must raise a clear ValueError naming the
    # missing key, not an uninformative bare KeyError.
    state_dict = {
        # Complete layer so the dict still looks like a Krea-2 LoRA.
        "transformer.text_fusion.0.attn.to_k.lora_A.weight": torch.ones(2, 4),
        "transformer.text_fusion.0.attn.to_k.lora_B.weight": torch.ones(4, 2),
        # Incomplete layer: lora_A present, lora_B missing.
        "transformer.text_fusion.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
    }

    with pytest.raises(ValueError, match="lora_B.weight"):
        lora_model_from_krea2_state_dict(state_dict)


def test_peft_dora_magnitude_vector_key_produces_dora_layer() -> None:
    # Standard PEFT / Diffusers DoRA stores the magnitude as `.lora_magnitude_vector.weight` (not the LyCORIS
    # `.dora_scale`). It must be recognized and produce a DoRALayer preserving the magnitude, so valid
    # Diffusers DoRA adapters load instead of being split into a bogus, unrecognized layer (review 4802322488).
    magnitude = torch.full((4, 1), 3.0)
    state_dict = {
        "transformer.text_fusion.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.text_fusion.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        "transformer.text_fusion.0.attn.to_q.lora_magnitude_vector.weight": magnitude,
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}text_fusion.0.attn.to_q"]
    assert isinstance(layer, DoRALayer)
    assert torch.equal(layer.dora_scale, magnitude)


def test_conflicting_transformer_and_diffusion_model_aliases_raise() -> None:
    # `transformer.` and `diffusion_model.` normalize to the same target key. Providing both aliases for one
    # logical layer (with different tensors) must raise, not silently drop one based on dict ordering.
    state_dict = {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": torch.full((2, 4), 2.0),
        "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight": torch.full((4, 2), 2.0),
    }

    with pytest.raises(ValueError, match="normalize to the same target"):
        lora_model_from_krea2_state_dict(state_dict)
