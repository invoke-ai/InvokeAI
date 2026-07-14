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
