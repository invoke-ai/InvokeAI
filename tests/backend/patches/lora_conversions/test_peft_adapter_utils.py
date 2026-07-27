import torch

from invokeai.backend.patches.lora_conversions.peft_adapter_utils import (
    has_peft_named_adapter_keys,
    normalize_peft_adapter_names,
)


def _t() -> torch.Tensor:
    return torch.zeros(1)


def test_no_op_when_no_named_adapter_keys():
    """State dicts without named-adapter keys are returned unchanged."""
    sd = {
        "transformer_blocks.0.attn.to_q.lora_A.weight": _t(),
        "transformer_blocks.0.attn.to_q.lora_B.weight": _t(),
        "transformer_blocks.0.attn.to_k.lora_down.weight": _t(),
        "single_blocks.0.lokr_w1": _t(),
    }
    assert not has_peft_named_adapter_keys(sd)
    assert normalize_peft_adapter_names(sd) is sd


def test_strips_default_adapter_name():
    """The common `default` adapter name gets stripped from lora_A/lora_B keys."""
    sd = {
        "transformer_blocks.0.attn.to_q.lora_A.default.weight": _t(),
        "transformer_blocks.0.attn.to_q.lora_B.default.weight": _t(),
        "transformer_blocks.0.attn.to_k.lora_A.default.weight": _t(),
        "transformer_blocks.0.attn.to_k.lora_B.default.weight": _t(),
    }
    assert has_peft_named_adapter_keys(sd)

    result = normalize_peft_adapter_names(sd)
    assert set(result.keys()) == {
        "transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer_blocks.0.attn.to_q.lora_B.weight",
        "transformer_blocks.0.attn.to_k.lora_A.weight",
        "transformer_blocks.0.attn.to_k.lora_B.weight",
    }


def test_strips_custom_adapter_name():
    """Non-default adapter names are also stripped, as long as only one is present."""
    sd = {
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.my_adapter.weight": _t(),
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_B.my_adapter.weight": _t(),
    }
    result = normalize_peft_adapter_names(sd)
    assert set(result.keys()) == {
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight",
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_B.weight",
    }


def test_leaves_multi_adapter_state_dict_untouched():
    """If multiple distinct adapter names are present, renaming would collide, so don't."""
    sd = {
        "transformer_blocks.0.attn.to_q.lora_A.default.weight": _t(),
        "transformer_blocks.0.attn.to_q.lora_A.other.weight": _t(),
    }
    assert has_peft_named_adapter_keys(sd)
    assert normalize_peft_adapter_names(sd) is sd


def test_preserves_non_lora_keys_alongside_named_adapter_keys():
    """Keys that aren't lora_A/lora_B PEFT keys pass through unchanged."""
    sd = {
        "transformer_blocks.0.attn.to_q.lora_A.default.weight": _t(),
        "transformer_blocks.0.attn.to_q.lora_B.default.weight": _t(),
        "transformer_blocks.0.attn.to_q.alpha": _t(),
        "metadata_like.dora_scale": _t(),
    }
    result = normalize_peft_adapter_names(sd)
    assert "transformer_blocks.0.attn.to_q.lora_A.weight" in result
    assert "transformer_blocks.0.attn.to_q.lora_B.weight" in result
    assert "transformer_blocks.0.attn.to_q.alpha" in result
    assert "metadata_like.dora_scale" in result


def test_preserves_integer_keys():
    """Non-string keys (some PyTorch state dicts use ints) are passed through."""
    sd: dict = {
        0: _t(),
        "x.lora_A.default.weight": _t(),
    }
    result = normalize_peft_adapter_names(sd)
    assert 0 in result
    assert "x.lora_A.weight" in result
