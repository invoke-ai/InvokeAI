"""LoRA on a BitsAndBytes-quantized Qwen2.5-VL encoder, loaded the way the Qwen-Image text encoder node loads it.

transformers builds plain `bnb.nn.Linear8bitLt` / `Linear4bit` layers, which cannot hold a sidecar patch; the node
converts them with `_make_bnb_layers_patchable`. Needs CUDA (bitsandbytes quantizes on the device).
"""

from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="BitsAndBytes quantization needs CUDA")

HIDDEN = 64
RANK = 4
WEIGHT = 0.6
DEVICE = torch.device("cuda:0")


def _tiny_config():
    from transformers import Qwen2_5_VLConfig

    return Qwen2_5_VLConfig(
        text_config={
            "hidden_size": HIDDEN,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 512,
            "max_position_embeddings": 256,
            "tie_word_embeddings": False,
            "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]},
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": HIDDEN,
            "fullatt_block_indexes": [1],
        },
    )


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from transformers import Qwen2_5_VLForConditionalGeneration

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_qwen_vl")
    Qwen2_5_VLForConditionalGeneration(_tiny_config()).to(torch.bfloat16).save_pretrained(path)
    return path


def _load(checkpoint: Path, quantization: str):
    """As `_load_quantized_encoder` loads it."""
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    if quantization == "nf4":
        config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    else:
        config = BitsAndBytesConfig(load_in_8bit=True)
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(checkpoint), quantization_config=config, device_map={"": DEVICE}, dtype=torch.bfloat16
    )


def _patch_for(layer_name: str, in_features: int, out_features: int):
    from invokeai.backend.patches.lora_conversions.qwen_image_lora_conversion_utils import (
        lora_model_from_qwen_image_state_dict,
    )

    generator = torch.Generator().manual_seed(3)
    down = torch.randn(RANK, in_features, generator=generator) * 0.1
    up = torch.randn(out_features, RANK, generator=generator) * 0.1
    patch = lora_model_from_qwen_image_state_dict(
        {f"text_encoder.{layer_name}.lora_A.weight": down, f"text_encoder.{layer_name}.lora_B.weight": up}
    )
    return patch, down, up


@pytest.mark.parametrize("quantization", ["int8", "nf4"])
@pytest.mark.parametrize(
    ("layer_name", "in_features", "out_features"),
    [
        ("model.language_model.layers.0.mlp.down_proj", 128, HIDDEN),
        ("model.language_model.layers.1.self_attn.q_proj", HIDDEN, HIDDEN),
        ("lm_head", HIDDEN, 512),  # not quantized by transformers: a plain Linear
    ],
)
def test_sidecar_lora_on_a_bnb_quantized_encoder(
    checkpoint: Path, quantization: str, layer_name: str, in_features: int, out_features: int
) -> None:
    import bitsandbytes as bnb

    from invokeai.app.invocations.text_encoder.qwen_image_text_encoder import _make_bnb_layers_patchable
    from invokeai.backend.patches.layer_patcher import LayerPatcher
    from invokeai.backend.patches.lora_conversions.qwen_image_lora_constants import QWEN_IMAGE_LORA_QWEN_VL_PREFIX

    input_ids = torch.arange(16, device=DEVICE).reshape(1, 16)
    # A separate instance: the plain int8 forward moves CB/SCB off the weight, so the model under test must be wrapped
    # before its first forward, as in the node. Quantization is deterministic, so the two instances agree.
    reference = _load(checkpoint, quantization)
    with torch.no_grad():
        reference_hidden = reference(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
    del reference

    model = _load(checkpoint, quantization)
    _make_bnb_layers_patchable(model)

    layer = model.get_submodule(layer_name)
    if layer_name != "lm_head":
        assert isinstance(layer, (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)), "wrapping must keep the bnb base class"
    assert hasattr(layer, "add_patch")
    x = torch.randn(1, 5, in_features, device=DEVICE, dtype=torch.bfloat16)
    patch, down, up = _patch_for(layer_name, in_features, out_features)

    with torch.no_grad():
        # Wrapped layers without a patch compute what the unwrapped model did.
        wrapped_hidden = model(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
        assert torch.equal(wrapped_hidden, reference_hidden)

        unpatched = layer(x)
        with LayerPatcher.apply_smart_model_patches(
            model=model,
            patches=[(patch, WEIGHT)],
            prefix=QWEN_IMAGE_LORA_QWEN_VL_PREFIX,
            dtype=torch.bfloat16,
            force_sidecar_patching=True,
        ):
            assert layer.get_num_patches() == 1
            patched = layer(x)
        after = layer(x)

    expected_shift = (x.float() @ (up @ down).T.to(DEVICE) * WEIGHT).to(torch.bfloat16)
    torch.testing.assert_close((patched - unpatched).float(), expected_shift.float(), atol=0.05, rtol=0.1)
    assert layer.get_num_patches() == 0
    assert torch.equal(after, unpatched)
