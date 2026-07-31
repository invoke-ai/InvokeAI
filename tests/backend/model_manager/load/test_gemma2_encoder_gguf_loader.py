"""Tests for the native GGMLTensor Gemma-2 GGUF encoder loader.

The loader keeps the large 2D projection weights quantized (as GGMLTensor, dequantized on demand by the
model cache) and materializes only the embedding and RMSNorm weights, so a GGUF Gemma-2 encoder no longer
costs the same VRAM as the unquantized model.
"""

from pathlib import Path

import pytest

from invokeai.backend.model_manager.load.model_loaders.gemma2_encoder import (
    _convert_gemma_llamacpp_to_pytorch,
    load_gemma2_model_from_gguf,
)

# A local Gemma-2-2b Q4_K_M GGUF for the (skipped-in-CI) end-to-end load/compare test.
_LOCAL_GGUF = Path("E:/ModelStuffSSD/gemma-2-2b-it-q4_k_m.gguf")


def test_convert_maps_every_gemma_component() -> None:
    sd = {
        "token_embd.weight": "E",
        "output_norm.weight": "N",
        "blk.0.attn_q.weight": 1,
        "blk.0.attn_k.weight": 2,
        "blk.0.attn_v.weight": 3,
        "blk.0.attn_output.weight": 4,
        "blk.0.ffn_gate.weight": 5,
        "blk.0.ffn_up.weight": 6,
        "blk.0.ffn_down.weight": 7,
        "blk.0.attn_norm.weight": 8,
        "blk.0.post_attention_norm.weight": 9,
        "blk.0.ffn_norm.weight": 10,
        "blk.0.post_ffw_norm.weight": 11,
    }
    assert _convert_gemma_llamacpp_to_pytorch(sd) == {
        "embed_tokens.weight": "E",
        "norm.weight": "N",
        "layers.0.self_attn.q_proj.weight": 1,
        "layers.0.self_attn.k_proj.weight": 2,
        "layers.0.self_attn.v_proj.weight": 3,
        "layers.0.self_attn.o_proj.weight": 4,
        "layers.0.mlp.gate_proj.weight": 5,
        "layers.0.mlp.up_proj.weight": 6,
        "layers.0.mlp.down_proj.weight": 7,
        "layers.0.input_layernorm.weight": 8,
        "layers.0.post_attention_layernorm.weight": 9,
        "layers.0.pre_feedforward_layernorm.weight": 10,
        "layers.0.post_feedforward_layernorm.weight": 11,
    }


@pytest.mark.parametrize("bad_key", ["blk.0.bogus.weight", "totally.unknown", "lm_head.weight"])
def test_convert_rejects_unmapped_keys(bad_key: str) -> None:
    with pytest.raises(ValueError, match="Unmapped"):
        _convert_gemma_llamacpp_to_pytorch({bad_key: 1})


@pytest.mark.skipif(not _LOCAL_GGUF.exists(), reason="requires a local Gemma-2-2b GGUF file")
def test_native_gguf_load_keeps_projections_quantized_and_matches_reference() -> None:
    import torch
    from transformers import AutoTokenizer, Gemma2ForCausalLM

    from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

    model = load_gemma2_model_from_gguf(_LOCAL_GGUF, torch.float32)

    # Projections stay quantized; embedding and norms are materialized; nothing left on meta.
    assert isinstance(model.layers[1].self_attn.q_proj.weight, GGMLTensor)
    assert not isinstance(model.embed_tokens.weight, GGMLTensor)
    assert not isinstance(model.norm.weight, GGMLTensor)
    assert not isinstance(model.layers[0].input_layernorm.weight, GGMLTensor)
    assert not any(p.is_meta for p in model.parameters())

    tok = AutoTokenizer.from_pretrained(str(_LOCAL_GGUF.parent), gguf_file=_LOCAL_GGUF.name, local_files_only=True)
    inputs = tok(["a golden retriever puppy in a sunlit meadow"], return_tensors="pt")

    with torch.no_grad():
        hs_native = model(inputs.input_ids, inputs.attention_mask)[0]

    # Reference: transformers' own GGUF loader, which fully dequantizes every weight.
    reference = Gemma2ForCausalLM.from_pretrained(
        str(_LOCAL_GGUF.parent), gguf_file=_LOCAL_GGUF.name, torch_dtype=torch.float32
    ).model
    reference.eval()
    with torch.no_grad():
        hs_ref = reference(inputs.input_ids, inputs.attention_mask)[0]

    # Both decode the same q4_k_m weights; the small residual is quantization-path noise.
    assert hs_native.shape == hs_ref.shape
    assert (hs_native.float() - hs_ref.float()).abs().mean().item() < 0.05
