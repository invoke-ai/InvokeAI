"""Tests for the native GGMLTensor Gemma-2 GGUF encoder loader.

The loader keeps the large 2D projection weights quantized (as GGMLTensor, dequantized on demand by the
model cache) and materializes only the embedding and RMSNorm weights, so a GGUF Gemma-2 encoder no longer
costs the same VRAM as the unquantized model.

The load tests run against a synthetic 2-layer Gemma-2 built from mocked GGUF tensors — small enough to
quantize, load and run a forward pass on in CI, while exercising the real code path: quantized retention,
the llama.cpp norm (+1) convention, meta-buffer repair and a forward with quantized weights in place.
Set ``INVOKEAI_TEST_GEMMA2_GGUF`` to a real Gemma-2-2b GGUF to additionally run the end-to-end comparison
against transformers' own (fully dequantizing) GGUF loader.
"""

import importlib
import os
from pathlib import Path
from typing import Any

import accelerate
import gguf
import numpy as np
import pytest
import torch

import invokeai.backend.quantization.gguf.loaders
from invokeai.backend.model_manager.load.model_loaders.gemma2_encoder import (
    _convert_gemma_llamacpp_to_pytorch,
    load_gemma2_model_from_gguf,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

# A real Gemma-2-2b GGUF for the opt-in end-to-end load/compare test.
_LOCAL_GGUF_ENV_VAR = "INVOKEAI_TEST_GEMMA2_GGUF"


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


# --------------------------------------------------------------------------------------------------
# Synthetic GGUF fixture
# --------------------------------------------------------------------------------------------------

# A Gemma-2 shaped small enough to quantize and run in CI. Every dimension that ends up as a tensor's
# last axis is a multiple of Q8_0's 32-element block size.
_VOCAB = 96
_HIDDEN = 64
_INTERMEDIATE = 128
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 16
_LAYERS = 2

_TINY_GEMMA_CONFIG: dict[str, Any] = {
    "model_type": "gemma2",
    "vocab_size": _VOCAB,
    "hidden_size": _HIDDEN,
    "intermediate_size": _INTERMEDIATE,
    "num_hidden_layers": _LAYERS,
    "num_attention_heads": _HEADS,
    "num_key_value_heads": _KV_HEADS,
    "head_dim": _HEAD_DIM,
    "max_position_embeddings": 128,
    "query_pre_attn_scalar": _HEAD_DIM,
    "sliding_window": 32,
    "rope_theta": 10000.0,
}


def _q8(*shape: int) -> GGMLTensor:
    """A Q8_0-quantized GGMLTensor, like the 2D projection weights in a real Gemma-2 GGUF."""
    rng = np.random.default_rng(seed=sum(shape))
    data = rng.normal(scale=0.05, size=shape).astype(np.float32)
    quantized = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0)
    return GGMLTensor(torch.from_numpy(quantized), gguf.GGMLQuantizationType.Q8_0, torch.Size(shape), torch.float32)


def _f32(*shape: int) -> GGMLTensor:
    """An unquantized (F32) GGMLTensor — how llama.cpp stores the RMSNorm weights."""
    rng = np.random.default_rng(seed=sum(shape) + 1)
    data = torch.from_numpy(rng.normal(loc=1.0, scale=0.05, size=shape).astype(np.float32))
    return GGMLTensor(data, gguf.GGMLQuantizationType.F32, torch.Size(shape), torch.float32)


def _tiny_gguf_state_dict() -> dict[str, GGMLTensor]:
    """llama.cpp-named tensors for the tiny Gemma-2 above, as `gguf_sd_loader` would return them."""
    sd: dict[str, GGMLTensor] = {
        "token_embd.weight": _q8(_VOCAB, _HIDDEN),
        "output_norm.weight": _f32(_HIDDEN),
    }
    for i in range(_LAYERS):
        sd |= {
            f"blk.{i}.attn_q.weight": _q8(_HEADS * _HEAD_DIM, _HIDDEN),
            f"blk.{i}.attn_k.weight": _q8(_KV_HEADS * _HEAD_DIM, _HIDDEN),
            f"blk.{i}.attn_v.weight": _q8(_KV_HEADS * _HEAD_DIM, _HIDDEN),
            f"blk.{i}.attn_output.weight": _q8(_HIDDEN, _HEADS * _HEAD_DIM),
            f"blk.{i}.ffn_gate.weight": _q8(_INTERMEDIATE, _HIDDEN),
            f"blk.{i}.ffn_up.weight": _q8(_INTERMEDIATE, _HIDDEN),
            f"blk.{i}.ffn_down.weight": _q8(_HIDDEN, _INTERMEDIATE),
            f"blk.{i}.attn_norm.weight": _f32(_HIDDEN),
            f"blk.{i}.post_attention_norm.weight": _f32(_HIDDEN),
            f"blk.{i}.ffn_norm.weight": _f32(_HIDDEN),
            f"blk.{i}.post_ffw_norm.weight": _f32(_HIDDEN),
        }
    return sd


@pytest.fixture
def tiny_gguf(monkeypatch: pytest.MonkeyPatch) -> dict[str, GGMLTensor]:
    """Mock out the two GGUF file readers so the loader runs against in-memory tensors."""
    sd = _tiny_gguf_state_dict()
    # transformers exposes its submodules lazily, so reach for the module object explicitly.
    monkeypatch.setattr(
        importlib.import_module("transformers.modeling_gguf_pytorch_utils"),
        "load_gguf_checkpoint",
        lambda path, return_tensors=True: {"config": dict(_TINY_GEMMA_CONFIG)},
    )
    monkeypatch.setattr(
        invokeai.backend.quantization.gguf.loaders, "gguf_sd_loader", lambda path, compute_dtype: dict(sd)
    )
    return sd


# --------------------------------------------------------------------------------------------------
# Loader behaviour
# --------------------------------------------------------------------------------------------------


class TestNativeGgufLoad:
    def test_projections_stay_quantized(self, tiny_gguf: dict[str, GGMLTensor]) -> None:
        """The point of the native loader: the large 2D weights are never dequantized at load time."""
        model = load_gemma2_model_from_gguf(Path("unused.gguf"), torch.float32)

        for layer in model.layers:
            for proj in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
                layer.mlp.down_proj,
            ):
                assert isinstance(proj.weight, GGMLTensor)
                # The dequantized shape is what the module's math sees, not the packed byte count.
                assert proj.weight.shape == (proj.out_features, proj.in_features)

    def test_embedding_and_norms_are_materialized(self, tiny_gguf: dict[str, GGMLTensor]) -> None:
        model = load_gemma2_model_from_gguf(Path("unused.gguf"), torch.float32)

        # nn.Embedding needs indexed access, so the embedding cannot stay quantized.
        assert not isinstance(model.embed_tokens.weight, GGMLTensor)
        assert model.embed_tokens.weight.shape == (_VOCAB, _HIDDEN)
        torch.testing.assert_close(model.embed_tokens.weight, tiny_gguf["token_embd.weight"].get_dequantized_tensor())

        # llama.cpp folds the +1 into the stored norm weight; Gemma2RMSNorm re-adds it at runtime, so
        # the loader must subtract it back out (matching transformers' Gemma2TensorProcessor).
        for name, gguf_key in (
            ("norm", "output_norm.weight"),
            ("layers.0.input_layernorm", "blk.0.attn_norm.weight"),
            ("layers.1.post_feedforward_layernorm", "blk.1.post_ffw_norm.weight"),
        ):
            weight = model.get_submodule(name).weight
            assert not isinstance(weight, GGMLTensor)
            torch.testing.assert_close(weight, tiny_gguf[gguf_key].get_dequantized_tensor() - 1.0)

    def test_nothing_is_left_on_the_meta_device(self, tiny_gguf: dict[str, GGMLTensor]) -> None:
        """The model is built with `init_empty_weights`, so a weight the GGUF does not supply would
        silently remain a meta tensor and only fail at the first forward."""
        model = load_gemma2_model_from_gguf(Path("unused.gguf"), torch.float32)

        assert not any(p.is_meta for p in model.parameters())
        assert not any(b.is_meta for b in model.buffers())

    def test_meta_buffers_are_repaired(self, monkeypatch: pytest.MonkeyPatch, tiny_gguf: dict[str, GGMLTensor]) -> None:
        """The rotary embedding's `inv_freq` is not in the GGUF. Depending on the accelerate version
        `init_empty_weights` also puts buffers on meta, and the loader has to rebuild them."""
        real_init_empty_weights = accelerate.init_empty_weights
        monkeypatch.setattr(
            accelerate, "init_empty_weights", lambda include_buffers=True: real_init_empty_weights(include_buffers=True)
        )

        model = load_gemma2_model_from_gguf(Path("unused.gguf"), torch.float32)

        inv_freq = model.rotary_emb.inv_freq
        assert not inv_freq.is_meta
        assert inv_freq.shape == (_HEAD_DIM // 2,)
        assert torch.isfinite(inv_freq).all()
        expected = 1.0 / (
            _TINY_GEMMA_CONFIG["rope_theta"] ** (torch.arange(0, _HEAD_DIM, 2, dtype=torch.float32) / _HEAD_DIM)
        )
        torch.testing.assert_close(inv_freq.float(), expected)

    def test_forward_runs_with_quantized_weights(self, tiny_gguf: dict[str, GGMLTensor]) -> None:
        """End of the line: the quantized projections have to survive an actual forward pass."""
        model = load_gemma2_model_from_gguf(Path("unused.gguf"), torch.float32)

        input_ids = torch.tensor([[3, 17, 42, 8]])
        with torch.no_grad():
            hidden_states = model(input_ids, torch.ones_like(input_ids))[0]

        assert hidden_states.shape == (1, 4, _HIDDEN)
        assert torch.isfinite(hidden_states).all()
        # The projections were not dequantized in place by running the model.
        assert isinstance(model.layers[0].self_attn.q_proj.weight, GGMLTensor)

    def test_unexpected_tensor_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tiny_gguf: dict[str, GGMLTensor]
    ) -> None:
        """A tensor that maps cleanly but has no home in the configured model (here: a block beyond
        `num_hidden_layers`) must fail loudly instead of being dropped."""
        sd = dict(tiny_gguf) | {f"blk.{_LAYERS}.attn_q.weight": _q8(_HEADS * _HEAD_DIM, _HIDDEN)}
        monkeypatch.setattr(
            invokeai.backend.quantization.gguf.loaders, "gguf_sd_loader", lambda path, compute_dtype: sd
        )

        with pytest.raises(RuntimeError, match="Unexpected keys"):
            load_gemma2_model_from_gguf(Path("unused.gguf"), torch.float32)


@pytest.mark.skipif(not os.environ.get(_LOCAL_GGUF_ENV_VAR), reason=f"set {_LOCAL_GGUF_ENV_VAR} to a Gemma-2-2b GGUF")
def test_native_gguf_load_keeps_projections_quantized_and_matches_reference() -> None:
    from transformers import AutoTokenizer, Gemma2ForCausalLM

    gguf_path = Path(os.environ[_LOCAL_GGUF_ENV_VAR])
    model = load_gemma2_model_from_gguf(gguf_path, torch.float32)

    # Projections stay quantized; embedding and norms are materialized; nothing left on meta.
    assert isinstance(model.layers[1].self_attn.q_proj.weight, GGMLTensor)
    assert not isinstance(model.embed_tokens.weight, GGMLTensor)
    assert not isinstance(model.norm.weight, GGMLTensor)
    assert not isinstance(model.layers[0].input_layernorm.weight, GGMLTensor)
    assert not any(p.is_meta for p in model.parameters())

    tok = AutoTokenizer.from_pretrained(str(gguf_path.parent), gguf_file=gguf_path.name, local_files_only=True)
    inputs = tok(["a golden retriever puppy in a sunlit meadow"], return_tensors="pt")

    with torch.no_grad():
        hs_native = model(inputs.input_ids, inputs.attention_mask)[0]

    # Reference: transformers' own GGUF loader, which fully dequantizes every weight.
    reference = Gemma2ForCausalLM.from_pretrained(
        str(gguf_path.parent), gguf_file=gguf_path.name, torch_dtype=torch.float32
    ).model
    reference.eval()
    with torch.no_grad():
        hs_ref = reference(inputs.input_ids, inputs.attention_mask)[0]

    # Both decode the same weights; the small residual is quantization-path noise.
    assert hs_native.shape == hs_ref.shape
    assert (hs_native.float() - hs_ref.float()).abs().mean().item() < 0.05
