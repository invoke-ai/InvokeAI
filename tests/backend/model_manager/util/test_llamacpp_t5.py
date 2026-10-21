"""The llama.cpp T5-encoder GGUF helpers shared by the FLUX/SD3 T5 v1.1 and the Wan UMT5 loaders.

The key mapping has to be pinned name by name: ``attn_q``/``attn_k`` and ``ffn_gate``/``ffn_up`` are
same-shaped pairs, so a swap loads cleanly and encodes wrongly with nothing logged.
"""

import gguf
import pytest
import torch
from transformers import T5Config, T5EncoderModel, UMT5Config, UMT5EncoderModel

from invokeai.backend.model_manager.util.llamacpp_t5 import (
    convert_llamacpp_t5_keys,
    has_per_layer_relative_bias,
    infer_t5_encoder_config,
    make_feed_forward_gguf_safe,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor


class TestConvertLlamacppT5Keys:
    def test_top_level_keys_are_remapped(self):
        sd = {"token_embd.weight": "embedding", "enc.output_norm.weight": "final norm"}

        assert convert_llamacpp_t5_keys(sd) == {
            "shared.weight": "embedding",
            "encoder.final_layer_norm.weight": "final norm",
        }

    def test_attention_keys_map_to_layer_0_including_every_blocks_relative_bias(self):
        sd = {
            "enc.blk.3.attn_q.weight": "q",
            "enc.blk.3.attn_k.weight": "k",
            "enc.blk.3.attn_v.weight": "v",
            "enc.blk.3.attn_o.weight": "o",
            "enc.blk.3.attn_norm.weight": "norm",
            "enc.blk.0.attn_rel_b.weight": "bias 0",
            # UMT5 has one per block.
            "enc.blk.3.attn_rel_b.weight": "bias 3",
        }

        assert convert_llamacpp_t5_keys(sd) == {
            "encoder.block.3.layer.0.SelfAttention.q.weight": "q",
            "encoder.block.3.layer.0.SelfAttention.k.weight": "k",
            "encoder.block.3.layer.0.SelfAttention.v.weight": "v",
            "encoder.block.3.layer.0.SelfAttention.o.weight": "o",
            "encoder.block.3.layer.0.layer_norm.weight": "norm",
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": "bias 0",
            "encoder.block.3.layer.0.SelfAttention.relative_attention_bias.weight": "bias 3",
        }

    def test_feed_forward_keys_map_to_layer_1_with_gate_as_the_activated_branch(self):
        sd = {
            "enc.blk.5.ffn_gate.weight": "gate",
            "enc.blk.5.ffn_up.weight": "up",
            "enc.blk.5.ffn_down.weight": "down",
            "enc.blk.5.ffn_norm.weight": "norm",
        }

        assert convert_llamacpp_t5_keys(sd) == {
            "encoder.block.5.layer.1.DenseReluDense.wi_0.weight": "gate",
            "encoder.block.5.layer.1.DenseReluDense.wi_1.weight": "up",
            "encoder.block.5.layer.1.DenseReluDense.wo.weight": "down",
            "encoder.block.5.layer.1.layer_norm.weight": "norm",
        }

    def test_unknown_keys_pass_through_for_the_completeness_check_to_name(self):
        sd = {0: "non-string", "some.unrelated.key": "x", "enc.blk.0.mystery_component.weight": "y"}

        assert convert_llamacpp_t5_keys(sd) == sd


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (["enc.blk.0.attn_rel_b.weight", "enc.blk.1.attn_q.weight", "enc.blk.12.attn_q.weight"], False),
        (["enc.blk.0.attn_rel_b.weight", "enc.blk.1.attn_rel_b.weight"], True),
        (["enc.blk.23.attn_rel_b.weight"], True),
        # A transformers-named T5 is not llama.cpp-named; its block-0-only bias says nothing here.
        (["encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight"], False),
    ],
    ids=["t5_v1_1", "umt5", "umt5_last_block", "transformers_naming"],
)
def test_per_layer_relative_bias_marks_umt5(keys: list[str], expected: bool):
    assert has_per_layer_relative_bias(keys) is expected


def _t5_state_dict(*, num_layers: int = 2) -> dict[str, torch.Tensor]:
    """A transformers-named encoder state dict with the shapes the inference reads."""
    sd: dict[str, torch.Tensor] = {
        "shared.weight": torch.empty(32, 8),
        "encoder.block.0.layer.0.SelfAttention.q.weight": torch.empty(16, 8),
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": torch.empty(4, 2),
        "encoder.block.0.layer.1.DenseReluDense.wi_0.weight": torch.empty(24, 8),
    }
    for block in range(1, num_layers):
        sd[f"encoder.block.{block}.layer.0.SelfAttention.q.weight"] = torch.empty(16, 8)
    return sd


class TestInferT5EncoderConfig:
    @pytest.mark.parametrize("config_class", [T5Config, UMT5Config])
    def test_infers_the_dimensions_from_tensor_shapes(self, config_class):
        config = infer_t5_encoder_config(_t5_state_dict(), config_class)

        assert type(config) is config_class
        assert (config.vocab_size, config.d_model, config.num_heads, config.d_kv) == (32, 8, 2, 8)
        assert (config.d_ff, config.num_layers, config.relative_attention_num_buckets) == (24, 2, 4)
        # The constants T5 v1.1 and UMT5 share.
        assert config.relative_attention_max_distance == 128
        assert config.layer_norm_epsilon == 1e-6
        assert config.is_gated_act is True
        assert config.dense_act_fn == "gelu_new"

    @pytest.mark.parametrize(
        "missing_key",
        [
            "shared.weight",
            "encoder.block.0.layer.0.SelfAttention.q.weight",
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            "encoder.block.0.layer.1.DenseReluDense.wi_0.weight",
        ],
    )
    def test_a_missing_required_tensor_raises(self, missing_key: str):
        sd = _t5_state_dict(num_layers=1)
        del sd[missing_key]

        with pytest.raises(ValueError, match="Could not find"):
            infer_t5_encoder_config(sd, T5Config)


def _tiny_encoder(model_class, config_class):
    config = config_class(
        vocab_size=32, d_model=32, d_kv=16, d_ff=64, num_layers=1, num_heads=2, feed_forward_proj="gated-gelu"
    )
    return model_class(config).eval()


_FAMILIES = [(T5EncoderModel, T5Config), (UMT5EncoderModel, UMT5Config)]


class TestMakeFeedForwardGGUFSafe:
    @pytest.mark.parametrize(("model_class", "config_class"), _FAMILIES, ids=["t5", "umt5"])
    def test_activations_reach_a_packed_wo_as_floats(self, model_class, config_class):
        """transformers casts the activations to ``wo.weight.dtype``; a quantized GGMLTensor reports
        ``uint8`` there, which turns them into integers before the projection."""
        model = _tiny_encoder(model_class, config_class)
        make_feed_forward_gguf_safe(model)
        feed_forward = model.encoder.block[0].layer[1].DenseReluDense
        received: list[torch.dtype] = []

        class _PackedWo(torch.nn.Module):
            weight = torch.zeros(32, 64, dtype=torch.uint8)

            def forward(self, x):
                received.append(x.dtype)
                return x[..., :32]

        feed_forward.wo = _PackedWo()
        feed_forward(torch.randn(1, 3, 32))

        assert received == [torch.float32]

    @pytest.mark.parametrize(("model_class", "config_class"), _FAMILIES, ids=["t5", "umt5"])
    def test_an_f16_gguf_wo_gives_the_dense_result(self, model_class, config_class):
        """An F16 GGUF stores ``wo`` unquantized, as a GGMLTensor reporting ``float16``. Casting the
        activations to that dtype left them float16 against the weight dequantized to the compute
        dtype, and the matmul failed."""
        model = _tiny_encoder(model_class, config_class)
        feed_forward = model.encoder.block[0].layer[1].DenseReluDense
        hidden_states = torch.randn(1, 3, 32)
        f16_weight = feed_forward.wo.weight.detach().to(torch.float16)
        feed_forward.wo.weight = torch.nn.Parameter(f16_weight.float(), requires_grad=False)
        with torch.no_grad():
            expected = feed_forward(hidden_states)

        feed_forward.wo.weight = torch.nn.Parameter(
            GGMLTensor(f16_weight, gguf.GGMLQuantizationType.F16, f16_weight.shape, torch.float32),
            requires_grad=False,
        )
        make_feed_forward_gguf_safe(model)
        with torch.no_grad():
            actual = feed_forward(hidden_states)

        torch.testing.assert_close(actual, expected)

    def test_a_plain_float_wo_still_gets_its_own_dtype(self):
        """The cast transformers makes is right for a dense ``wo`` of another dtype -- float32 under a
        float16 load, by ``_keep_in_fp32_modules``. Shown with float64 to keep half precision off the CPU."""
        model = _tiny_encoder(UMT5EncoderModel, UMT5Config)
        make_feed_forward_gguf_safe(model)
        feed_forward = model.encoder.block[0].layer[1].DenseReluDense
        feed_forward.wo.to(torch.float64)

        output = feed_forward(torch.randn(1, 3, 32))

        assert output.dtype == torch.float64

    def test_raises_when_no_feed_forward_modules_match(self):
        """A rename in transformers would otherwise make the patch a silent no-op."""
        with pytest.raises(RuntimeError, match="Failed to patch any T5 feed-forward modules"):
            make_feed_forward_gguf_safe(torch.nn.Linear(2, 2))
