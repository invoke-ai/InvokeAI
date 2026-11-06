"""Unit tests for building a T5 encoder from a bare state dict (shared by the GGUF and single-file loaders)."""

import pytest
import torch

from invokeai.backend.t5.t5_encoder import infer_t5_encoder_config, make_t5_feed_forward_safe


def _synthetic_t5_state_dict(
    *,
    vocab_size: int = 32,
    d_model: int = 8,
    inner_dim: int = 16,
    num_buckets: int = 4,
    num_heads: int = 2,
    d_ff: int = 24,
    num_layers: int = 2,
) -> dict[str, torch.Tensor]:
    """A minimal HF-named T5 encoder state dict with the shapes the inference reads."""
    sd: dict[str, torch.Tensor] = {
        "shared.weight": torch.empty(vocab_size, d_model),
        "encoder.block.0.layer.0.SelfAttention.q.weight": torch.empty(inner_dim, d_model),
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": torch.empty(num_buckets, num_heads),
        "encoder.block.0.layer.1.DenseReluDense.wi_0.weight": torch.empty(d_ff, d_model),
    }
    # Add keys for the remaining blocks so num_layers is inferred correctly.
    for i in range(1, num_layers):
        sd[f"encoder.block.{i}.layer.0.SelfAttention.q.weight"] = torch.empty(inner_dim, d_model)
    return sd


class TestInferT5ConfigFromStateDict:
    def test_infers_expected_dimensions(self):
        sd = _synthetic_t5_state_dict()
        config = infer_t5_encoder_config(sd)

        assert config.vocab_size == 32
        assert config.d_model == 8
        assert config.num_heads == 2
        assert config.d_kv == 8  # inner_dim (16) // num_heads (2)
        assert config.d_ff == 24
        assert config.num_layers == 2
        assert config.relative_attention_num_buckets == 4
        # Fixed values expected for the targeted T5 v1.1 XXL family.
        assert config.feed_forward_proj == "gated-gelu"
        assert config.is_gated_act is True

    @pytest.mark.parametrize(
        "missing_key",
        [
            "shared.weight",
            "encoder.block.0.layer.0.SelfAttention.q.weight",
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            "encoder.block.0.layer.1.DenseReluDense.wi_0.weight",
        ],
    )
    def test_missing_required_key_raises(self, missing_key: str):
        sd = _synthetic_t5_state_dict(num_layers=1)
        del sd[missing_key]
        with pytest.raises(ValueError):
            infer_t5_encoder_config(sd)


class TestMakeT5FeedForwardSafe:
    def _tiny_t5_encoder(self):
        from transformers import T5Config, T5EncoderModel

        config = T5Config(
            vocab_size=32,
            d_model=8,
            d_kv=4,
            d_ff=16,
            num_layers=1,
            num_heads=2,
            feed_forward_proj="gated-gelu",
            is_gated_act=True,
            dense_act_fn="gelu_new",
        )
        return T5EncoderModel(config)

    def test_gated_feed_forward_is_patched(self):
        model = self._tiny_t5_encoder()
        make_t5_feed_forward_safe(model)

        patched = [m for m in model.modules() if m.__class__.__name__ == "T5DenseGatedActDense"]
        assert patched, "expected at least one gated feed-forward module in a gated-gelu T5"
        for module in patched:
            # The forward was rebound to the module-local ``gated_forward`` closure.
            assert module.forward.__func__.__name__ == "gated_forward"

    @pytest.mark.parametrize("storage_dtype", [torch.uint8, torch.float8_e4m3fn])
    def test_patched_forward_keeps_activations_out_of_quantized_storage_dtypes(self, storage_dtype: torch.dtype):
        # Regression guard: GGML stores wo as uint8 and a single-file fp8 checkpoint can keep it in float8.
        # Neither may become the activations' dtype; the Linear dequantizes its weight instead.
        model = self._tiny_t5_encoder()
        make_t5_feed_forward_safe(model)
        ff = next(m for m in model.modules() if m.__class__.__name__ == "T5DenseGatedActDense")

        received_dtypes = []

        class _RecordingWo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.zeros(8, 16, dtype=storage_dtype)

            def forward(self, x):
                received_dtypes.append(x.dtype)
                return x.to(torch.float32)[..., :8]

        ff.wo = _RecordingWo()
        ff(torch.randn(1, 3, 8))

        assert received_dtypes == [torch.float32]

    def test_patched_forward_casts_to_a_compute_dtype_wo(self):
        # The cast transformers performs for a wo held in another float compute dtype is preserved.
        model = self._tiny_t5_encoder()
        make_t5_feed_forward_safe(model)
        ff = next(m for m in model.modules() if m.__class__.__name__ == "T5DenseGatedActDense")
        ff.wo = ff.wo.to(torch.float64)

        out = ff(torch.randn(1, 3, 8))

        assert out.dtype == torch.float64

    def test_raises_when_no_feed_forward_modules_match(self):
        # If transformers ever renames the T5 feed-forward classes, patching would be a silent no-op.
        # The guard must fail loudly instead.
        model = torch.nn.Linear(2, 2)
        with pytest.raises(RuntimeError, match="Failed to patch any T5 feed-forward modules"):
            make_t5_feed_forward_safe(model)
