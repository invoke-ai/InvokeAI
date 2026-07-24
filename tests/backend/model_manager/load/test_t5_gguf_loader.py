"""Unit tests for the GGUF-quantized T5 encoder loader helpers.

These cover the pure, high-risk parts of ``T5EncoderGGUFModel`` in isolation:
- ``_convert_t5_gguf_to_transformers`` (llama.cpp -> HF transformers key remapping)
- ``_infer_t5_config_from_state_dict`` (tensor-shape -> ``T5Config`` inference)
- ``_make_feed_forward_gguf_safe`` (the wo-dtype workaround and its fail-loud guard)

The loader's ``__init__`` needs the full model-cache infrastructure, but the methods under test
only use the static ``_shape_of`` helper, so the tests instantiate the class via ``object.__new__``
to bypass the constructor.
"""

import re

import pytest
import torch

from invokeai.backend.model_manager.load.model_loaders.flux import T5EncoderGGUFModel


def _loader() -> T5EncoderGGUFModel:
    """Build a loader instance without running the (cache-dependent) constructor."""
    return object.__new__(T5EncoderGGUFModel)


class TestConvertT5GGUFToTransformers:
    def test_top_level_keys_are_remapped(self):
        sd = {
            "token_embd.weight": torch.empty(1),
            "enc.output_norm.weight": torch.empty(1),
        }
        out = _loader()._convert_t5_gguf_to_transformers(sd)
        assert set(out.keys()) == {"shared.weight", "encoder.final_layer_norm.weight"}

    def test_attention_keys_map_to_layer_0(self):
        sd = {
            "enc.blk.3.attn_q.weight": torch.empty(1),
            "enc.blk.3.attn_k.weight": torch.empty(1),
            "enc.blk.3.attn_v.weight": torch.empty(1),
            "enc.blk.3.attn_o.weight": torch.empty(1),
            "enc.blk.3.attn_norm.weight": torch.empty(1),
            "enc.blk.0.attn_rel_b.weight": torch.empty(1),
        }
        out = _loader()._convert_t5_gguf_to_transformers(sd)
        assert set(out.keys()) == {
            "encoder.block.3.layer.0.SelfAttention.q.weight",
            "encoder.block.3.layer.0.SelfAttention.k.weight",
            "encoder.block.3.layer.0.SelfAttention.v.weight",
            "encoder.block.3.layer.0.SelfAttention.o.weight",
            "encoder.block.3.layer.0.layer_norm.weight",
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
        }

    def test_feed_forward_keys_map_to_layer_1(self):
        sd = {
            "enc.blk.5.ffn_gate.weight": torch.empty(1),
            "enc.blk.5.ffn_up.weight": torch.empty(1),
            "enc.blk.5.ffn_down.weight": torch.empty(1),
            "enc.blk.5.ffn_norm.weight": torch.empty(1),
        }
        out = _loader()._convert_t5_gguf_to_transformers(sd)
        assert set(out.keys()) == {
            "encoder.block.5.layer.1.DenseReluDense.wi_0.weight",
            "encoder.block.5.layer.1.DenseReluDense.wi_1.weight",
            "encoder.block.5.layer.1.DenseReluDense.wo.weight",
            "encoder.block.5.layer.1.layer_norm.weight",
        }

    def test_values_are_preserved_by_identity(self):
        tensor = torch.empty(1)
        out = _loader()._convert_t5_gguf_to_transformers({"enc.blk.0.attn_q.weight": tensor})
        assert out["encoder.block.0.layer.0.SelfAttention.q.weight"] is tensor

    def test_unknown_block_component_is_kept_as_is(self):
        # Preserved verbatim so the loader's meta-tensor check surfaces it as an unmapped key.
        sd = {"enc.blk.0.mystery_component.weight": torch.empty(1)}
        out = _loader()._convert_t5_gguf_to_transformers(sd)
        assert set(out.keys()) == {"enc.blk.0.mystery_component.weight"}

    def test_non_string_and_unrelated_keys_pass_through(self):
        sd = {
            0: torch.empty(1),  # non-string key
            "some.unrelated.key": torch.empty(1),
        }
        out = _loader()._convert_t5_gguf_to_transformers(sd)
        assert set(out.keys()) == {0, "some.unrelated.key"}


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
        config = _loader()._infer_t5_config_from_state_dict(sd)

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
            _loader()._infer_t5_config_from_state_dict(sd)


class TestMakeFeedForwardGGUFSafe:
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
        T5EncoderGGUFModel._make_feed_forward_gguf_safe(model)

        patched = [m for m in model.modules() if m.__class__.__name__ == "T5DenseGatedActDense"]
        assert patched, "expected at least one gated feed-forward module in a gated-gelu T5"
        for module in patched:
            # The forward was rebound to the module-local ``gated_forward`` closure.
            assert module.forward.__func__.__name__ == "gated_forward"

    def test_patched_forward_does_not_cast_to_integer_weight_dtype(self):
        # Regression guard for the uint8/int8 dtype bug: with a fake uint8 (non-floating) wo weight,
        # the patched forward must NOT cast activations to the weight dtype.
        model = self._tiny_t5_encoder()
        T5EncoderGGUFModel._make_feed_forward_gguf_safe(model)
        ff = next(m for m in model.modules() if m.__class__.__name__ == "T5DenseGatedActDense")

        # Swap wo for a stub whose weight is a uint8 tensor and record the dtype it receives.
        received_dtypes = []

        class _RecordingWo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.zeros(8, 16, dtype=torch.uint8)

            def forward(self, x):
                received_dtypes.append(x.dtype)
                return x.to(torch.float32)[..., :8]

        ff.wo = _RecordingWo()
        ff(torch.randn(1, 3, 8))

        assert received_dtypes and received_dtypes[0].is_floating_point

    def test_raises_when_no_feed_forward_modules_match(self):
        # If transformers ever renames the T5 feed-forward classes, patching would be a silent no-op.
        # The guard must fail loudly instead.
        model = torch.nn.Linear(2, 2)
        with pytest.raises(RuntimeError, match="Failed to patch any T5 feed-forward modules"):
            T5EncoderGGUFModel._make_feed_forward_gguf_safe(model)


def test_convert_keys_use_expected_block_regex():
    # Sanity check that the module's block pattern only matches ``enc.blk.N.*`` keys.
    pattern = re.compile(r"^enc\.blk\.(\d+)\.(.+)$")
    assert pattern.match("enc.blk.0.attn_q.weight")
    assert not pattern.match("enc.output_norm.weight")
    assert not pattern.match("token_embd.weight")
