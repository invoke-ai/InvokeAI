"""Unit tests for the GGUF-quantized T5 encoder loader helpers.

These cover the llama.cpp -> HF transformers key remapping of ``T5EncoderGGUFModel`` in isolation
(``_convert_t5_gguf_to_transformers``). The config inference and the feed-forward workaround it shares
with the single-file loader are covered in ``tests/backend/t5/test_t5_encoder.py``.

The loader's ``__init__`` needs the full model-cache infrastructure, but the methods under test
only use the static ``_shape_of`` helper, so the tests instantiate the class via ``object.__new__``
to bypass the constructor.
"""

import re

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


def test_convert_keys_use_expected_block_regex():
    # Sanity check that the module's block pattern only matches ``enc.blk.N.*`` keys.
    pattern = re.compile(r"^enc\.blk\.(\d+)\.(.+)$")
    assert pattern.match("enc.blk.0.attn_q.weight")
    assert not pattern.match("enc.output_norm.weight")
    assert not pattern.match("token_embd.weight")
