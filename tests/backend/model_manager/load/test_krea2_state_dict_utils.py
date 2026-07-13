"""Unit tests for the Krea-2 loader state-dict helpers.

These cover the pure key/tensor transforms that the single-file, GGUF and Qwen3-VL encoder loaders
run before ``load_state_dict`` (prefix stripping, native<->diffusers key conversion, scaled-fp8
dequantization, encoder key remapping) plus the shared ``_reject_incomplete_load`` guard that turns a
silent partial load into an actionable error. They exercise the conversion logic without needing the
real (diffusers ``Krea2Transformer2DModel`` / transformers ``Qwen3VLModel``) architectures or weights.
"""

import re

import accelerate
import pytest
import torch

from invokeai.backend.model_manager.load.model_loaders.krea2 import (
    _convert_krea2_native_to_diffusers,
    _dequantize_scaled_fp8,
    _is_native_krea2_format,
    _reject_incomplete_load,
    _remap_qwen3vl_singlefile_keys,
    _strip_comfyui_prefix,
)


class TestStripComfyuiPrefix:
    @pytest.mark.parametrize("prefix", ["model.diffusion_model.", "diffusion_model."])
    def test_strips_known_prefixes(self, prefix: str) -> None:
        sd = {f"{prefix}blocks.0.weight": torch.zeros(1), f"{prefix}first.weight": torch.zeros(1)}
        out = _strip_comfyui_prefix(sd)
        assert set(out.keys()) == {"blocks.0.weight", "first.weight"}

    def test_noop_when_no_prefix(self) -> None:
        sd = {"blocks.0.weight": torch.zeros(1), "img_in.weight": torch.zeros(1)}
        out = _strip_comfyui_prefix(sd)
        assert set(out.keys()) == set(sd.keys())

    def test_only_the_first_matching_prefix_is_used(self) -> None:
        # "model.diffusion_model." is checked before "diffusion_model.", so both strip to the same tail.
        sd = {"model.diffusion_model.blocks.0.weight": torch.zeros(1)}
        out = _strip_comfyui_prefix(sd)
        assert list(out.keys()) == ["blocks.0.weight"]


class TestIsNativeKrea2Format:
    @pytest.mark.parametrize(
        "key",
        ["blocks.0.attn.wq.weight", "txtfusion.0.mlp.up.weight", "first.weight", "blocks.0.mod.lin"],
    )
    def test_true_for_native_keys(self, key: str) -> None:
        assert _is_native_krea2_format({key: torch.zeros(1)}) is True

    @pytest.mark.parametrize(
        "key",
        ["transformer_blocks.0.attn.to_q.weight", "img_in.weight", "text_fusion.0.ff.up.weight"],
    )
    def test_false_for_diffusers_keys(self, key: str) -> None:
        assert _is_native_krea2_format({key: torch.zeros(1)}) is False


class TestDequantizeScaledFp8:
    def test_folds_scale_into_weight_and_drops_scale_key(self) -> None:
        sd = {
            "layer.weight": torch.tensor([2.0, 4.0]),
            "layer.weight_scale": torch.tensor(0.5),
        }
        out = _dequantize_scaled_fp8(sd)
        assert "layer.weight_scale" not in out
        assert torch.allclose(out["layer.weight"], torch.tensor([1.0, 2.0]))

    def test_noop_without_scale_keys(self) -> None:
        sd = {"layer.weight": torch.tensor([2.0, 4.0])}
        out = _dequantize_scaled_fp8(sd)
        assert out is sd

    def test_orphan_scale_key_is_dropped(self) -> None:
        # A scale key with no matching weight is simply removed (nothing to multiply).
        sd = {"other.weight": torch.tensor([1.0]), "layer.weight_scale": torch.tensor(0.5)}
        out = _dequantize_scaled_fp8(sd)
        assert "layer.weight_scale" not in out
        assert "other.weight" in out


class TestConvertKrea2NativeToDiffusers:
    def test_top_level_module_renames(self) -> None:
        sd = {
            "first.weight": torch.zeros(1),
            "tmlp.0.weight": torch.zeros(1),
            "tmlp.2.weight": torch.zeros(1),
            "tproj.1.weight": torch.zeros(1),
            "txtmlp.0.scale": torch.zeros(1),
            "txtmlp.1.weight": torch.zeros(1),
            "txtmlp.3.weight": torch.zeros(1),
            "last.linear.weight": torch.zeros(1),
            "last.norm.scale": torch.zeros(1),
        }
        out = _convert_krea2_native_to_diffusers(sd)
        assert "img_in.weight" in out
        assert "time_embed.linear_1.weight" in out
        assert "time_embed.linear_2.weight" in out
        assert "time_mod_proj.weight" in out
        assert "txt_in.norm.weight" in out
        assert "txt_in.linear_1.weight" in out
        assert "txt_in.linear_2.weight" in out
        assert "final_layer.linear.weight" in out
        assert "final_layer.norm.weight" in out

    def test_within_block_renames(self) -> None:
        sd = {
            "blocks.0.attn.wq.weight": torch.zeros(1),
            "blocks.0.attn.wk.weight": torch.zeros(1),
            "blocks.0.attn.wv.weight": torch.zeros(1),
            "blocks.0.attn.wo.weight": torch.zeros(1),
            "blocks.0.attn.gate.weight": torch.zeros(1),
            "blocks.0.attn.qknorm.qnorm.scale": torch.zeros(1),
            "blocks.0.attn.qknorm.knorm.scale": torch.zeros(1),
            "blocks.0.mlp.gate.weight": torch.zeros(1),
            "blocks.0.mlp.up.weight": torch.zeros(1),
            "blocks.0.mlp.down.weight": torch.zeros(1),
            "blocks.0.prenorm.scale": torch.zeros(1),
            "blocks.0.postnorm.scale": torch.zeros(1),
            "txtfusion.1.attn.wq.weight": torch.zeros(1),
        }
        out = _convert_krea2_native_to_diffusers(sd)
        assert "transformer_blocks.0.attn.to_q.weight" in out
        assert "transformer_blocks.0.attn.to_k.weight" in out
        assert "transformer_blocks.0.attn.to_v.weight" in out
        assert "transformer_blocks.0.attn.to_out.0.weight" in out
        assert "transformer_blocks.0.attn.to_gate.weight" in out
        assert "transformer_blocks.0.attn.norm_q.weight" in out
        assert "transformer_blocks.0.attn.norm_k.weight" in out
        assert "transformer_blocks.0.ff.gate.weight" in out
        assert "transformer_blocks.0.ff.up.weight" in out
        assert "transformer_blocks.0.ff.down.weight" in out
        assert "transformer_blocks.0.norm1.weight" in out
        assert "transformer_blocks.0.norm2.weight" in out
        # text_fusion tower renamed the same way.
        assert "text_fusion.1.attn.to_q.weight" in out
        # No native names survive.
        assert not any(".wq." in k or ".qknorm." in k or ".mlp." in k or "prenorm" in k for k in out)

    def test_final_block_projections_are_dropped(self) -> None:
        sd = {"last.down.weight": torch.zeros(2, 2), "last.up.weight": torch.zeros(2, 2)}
        out = _convert_krea2_native_to_diffusers(sd)
        assert out == {}

    def test_mod_lin_is_reshaped_to_scale_shift_table(self) -> None:
        # A flat (6*H,) per-block modulation vector becomes a (6, H) scale_shift_table.
        sd = {"blocks.0.mod.lin": torch.arange(12, dtype=torch.float32)}
        out = _convert_krea2_native_to_diffusers(sd)
        assert "transformer_blocks.0.scale_shift_table" in out
        table = out["transformer_blocks.0.scale_shift_table"]
        assert table.shape == (6, 2)
        assert torch.equal(table, torch.arange(12, dtype=torch.float32).reshape(6, 2))

    def test_non_string_keys_pass_through(self) -> None:
        sentinel = object()
        out = _convert_krea2_native_to_diffusers({sentinel: torch.zeros(1)})  # type: ignore[dict-item]
        assert sentinel in out


class TestRemapQwen3vlSinglefileKeys:
    def test_routes_towers_and_prefixes_bare_language_model_keys(self) -> None:
        sd = {
            "model.visual.blocks.0.weight": torch.zeros(1),
            "model.language_model.layers.0.weight": torch.zeros(1),
            "model.layers.1.weight": torch.zeros(1),  # bare LM key under a model. prefix
            "model.embed_tokens.weight": torch.zeros(1),
            "model.norm.weight": torch.zeros(1),
            "visual.blocks.1.weight": torch.zeros(1),  # already un-prefixed
            "layers.2.weight": torch.zeros(1),  # bare, no model. prefix
        }
        out = _remap_qwen3vl_singlefile_keys(sd)
        assert "visual.blocks.0.weight" in out
        assert "language_model.layers.0.weight" in out
        assert "language_model.layers.1.weight" in out
        assert "language_model.embed_tokens.weight" in out
        assert "language_model.norm.weight" in out
        assert "visual.blocks.1.weight" in out
        assert "language_model.layers.2.weight" in out
        # No key retains the leading model. prefix.
        assert not any(k.startswith("model.") for k in out)


class TestRejectIncompleteLoad:
    @pytest.mark.parametrize(
        "what",
        ["Krea-2 single-file checkpoint", "Krea-2 GGUF checkpoint", "Qwen3-VL encoder checkpoint"],
    )
    def test_raises_when_parameters_remain_on_meta_device(self, what: str) -> None:
        # accelerate.init_empty_weights() leaves every parameter on the meta device — the exact state a
        # strict=False load produces for a checkpoint that omits required weights. All three Krea-2 loaders
        # feed their `what` label through this guard, so parametrize over the real call-site messages.
        with accelerate.init_empty_weights():
            model = torch.nn.Linear(4, 4)
        with pytest.raises(RuntimeError, match=re.escape(f"{what} is incomplete")):
            _reject_incomplete_load(model, what=what)

    def test_does_not_raise_for_a_fully_materialized_model(self) -> None:
        model = torch.nn.Linear(4, 4)  # normal construction — no meta tensors
        _reject_incomplete_load(model, what="Krea-2 single-file checkpoint")

    def test_names_the_missing_parameters(self) -> None:
        # Materialize only the weight; the bias stays on meta and must be named in the error.
        with accelerate.init_empty_weights():
            model = torch.nn.Linear(4, 4)
        model.load_state_dict({"weight": torch.zeros(4, 4)}, strict=False, assign=True)
        with pytest.raises(RuntimeError, match="bias") as exc_info:
            _reject_incomplete_load(model, what="Krea-2 single-file checkpoint")
        assert "1 parameter(s)" in str(exc_info.value)
