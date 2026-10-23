"""Unit tests for the Z-Image GGUF/ComfyUI -> diffusers state-dict converter."""

import gguf
import pytest
import torch

from invokeai.backend.model_manager.load.model_loaders.z_image import (
    _convert_z_image_gguf_to_diffusers,
    _remap_z_image_layer_paths,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.int8_convrot import extract_int8_convrot_markers
from tests.backend.model_manager.load.state_dicts.utils import keys_to_mock_state_dict
from tests.backend.model_manager.load.state_dicts.z_image_transformer_comfyui_keys import (
    state_dict_keys as z_image_keys,
)
from tests.fixtures.quantized_payloads import comfy_quant_marker


def _gguf_f32(values: torch.Tensor) -> GGMLTensor:
    """An F32 tensor as `gguf_sd_loader` hands it over: still a GGMLTensor, though nothing is quantized."""
    return GGMLTensor(values, gguf.GGMLQuantizationType.F32, values.shape, torch.float32)


class TestConvertZImageGgufToDiffusers:
    def test_fused_qkv_split(self):
        sd = keys_to_mock_state_dict(z_image_keys)
        n_qkv = sum(1 for k in sd if k.endswith(".attention.qkv.weight"))
        assert n_qkv > 0

        out = _convert_z_image_gguf_to_diffusers(sd)

        # Each fused qkv weight becomes three separate projections.
        assert sum(1 for k in out if k.endswith(".attention.to_q.weight")) == n_qkv
        assert sum(1 for k in out if k.endswith(".attention.to_k.weight")) == n_qkv
        assert sum(1 for k in out if k.endswith(".attention.to_v.weight")) == n_qkv
        assert not any(".attention.qkv." in k for k in out)

    def test_key_renames(self):
        out = _convert_z_image_gguf_to_diffusers(keys_to_mock_state_dict(z_image_keys))
        # q_norm/k_norm -> norm_q/norm_k, attention.out -> attention.to_out.0
        assert any(k.endswith(".attention.norm_q.weight") for k in out)
        assert any(k.endswith(".attention.norm_k.weight") for k in out)
        assert any(k.endswith(".attention.to_out.0.weight") for k in out)
        assert not any(".q_norm." in k or ".k_norm." in k for k in out)
        assert not any(".attention.out." in k for k in out)

    def test_embedder_and_final_layer_renamed(self):
        out = _convert_z_image_gguf_to_diffusers(keys_to_mock_state_dict(z_image_keys))
        assert any(k.startswith("all_x_embedder.2-1.") for k in out)
        assert any(k.startswith("all_final_layer.2-1.") for k in out)
        assert not any(k.startswith("x_embedder.") or k.startswith("final_layer.") for k in out)

    def test_norm_final_is_dropped(self):
        # The diffusers model uses a non-learnable final LayerNorm, so norm_final.* is skipped.
        assert any(k.startswith("norm_final.") for k in z_image_keys)
        out = _convert_z_image_gguf_to_diffusers(keys_to_mock_state_dict(z_image_keys))
        assert not any(k.startswith("norm_final.") for k in out)

    def test_pad_tokens_are_2d_after_conversion(self):
        # The diffusers model expects a leading batch dim on the pad tokens. The checkpoint
        # already stores them 2D; GGUF releases ship them 1D or 2D (see the tests below).
        out = _convert_z_image_gguf_to_diffusers(keys_to_mock_state_dict(z_image_keys))
        for pad in ("x_pad_token", "cap_pad_token"):
            assert out[pad].dim() == 2
            assert out[pad].shape[0] == 1

    @pytest.mark.parametrize(
        "stored",
        [
            pytest.param(lambda row: row, id="plain-1d"),
            pytest.param(lambda row: _gguf_f32(row), id="gguf-1d"),
            # The released Z-Image Turbo GGUFs: F32 with GGUF shape [dim, 1], which the reader hands over as (1, dim).
            pytest.param(lambda row: _gguf_f32(row.reshape(1, -1)), id="gguf-2d"),
        ],
    )
    def test_a_pad_token_becomes_a_plain_row(self, stored):
        # A pad token left packed reaches the forward's `torch.where`, which GGMLTensor does not dispatch.
        rows = {"x_pad_token": torch.arange(4.0), "cap_pad_token": torch.arange(4.0) + 10}
        out = _convert_z_image_gguf_to_diffusers({key: stored(row) for key, row in rows.items()})
        for key, row in rows.items():
            assert type(out[key]) is torch.Tensor
            assert torch.equal(out[key], row.reshape(1, 4))

    def test_a_pad_token_that_is_not_one_row_is_refused(self):
        with pytest.raises(ValueError, match=r"x_pad_token has shape \(2, 2\)"):
            _convert_z_image_gguf_to_diffusers({"x_pad_token": torch.zeros(2, 2)})

    def test_qkv_split_preserves_values(self):
        # A [6, 2] fused qkv splits into three [2, 2] chunks in order q, k, v.
        qkv = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        out = _convert_z_image_gguf_to_diffusers({"blk.attention.qkv.weight": qkv})
        assert torch.allclose(out["blk.attention.to_q.weight"], qkv[0:2])
        assert torch.allclose(out["blk.attention.to_k.weight"], qkv[2:4])
        assert torch.allclose(out["blk.attention.to_v.weight"], qkv[4:6])


MARKER = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}


class TestQkvQuantizationSideChannel:
    """A scaled-fp8 checkpoint puts a `scale_weight` next to the fused `qkv.weight`.

    Left on `...attention.qkv`, the recovered scale is keyed on a module path the diffusers model
    does not have, so `attach_fp8_scales` finds nothing and the three split weights stay quantized
    but *unscaled* — off by 1/weight_scale, with no error anywhere.
    """

    def test_per_tensor_scale_reaches_all_three_projections(self):
        out = _convert_z_image_gguf_to_diffusers(
            {
                "blk.attention.qkv.weight": torch.arange(12, dtype=torch.float32).reshape(6, 2),
                "blk.attention.qkv.scale_weight": torch.tensor(0.25),
            }
        )
        assert not any(".attention.qkv." in k for k in out)
        for name in ("to_q", "to_k", "to_v"):
            assert torch.equal(out[f"blk.attention.{name}.scale_weight"], torch.tensor(0.25))

    def test_per_channel_scale_is_split_like_the_weight(self):
        out = _convert_z_image_gguf_to_diffusers(
            {
                "blk.attention.qkv.weight": torch.arange(12, dtype=torch.float32).reshape(6, 2),
                "blk.attention.qkv.weight_scale": torch.arange(6, dtype=torch.float32),
            }
        )
        assert torch.equal(out["blk.attention.to_q.weight_scale"], torch.tensor([0.0, 1.0]))
        assert torch.equal(out["blk.attention.to_k.weight_scale"], torch.tensor([2.0, 3.0]))
        assert torch.equal(out["blk.attention.to_v.weight_scale"], torch.tensor([4.0, 5.0]))

    def test_marker_blob_is_copied_not_split(self):
        # `.comfy_quant` is a 1-D JSON byte string describing the layer, not a per-channel vector.
        blob = comfy_quant_marker({"format": "float8_e4m3fn"})
        out = _convert_z_image_gguf_to_diffusers(
            {
                "blk.attention.qkv.weight": torch.arange(12, dtype=torch.float32).reshape(6, 2),
                "blk.attention.qkv.comfy_quant": blob,
            }
        )
        for name in ("to_q", "to_k", "to_v"):
            assert torch.equal(out[f"blk.attention.{name}.comfy_quant"], blob)

    def test_unknown_suffix_is_left_alone(self):
        out = _convert_z_image_gguf_to_diffusers(
            {
                "blk.attention.qkv.weight": torch.arange(12, dtype=torch.float32).reshape(6, 2),
                "blk.attention.qkv.something_else": torch.tensor(1.0),
            }
        )
        assert "blk.attention.qkv.something_else" in out

    def test_undivisible_scale_is_rejected_rather_than_mis_split(self):
        with pytest.raises(ValueError, match="Cannot split fused QKV quantization data"):
            _convert_z_image_gguf_to_diffusers(
                {
                    "blk.attention.qkv.weight": torch.arange(12, dtype=torch.float32).reshape(6, 2),
                    "blk.attention.qkv.weight_scale": torch.arange(4, dtype=torch.float32),
                }
            )

    def test_the_markers_are_readable_after_the_conversion(self) -> None:
        """Why Z-Image reads them after converting rather than before: unlike Krea-2's converter,
        this one carries `.comfy_quant` onto the final module names, so no re-keying is needed."""
        prefix = "layers.0.attention"
        sd = {
            f"{prefix}.qkv.weight": torch.zeros(3 * 4, 256, dtype=torch.int8),
            f"{prefix}.qkv.weight_scale": torch.ones(3 * 4, 1),
            f"{prefix}.qkv.comfy_quant": comfy_quant_marker(MARKER),
            "x_embedder.weight": torch.zeros(2, 2),
        }
        markers = extract_int8_convrot_markers(_convert_z_image_gguf_to_diffusers(sd))
        assert set(markers) == {f"{prefix}.to_q", f"{prefix}.to_k", f"{prefix}.to_v"}
        assert all(m == MARKER for m in markers.values())


class TestMetadataPathRemap:
    """`_quantization_metadata` names layers in the checkpoint's scheme; the scales are recovered
    after the rename, so the per-layer hints have to follow the same route."""

    def test_renamed_layers_map_one_to_one(self):
        mapping = _remap_z_image_layer_paths(["x_embedder", "final_layer.linear", "layers.0.attention.out"])
        assert mapping["x_embedder"] == ["all_x_embedder.2-1"]
        assert mapping["final_layer.linear"] == ["all_final_layer.2-1.linear"]
        assert mapping["layers.0.attention.out"] == ["layers.0.attention.to_out.0"]

    def test_fused_qkv_maps_to_all_three_projections(self):
        mapping = _remap_z_image_layer_paths(["layers.0.attention.qkv"])
        assert mapping["layers.0.attention.qkv"] == [
            "layers.0.attention.to_q",
            "layers.0.attention.to_k",
            "layers.0.attention.to_v",
        ]
