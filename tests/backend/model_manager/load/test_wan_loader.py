"""Tests for Wan loader helpers (native -> diffusers key conversion)."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import gguf
import pytest
import torch

from invokeai.backend.model_manager.load.model_loaders.wan import (
    WanGGUFCheckpointModel,
    _convert_wan_native_to_diffusers,
    _unwrap_unquantized_to_compute_dtype,
)
from invokeai.backend.model_manager.taxonomy import WanVariantType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor


def test_converts_text_and_time_embedders():
    sd = {
        "text_embedding.0.weight": "a",
        "text_embedding.0.bias": "b",
        "text_embedding.2.weight": "c",
        "time_embedding.0.weight": "d",
        "time_embedding.2.weight": "e",
        "time_projection.1.weight": "f",
    }
    out = _convert_wan_native_to_diffusers(sd)
    assert "condition_embedder.text_embedder.linear_1.weight" in out
    assert "condition_embedder.text_embedder.linear_1.bias" in out
    assert "condition_embedder.text_embedder.linear_2.weight" in out
    assert "condition_embedder.time_embedder.linear_1.weight" in out
    assert "condition_embedder.time_embedder.linear_2.weight" in out
    assert "condition_embedder.time_proj.weight" in out


def test_converts_attention_blocks():
    sd = {
        "blocks.0.self_attn.q.weight": 1,
        "blocks.0.self_attn.k.weight": 2,
        "blocks.0.self_attn.v.weight": 3,
        "blocks.0.self_attn.o.weight": 4,
        "blocks.0.self_attn.norm_q.weight": 5,
        "blocks.0.self_attn.norm_k.weight": 6,
        "blocks.0.cross_attn.q.weight": 7,
        "blocks.0.cross_attn.k.weight": 8,
        "blocks.0.cross_attn.v.weight": 9,
        "blocks.0.cross_attn.o.weight": 10,
    }
    out = _convert_wan_native_to_diffusers(sd)
    assert "blocks.0.attn1.to_q.weight" in out
    assert "blocks.0.attn1.to_k.weight" in out
    assert "blocks.0.attn1.to_v.weight" in out
    assert "blocks.0.attn1.to_out.0.weight" in out
    assert "blocks.0.attn1.norm_q.weight" in out
    assert "blocks.0.attn1.norm_k.weight" in out
    assert "blocks.0.attn2.to_q.weight" in out
    assert "blocks.0.attn2.to_out.0.weight" in out


def test_converts_ffn_and_modulation():
    sd = {
        "blocks.0.ffn.0.weight": 1,
        "blocks.0.ffn.0.bias": 2,
        "blocks.0.ffn.2.weight": 3,
        "blocks.0.modulation": 4,
    }
    out = _convert_wan_native_to_diffusers(sd)
    assert "blocks.0.ffn.net.0.proj.weight" in out
    assert "blocks.0.ffn.net.0.proj.bias" in out
    assert "blocks.0.ffn.net.2.weight" in out
    assert "blocks.0.scale_shift_table" in out


def test_swaps_norm2_and_norm3():
    """Native norm3 has params (cross-attn norm in diffusers norm2 slot)
    while native norm2 is the elementwise-affine-False norm. The swap
    via placeholder must not collide."""
    sd = {
        "blocks.0.norm2.weight": "native_norm2",
        "blocks.0.norm3.weight": "native_norm3",
    }
    out = _convert_wan_native_to_diffusers(sd)
    assert out["blocks.0.norm3.weight"] == "native_norm2"
    assert out["blocks.0.norm2.weight"] == "native_norm3"


def test_converts_head_keys():
    sd = {
        "head.head.weight": 1,
        "head.head.bias": 2,
        "head.modulation": 3,
    }
    out = _convert_wan_native_to_diffusers(sd)
    assert "proj_out.weight" in out
    assert "proj_out.bias" in out
    assert "scale_shift_table" in out


def test_diffusers_keys_pass_through_unchanged():
    """If a state dict is already in diffusers form, the substring rules
    must be no-ops — none of the native fingerprints are present."""
    sd = {
        "patch_embedding.weight": 1,
        "condition_embedder.text_embedder.linear_1.weight": 2,
        "blocks.0.attn1.to_q.weight": 3,
        "blocks.0.ffn.net.0.proj.weight": 4,
        "scale_shift_table": 5,
        "proj_out.weight": 6,
    }
    out = _convert_wan_native_to_diffusers(sd)
    assert set(out.keys()) == set(sd.keys())
    assert all(out[k] == sd[k] for k in sd)


def test_does_not_mutate_input():
    sd = {"text_embedding.0.weight": 1}
    snapshot = dict(sd)
    _convert_wan_native_to_diffusers(sd)
    assert sd == snapshot


def test_non_string_keys_pass_through():
    sd = {0: "ignored", "text_embedding.0.weight": "renamed"}
    out = _convert_wan_native_to_diffusers(sd)
    assert out[0] == "ignored"
    assert "condition_embedder.text_embedder.linear_1.weight" in out


def _ggml(data: torch.Tensor, qtype: gguf.GGMLQuantizationType, compute_dtype: torch.dtype) -> GGMLTensor:
    return GGMLTensor(
        data=data,
        ggml_quantization_type=qtype,
        tensor_shape=data.shape,
        compute_dtype=compute_dtype,
    )


class TestUnwrapUnquantized:
    """The QuantStack GGUFs store ``patch_embedding.bias`` as F16 while latents
    flow through the model as bf16. Conv3d isn't in GGMLTensor's dispatch table,
    so without unwrapping the F16 wrapper goes into conv3d as-is and crashes
    with ``Input type (c10::BFloat16) and bias type (c10::Half) should be the same``.
    These tests guard the unwrap step that prevents that."""

    def test_f16_compatible_qtype_is_unwrapped_and_cast(self):
        # F16 storage that should become bf16 plain tensor.
        f16_data = torch.zeros((4,), dtype=torch.float16)
        sd = {"bias": _ggml(f16_data, gguf.GGMLQuantizationType.F16, torch.bfloat16)}
        out = _unwrap_unquantized_to_compute_dtype(sd)

        result = out["bias"]
        assert not isinstance(result, GGMLTensor)
        assert result.dtype == torch.bfloat16

    def test_f32_compatible_qtype_is_unwrapped_and_cast(self):
        # patch_embedding.weight in QuantStack is F32 — same path.
        f32_data = torch.zeros((4,), dtype=torch.float32)
        sd = {"weight": _ggml(f32_data, gguf.GGMLQuantizationType.F32, torch.bfloat16)}
        out = _unwrap_unquantized_to_compute_dtype(sd)

        result = out["weight"]
        assert not isinstance(result, GGMLTensor)
        assert result.dtype == torch.bfloat16

    def test_quantized_tensor_stays_wrapped(self):
        # Q4_K and friends must remain GGMLTensor so on-demand dequant works
        # via the linear/addmm dispatch path. The byte storage shape is fake
        # but irrelevant for this test.
        q4_data = torch.zeros((1,), dtype=torch.uint8)
        sd = {"linear.weight": _ggml(q4_data, gguf.GGMLQuantizationType.Q4_K, torch.bfloat16)}
        out = _unwrap_unquantized_to_compute_dtype(sd)

        assert isinstance(out["linear.weight"], GGMLTensor)
        assert out["linear.weight"]._ggml_quantization_type == gguf.GGMLQuantizationType.Q4_K

    def test_plain_torch_tensor_passes_through(self):
        plain = torch.zeros((4,), dtype=torch.bfloat16)
        sd = {"plain": plain}
        out = _unwrap_unquantized_to_compute_dtype(sd)
        assert out["plain"] is plain


def _run_gguf_loader(extra_keys: list[str], native_layout: bool = False) -> dict:
    """Drive WanGGUFCheckpointModel over a state dict carrying `extra_keys`.

    The extras go into the *state dict*, not into a mocked `unexpected_keys`, so the
    loader's own classification runs. Returns the dict actually handed to
    `load_state_dict`, which is what proves a key was dropped rather than merely
    tolerated.

    With `native_layout`, the base keys use the upstream ComfyUI/QuantStack naming so
    `_convert_wan_native_to_diffusers` runs first — a different path to the same gate,
    and the one where an unmapped key is possible at all.
    """
    if native_layout:
        state_dict = {
            "patch_embedding.weight": torch.zeros(128, 16, 1, 2, 2),
            "text_embedding.0.weight": torch.zeros(128, 4096),
            "blocks.0.ffn.0.weight": torch.zeros(256, 128),
            "head.head.weight": torch.zeros(64, 128),
        }
    else:
        state_dict = {
            "patch_embedding.weight": torch.zeros(128, 16, 1, 2, 2),
            "blocks.0.ffn.net.0.proj.weight": torch.zeros(256, 128),
            "proj_out.weight": torch.zeros(64, 128),
        }
    for key in extra_keys:
        state_dict[key] = torch.zeros(4, 4)
    model = MagicMock()
    # Report as unexpected whatever the loader still hands over that isn't a real param.
    # Named post-conversion, since that is what reaches `load_state_dict`.
    real_params = {
        "patch_embedding.weight",
        "condition_embedder.text_embedder.linear_1.weight",
        "blocks.0.ffn.net.0.proj.weight",
        "proj_out.weight",
    }
    model.load_state_dict.side_effect = lambda sd, **_: SimpleNamespace(
        missing_keys=[], unexpected_keys=[k for k in sd if k not in real_params]
    )
    config = SimpleNamespace(path="/models/wan.gguf", variant=WanVariantType.T2V_A14B)
    loader = object.__new__(WanGGUFCheckpointModel)

    with (
        patch("invokeai.backend.model_manager.load.model_loaders.wan.gguf_sd_loader", return_value=state_dict),
        patch(
            "invokeai.backend.model_manager.load.model_loaders.wan._unwrap_unquantized_to_compute_dtype",
            side_effect=lambda value: value,
        ),
        patch("invokeai.backend.model_manager.load.model_loaders.wan.TorchDevice.choose_torch_device"),
        patch(
            "invokeai.backend.model_manager.load.model_loaders.wan.TorchDevice.choose_bfloat16_safe_dtype",
            return_value=torch.bfloat16,
        ),
        patch("accelerate.init_empty_weights", return_value=nullcontext()),
        patch("diffusers.WanTransformer3DModel", return_value=model),
    ):
        loader._load_from_singlefile(config)

    return model.load_state_dict.call_args.args[0]


def test_gguf_loader_drops_all_in_one_bundled_components_before_loading() -> None:
    """The "all-in-one" GGUF convention bundles the VAE and text encoder alongside the
    transformer — befox/WAN2.2-14B-Rapid-AllInOne-GGUF ships ~110 such files, converted
    from Phr00t/WAN2.2-14B-Rapid-AllInOne.

    They must load (refusing them regressed a path that worked before the unexpected-key
    backstop existed) *and* the bundled weights must be gone before the compute-dtype
    cast and the RAM-cache reservation, not merely tolerated at load_state_dict — the
    bundled UMT5-XXL alone is several GB that would otherwise be upcast and reserved.
    """
    bundled = ["vae.decoder.conv_in.weight", "text_encoders.umt5xxl.shared.weight", "model_ema.patch_embedding.weight"]

    handed_over = _run_gguf_loader(bundled)

    assert [key for key in bundled if key in handed_over] == []


def test_gguf_loader_still_refuses_an_unknown_conditioning_branch() -> None:
    """The allowlist must not switch the backstop off — an unenumerated Wan derivative
    still has to fail loudly rather than generate with its conditioning branch absent."""
    with pytest.raises(RuntimeError, match="audio_injector"):
        _run_gguf_loader(["vae.decoder.conv_in.weight", "audio_injector.0.proj.weight"])


def test_gguf_loader_refuses_a_native_layout_key_the_rename_table_does_not_map() -> None:
    """Pins the intended outcome for the one case the unexpected-key backstop newly
    changes for GGUF: a native-layout key that survives `_convert_wan_native_to_diffusers`
    unrenamed.

    Before the backstop the GGUF loader checked `missing_keys` only, so such a key was
    silently discarded by `load_state_dict(strict=False)`. Refusing is deliberate, and
    the blast radius is narrower than it looks: an unmapped key that *should* have become
    a real parameter also leaves that parameter unfilled, which the pre-existing
    `missing_keys` check already caught. What is genuinely new is the case below —
    a whole extra branch, here VACE, whose conversion we deliberately do not ship
    (see `_WAN_NATIVE_TO_DIFFUSERS_RENAMES`: "T2V subset; we don't ship VACE / motion /
    face-adapter conversion"). Generating with it quietly absent is worse than refusing.
    """
    with pytest.raises(RuntimeError, match="vace_blocks"):
        _run_gguf_loader(["vace_blocks.0.after_proj.weight"], native_layout=True)


def test_gguf_loader_accepts_a_native_layout_all_in_one_bundle() -> None:
    """The benign-extras drop has to survive the native-layout rewrite too. The rename
    table is blind substring replacement over every key, so it runs across the bundled
    VAE/encoder names as well — this pins that they are still recognised and dropped."""
    bundled = ["vae.decoder.conv_in.weight", "text_encoders.umt5xxl.shared.weight"]

    handed_over = _run_gguf_loader(bundled, native_layout=True)

    assert [key for key in bundled if key in handed_over] == []


def test_gguf_loader_rejects_missing_model_parameter() -> None:
    state_dict = {
        "patch_embedding.weight": torch.zeros(128, 16, 1, 2, 2),
        "blocks.0.ffn.net.0.proj.weight": torch.zeros(256, 128),
        "proj_out.weight": torch.zeros(64, 128),
    }
    model = MagicMock()
    model.load_state_dict.return_value = SimpleNamespace(
        missing_keys=["blocks.0.attn1.to_q.weight"],
        unexpected_keys=[],
    )
    config = SimpleNamespace(path="/models/wan.gguf", variant=WanVariantType.T2V_A14B)
    loader = object.__new__(WanGGUFCheckpointModel)

    with (
        patch("invokeai.backend.model_manager.load.model_loaders.wan.gguf_sd_loader", return_value=state_dict),
        patch(
            "invokeai.backend.model_manager.load.model_loaders.wan._unwrap_unquantized_to_compute_dtype",
            side_effect=lambda value: value,
        ),
        patch("invokeai.backend.model_manager.load.model_loaders.wan.TorchDevice.choose_torch_device"),
        patch(
            "invokeai.backend.model_manager.load.model_loaders.wan.TorchDevice.choose_bfloat16_safe_dtype",
            return_value=torch.bfloat16,
        ),
        patch("accelerate.init_empty_weights", return_value=nullcontext()),
        patch("diffusers.WanTransformer3DModel", return_value=model),
        pytest.raises(RuntimeError, match="missing"),
    ):
        loader._load_from_singlefile(config)
