"""Tests for the Anima ControlNet-LLLite adapter — construction from a saved
state dict, exact-passthrough guarantees, forward-swap binding/restore,
multi-adapter composition, and the conditioning image preprocessing helpers."""

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from invokeai.app.invocations.anima_denoise import AnimaDenoiseInvocation
from invokeai.app.invocations.anima_lllite import AnimaLLLiteField
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.backend.anima.control_net_lllite import (
    AnimaControlNetLLLite,
    build_inpaint_cond_image,
    prepare_cond_image,
    prepare_mask,
    target_cond_hw,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType

# Opt-in test against the real adapter weights (anima-lllite-inpainting-v2.safetensors).
_REAL_WEIGHTS_ENV_VAR = "ANIMA_LLLITE_WEIGHTS_PATH"
REAL_WEIGHTS_PATH = Path(os.environ[_REAL_WEIGHTS_ENV_VAR]) if _REAL_WEIGHTS_ENV_VAR in os.environ else None

COND_DIM = 16
COND_EMB_DIM = 8
MLP_DIM = 8
IN_DIM = 16
N_BLOCKS = 2
KINDS = ("self_attn_q_proj", "self_attn_k_proj", "self_attn_v_proj", "mlp_layer1")


def make_synthetic_state_dict(seed: int = 0, mlp_dim: int = MLP_DIM) -> dict[str, torch.Tensor]:
    """Saved-format (v2 named-key) state dict for a tiny 2-block adapter with
    4-channel (inpaint) conditioning and one trunk resblock."""
    g = torch.Generator().manual_seed(seed)

    def t(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    ch_half = COND_DIM // 2
    sd = {
        "lllite_conditioning1.conv1.weight": t(ch_half, 4, 4, 4),
        "lllite_conditioning1.conv1.bias": t(ch_half),
        "lllite_conditioning1.norm1.weight": t(ch_half),
        "lllite_conditioning1.norm1.bias": t(ch_half),
        "lllite_conditioning1.conv2.weight": t(ch_half, ch_half, 3, 3),
        "lllite_conditioning1.conv2.bias": t(ch_half),
        "lllite_conditioning1.norm2.weight": t(ch_half),
        "lllite_conditioning1.norm2.bias": t(ch_half),
        "lllite_conditioning1.conv3.weight": t(COND_DIM, ch_half, 4, 4),
        "lllite_conditioning1.conv3.bias": t(COND_DIM),
        "lllite_conditioning1.norm3.weight": t(COND_DIM),
        "lllite_conditioning1.norm3.bias": t(COND_DIM),
        "lllite_conditioning1.resblocks.0.norm1.weight": t(COND_DIM),
        "lllite_conditioning1.resblocks.0.norm1.bias": t(COND_DIM),
        "lllite_conditioning1.resblocks.0.conv1.weight": t(COND_DIM, COND_DIM, 3, 3),
        "lllite_conditioning1.resblocks.0.conv1.bias": t(COND_DIM),
        "lllite_conditioning1.resblocks.0.norm2.weight": t(COND_DIM),
        "lllite_conditioning1.resblocks.0.norm2.bias": t(COND_DIM),
        "lllite_conditioning1.resblocks.0.conv2.weight": t(COND_DIM, COND_DIM, 3, 3),
        "lllite_conditioning1.resblocks.0.conv2.bias": t(COND_DIM),
        "lllite_conditioning1.proj.weight": t(COND_EMB_DIM, COND_DIM, 1, 1),
        "lllite_conditioning1.proj.bias": t(COND_EMB_DIM),
        "lllite_conditioning1.out_norm.weight": t(COND_EMB_DIM),
        "lllite_conditioning1.out_norm.bias": t(COND_EMB_DIM),
    }
    for i in range(N_BLOCKS):
        for kind in KINDS:
            p = f"lllite_dit_blocks_{i}_{kind}"
            sd[f"{p}.down.weight"] = t(mlp_dim, IN_DIM)
            sd[f"{p}.down.bias"] = t(mlp_dim)
            sd[f"{p}.mid.weight"] = t(mlp_dim, mlp_dim + COND_EMB_DIM)
            sd[f"{p}.mid.bias"] = t(mlp_dim)
            sd[f"{p}.cond_to_film.weight"] = t(2 * mlp_dim, COND_EMB_DIM)
            sd[f"{p}.cond_to_film.bias"] = t(2 * mlp_dim)
            sd[f"{p}.up.weight"] = t(IN_DIM, mlp_dim)
            sd[f"{p}.up.bias"] = t(IN_DIM)
            sd[f"{p}.depth_embed"] = t(COND_EMB_DIM)
    return sd


class FakeAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)


class FakeMlp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim * 2, bias=False)


class FakeBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = FakeAttention(dim)
        self.cross_attn = FakeAttention(dim)
        self.mlp = FakeMlp(dim)


class FakeTransformer(nn.Module):
    def __init__(self, dim: int, n_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock(dim) for _ in range(n_blocks)])


def make_model_and_transformer(dim: int = IN_DIM) -> tuple[AnimaControlNetLLLite, FakeTransformer]:
    model = AnimaControlNetLLLite.from_state_dict(make_synthetic_state_dict(), None)
    torch.manual_seed(123)
    transformer = FakeTransformer(dim, N_BLOCKS)
    return model, transformer


def matching_cond_image() -> torch.Tensor:
    """4ch cond image sized for latent 4x4 -> trunk tokens S = 2*2 = 4."""
    return torch.randn(1, 4, 32, 32, generator=torch.Generator().manual_seed(7)).clamp(-1, 1)


def plain_linear(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, linear.weight, linear.bias)


# ----------------------------------------------------------------------------
# Preprocessing helpers
# ----------------------------------------------------------------------------


def test_target_cond_hw_even_latent() -> None:
    assert target_cond_hw(128, 128) == (1024, 1024)
    assert target_cond_hw(64, 96) == (512, 768)


def test_target_cond_hw_odd_latent_pads_to_patch_multiple() -> None:
    assert target_cond_hw(129, 129) == (1040, 1040)
    assert target_cond_hw(129, 64) == (1040, 512)
    assert target_cond_hw(5, 5, patch_spatial=4) == (64, 64)


def test_prepare_cond_image_no_resize_is_exact_rescale() -> None:
    rgb = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(0))
    out = prepare_cond_image(rgb, latent_h=2, latent_w=2)
    assert torch.equal(out, rgb * 2.0 - 1.0)


def test_prepare_cond_image_resizes_takes_first_frame_and_stays_in_range() -> None:
    rgb = torch.rand(2, 3, 100, 100, generator=torch.Generator().manual_seed(1))
    out = prepare_cond_image(rgb, latent_h=9, latent_w=8)
    assert out.shape == (1, 3, 80, 64)
    assert out.min().item() >= -1.0
    assert out.max().item() <= 1.0


def test_prepare_cond_image_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="Unexpected cond image shape"):
        prepare_cond_image(torch.rand(1, 4, 16, 16), latent_h=2, latent_w=2)


def test_prepare_mask_binarizes_at_half() -> None:
    mask = torch.full((1, 1, 16, 16), 0.49)
    mask[:, :, 8:, :] = 0.5
    out = prepare_mask(mask, latent_h=2, latent_w=2)
    assert torch.equal(out[:, :, :8, :], torch.zeros(1, 1, 8, 16))
    assert torch.equal(out[:, :, 8:, :], torch.ones(1, 1, 8, 16))


def test_prepare_mask_accepts_3d_and_resizes_nearest() -> None:
    mask = torch.zeros(1, 8, 8)
    mask[:, :4, :] = 1.0
    out = prepare_mask(mask, latent_h=4, latent_w=4)
    assert out.shape == (1, 1, 32, 32)
    assert set(out.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(out[:, :, :16, :], torch.ones(1, 1, 16, 32))
    assert torch.equal(out[:, :, 16:, :], torch.zeros(1, 1, 16, 32))


def test_prepare_mask_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="Unexpected mask shape"):
        prepare_mask(torch.rand(1, 3, 16, 16), latent_h=2, latent_w=2)


def test_build_inpaint_cond_image_polarity_and_range() -> None:
    rgb = torch.full((1, 3, 4, 4), 0.5)
    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, :2, :] = 1.0  # white = inpaint area

    out = build_inpaint_cond_image(rgb, mask, masked_input=True)
    assert out.shape == (1, 4, 4, 4)
    # RGB zeroed under the inpaint area, untouched elsewhere.
    assert torch.equal(out[:, :3, :2, :], torch.zeros(1, 3, 2, 4))
    assert torch.equal(out[:, :3, 2:, :], rgb[:, :, 2:, :])
    # Mask channel rescaled to [-1, 1]: +1 = inpaint, -1 = keep.
    assert torch.equal(out[:, 3:, :2, :], torch.ones(1, 1, 2, 4))
    assert torch.equal(out[:, 3:, 2:, :], -torch.ones(1, 1, 2, 4))


def test_build_inpaint_cond_image_without_masked_input_keeps_rgb() -> None:
    rgb = torch.full((1, 3, 4, 4), 0.5)
    mask = torch.ones(1, 1, 4, 4)
    out = build_inpaint_cond_image(rgb, mask, masked_input=False)
    assert torch.equal(out[:, :3], rgb)
    assert torch.equal(out[:, 3:], torch.ones(1, 1, 4, 4))


# ----------------------------------------------------------------------------
# Construction from state dict
# ----------------------------------------------------------------------------


def test_from_state_dict_synthetic_shape_fallbacks() -> None:
    sd = make_synthetic_state_dict()
    model = AnimaControlNetLLLite.from_state_dict(sd, None)

    assert len(model.lllite_modules) == N_BLOCKS * len(KINDS)
    assert model.cond_in_channels == 4
    assert model.cond_emb_dim == COND_EMB_DIM
    assert model.mlp_dim == MLP_DIM
    assert model.cond_dim == COND_DIM
    assert model.cond_resblocks == 1
    assert model.use_aspp is False
    assert model.inpaint_masked_input is False  # metadata-only, defaults False

    # Weights actually landed where they belong.
    assert torch.equal(model.conditioning1.conv1.weight, sd["lllite_conditioning1.conv1.weight"])
    by_name = {m.lllite_name: m for m in model.lllite_modules}
    m0 = by_name["lllite_dit_blocks_0_self_attn_q_proj"]
    assert torch.equal(m0.depth_embed, sd["lllite_dit_blocks_0_self_attn_q_proj.depth_embed"])
    assert torch.equal(m0.up.weight, sd["lllite_dit_blocks_0_self_attn_q_proj.up.weight"])
    m1 = by_name["lllite_dit_blocks_1_mlp_layer1"]
    assert torch.equal(m1.down.weight, sd["lllite_dit_blocks_1_mlp_layer1.down.weight"])

    assert not any(p.requires_grad for p in model.parameters())
    assert not model.training


def test_from_state_dict_metadata_wins() -> None:
    metadata = {
        "lllite.cond_emb_dim": str(COND_EMB_DIM),
        "lllite.mlp_dim": str(MLP_DIM),
        "lllite.cond_dim": str(COND_DIM),
        "lllite.cond_resblocks": "1",
        "lllite.use_aspp": "false",
        "lllite.cond_in_channels": "4",
        "lllite.inpaint_masked_input": "true",
    }
    model = AnimaControlNetLLLite.from_state_dict(make_synthetic_state_dict(), metadata)
    assert model.inpaint_masked_input is True
    assert model.cond_in_channels == 4


def test_from_state_dict_rejects_legacy_format() -> None:
    sd = make_synthetic_state_dict()
    sd["lllite_modules.0.down.weight"] = torch.zeros(MLP_DIM, IN_DIM)
    with pytest.raises(ValueError, match="legacy"):
        AnimaControlNetLLLite.from_state_dict(sd, None)


def test_from_state_dict_strict_on_unknown_keys() -> None:
    sd = make_synthetic_state_dict()
    sd["some_unrelated_key"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="some_unrelated_key"):
        AnimaControlNetLLLite.from_state_dict(sd, None)

    sd = make_synthetic_state_dict()
    sd["lllite_dit_blocks_0_self_attn_q_proj.extra.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="extra"):
        AnimaControlNetLLLite.from_state_dict(sd, None)


def test_from_state_dict_strict_on_missing_keys() -> None:
    sd = make_synthetic_state_dict()
    del sd["lllite_dit_blocks_0_self_attn_q_proj.depth_embed"]
    with pytest.raises(RuntimeError, match="depth_embed"):
        AnimaControlNetLLLite.from_state_dict(sd, None)


def test_from_state_dict_requires_modules() -> None:
    sd = {k: v for k, v in make_synthetic_state_dict().items() if k.startswith("lllite_conditioning1.")}
    with pytest.raises(ValueError, match="no LLLite modules"):
        AnimaControlNetLLLite.from_state_dict(sd, None)


def test_from_state_dict_missing_down_weight_raises_value_error() -> None:
    sd = make_synthetic_state_dict()
    del sd["lllite_dit_blocks_0_self_attn_q_proj.down.weight"]
    with pytest.raises(ValueError, match="missing key 'lllite_dit_blocks_0_self_attn_q_proj.down.weight'"):
        AnimaControlNetLLLite.from_state_dict(sd, None)


# ----------------------------------------------------------------------------
# Binding / restore
# ----------------------------------------------------------------------------


def test_apply_to_swaps_forward_and_restore_is_bit_exact() -> None:
    model, transformer = make_model_and_transformer()
    x = torch.randn(1, 4, IN_DIM, generator=torch.Generator().manual_seed(2))
    q_proj = transformer.blocks[0].self_attn.q_proj
    expected = plain_linear(q_proj, x)

    model.apply_to(transformer)
    model.set_multiplier(1.0)
    model.set_cond_image(matching_cond_image())
    assert not torch.equal(q_proj(x), expected)

    model.restore()
    assert torch.equal(q_proj(x), expected)


def test_apply_to_is_idempotent() -> None:
    model, transformer = make_model_and_transformer()
    x = torch.randn(1, 4, IN_DIM, generator=torch.Generator().manual_seed(3))
    v_proj = transformer.blocks[1].self_attn.v_proj
    expected = plain_linear(v_proj, x)

    model.apply_to(transformer)
    model.apply_to(transformer)  # must not double-wrap
    model.restore()
    assert torch.equal(v_proj(x), expected)


def test_restore_without_apply_is_safe() -> None:
    model, _ = make_model_and_transformer()
    model.restore()
    model.restore()


def test_restore_does_not_pin_instance_level_forward() -> None:
    # An instance-level `forward` left behind after restore() would silently
    # bypass class-level forward swaps that share the module __dict__ (see
    # wrap_custom_layer notes in model_manager/load/load_default.py).
    model, transformer = make_model_and_transformer()
    q_proj = transformer.blocks[0].self_attn.q_proj
    assert "forward" not in q_proj.__dict__

    model.apply_to(transformer)
    assert "forward" in q_proj.__dict__
    model.restore()
    assert "forward" not in q_proj.__dict__


def test_restore_preserves_preexisting_instance_level_forward() -> None:
    model, transformer = make_model_and_transformer()
    q_proj = transformer.blocks[0].self_attn.q_proj
    sentinel = q_proj.forward
    q_proj.forward = sentinel  # pre-existing instance-level forward

    model.apply_to(transformer)
    model.restore()
    assert q_proj.__dict__.get("forward") is sentinel


def test_apply_to_rejects_in_features_mismatch() -> None:
    model, _ = make_model_and_transformer()
    transformer = FakeTransformer(IN_DIM * 2, N_BLOCKS)
    with pytest.raises(ValueError, match="in_features"):
        model.apply_to(transformer)


def test_apply_to_rejects_missing_block() -> None:
    model, _ = make_model_and_transformer()
    transformer = FakeTransformer(IN_DIM, 1)
    with pytest.raises(ValueError, match="block"):
        model.apply_to(transformer)


# ----------------------------------------------------------------------------
# Forward: passthrough guarantees and active path
# ----------------------------------------------------------------------------


def test_passthrough_multiplier_zero_is_bit_exact() -> None:
    model, transformer = make_model_and_transformer()
    model.apply_to(transformer)
    model.set_cond_image(matching_cond_image())
    model.set_multiplier(0.0)
    x = torch.randn(1, 4, IN_DIM, generator=torch.Generator().manual_seed(4))
    for block in transformer.blocks:
        for linear in (block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj):
            assert torch.equal(linear(x), plain_linear(linear, x))


def test_passthrough_no_cond_is_bit_exact() -> None:
    model, transformer = make_model_and_transformer()
    model.apply_to(transformer)
    model.set_multiplier(1.0)
    q_proj = transformer.blocks[0].self_attn.q_proj
    x = torch.randn(1, 4, IN_DIM, generator=torch.Generator().manual_seed(5))
    assert torch.equal(q_proj(x), plain_linear(q_proj, x))

    model.set_cond_image(matching_cond_image())
    assert not torch.equal(q_proj(x), plain_linear(q_proj, x))
    model.set_cond_image(None)  # clearing re-enables passthrough
    assert torch.equal(q_proj(x), plain_linear(q_proj, x))


def test_passthrough_on_seq_len_mismatch_is_bit_exact() -> None:
    model, transformer = make_model_and_transformer()
    model.apply_to(transformer)
    model.set_multiplier(1.0)
    # Cond image for latent 6x6 -> S = 3*3 = 9, but x has S = 4.
    model.set_cond_image(torch.randn(1, 4, 48, 48, generator=torch.Generator().manual_seed(6)))
    q_proj = transformer.blocks[0].self_attn.q_proj
    x = torch.randn(1, 4, IN_DIM, generator=torch.Generator().manual_seed(7))
    assert torch.equal(q_proj(x), plain_linear(q_proj, x))


def test_passthrough_on_non_divisible_batch_is_bit_exact() -> None:
    model, transformer = make_model_and_transformer()
    model.apply_to(transformer)
    model.set_multiplier(1.0)
    model.set_cond_image(matching_cond_image().repeat(2, 1, 1, 1))  # cond batch 2
    q_proj = transformer.blocks[0].self_attn.q_proj
    x = torch.randn(3, 4, IN_DIM, generator=torch.Generator().manual_seed(8))  # 3 % 2 != 0
    assert torch.equal(q_proj(x), plain_linear(q_proj, x))


def test_cfg_batch_broadcast_matches_single_sample() -> None:
    model, transformer = make_model_and_transformer()
    model.apply_to(transformer)
    model.set_multiplier(1.0)
    model.set_cond_image(matching_cond_image())  # cond batch 1
    q_proj = transformer.blocks[0].self_attn.q_proj
    xa = torch.randn(1, 4, IN_DIM, generator=torch.Generator().manual_seed(9))
    xb = torch.randn(1, 4, IN_DIM, generator=torch.Generator().manual_seed(10))
    y_batched = q_proj(torch.cat([xa, xb], dim=0))
    assert not torch.equal(y_batched, plain_linear(q_proj, torch.cat([xa, xb], dim=0)))
    assert torch.allclose(y_batched[0:1], q_proj(xa), rtol=1e-5, atol=1e-6)
    assert torch.allclose(y_batched[1:2], q_proj(xb), rtol=1e-5, atol=1e-6)


def test_5d_mlp_input_matches_flattened_3d_path() -> None:
    model, transformer = make_model_and_transformer()
    model.apply_to(transformer)
    model.set_multiplier(1.0)
    model.set_cond_image(matching_cond_image())
    layer1 = transformer.blocks[0].mlp.layer1
    x5 = torch.randn(1, 1, 2, 2, IN_DIM, generator=torch.Generator().manual_seed(11))
    y5 = layer1(x5)
    assert y5.shape == (1, 1, 2, 2, IN_DIM * 2)
    y3 = layer1(x5.reshape(1, 4, IN_DIM))
    assert torch.equal(y5, y3.reshape(1, 1, 2, 2, -1))
    assert not torch.equal(y5, plain_linear(layer1, x5))


def test_5d_passthrough_keeps_shape() -> None:
    model, transformer = make_model_and_transformer()
    model.apply_to(transformer)
    model.set_multiplier(1.0)
    # Seq mismatch: trunk S = 9, x flattened S = 4 -> identity fallback.
    model.set_cond_image(torch.randn(1, 4, 48, 48, generator=torch.Generator().manual_seed(12)))
    layer1 = transformer.blocks[0].mlp.layer1
    x5 = torch.randn(1, 1, 2, 2, IN_DIM, generator=torch.Generator().manual_seed(13))
    assert torch.equal(layer1(x5), plain_linear(layer1, x5))


def test_forward_casts_mismatched_input_dtype() -> None:
    model, transformer = make_model_and_transformer()
    model.apply_to(transformer)
    model.set_multiplier(1.0)
    model.set_cond_image(matching_cond_image())
    transformer.to(torch.bfloat16)
    x = torch.randn(1, 4, IN_DIM, generator=torch.Generator().manual_seed(14)).to(torch.bfloat16)
    q_proj = transformer.blocks[0].self_attn.q_proj
    y = q_proj(x)
    assert y.dtype == torch.bfloat16
    assert not torch.equal(y, plain_linear(q_proj, x))


# ----------------------------------------------------------------------------
# Multi-adapter composition (denoise applies in list order, restores LIFO)
# ----------------------------------------------------------------------------


class _IdentityCapture(nn.Linear):
    """nn.Linear whose forward returns its input unchanged — binding a LLLite
    module to it reads out the perturbed input the module passes to the
    forward it wrapped."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def make_two_adapters() -> tuple[AnimaControlNetLLLite, AnimaControlNetLLLite, FakeTransformer]:
    """Two adapters with different mlp dims (8 and 16) plus one fake tree."""
    model_a = AnimaControlNetLLLite.from_state_dict(make_synthetic_state_dict(seed=0, mlp_dim=8), None)
    model_b = AnimaControlNetLLLite.from_state_dict(make_synthetic_state_dict(seed=1, mlp_dim=16), None)
    torch.manual_seed(123)
    transformer = FakeTransformer(IN_DIM, N_BLOCKS)
    return model_a, model_b, transformer


def cond_image_for(seed: int) -> torch.Tensor:
    return torch.randn(1, 4, 32, 32, generator=torch.Generator().manual_seed(seed)).clamp(-1, 1)


def test_two_adapters_chain_with_second_wrapping_first() -> None:
    model_a, model_b, transformer = make_two_adapters()
    assert model_a.mlp_dim != model_b.mlp_dim
    q_proj = transformer.blocks[0].self_attn.q_proj
    x = torch.randn(1, 4, IN_DIM, generator=torch.Generator().manual_seed(20))

    model_a.set_multiplier(1.0)
    model_a.set_cond_image(cond_image_for(30))
    model_b.set_multiplier(1.0)
    model_b.set_cond_image(cond_image_for(31))

    # Capture x + delta_B(x): the input B hands to the forward it wrapped.
    capture = _IdentityCapture(IN_DIM, IN_DIM, bias=False)
    module_b = next(m for m in model_b.lllite_modules if m.lllite_name == "lllite_dit_blocks_0_self_attn_q_proj")
    module_b.bind(capture)
    try:
        x_plus_delta_b = capture(x)
    finally:
        module_b.unbind()
    assert not torch.equal(x_plus_delta_b, x)

    model_a.apply_to(transformer)
    expected = q_proj(x_plus_delta_b)  # A's wrapped forward on B's perturbed input

    model_b.apply_to(transformer)  # B wraps A
    assert torch.equal(q_proj(x), expected)

    model_b.restore()
    model_a.restore()


def test_two_adapters_reverse_restore_returns_pristine_dispatch() -> None:
    model_a, model_b, transformer = make_two_adapters()
    x = torch.randn(1, 4, IN_DIM, generator=torch.Generator().manual_seed(21))

    targets = [
        linear
        for block in transformer.blocks
        for linear in (block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj, block.mlp.layer1)
    ]
    pre_outputs = [linear(x) for linear in targets]

    for model, seed in ((model_a, 32), (model_b, 33)):
        model.apply_to(transformer)
        model.set_multiplier(1.0)
        model.set_cond_image(cond_image_for(seed))
    assert all("forward" in linear.__dict__ for linear in targets)

    # LIFO: the last adapter applied is restored first.
    model_b.restore()
    model_a.restore()

    for linear, expected in zip(targets, pre_outputs, strict=True):
        assert "forward" not in linear.__dict__
        assert torch.equal(linear(x), expected)


def test_each_adapter_multiplier_gates_only_its_own_contribution() -> None:
    model_a, model_b, transformer = make_two_adapters()
    q_proj = transformer.blocks[0].self_attn.q_proj
    x = torch.randn(1, 4, IN_DIM, generator=torch.Generator().manual_seed(22))
    y_plain = plain_linear(q_proj, x)
    cond_a = cond_image_for(34)
    cond_b = cond_image_for(35)

    # Single-adapter baselines.
    model_a.apply_to(transformer)
    model_a.set_multiplier(1.0)
    model_a.set_cond_image(cond_a)
    y_a_only = q_proj(x)
    model_a.restore()

    model_b.apply_to(transformer)
    model_b.set_multiplier(1.0)
    model_b.set_cond_image(cond_b)
    y_b_only = q_proj(x)
    model_b.restore()

    assert not torch.equal(y_a_only, y_plain)
    assert not torch.equal(y_b_only, y_plain)
    assert not torch.equal(y_a_only, y_b_only)

    # Both applied: zeroing one adapter's multiplier removes exactly its contribution.
    model_a.apply_to(transformer)
    model_b.apply_to(transformer)
    model_a.set_cond_image(cond_a)
    model_b.set_cond_image(cond_b)

    model_a.set_multiplier(1.0)
    model_b.set_multiplier(0.0)
    assert torch.equal(q_proj(x), y_a_only)

    model_a.set_multiplier(0.0)
    model_b.set_multiplier(1.0)
    assert torch.equal(q_proj(x), y_b_only)

    model_a.set_multiplier(0.0)
    model_b.set_multiplier(0.0)
    assert torch.equal(q_proj(x), y_plain)

    model_a.set_multiplier(1.0)
    model_b.set_multiplier(1.0)
    y_both = q_proj(x)
    assert not torch.equal(y_both, y_a_only)
    assert not torch.equal(y_both, y_b_only)

    model_b.restore()
    model_a.restore()


# ----------------------------------------------------------------------------
# Denoise input normalization (duplicate models share one cached instance)
# ----------------------------------------------------------------------------


def _lllite_field(key: str) -> AnimaLLLiteField:
    return AnimaLLLiteField(
        image_name="cond_image",
        control_model=ModelIdentifierField(
            key=key,
            hash="blake3:0000",
            name=f"adapter-{key}",
            base=BaseModelType.Anima,
            type=ModelType.ControlNet,
        ),
    )


def test_normalize_control_lllite_to_list() -> None:
    assert AnimaDenoiseInvocation._normalize_control_lllite(None) == []
    single = _lllite_field("key-a")
    assert AnimaDenoiseInvocation._normalize_control_lllite(single) == [single]
    pair = [_lllite_field("key-a"), _lllite_field("key-b")]
    assert AnimaDenoiseInvocation._normalize_control_lllite(pair) == pair


def test_normalize_control_lllite_sorts_by_model_key() -> None:
    """Apply order must be deterministic: collect-node input order follows random node ids."""
    field_a = _lllite_field("key-a")
    field_b = _lllite_field("key-b")
    assert AnimaDenoiseInvocation._normalize_control_lllite([field_b, field_a]) == [field_a, field_b]


def test_normalize_control_lllite_rejects_duplicate_model_keys() -> None:
    pair = [_lllite_field("key-a"), _lllite_field("key-a")]
    with pytest.raises(ValueError, match="used by more than one"):
        AnimaDenoiseInvocation._normalize_control_lllite(pair)


# ----------------------------------------------------------------------------
# Step-range gate (_get_lllite_multiplier)
# ----------------------------------------------------------------------------


def _stepped_field(weight: float, begin_step_percent: float, end_step_percent: float) -> AnimaLLLiteField:
    return AnimaLLLiteField(
        image_name="cond_image",
        control_model=ModelIdentifierField(
            key="key-a",
            hash="blake3:0000",
            name="adapter",
            base=BaseModelType.Anima,
            type=ModelType.ControlNet,
        ),
        weight=weight,
        begin_step_percent=begin_step_percent,
        end_step_percent=end_step_percent,
    )


def test_lllite_multiplier_full_range_returns_weight() -> None:
    """A 0%..100% window returns the field weight at every step, boundaries included."""
    field = _stepped_field(weight=0.7, begin_step_percent=0.0, end_step_percent=1.0)
    for step_index in (0, 1, 10, 19, 20):
        assert AnimaDenoiseInvocation._get_lllite_multiplier(field, step_index, total_steps=20) == 0.7


def test_lllite_multiplier_zero_outside_window() -> None:
    """A 25%..75% window over 20 steps gates to first_step=5, last_step=15 (both inclusive)."""
    field = _stepped_field(weight=0.5, begin_step_percent=0.25, end_step_percent=0.75)
    # Below the window.
    assert AnimaDenoiseInvocation._get_lllite_multiplier(field, 4, total_steps=20) == 0.0
    # First applied step (floor(0.25 * 20) = 5) is inclusive.
    assert AnimaDenoiseInvocation._get_lllite_multiplier(field, 5, total_steps=20) == 0.5
    assert AnimaDenoiseInvocation._get_lllite_multiplier(field, 10, total_steps=20) == 0.5
    # Last applied step (ceil(0.75 * 20) = 15) is inclusive.
    assert AnimaDenoiseInvocation._get_lllite_multiplier(field, 15, total_steps=20) == 0.5
    # Above the window.
    assert AnimaDenoiseInvocation._get_lllite_multiplier(field, 16, total_steps=20) == 0.0


def test_lllite_multiplier_floor_ceil_rounding() -> None:
    """A non-integer step boundary floors the start and ceils the end (widening, not truncating)."""
    # 0.34 * 10 = 3.4 -> first_step = floor(3.4) = 3, last_step = ceil(3.4) = 4.
    field = _stepped_field(weight=1.0, begin_step_percent=0.34, end_step_percent=0.34)
    assert AnimaDenoiseInvocation._get_lllite_multiplier(field, 2, total_steps=10) == 0.0
    assert AnimaDenoiseInvocation._get_lllite_multiplier(field, 3, total_steps=10) == 1.0
    assert AnimaDenoiseInvocation._get_lllite_multiplier(field, 4, total_steps=10) == 1.0
    assert AnimaDenoiseInvocation._get_lllite_multiplier(field, 5, total_steps=10) == 0.0


# ----------------------------------------------------------------------------
# Real weight file (optional; skipped when the file is not present)
# ----------------------------------------------------------------------------


@pytest.mark.skipif(
    REAL_WEIGHTS_PATH is None,
    reason=f"set {_REAL_WEIGHTS_ENV_VAR} to the real LLLite weights file to run",
)
def test_from_state_dict_real_file() -> None:
    from safetensors import safe_open
    from safetensors.torch import load_file

    assert REAL_WEIGHTS_PATH is not None
    # A stale path must fail loudly, not silently skip.
    assert REAL_WEIGHTS_PATH.is_file(), f"{_REAL_WEIGHTS_ENV_VAR} points to a missing file: {REAL_WEIGHTS_PATH}"
    sd = load_file(str(REAL_WEIGHTS_PATH))
    with safe_open(str(REAL_WEIGHTS_PATH), framework="pt") as f:
        metadata = f.metadata()

    model = AnimaControlNetLLLite.from_state_dict(sd, metadata)

    assert len(model.lllite_modules) == 112
    assert model.cond_in_channels == 4
    assert model.inpaint_masked_input is True
    assert model.cond_emb_dim == 64
    assert model.mlp_dim == 64
    assert model.cond_dim == 128
    assert model.cond_resblocks == 4
    assert model.use_aspp is False

    for m in model.lllite_modules:
        assert m.in_dim == 2048
        assert m.down.weight.shape == (64, 2048)
        assert m.mid.weight.shape == (64, 128)
        assert m.cond_to_film.weight.shape == (128, 64)
        assert m.up.weight.shape == (2048, 64)
        assert m.depth_embed.shape == (64,)

    # Strict loading consumed every saved key (1056 = 48 trunk + 112 * 9).
    assert len(model.state_dict()) == len(sd) == 1056

    # Forward smoke test on one module bound to a 2048-wide Linear.
    model = model.to(torch.float32)
    linear = nn.Linear(2048, 2048)
    module = model.lllite_modules[0]
    module.bind(linear)
    try:
        model.set_multiplier(1.0)
        # Cond image for latent 4x4 -> S = 2*2 = 4 trunk tokens.
        model.set_cond_image(torch.randn(1, 4, 32, 32, generator=torch.Generator().manual_seed(15)))
        assert module.cond_emb is not None
        assert module.cond_emb.shape == (1, 4, 64)
        x = torch.randn(1, 4, 2048, generator=torch.Generator().manual_seed(16))
        y = linear(x)
        assert y.shape == (1, 4, 2048)
        assert torch.isfinite(y).all()
        assert not torch.equal(y, plain_linear(linear, x))
    finally:
        module.unbind()
