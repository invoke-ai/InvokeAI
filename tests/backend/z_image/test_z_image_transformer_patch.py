import diffusers
import torch

from invokeai.backend.z_image.z_image_transformer_patch import patch_transformer_for_regional_prompting


def _tiny_z_image_transformer() -> diffusers.ZImageTransformer2DModel:
    torch.manual_seed(0)
    return diffusers.ZImageTransformer2DModel(
        all_patch_size=(1,),
        all_f_patch_size=(1,),
        in_channels=4,
        dim=8,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=1,
        n_kv_heads=1,
        cap_feat_dim=8,
        axes_dims=[2, 2, 4],
        axes_lens=[64, 64, 64],
    ).eval()


def _run(
    model: diffusers.ZImageTransformer2DModel,
    x: list[torch.Tensor],
    cap_feats: torch.Tensor,
    regional_mask: torch.Tensor,
    img_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.tensor([0.5])
    with torch.no_grad():
        baseline = model(x, t, cap_feats=[cap_feats], patch_size=1)[0][0]
        with patch_transformer_for_regional_prompting(
            model, regional_mask, img_len, positive_cap_feats=cap_feats
        ) as patched:
            out = patched(x, t, cap_feats=[cap_feats], patch_size=1)[0][0]
    return baseline, out


def test_regional_forward_single_item_batch() -> None:
    """Regression test for #9612.

    z_image_denoise always calls the transformer with a single item. With diffusers 0.40 the unified attention mask
    is ``None`` when all items have the same length, which must not crash the regional forward.
    """
    model = _tiny_z_image_transformer()
    x = [torch.randn(4, 1, 4, 4)]  # 16 image tokens
    cap_feats = torch.randn(5, 8)
    img_len, txt_len = 16, 5
    regional_mask = torch.ones(img_len + txt_len, img_len + txt_len, dtype=torch.bool)
    regional_mask[:8, img_len:] = False  # top half of the image does not attend to the prompt

    baseline, out = _run(model, x, cap_feats, regional_mask, img_len)

    assert out.shape == baseline.shape
    assert torch.isfinite(out).all()


def test_regional_forward_single_item_batch_matches_unpatched_without_padding() -> None:
    """With no SEQ_MULTI_OF padding, an all-True regional mask must reproduce the unpatched forward, and a
    restrictive one must still change the output."""
    model = _tiny_z_image_transformer()
    x = [torch.randn(4, 1, 8, 8)]  # 64 image tokens, no padding
    cap_feats = torch.randn(32, 8)  # 32 text tokens, no padding
    img_len, txt_len = 64, 32
    seq_len = img_len + txt_len

    baseline, out = _run(model, x, cap_feats, torch.ones(seq_len, seq_len, dtype=torch.bool), img_len)
    torch.testing.assert_close(out, baseline)

    restrictive = torch.ones(seq_len, seq_len, dtype=torch.bool)
    restrictive[: img_len // 2, img_len:] = False
    _, out_restricted = _run(model, x, cap_feats, restrictive, img_len)
    assert not torch.allclose(out_restricted, baseline)
