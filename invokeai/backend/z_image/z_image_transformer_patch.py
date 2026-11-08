"""Utilities for patching the ZImageTransformer2DModel to support regional attention masks."""

from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple

import torch


def create_regional_forward(
    original_forward: Callable,
    regional_attn_mask: torch.Tensor | None,
    img_seq_len: int,
    positive_cap_feats: torch.Tensor | None,
    negative_regional_attn_mask: torch.Tensor | None = None,
    negative_cap_feats: torch.Tensor | None = None,
) -> Callable:
    """Create a modified forward function that uses a regional attention mask.

    The regional attention mask replaces the internally computed padding mask on the
    main transformer layers (alternating with the plain padding mask), allowing for
    regional prompting where different image regions attend to different text prompts.

    This delegates to the model's own helper methods (``patchify_and_embed``,
    ``_prepare_sequence``, ``_build_unified_sequence``) so it stays in sync with the
    upstream diffusers ``ZImageTransformer2DModel.forward`` implementation. Only the
    main-layer attention mask is overridden.

    Args:
        original_forward: The original forward method of ZImageTransformer2DModel
            (kept for signature compatibility; not used directly).
        regional_attn_mask: Boolean attention mask of shape (seq_len, seq_len) where
                           seq_len = img_seq_len + txt_seq_len, ordered [img, txt].
        img_seq_len: Number of (unpadded) image tokens in the sequence.
        positive_cap_feats: The caption tensor paired with ``regional_attn_mask``.
        negative_regional_attn_mask: The negative pass's independent regional mask, if any.
        negative_cap_feats: The caption tensor paired with ``negative_regional_attn_mask``.
            Tensor identity selects each pass's mask, even when their token lengths coincide.

    Returns:
        A modified forward function with regional attention support.
    """

    def regional_forward(
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        cap_feats: List[torch.Tensor],
        patch_size: int = 2,
        f_patch_size: int = 1,
    ) -> Tuple[List[torch.Tensor], dict]:
        """Modified forward with regional attention mask injection.

        Mirrors the basic (non-omni) path of ZImageTransformer2DModel.forward but
        injects a regional attention mask into the main transformer layers.
        """
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        device = x[0].device

        # Select before patchify_and_embed replaces the caption tensors. Token lengths cannot
        # distinguish the CFG passes, and either pass may be the only one with regions.
        regional_masks = [
            regional_attn_mask
            if caption is positive_cap_feats
            else negative_regional_attn_mask
            if caption is negative_cap_feats
            else None
            for caption in cap_feats
        ]

        # Single adaLN embedding for all tokens (basic mode).
        adaln_input = self.t_embedder(t * self.t_scale).type_as(x[0])

        # Patchify & embed (basic mode: single image per batch item).
        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_pad_mask,
            cap_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # X embed & refine.
        x_seqlens = [len(xi) for xi in x]
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(x, dim=0))
        x, x_freqs, x_mask, _, _ = self._prepare_sequence(
            list(x.split(x_seqlens, dim=0)), x_pos_ids, x_pad_mask, self.x_pad_token, None, device
        )
        for layer in self.noise_refiner:
            x = layer(x, x_mask, x_freqs, adaln_input, None, None, None)

        # Cap embed & refine.
        cap_seqlens = [len(ci) for ci in cap_feats]
        cap_feats = self.cap_embedder(torch.cat(cap_feats, dim=0))
        cap_feats, cap_freqs, cap_mask, _, _ = self._prepare_sequence(
            list(cap_feats.split(cap_seqlens, dim=0)), cap_pos_ids, cap_pad_mask, self.cap_pad_token, None, device
        )
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, cap_freqs)

        # Unified sequence: basic mode order [x, cap].
        unified, unified_freqs, unified_mask, _ = self._build_unified_sequence(
            x,
            x_freqs,
            x_seqlens,
            None,
            cap_feats,
            cap_freqs,
            cap_seqlens,
            None,
            None,
            None,
            None,
            None,
            False,  # omni_mode
            device,
        )

        bsz = unified.shape[0]
        unified_seqlen = unified.shape[1]

        # --- REGIONAL ATTENTION MASK INJECTION ---
        # The regional mask is (S, S) with S = img_seq_len + txt_seq_len, ordered [img, txt],
        # using the *unpadded* image and text token counts. In the unified sequence, however,
        # both the image block and the caption block are individually padded to a multiple of
        # SEQ_MULTI_OF, so the real layout per item is:
        #     [ img_real | img_pad | txt_real | txt_pad ]
        # We therefore scatter the four regional sub-blocks (img-img, img-txt, txt-img, txt-txt)
        # into their padding-aware positions instead of assuming a contiguous top-left block.
        #
        use_regional = any(mask is not None for mask in regional_masks)

        # Rebuilt on every forward rather than cached per CFG pass: caching both passes' (S, S) biases keeps them
        # resident (hundreds of MB at 2048 px), while the rebuild costs only a few milliseconds per forward.
        float_mask = None
        if use_regional:
            # Build a per-item additive float mask. Start from the plain padding mask (0 where a
            # token is valid, -inf where it is padding) so non-matching items behave normally.
            neg_inf = torch.finfo(unified.dtype).min
            zero = torch.zeros((), dtype=unified.dtype, device=device)
            if unified_mask is None:
                # diffusers builds no padding mask when every item has the same length, which is
                # always the case at batch size 1: no position is batch padding.
                float_mask = torch.zeros((bsz, 1, unified_seqlen, unified_seqlen), dtype=unified.dtype, device=device)
            else:
                float_mask = (
                    torch.where(
                        unified_mask.bool().unsqueeze(1).unsqueeze(1),  # (bsz, 1, 1, S)
                        zero,
                        torch.full((), neg_inf, dtype=unified.dtype, device=device),
                    )
                    .expand(bsz, 1, unified_seqlen, unified_seqlen)
                    .clone()
                )

            for i, regional in enumerate(regional_masks):
                if regional is None:
                    continue
                regional = regional.to(device=device, dtype=torch.bool)
                txt_seq_len = regional.shape[0] - img_seq_len
                x_len = x_seqlens[i]

                ii, it = slice(0, img_seq_len), slice(img_seq_len, img_seq_len + txt_seq_len)
                ui = slice(0, img_seq_len)  # real image positions in unified item
                ut = slice(x_len, x_len + txt_seq_len)  # real text positions in unified item

                # Reset the masked region so only regional rules apply to real img/txt tokens;
                # their rows start fully blocked and we open the allowed sub-blocks below.
                float_mask[i, 0, ui, :] = neg_inf
                float_mask[i, 0, ut, :] = neg_inf

                float_mask[i, 0, ui, ui] = torch.where(regional[ii, ii], zero, neg_inf)  # img -> img
                float_mask[i, 0, ui, ut] = torch.where(regional[ii, it], zero, neg_inf)  # img -> txt
                float_mask[i, 0, ut, ui] = torch.where(regional[it, ii], zero, neg_inf)  # txt -> img
                float_mask[i, 0, ut, ut] = torch.where(regional[it, it], zero, neg_inf)  # txt -> txt

        for layer_idx, layer in enumerate(self.layers):
            attn_mask = float_mask if (use_regional and layer_idx % 2 == 0) else unified_mask
            unified = layer(unified, attn_mask, unified_freqs, adaln_input, None, None, None)

        # Final layer + unpatchify.
        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, c=adaln_input)
        x_out = self.unpatchify(list(unified.unbind(dim=0)), x_size, patch_size, f_patch_size)

        return x_out, {}

    return regional_forward


@contextmanager
def patch_transformer_for_regional_prompting(
    transformer,
    regional_attn_mask: Optional[torch.Tensor],
    img_seq_len: int,
    positive_cap_feats: Optional[torch.Tensor] = None,
    negative_regional_attn_mask: Optional[torch.Tensor] = None,
    negative_cap_feats: Optional[torch.Tensor] = None,
):
    """Context manager to temporarily patch the transformer for regional prompting.

    Args:
        transformer: The ZImageTransformer2DModel instance.
        regional_attn_mask: Regional attention mask of shape (seq_len, seq_len).
                           If both regional masks are None, the transformer is not patched.
        img_seq_len: Number of image tokens.
        positive_cap_feats: The caption-embedding tensor the regional mask was built for.
            Required when ``regional_attn_mask`` is provided; the mask is applied only to
            forward calls whose ``cap_feats`` is this exact object (the conditioned pass).
        negative_regional_attn_mask: The negative pass's regional mask.
        negative_cap_feats: The corresponding negative caption tensor; required for a negative mask.

    Yields:
        The (possibly patched) transformer.
    """
    if regional_attn_mask is None and negative_regional_attn_mask is None:
        # No regional prompting, use original forward
        yield transformer
        return

    if regional_attn_mask is not None and positive_cap_feats is None:
        raise ValueError("positive_cap_feats is required when regional_attn_mask is provided")
    if negative_regional_attn_mask is not None and negative_cap_feats is None:
        raise ValueError("negative_cap_feats is required when negative_regional_attn_mask is provided")

    # Store original forward
    original_forward = transformer.forward

    # Create and bind the regional forward
    regional_fwd = create_regional_forward(
        original_forward,
        regional_attn_mask,
        img_seq_len,
        positive_cap_feats,
        negative_regional_attn_mask,
        negative_cap_feats,
    )
    transformer.forward = lambda *args, **kwargs: regional_fwd(transformer, *args, **kwargs)

    try:
        yield transformer
    finally:
        # Restore original forward
        transformer.forward = original_forward
