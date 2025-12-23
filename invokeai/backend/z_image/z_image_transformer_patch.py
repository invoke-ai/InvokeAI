"""Utilities for patching the ZImageTransformer2DModel to support regional attention masks."""

from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def create_regional_forward(
    original_forward: Callable,
    regional_attn_mask: torch.Tensor,
    img_seq_len: int,
) -> Callable:
    """Create a modified forward function that uses a regional attention mask.

    The regional attention mask replaces the internally computed padding mask,
    allowing for regional prompting where different image regions attend to
    different text prompts.

    Args:
        original_forward: The original forward method of ZImageTransformer2DModel.
        regional_attn_mask: Attention mask of shape (seq_len, seq_len) where
                           seq_len = img_seq_len + txt_seq_len.
        img_seq_len: Number of image tokens in the sequence.

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

        This is based on the original ZImageTransformer2DModel.forward but
        replaces the padding-based attention mask with a regional attention mask.
        """
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        bsz = len(x)
        device = x[0].device
        t_scaled = t * self.t_scale
        t_emb = self.t_embedder(t_scaled)

        SEQ_MULTI_OF = 32  # From diffusers transformer_z_image.py

        # Patchify and embed (reusing the original method)
        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # x embed & refine
        x_item_seqlens = [len(_) for _ in x]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        x_cat = torch.cat(x, dim=0)
        x_cat = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x_cat)

        adaln_input = t_emb.type_as(x_cat)
        x_cat[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        x_list = list(x_cat.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

        x_padded = pad_sequence(x_list, batch_first=True, padding_value=0.0)
        x_freqs_cis_padded = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        # Process through noise_refiner
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.noise_refiner:
                x_padded = self._gradient_checkpointing_func(
                    layer, x_padded, x_attn_mask, x_freqs_cis_padded, adaln_input
                )
        else:
            for layer in self.noise_refiner:
                x_padded = layer(x_padded, x_attn_mask, x_freqs_cis_padded, adaln_input)

        # cap embed & refine
        cap_item_seqlens = [len(_) for _ in cap_feats]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in cap_item_seqlens)
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_cat = torch.cat(cap_feats, dim=0)
        cap_cat = self.cap_embedder(cap_cat)
        cap_cat[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
        cap_list = list(cap_cat.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(cap_item_seqlens, dim=0))

        cap_padded = pad_sequence(cap_list, batch_first=True, padding_value=0.0)
        cap_freqs_cis_padded = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1

        # Process through context_refiner
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.context_refiner:
                cap_padded = self._gradient_checkpointing_func(layer, cap_padded, cap_attn_mask, cap_freqs_cis_padded)
        else:
            for layer in self.context_refiner:
                cap_padded = layer(cap_padded, cap_attn_mask, cap_freqs_cis_padded)

        # Unified sequence: [img_tokens, txt_tokens]
        unified = []
        unified_freqs_cis = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x_padded[i][:x_len], cap_padded[i][:cap_len]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis_padded[i][:x_len], cap_freqs_cis_padded[i][:cap_len]]))

        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens, strict=False)]
        assert unified_item_seqlens == [len(_) for _ in unified]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified_padded = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis_padded = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)

        # --- REGIONAL ATTENTION MASK INJECTION ---
        # Instead of using the padding mask, we use the regional attention mask
        # The regional mask is (seq_len, seq_len), we need to expand it to (batch, seq_len, seq_len)
        # and then add the batch dimension for broadcasting: (batch, 1, seq_len, seq_len)

        # Expand regional mask to match the actual sequence length (may include padding)
        if regional_attn_mask.shape[0] != unified_max_item_seqlen:
            # Pad the regional mask to match unified sequence length
            padded_regional_mask = torch.zeros(
                (unified_max_item_seqlen, unified_max_item_seqlen),
                dtype=regional_attn_mask.dtype,
                device=device,
            )
            mask_size = min(regional_attn_mask.shape[0], unified_max_item_seqlen)
            padded_regional_mask[:mask_size, :mask_size] = regional_attn_mask[:mask_size, :mask_size]
        else:
            padded_regional_mask = regional_attn_mask.to(device)

        # Convert boolean mask to additive float mask for attention
        # True (attend) -> 0.0, False (block) -> -inf
        # This is required because the attention backend expects additive masks for 4D inputs
        # Use bfloat16 to match the transformer's query dtype
        float_mask = torch.zeros_like(padded_regional_mask, dtype=torch.bfloat16)
        float_mask[~padded_regional_mask] = float("-inf")

        # Expand to (batch, 1, seq_len, seq_len) for attention
        unified_attn_mask = float_mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1)

        # Process through main layers with regional attention mask
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer_idx, layer in enumerate(self.layers):
                # Alternate between regional mask and full attention
                if layer_idx % 2 == 0:
                    unified_padded = self._gradient_checkpointing_func(
                        layer, unified_padded, unified_attn_mask, unified_freqs_cis_padded, adaln_input
                    )
                else:
                    # Use padding mask only for odd layers (allows global coherence)
                    padding_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
                    for i, seq_len in enumerate(unified_item_seqlens):
                        padding_mask[i, :seq_len] = 1
                    unified_padded = self._gradient_checkpointing_func(
                        layer, unified_padded, padding_mask, unified_freqs_cis_padded, adaln_input
                    )
        else:
            for layer_idx, layer in enumerate(self.layers):
                # Alternate between regional mask and full attention
                if layer_idx % 2 == 0:
                    unified_padded = layer(unified_padded, unified_attn_mask, unified_freqs_cis_padded, adaln_input)
                else:
                    # Use padding mask only for odd layers (allows global coherence)
                    padding_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
                    for i, seq_len in enumerate(unified_item_seqlens):
                        padding_mask[i, :seq_len] = 1
                    unified_padded = layer(unified_padded, padding_mask, unified_freqs_cis_padded, adaln_input)

        # Final layer
        unified_out = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified_padded, adaln_input)
        unified_list = list(unified_out.unbind(dim=0))
        x_out = self.unpatchify(unified_list, x_size, patch_size, f_patch_size)

        return x_out, {}

    return regional_forward


@contextmanager
def patch_transformer_for_regional_prompting(
    transformer,
    regional_attn_mask: Optional[torch.Tensor],
    img_seq_len: int,
):
    """Context manager to temporarily patch the transformer for regional prompting.

    Args:
        transformer: The ZImageTransformer2DModel instance.
        regional_attn_mask: Regional attention mask of shape (seq_len, seq_len).
                           If None, the transformer is not patched.
        img_seq_len: Number of image tokens.

    Yields:
        The (possibly patched) transformer.
    """
    if regional_attn_mask is None:
        # No regional prompting, use original forward
        yield transformer
        return

    # Store original forward
    original_forward = transformer.forward

    # Create and bind the regional forward
    regional_fwd = create_regional_forward(original_forward, regional_attn_mask, img_seq_len)
    transformer.forward = lambda *args, **kwargs: regional_fwd(transformer, *args, **kwargs)

    try:
        yield transformer
    finally:
        # Restore original forward
        transformer.forward = original_forward
