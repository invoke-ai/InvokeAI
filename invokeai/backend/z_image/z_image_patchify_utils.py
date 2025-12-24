# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Utility functions for Z-Image patchify operations."""

from typing import List, Tuple

import torch

# Sequence must be multiple of this value (from diffusers transformer_z_image)
SEQ_MULTI_OF = 32


def create_coordinate_grid(
    size: Tuple[int, ...],
    start: Tuple[int, ...] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a coordinate grid for position embeddings.

    Args:
        size: Size of the grid (e.g., (F, H, W))
        start: Starting coordinates (default: all zeros)
        device: Target device

    Returns:
        Coordinate grid tensor of shape (*size, len(size))
    """
    if start is None:
        start = tuple(0 for _ in size)

    axes = [
        torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size, strict=False)
    ]
    grids = torch.meshgrid(axes, indexing="ij")
    return torch.stack(grids, dim=-1)


def patchify_control_context(
    all_image: List[torch.Tensor],
    patch_size: int,
    f_patch_size: int,
    cap_seq_len: int,
) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int]], List[torch.Tensor], List[torch.Tensor]]:
    """Patchify control images without embedding.

    This function extracts patches from control images for control context processing.
    It handles padding and position ID creation for the control signal.

    Args:
        all_image: List of control image tensors [C, F, H, W]
        patch_size: Spatial patch size (height and width)
        f_patch_size: Frame patch size
        cap_seq_len: Caption sequence length (for position ID offset)

    Returns:
        Tuple of:
        - all_image_out: List of patchified image tensors
        - all_image_size: List of (F, H, W) tuples
        - all_image_pos_ids: List of position ID tensors
        - all_image_pad_mask: List of padding mask tensors
    """
    pH = pW = patch_size
    pF = f_patch_size
    device = all_image[0].device

    all_image_out: List[torch.Tensor] = []
    all_image_size: List[Tuple[int, int, int]] = []
    all_image_pos_ids: List[torch.Tensor] = []
    all_image_pad_mask: List[torch.Tensor] = []

    # Calculate padded caption length for position offset
    cap_padding_len = (-cap_seq_len) % SEQ_MULTI_OF
    cap_padded_len = cap_seq_len + cap_padding_len

    for image in all_image:
        C, F, H, W = image.size()
        all_image_size.append((F, H, W))
        F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

        # Patchify: [C, F, H, W] -> [(F_tokens*H_tokens*W_tokens), (pF*pH*pW*C)]
        # Step 1: Rearrange to put spatial dims together for proper patching
        # [C, F, H, W] -> [F, H, W, C]
        image = image.permute(1, 2, 3, 0).contiguous()

        # Step 2: Split H and W into tokens and patch sizes
        # [F, H, W, C] -> [F, H_tokens, pH, W_tokens, pW, C]
        image = image.view(F, H_tokens, pH, W_tokens, pW, C)

        # Step 3: Rearrange to group patches and features
        # [F, H_tokens, pH, W_tokens, pW, C] -> [F, H_tokens, W_tokens, pH, pW, C]
        image = image.permute(0, 1, 3, 2, 4, 5).contiguous()

        # Step 4: For F > 1, we'd need to handle F similarly, but for F=1 this is simpler
        # Final reshape: [F*H_tokens*W_tokens, pH*pW*C]
        num_patches = F_tokens * H_tokens * W_tokens
        patch_features = pF * pH * pW * C
        image = image.reshape(num_patches, patch_features)

        image_ori_len = len(image)
        image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

        # Create position IDs
        image_ori_pos_ids = create_coordinate_grid(
            size=(F_tokens, H_tokens, W_tokens),
            start=(cap_padded_len + 1, 0, 0),
            device=device,
        ).flatten(0, 2)

        image_padding_pos_ids = (
            create_coordinate_grid(
                size=(1, 1, 1),
                start=(0, 0, 0),
                device=device,
            )
            .flatten(0, 2)
            .repeat(image_padding_len, 1)
        )
        image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
        all_image_pos_ids.append(image_padded_pos_ids)

        # Padding mask
        all_image_pad_mask.append(
            torch.cat(
                [
                    torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                    torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                ],
                dim=0,
            )
        )

        # Padded feature
        image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
        all_image_out.append(image_padded_feat)

    return all_image_out, all_image_size, all_image_pos_ids, all_image_pad_mask
