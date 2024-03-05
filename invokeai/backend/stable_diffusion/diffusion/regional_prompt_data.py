import torch
import torch.nn.functional as F

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    TextConditioningRegions,
)


class RegionalPromptData:
    def __init__(self, regions: list[TextConditioningRegions], max_downscale_factor: int = 8):
        """Initialize a `RegionalPromptData` object.

        Args:
            regions (list[TextConditioningRegions]): regions[i] contains the prompt regions for the i'th sample in the
                batch.
            max_downscale_factor: The maximum downscale factor to use when preparing the spatial masks.
        """
        self._regions = regions
        # self._spatial_masks_by_seq_len[b][s] contains the spatial masks for the b'th batch sample with a query
        # sequence length of s.
        self._spatial_masks_by_seq_len: list[dict[int, torch.Tensor]] = self._prepare_spatial_masks(
            regions, max_downscale_factor
        )

        self.negative_cross_attn_mask_score = -10000

    def _prepare_spatial_masks(
        self, regions: list[TextConditioningRegions], max_downscale_factor: int = 8
    ) -> list[dict[int, torch.Tensor]]:
        """Prepare the spatial masks for all downscaling factors."""
        # TODO(ryand): Pass in a list of downscale factors? IIRC, SDXL does not apply attention at all downscaling
        # levels, but I need to double check that.

        # batch_masks_by_seq_len[b][s] contains the spatial masks for the b'th batch sample with a query sequence length
        # of s.
        batch_sample_masks_by_seq_len: list[dict[int, torch.Tensor]] = []

        for batch_sample_regions in regions:
            batch_sample_masks_by_seq_len.append({})

            # Convert the bool masks to float masks so that max pooling can be applied.
            batch_sample_masks = batch_sample_regions.masks.to(dtype=torch.float32)

            # Downsample the spatial dimensions by factors of 2 until max_downscale_factor is reached.
            downscale_factor = 1
            while downscale_factor <= max_downscale_factor:
                b, num_prompts, h, w = batch_sample_masks.shape
                assert b == 1
                query_seq_len = h * w

                batch_sample_masks_by_seq_len[-1][query_seq_len] = batch_sample_masks

                downscale_factor *= 2
                if downscale_factor <= max_downscale_factor:
                    # We use max pooling because we downscale to a pretty low resolution, so we don't want small prompt
                    # regions to be lost entirely.
                    # TODO(ryand): In the future, we may want to experiment with other downsampling methods, and could
                    # potentially use a weighted mask rather than a binary mask.
                    batch_sample_masks = F.max_pool2d(batch_sample_masks, kernel_size=2, stride=2)

        return batch_sample_masks_by_seq_len

    def get_cross_attn_mask(self, query_seq_len: int, key_seq_len: int) -> torch.Tensor:
        """Get the cross-attention mask for the given query sequence length.

        Args:
            query_seq_len: The length of the flattened spatial features at the current downscaling level.
            key_seq_len (int): The sequence length of the prompt embeddings (which act as the key in the cross-attention
                layers). This is most likely equal to the max embedding range end, but we pass it explicitly to be sure.

        Returns:
            torch.Tensor: The masks.
                shape: (batch_size, query_seq_len, key_seq_len).
                dtype: float
                The mask is a binary mask with values of 0.0 and 1.0.
        """
        batch_size = len(self._spatial_masks_by_seq_len)
        batch_spatial_masks = [self._spatial_masks_by_seq_len[b][query_seq_len] for b in range(batch_size)]

        # Create an empty attention mask with the correct shape.
        attn_mask = torch.zeros((batch_size, query_seq_len, key_seq_len))

        for batch_idx in range(batch_size):
            batch_sample_spatial_masks = batch_spatial_masks[batch_idx]
            batch_sample_regions = self._regions[batch_idx]

            # Flatten the spatial dimensions of the mask by reshaping to (1, num_prompts, query_seq_len, 1).
            _, num_prompts, _, _ = batch_sample_spatial_masks.shape
            batch_sample_query_masks = batch_sample_spatial_masks.view((1, num_prompts, query_seq_len, 1))

            for prompt_idx, embedding_range in enumerate(batch_sample_regions.ranges):
                batch_sample_query_scores = batch_sample_query_masks[0, prompt_idx, :, :].clone()
                batch_sample_query_mask = batch_sample_query_scores > 0.5
                batch_sample_query_scores[
                    batch_sample_query_mask
                ] = batch_sample_regions.positive_cross_attn_mask_scores[prompt_idx]
                batch_sample_query_scores[~batch_sample_query_mask] = self.negative_cross_attn_mask_score  # TODO(ryand)
                attn_mask[batch_idx, :, embedding_range.start : embedding_range.end] = batch_sample_query_scores

        return attn_mask

    def get_self_attn_mask(
        self, query_seq_len: int, percent_through: float, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Get the self-attention mask for the given query sequence length.

        Args:
            query_seq_len: The length of the flattened spatial features at the current downscaling level.

        Returns:
            torch.Tensor: The masks.
                shape: (batch_size, query_seq_len, query_seq_len).
                dtype: float
                The mask is a binary mask with values of 0.0 and 1.0.
        """
        # TODO(ryand): Manage dtype and device properly. There's a lot of inefficient copying, conversion, and
        # unnecessary CPU operations happening in this class.
        batch_size = len(self._spatial_masks_by_seq_len)
        batch_spatial_masks = [
            self._spatial_masks_by_seq_len[b][query_seq_len].to(device=device, dtype=dtype) for b in range(batch_size)
        ]

        # Create an empty attention mask with the correct shape.
        attn_mask = torch.zeros((batch_size, query_seq_len, query_seq_len), dtype=dtype, device=device)

        for batch_idx in range(batch_size):
            batch_sample_spatial_masks = batch_spatial_masks[batch_idx]
            batch_sample_regions = self._regions[batch_idx]

            # Flatten the spatial dimensions of the mask by reshaping to (1, num_prompts, query_seq_len, 1).
            _, num_prompts, _, _ = batch_sample_spatial_masks.shape
            batch_sample_query_masks = batch_sample_spatial_masks.view((1, num_prompts, query_seq_len, 1))

            for prompt_idx in range(num_prompts):
                if percent_through > batch_sample_regions.self_attn_adjustment_end_step_percents[prompt_idx]:
                    continue
                prompt_query_mask = batch_sample_query_masks[0, prompt_idx, :, 0]  # Shape: (query_seq_len,)
                # Multiply a (1, query_seq_len) mask by a (query_seq_len, 1) mask to get a (query_seq_len,
                # query_seq_len) mask.
                # TODO(ryand): Is += really the best option here?
                attn_mask[batch_idx, :, :] += (
                    prompt_query_mask.unsqueeze(0)
                    * prompt_query_mask.unsqueeze(1)
                    * batch_sample_regions.positive_self_attn_mask_scores[prompt_idx]
                )

            attn_mask_min = attn_mask[batch_idx].min()

            # Adjust so that the minimum value is 0.0 regardless of whether all pixels are covered or not.
            if abs(attn_mask_min) > 0.0001:
                attn_mask[batch_idx] = attn_mask[batch_idx] - attn_mask_min
        return attn_mask
