import torch
import torch.nn.functional as F

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import TextConditioningRegions


class RegionalPromptData:
    """A class to manage the prompt data for regional conditioning."""

    def __init__(
        self,
        text_embeds: list[list[torch.Tensor]],
        masks: list[list[torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Initialize a `RegionalPromptData` object.
        Args:
            TODO(ryand): Update these docs.

            text_embeds (list[list[torch.Tensor]]): The text prompt embeddings. text_embeds[b][i] contains the embedding
                for prompt i to be applied to batch image b.
            masks (list[list[torch.Tensor]]): The masks indicating the spatial regions of the image that each prompt
                applies to. masks[b][i] contains the mask for text_embeds[b][i].

            device (torch.device): The device to use for the attention masks.
            dtype (torch.dtype): The data type to use for the attention masks.
            max_downscale_factor: Spatial masks will be prepared for downscale factors from 1 to max_downscale_factor
                in steps of 2x.
        """

        assert len(text_embeds) == len(masks)
        for text_embeds_batch, masks_batch in zip(text_embeds, masks, strict=True):
            assert len(text_embeds_batch) == len(masks_batch)

        self.prompt_count_by_batch_element = [len(text_embeds_batch) for text_embeds_batch in text_embeds]

        # Flattenand concat text_embeds.
        text_embeds_flat_list: list[torch.Tensor] = []
        for text_embeds_batch in text_embeds:
            text_embeds_flat_list.extend(text_embeds_batch)
        # TODO(ryand): Or stack?
        # TODO(ryand): Text embeds might not all be the same size (if there were long prompts).
        self.text_embeds = torch.cat(text_embeds_flat_list, dim=0)

        #  Flatten and concat masks.
        masks_flat_list = []
        for mask_batch in masks:
            masks_flat_list.extend(mask_batch)
        self._masks = torch.cat(masks_flat_list, dim=0)
        # TODO(ryand): Is this necessary? Do we need to do the same for text_embeds?
        self._masks = self._masks.to(dtype=dtype, device=device)

        self._device = device
        self._dtype = dtype

    def get_masks(self, query_seq_len: int, max_downscale_factor: int = 8) -> torch.Tensor:
        _, _, h, w = self._masks.shape

        # Determine the downscaling factor for the given query sequence length.
        downscale_factor = 1
        while downscale_factor <= max_downscale_factor:
            if query_seq_len == (h // downscale_factor) * (w // downscale_factor):
                break
            downscale_factor *= 2

        if query_seq_len != (h // downscale_factor) * (w // downscale_factor):
            raise ValueError(f"Failed to find a mask downsampling factor for query sequence length: {query_seq_len}")

        target_h = h // downscale_factor
        target_w = w // downscale_factor
        mask_downscaled = torch.nn.functional.interpolate(self._masks, size=(target_h, target_w), mode="nearest")

        return mask_downscaled

    def _prepare_spatial_masks_old(
        self, regions: list[TextConditioningRegions], max_downscale_factor: int = 8
    ) -> list[dict[int, torch.Tensor]]:
        """Prepare the spatial masks for all downscaling factors."""
        # batch_masks_by_seq_len[b][s] contains the spatial masks for the b'th batch sample with a query sequence length
        # of s.
        batch_sample_masks_by_seq_len: list[dict[int, torch.Tensor]] = []

        for batch_sample_regions in regions:
            batch_sample_masks_by_seq_len.append({})

            batch_sample_masks = batch_sample_regions.masks.to(device=self._device, dtype=self._dtype)

            # Downsample the spatial dimensions by factors of 2 until max_downscale_factor is reached.
            downscale_factor = 1
            while downscale_factor <= max_downscale_factor:
                b, _num_prompts, h, w = batch_sample_masks.shape
                assert b == 1
                query_seq_len = h * w

                batch_sample_masks_by_seq_len[-1][query_seq_len] = batch_sample_masks

                downscale_factor *= 2
                if downscale_factor <= max_downscale_factor:
                    # We use max pooling because we downscale to a pretty low resolution, so we don't want small prompt
                    # regions to be lost entirely.
                    #
                    # ceil_mode=True is set to mirror the downsampling behavior of SD and SDXL.
                    #
                    # TODO(ryand): In the future, we may want to experiment with other downsampling methods (e.g.
                    # nearest interpolation), and could potentially use a weighted mask rather than a binary mask.
                    batch_sample_masks = F.max_pool2d(batch_sample_masks, kernel_size=2, stride=2, ceil_mode=True)

        return batch_sample_masks_by_seq_len

    def get_cross_attn_mask(self, query_seq_len: int, key_seq_len: int) -> torch.Tensor:
        """Get the cross-attention mask for the given query sequence length.
        Args:
            query_seq_len: The length of the flattened spatial features at the current downscaling level.
            key_seq_len (int): The sequence length of the prompt embeddings (which act as the key in the cross-attention
                layers). This is most likely equal to the max embedding range end, but we pass it explicitly to be sure.
        Returns:
            torch.Tensor: The cross-attention score mask.
                shape: (batch_size, query_seq_len, key_seq_len).
                dtype: float
        """
        batch_size = len(self._spatial_masks_by_seq_len)
        batch_spatial_masks = [self._spatial_masks_by_seq_len[b][query_seq_len] for b in range(batch_size)]

        # Create an empty attention mask with the correct shape.
        attn_mask = torch.zeros((batch_size, query_seq_len, key_seq_len), dtype=self._dtype, device=self._device)

        for batch_idx in range(batch_size):
            batch_sample_spatial_masks = batch_spatial_masks[batch_idx]
            batch_sample_regions = self._regions[batch_idx]

            # Flatten the spatial dimensions of the mask by reshaping to (1, num_prompts, query_seq_len, 1).
            _, num_prompts, _, _ = batch_sample_spatial_masks.shape
            batch_sample_query_masks = batch_sample_spatial_masks.view((1, num_prompts, query_seq_len, 1))

            for prompt_idx, embedding_range in enumerate(batch_sample_regions.ranges):
                batch_sample_query_scores = batch_sample_query_masks[0, prompt_idx, :, :].clone()
                batch_sample_query_mask = batch_sample_query_scores > 0.5
                batch_sample_query_scores[batch_sample_query_mask] = 0.0
                batch_sample_query_scores[~batch_sample_query_mask] = self._negative_cross_attn_mask_score
                attn_mask[batch_idx, :, embedding_range.start : embedding_range.end] = batch_sample_query_scores

        return attn_mask
