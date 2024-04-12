import torch


class RegionalIPData:
    """A class to manage the data for regional IP-Adapter conditioning."""

    def __init__(
        self,
        image_prompt_embeds: list[torch.Tensor],
        scales: list[float],
        masks: list[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
        max_downscale_factor: int = 8,
    ):
        """Initialize a `IPAdapterConditioningData` object."""
        assert len(image_prompt_embeds) == len(scales) == len(masks)

        # The image prompt embeddings.
        # regional_ip_data[i] contains the image prompt embeddings for the i'th IP-Adapter. Each tensor
        # has shape (batch_size, num_ip_images, seq_len, ip_embedding_len).
        self.image_prompt_embeds = image_prompt_embeds

        # The scales for the IP-Adapter attention.
        # scales[i] contains the attention scale for the i'th IP-Adapter.
        self.scales = scales

        # The IP-Adapter masks.
        # self._masks_by_seq_len[s] contains the spatial masks for the downsampling level with query sequence length of
        # s. It has shape (batch_size, num_ip_images, query_seq_len, 1). The masks have values of 1.0 for included
        # regions and 0.0 for excluded regions.
        self._masks_by_seq_len = self._prepare_masks(masks, max_downscale_factor, device, dtype)

    def _prepare_masks(
        self, masks: list[torch.Tensor], max_downscale_factor: int, device: torch.device, dtype: torch.dtype
    ) -> dict[int, torch.Tensor]:
        """Prepare the masks for the IP-Adapter attention."""
        # Concatenate the masks so that they can be processed more efficiently.
        mask_tensor = torch.cat(masks, dim=1)

        mask_tensor = mask_tensor.to(device=device, dtype=dtype)

        masks_by_seq_len: dict[int, torch.Tensor] = {}

        # Downsample the spatial dimensions by factors of 2 until max_downscale_factor is reached.
        downscale_factor = 1
        while downscale_factor <= max_downscale_factor:
            b, num_ip_adapters, h, w = mask_tensor.shape
            # Assert that the batch size is 1, because I haven't thought through batch handling for this feature yet.
            assert b == 1

            # The IP-Adapters are applied in the cross-attention layers, where the query sequence length is the h * w of
            # the spatial features.
            query_seq_len = h * w

            masks_by_seq_len[query_seq_len] = mask_tensor.view((b, num_ip_adapters, -1, 1))

            downscale_factor *= 2
            if downscale_factor <= max_downscale_factor:
                # We use max pooling because we downscale to a pretty low resolution, so we don't want small mask
                # regions to be lost entirely.
                #
                # ceil_mode=True is set to mirror the downsampling behavior of SD and SDXL.
                #
                # TODO(ryand): In the future, we may want to experiment with other downsampling methods.
                mask_tensor = torch.nn.functional.max_pool2d(mask_tensor, kernel_size=2, stride=2, ceil_mode=True)

        return masks_by_seq_len

    def get_masks(self, query_seq_len: int) -> torch.Tensor:
        """Get the mask for the given query sequence length."""
        return self._masks_by_seq_len[query_seq_len]
