import torch


class RegionalIPData:
    """A class to manage the data for regional IP-Adapter conditioning."""

    def __init__(
        self,
        image_prompt_embeds: list[torch.Tensor],
        scales: list[float],
    ):
        """Initialize a `IPAdapterConditioningData` object."""
        # The image prompt embeddings.
        # regional_ip_data[i] contains the image prompt embeddings for the i'th IP-Adapter. Each tensor
        # has shape (batch_size, num_ip_images, seq_len, ip_embedding_len).
        self.image_prompt_embeds = image_prompt_embeds

        # The scales for the IP-Adapter attention.
        # scales[i] contains the attention scale for the i'th IP-Adapter.
        self.scales = scales
