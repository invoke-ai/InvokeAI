import torch
import torch.nn.functional as F

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    TextConditioningRegions,
)


class RegionalPromptData:
    def __init__(self, attn_masks_by_seq_len: dict[int, torch.Tensor]):
        self._attn_masks_by_seq_len = attn_masks_by_seq_len

    @classmethod
    def from_regions(
        cls,
        regions: list[TextConditioningRegions],
        key_seq_len: int,
        # TODO(ryand): Pass in a list of downscale factors?
        max_downscale_factor: int = 8,
    ):
        """Construct a `RegionalPromptData` object.

        Args:
            regions (list[TextConditioningRegions]): regions[i] contains the prompt regions for the i'th sample in the
                batch.
            key_seq_len (int): The sequence length of the expected prompt embeddings (which act as the key in the
                cross-attention layers). This is most likely equal to the max embedding range end, but we pass it
                explicitly to be sure.
        """
        attn_masks_by_seq_len = {}

        # batch_attn_mask_by_seq_len[b][s] contains the attention mask for the b'th batch sample with a query sequence
        # length of s.
        batch_attn_masks_by_seq_len: list[dict[int, torch.Tensor]] = []
        for batch_sample_regions in regions:
            batch_attn_masks_by_seq_len.append({})

            # Convert the bool masks to float masks so that max pooling can be applied.
            batch_masks = batch_sample_regions.masks.to(dtype=torch.float32)

            # Downsample the spatial dimensions by factors of 2 until max_downscale_factor is reached.
            downscale_factor = 1
            while downscale_factor <= max_downscale_factor:
                _, num_prompts, h, w = batch_masks.shape
                query_seq_len = h * w

                # Flatten the spatial dimensions of the mask by reshaping to (1, num_prompts, query_seq_len, 1).
                batch_query_masks = batch_masks.reshape((1, num_prompts, -1, 1))

                # Create a cross-attention mask for each prompt that selects the corresponding embeddings from
                # `encoder_hidden_states`.
                # attn_mask shape: (batch_size, query_seq_len, key_seq_len)
                # TODO(ryand): What device / dtype should this be?
                attn_mask = torch.zeros((1, query_seq_len, key_seq_len))

                for prompt_idx, embedding_range in enumerate(batch_sample_regions.ranges):
                    attn_mask[0, :, embedding_range.start : embedding_range.end] = batch_query_masks[
                        :, prompt_idx, :, :
                    ]

                batch_attn_masks_by_seq_len[-1][query_seq_len] = attn_mask

                downscale_factor *= 2
                if downscale_factor <= max_downscale_factor:
                    # We use max pooling because we downscale to a pretty low resolution, so we don't want small prompt
                    # regions to be lost entirely.
                    # TODO(ryand): In the future, we may want to experiment with other downsampling methods, and could
                    # potentially use a weighted mask rather than a binary mask.
                    batch_masks = F.max_pool2d(batch_masks, kernel_size=2, stride=2)

        # Merge the batch_attn_masks_by_seq_len into a single attn_masks_by_seq_len.
        for query_seq_len in batch_attn_masks_by_seq_len[0].keys():
            attn_masks_by_seq_len[query_seq_len] = torch.cat(
                [batch_attn_masks_by_seq_len[i][query_seq_len] for i in range(len(batch_attn_masks_by_seq_len))]
            )

        return cls(attn_masks_by_seq_len)

    def get_attn_mask(self, query_seq_len: int) -> torch.Tensor:
        """Get the attention mask for the given query sequence length (i.e. downscaling level).

        This is called during cross-attention, where query_seq_len is the length of the flattened spatial features, so
        it changes at each downscaling level in the model.

        key_seq_len is the length of the expected prompt embeddings.

        Returns:
            torch.Tensor: The masks.
                shape: (batch_size, query_seq_len, key_seq_len).
                dtype: float
                The mask is a binary mask with values of 0.0 and 1.0.
        """
        return self._attn_masks_by_seq_len[query_seq_len]
