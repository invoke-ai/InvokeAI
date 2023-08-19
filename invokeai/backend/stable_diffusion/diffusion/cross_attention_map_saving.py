import math

import PIL
import torch
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import resize as tv_resize

from .cross_attention_control import CrossAttentionType, get_cross_attention_modules


class AttentionMapSaver:
    def __init__(self, token_ids: range, latents_shape: torch.Size):
        self.token_ids = token_ids
        self.latents_shape = latents_shape
        # self.collated_maps = #torch.zeros([len(token_ids), latents_shape[0], latents_shape[1]])
        self.collated_maps = {}

    def clear_maps(self):
        self.collated_maps = {}

    def add_attention_maps(self, maps: torch.Tensor, key: str):
        """
        Accumulate the given attention maps and store by summing with existing maps at the passed-in key (if any).
        :param maps: Attention maps to store. Expected shape [A, (H*W), N] where A is attention heads count, H and W are the map size (fixed per-key) and N is the number of tokens (typically 77).
        :param key: Storage key. If a map already exists for this key it will be summed with the incoming data. In this case the maps sizes (H and W) should match.
        :return: None
        """
        key_and_size = f"{key}_{maps.shape[1]}"

        # extract desired tokens
        maps = maps[:, :, self.token_ids]

        # merge attention heads to a single map per token
        maps = torch.sum(maps, 0)

        # store
        if key_and_size not in self.collated_maps:
            self.collated_maps[key_and_size] = torch.zeros_like(maps, device="cpu")
        self.collated_maps[key_and_size] += maps.cpu()

    def write_maps_to_disk(self, path: str):
        pil_image = self.get_stacked_maps_image()
        pil_image.save(path, "PNG")

    def get_stacked_maps_image(self) -> PIL.Image:
        """
        Scale all collected attention maps to the same size, blend them together and return as an image.
        :return: An image containing a vertical stack of blended attention maps, one for each requested token.
        """
        num_tokens = len(self.token_ids)
        if num_tokens == 0:
            return None

        latents_height = self.latents_shape[0]
        latents_width = self.latents_shape[1]

        merged = None

        for key, maps in self.collated_maps.items():
            # maps has shape [(H*W), N] for N tokens
            # but we want [N, H, W]
            this_scale_factor = math.sqrt(maps.shape[0] / (latents_width * latents_height))
            this_maps_height = int(float(latents_height) * this_scale_factor)
            this_maps_width = int(float(latents_width) * this_scale_factor)
            # and we need to do some dimension juggling
            maps = torch.reshape(
                torch.swapdims(maps, 0, 1),
                [num_tokens, this_maps_height, this_maps_width],
            )

            # scale to output size if necessary
            if this_scale_factor != 1:
                maps = tv_resize(maps, [latents_height, latents_width], InterpolationMode.BICUBIC)

            # normalize
            maps_min = torch.min(maps)
            maps_range = torch.max(maps) - maps_min
            # print(f"map {key} size {[this_maps_width, this_maps_height]} range {[maps_min, maps_min + maps_range]}")
            maps_normalized = (maps - maps_min) / maps_range
            # expand to (-0.1, 1.1) and clamp
            maps_normalized_expanded = maps_normalized * 1.1 - 0.05
            maps_normalized_expanded_clamped = torch.clamp(maps_normalized_expanded, 0, 1)

            # merge together, producing a vertical stack
            maps_stacked = torch.reshape(
                maps_normalized_expanded_clamped,
                [num_tokens * latents_height, latents_width],
            )

            if merged is None:
                merged = maps_stacked
            else:
                # screen blend
                merged = 1 - (1 - maps_stacked) * (1 - merged)

        if merged is None:
            return None

        merged_bytes = merged.mul(0xFF).byte()
        return PIL.Image.fromarray(merged_bytes.numpy(), mode="L")
