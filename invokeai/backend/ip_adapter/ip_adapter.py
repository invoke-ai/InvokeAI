# copied from https://github.com/tencent-ailab/IP-Adapter (Apache License 2.0)
#   and modified as needed

from contextlib import contextmanager
from typing import Optional, Union

import torch
from diffusers.models import UNet2DConditionModel

# FIXME: Getting errors when trying to use PyTorch 2.0 versions of IPAttnProcessor and AttnProcessor
#   so for now falling back to the default versions
# from .utils import is_torch2_available
# if is_torch2_available:
#     from .attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
# else:
#     from .attention_processor import IPAttnProcessor, AttnProcessor
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .attention_processor import AttnProcessor, IPAttnProcessor
from .resampler import Resampler


class ImageProjModel(torch.nn.Module):
    """Image Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    @classmethod
    def from_state_dict(cls, state_dict: dict[torch.Tensor], clip_extra_context_tokens=4):
        """Initialize an ImageProjModel from a state_dict.

        The cross_attention_dim and clip_embeddings_dim are inferred from the shape of the tensors in the state_dict.

        Args:
            state_dict (dict[torch.Tensor]): The state_dict of model weights.
            clip_extra_context_tokens (int, optional): Defaults to 4.

        Returns:
            ImageProjModel
        """
        cross_attention_dim = state_dict["norm.weight"].shape[0]
        clip_embeddings_dim = state_dict["proj.weight"].shape[-1]

        model = cls(cross_attention_dim, clip_embeddings_dim, clip_extra_context_tokens)

        model.load_state_dict(state_dict)
        return model

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class IPAdapter:
    """IP-Adapter: https://arxiv.org/pdf/2308.06721.pdf"""

    def __init__(
        self,
        state_dict: dict[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        num_tokens: int = 4,
    ):
        self.device = device
        self.dtype = dtype

        self._num_tokens = num_tokens

        self._clip_image_processor = CLIPImageProcessor()

        self._state_dict = state_dict

        self._image_proj_model = self._init_image_proj_model(self._state_dict["image_proj"])

        # The _attn_processors will be initialized later when we have access to the UNet.
        self._attn_processors = None

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype

        self._image_proj_model.to(device=self.device, dtype=self.dtype)
        if self._attn_processors is not None:
            torch.nn.ModuleList(self._attn_processors.values()).to(device=self.device, dtype=self.dtype)

    def _init_image_proj_model(self, state_dict):
        return ImageProjModel.from_state_dict(state_dict, self._num_tokens).to(self.device, dtype=self.dtype)

    def _prepare_attention_processors(self, unet: UNet2DConditionModel):
        """Prepare a dict of attention processors that can later be injected into a unet, and load the IP-Adapter
        attention weights into them.

        Note that the `unet` param is only used to determine attention block dimensions and naming.
        TODO(ryand): As a future improvement, this could all be inferred from the state_dict when the IPAdapter is
        intialized.
        """
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                ).to(self.device, dtype=self.dtype)

        ip_layers = torch.nn.ModuleList(attn_procs.values())
        ip_layers.load_state_dict(self._state_dict["ip_adapter"])
        self._attn_processors = attn_procs
        self._state_dict = None

    @contextmanager
    def apply_ip_adapter_attention(self, unet: UNet2DConditionModel, scale: int):
        """A context manager that patches `unet` with this IP-Adapter's attention processors while it is active.

        Yields:
            None
        """
        if self._attn_processors is None:
            # We only have to call _prepare_attention_processors(...) once, and then the result is cached and can be
            # used on any UNet model (with the same dimensions).
            self._prepare_attention_processors(unet)

        # Set scale.
        for attn_processor in self._attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

        orig_attn_processors = unet.attn_processors

        # Make a (moderately-) shallow copy of the self._attn_processors dict, because unet.set_attn_processor(...)
        # actually pops elements from the passed dict.
        ip_adapter_attn_processors = {k: v for k, v in self._attn_processors.items()}

        try:
            unet.set_attn_processor(ip_adapter_attn_processors)
            yield None
        finally:
            unet.set_attn_processor(orig_attn_processors)

    @torch.inference_mode()
    def get_image_embeds(self, pil_image, image_encoder: CLIPVisionModelWithProjection):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self._clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image_embeds = image_encoder(clip_image.to(self.device, dtype=self.dtype)).image_embeds
        image_prompt_embeds = self._image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self._image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def _init_image_proj_model(self, state_dict):
        return Resampler.from_state_dict(
            state_dict=state_dict,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self._num_tokens,
            ff_mult=4,
        ).to(self.device, dtype=self.dtype)

    @torch.inference_mode()
    def get_image_embeds(self, pil_image, image_encoder: CLIPVisionModelWithProjection):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self._clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=self.dtype)
        clip_image_embeds = image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self._image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[
            -2
        ]
        uncond_image_prompt_embeds = self._image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


def build_ip_adapter(
    ip_adapter_ckpt_path: str, device: torch.device, dtype: torch.dtype = torch.float16
) -> Union[IPAdapter, IPAdapterPlus]:
    state_dict = torch.load(ip_adapter_ckpt_path, map_location="cpu")

    # Determine if the state_dict is from an IPAdapter or IPAdapterPlus based on the image_proj weights that it
    # contains.
    is_plus = "proj.weight" not in state_dict["image_proj"]

    if is_plus:
        return IPAdapterPlus(state_dict, device=device, dtype=dtype)
    else:
        return IPAdapter(state_dict, device=device, dtype=dtype)
