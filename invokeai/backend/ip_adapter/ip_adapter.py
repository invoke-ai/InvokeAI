# copied from https://github.com/tencent-ailab/IP-Adapter (Apache License 2.0)
#   and modified as needed

from contextlib import contextmanager

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
        unet: UNet2DConditionModel,
        image_encoder_path: str,
        ip_adapter_ckpt_path: str,
        device: torch.device,
        num_tokens: int = 4,
    ):
        self._unet = unet
        self._device = device
        self._image_encoder_path = image_encoder_path
        self._ip_adapter_ckpt_path = ip_adapter_ckpt_path
        self._num_tokens = num_tokens

        self._attn_processors = self._prepare_attention_processors()

        # load image encoder
        self._image_encoder = CLIPVisionModelWithProjection.from_pretrained(self._image_encoder_path).to(
            self._device, dtype=torch.float16
        )
        self._clip_image_processor = CLIPImageProcessor()
        # image proj model
        self._image_proj_model = self._init_image_proj_model()

        self._load_weights()

    def _init_image_proj_model(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self._unet.config.cross_attention_dim,
            clip_embeddings_dim=self._image_encoder.config.projection_dim,
            clip_extra_context_tokens=self._num_tokens,
        ).to(self._device, dtype=torch.float16)
        return image_proj_model

    def _prepare_attention_processors(self):
        """Creates a dict of attention processors that can later be injected into `self.unet`, and loads the IP-Adapter
        attention weights into them.
        """
        attn_procs = {}
        for name in self._unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self._unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self._unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self._unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self._unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                ).to(self._device, dtype=torch.float16)
        return attn_procs

    @contextmanager
    def apply_ip_adapter_attention(self):
        """A context manager that patches `self._unet` with this IP-Adapter's attention processors while it is active.

        Yields:
            None
        """
        orig_attn_processors = self._unet.attn_processors
        try:
            self._unet.set_attn_processor(self._attn_processors)
            yield None
        finally:
            self._unet.set_attn_processor(orig_attn_processors)

    def _load_weights(self):
        state_dict = torch.load(self._ip_adapter_ckpt_path, map_location="cpu")
        self._image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self._attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self._clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image_embeds = self._image_encoder(clip_image.to(self._device, dtype=torch.float16)).image_embeds
        image_prompt_embeds = self._image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self._image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self._attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale


class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def _init_image_proj_model(self):
        image_proj_model = Resampler(
            dim=self._unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self._num_tokens,
            embedding_dim=self._image_encoder.config.hidden_size,
            output_dim=self._unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self._device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self._clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self._device, dtype=torch.float16)
        clip_image_embeds = self._image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self._image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self._image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self._image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds
