from typing import Optional

import torch
import torchvision

from invokeai.backend.flux2.text_conditioning import Flux2RegionalTextConditioning, Flux2TextConditioning
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Range
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.mask import to_standard_float_mask


class Flux2RegionalPromptingExtension:
    """Manages regional prompting for FLUX.2 Klein.

    FLUX.2 Klein uses the diffusers `Flux2Transformer2DModel`, whose attention processors
    accept an additive `attention_mask` via `joint_attention_kwargs`. Unlike FLUX 1 where
    we own the forward and can swap masks per block, here a single mask is applied to
    every double- and single-stream block. The mask construction mirrors FLUX 1's
    'restricted' variant since the unified attention sequence order ([txt, img]) and the
    SDPA additive-mask semantics are identical.
    """

    def __init__(
        self,
        regional_text_conditioning: Flux2RegionalTextConditioning,
        restricted_attn_mask: torch.Tensor | None = None,
    ):
        self.regional_text_conditioning = regional_text_conditioning
        self.restricted_attn_mask = restricted_attn_mask

    def get_joint_attention_kwargs(self, dtype: torch.dtype) -> dict | None:
        """Return joint_attention_kwargs to pass to the transformer forward, or None.

        SDPA expects an additive mask (0 where attention is allowed, -inf where blocked)
        broadcastable to (B, H, S, S). We return shape (1, 1, S, S) so it broadcasts
        across batch and head dims.
        """
        if self.restricted_attn_mask is None:
            return None

        bool_mask = self.restricted_attn_mask.to(device=TorchDevice.choose_torch_device())
        additive_mask = torch.zeros_like(bool_mask, dtype=dtype)
        additive_mask.masked_fill_(~bool_mask, float("-inf"))
        return {"attention_mask": additive_mask.unsqueeze(0).unsqueeze(0)}

    @classmethod
    def from_text_conditionings(
        cls,
        text_conditionings: list[Flux2TextConditioning],
        img_seq_len: int,
    ) -> "Flux2RegionalPromptingExtension":
        regional_text_conditioning = cls._concat_regional_text_conditioning(text_conditionings)
        attn_mask = cls._prepare_restricted_attn_mask(regional_text_conditioning, img_seq_len)
        return cls(
            regional_text_conditioning=regional_text_conditioning,
            restricted_attn_mask=attn_mask,
        )

    @classmethod
    def _prepare_restricted_attn_mask(
        cls,
        regional_text_conditioning: Flux2RegionalTextConditioning,
        img_seq_len: int,
    ) -> torch.Tensor | None:
        """Build a restricted attention mask.

        - img self-attention is allowed only within regions
        - img regions only attend to txt within their own region, not to global prompts
        - background regions (uncovered area) attend to global prompts and themselves
        Concatenation order in attention: [txt_seq, img_seq] — matches FLUX 1.
        """
        background_region_mask: torch.Tensor | None = None
        for image_mask in regional_text_conditioning.image_masks:
            if image_mask is not None:
                if background_region_mask is None:
                    background_region_mask = torch.ones_like(image_mask)
                background_region_mask *= 1 - image_mask

        if background_region_mask is None:
            # No regional masks present — fall back to default attention.
            return None

        device = TorchDevice.choose_torch_device()
        txt_seq_len = regional_text_conditioning.txt_embeddings.shape[1]

        regional_attention_mask = torch.zeros(
            (txt_seq_len + img_seq_len, txt_seq_len + img_seq_len), device=device, dtype=torch.float16
        )

        for image_mask, embedding_range in zip(
            regional_text_conditioning.image_masks, regional_text_conditioning.embedding_ranges, strict=True
        ):
            # 1. txt attends to itself
            regional_attention_mask[
                embedding_range.start : embedding_range.end, embedding_range.start : embedding_range.end
            ] = 1.0

            if image_mask is not None:
                regional_attention_mask[embedding_range.start : embedding_range.end, txt_seq_len:] = image_mask.view(
                    1, img_seq_len
                )
                regional_attention_mask[txt_seq_len:, embedding_range.start : embedding_range.end] = image_mask.view(
                    img_seq_len, 1
                )
                image_mask_col = image_mask.view(img_seq_len, 1)
                regional_attention_mask[txt_seq_len:, txt_seq_len:] += image_mask_col @ image_mask_col.T
            else:
                # Global prompt: allowed to attend to / be attended to by background only.
                regional_attention_mask[embedding_range.start : embedding_range.end, txt_seq_len:] = (
                    background_region_mask.view(1, img_seq_len)
                )
                regional_attention_mask[txt_seq_len:, embedding_range.start : embedding_range.end] = (
                    background_region_mask.view(img_seq_len, 1)
                )

        # Allow background to attend to itself.
        regional_attention_mask[txt_seq_len:, txt_seq_len:] += background_region_mask.view(img_seq_len, 1)
        regional_attention_mask[txt_seq_len:, txt_seq_len:] += background_region_mask.view(1, img_seq_len)

        return regional_attention_mask > 0.5

    @classmethod
    def _concat_regional_text_conditioning(
        cls,
        text_conditionings: list[Flux2TextConditioning],
    ) -> Flux2RegionalTextConditioning:
        concat_embeds: list[torch.Tensor] = []
        concat_ranges: list[Range] = []
        image_masks: list[torch.Tensor | None] = []

        cur_len = 0
        for tc in text_conditionings:
            concat_embeds.append(tc.txt_embeddings)
            seq_len = tc.txt_embeddings.shape[1]
            concat_ranges.append(Range(start=cur_len, end=cur_len + seq_len))
            image_masks.append(tc.mask)
            cur_len += seq_len

        txt_embeddings = torch.cat(concat_embeds, dim=1)

        # 4D position IDs (T, H, W, L). FLUX.2 only varies L for text tokens.
        bs, total_len, _ = txt_embeddings.shape
        device = TorchDevice.choose_torch_device()
        txt_ids = torch.zeros(bs, total_len, 4, dtype=torch.long, device=device)
        txt_ids[..., 3] = torch.arange(total_len, device=device, dtype=torch.long)

        return Flux2RegionalTextConditioning(
            txt_embeddings=txt_embeddings,
            txt_ids=txt_ids,
            image_masks=image_masks,
            embedding_ranges=concat_ranges,
        )

    @staticmethod
    def preprocess_regional_prompt_mask(
        mask: Optional[torch.Tensor],
        packed_height: int,
        packed_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Resize a regional prompt mask to the packed latent grid and flatten to (1, 1, seq).

        If `mask` is None, return an all-ones mask of the expected shape.
        """
        if mask is None:
            return torch.ones((1, 1, packed_height * packed_width), dtype=dtype, device=device)

        mask = to_standard_float_mask(mask, out_dtype=dtype)
        tf = torchvision.transforms.Resize(
            (packed_height, packed_width), interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )
        mask = mask.unsqueeze(0)
        resized = tf(mask)
        return resized.flatten(start_dim=2).to(device=device)
