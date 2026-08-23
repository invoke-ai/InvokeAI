"""VAE-encode a reference image into Krea-2 style-reference conditioning.

Training-free style transfer: the reference image's attention keys/values are spliced into the target's
in a band of transformer blocks, so the generation picks up the reference's palette, texture and
rendering without picking up its content. Ported from
https://github.com/nkxx188/ComfyUI-Krea2-StyleTransfer (MIT).

The reference is encoded at exactly the denoise node's resolution -- its image tokens are appended to the
target's and share the target's rotary embedding, so the token counts have to match.
"""

from typing import Literal

import einops
import torch
from PIL import Image as PILImage

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
    Krea2StyleReferenceField,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import Krea2StyleReferenceOutput
from invokeai.app.invocations.qwen_image_image_to_latents import QwenImageImageToLatentsInvocation
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.krea2.style_reference import KREA2_DEFAULT_STYLE_BLOCKS, KREA2_NUM_BLOCKS, parse_block_spec
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.util.devices import TorchDevice

Krea2StyleReferenceFit = Literal["crop", "contain", "stretch"]


def fit_image_to_box(image: PILImage.Image, width: int, height: int, fit: Krea2StyleReferenceFit) -> PILImage.Image:
    """Resize a reference image to exactly ``width`` x ``height``.

    ``crop`` scales to cover and center-crops (upstream's default -- it never introduces synthetic pixels,
    which matters because everything in the reference feeds the style statistics). ``contain`` scales to
    fit and letterboxes on white. ``stretch`` ignores the aspect ratio.
    """
    image = image.convert("RGB")
    source_width, source_height = image.size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("The style reference image has invalid dimensions.")

    if fit == "stretch":
        return image.resize((width, height), resample=PILImage.LANCZOS)

    if fit == "crop":
        scale = max(width / source_width, height / source_height)
        scaled = image.resize(
            (max(width, round(source_width * scale)), max(height, round(source_height * scale))),
            resample=PILImage.LANCZOS,
        )
        left = (scaled.width - width) // 2
        top = (scaled.height - height) // 2
        return scaled.crop((left, top, left + width, top + height))

    scale = min(width / source_width, height / source_height)
    scaled = image.resize(
        (max(1, min(width, round(source_width * scale))), max(1, min(height, round(source_height * scale)))),
        resample=PILImage.LANCZOS,
    )
    canvas = PILImage.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(scaled, ((width - scaled.width) // 2, (height - scaled.height) // 2))
    return canvas


@invocation(
    "krea2_style_reference",
    title="Style Reference - Krea-2",
    tags=["image", "conditioning", "krea2", "krea-2", "style"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Krea2StyleReferenceInvocation(BaseInvocation):
    """Encode a reference image into Krea-2 style-reference conditioning.

    Transfers the *look* of the reference image -- palette, texture, rendering -- while the prompt keeps
    driving the content. No adapter model or LoRA is involved.

    ``width`` and ``height`` must match the Krea-2 denoise node. Everything below ``style_strength`` is
    for tuning and should be left at its default; ``style_strength`` already modulates several of them.
    """

    image: ImageField = InputField(description="Reference image whose style should be transferred.")
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection, title="VAE")
    width: int = InputField(
        default=1024,
        gt=0,
        multiple_of=16,
        description="Width to encode the reference at (must match the denoise node's width).",
    )
    height: int = InputField(
        default=1024,
        gt=0,
        multiple_of=16,
        description="Height to encode the reference at (must match the denoise node's height).",
    )
    fit: Krea2StyleReferenceFit = InputField(
        default="crop",
        description="How to reconcile the reference's aspect ratio with the target size. 'crop' scales to "
        "cover and center-crops, 'contain' letterboxes on white, 'stretch' distorts.",
    )
    style_strength: float = InputField(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Overall style strength. 1.0 is the recommended setting. 0 makes the denoise node ignore "
        "the reference entirely - no capture pass, no retained keys/values - though this node still encodes "
        "it; disconnect the node to skip that too.",
    )
    blocks: str = InputField(
        default=KREA2_DEFAULT_STYLE_BLOCKS,
        description="Transformer blocks to inject the reference into, e.g. '7-27'. Styling the earliest "
        "blocks damages composition.",
        ui_order=10,
    )
    ref_k_strength: float = InputField(
        default=1.06,
        ge=0.0,
        le=5.0,
        description="Multiplier on the reference key path. This is the knob that makes the style visible "
        "without raising low_scale_end (which would also let reference content leak in).",
        ui_order=11,
    )
    adain_strength: float = InputField(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="How strongly the reference's query/key statistics are applied to the target.",
        ui_order=12,
    )
    value_mode: Literal["target", "raw_reference", "ref_mean", "target_adain", "target_adain_plus_ref"] = InputField(
        default="target_adain_plus_ref",
        description="How the reference value vectors are built.",
        ui_order=13,
    )
    value_adain_strength: float = InputField(
        default=0.65,
        ge=0.0,
        le=1.5,
        description="Reference statistics applied to the target value path. Has no effect while ref_value_mix is 1.0.",
        ui_order=14,
    )
    ref_value_mix: float = InputField(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How much raw reference value signal is kept. Higher usually preserves style material.",
        ui_order=15,
    )
    high_scale_start: float = InputField(
        default=1.04,
        description="Scale on the reference key's high-frequency bands at the first step.",
        ui_order=16,
    )
    high_scale_end: float = InputField(
        default=0.0,
        description="Scale on the reference key's high-frequency bands at the last step. 0 decays them "
        "away, which is what keeps reference content from leaking in.",
        ui_order=17,
    )
    low_scale_start: float = InputField(
        default=1.0,
        description="Scale on the reference key's low-frequency bands at the first step.",
        ui_order=18,
    )
    low_scale_end: float = InputField(
        default=1.10,
        description="Scale on the reference key's low-frequency bands at the last step. Raising this "
        "strengthens the style but also invites content leakage and quality loss.",
        ui_order=19,
    )
    beta: float = InputField(
        default=2.5,
        gt=0.0,
        le=20.0,
        description="Exponent of the high-to-low frequency falloff curve.",
        ui_order=20,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> Krea2StyleReferenceOutput:
        # Fail on a malformed block spec here rather than several nodes later in the denoise loop.
        parse_block_spec(self.blocks, KREA2_NUM_BLOCKS)

        image = context.images.get_pil(self.image.image_name, "RGB")
        image = fit_image_to_box(image, self.width, self.height, self.fit)

        # multiple_of=16 keeps the post-VAE latents even, which the transformer's 2x2 patch packing needs.
        # width/height are already multiples of 16, so this only converts and normalizes to [-1, 1].
        image_tensor = image_resized_to_grid_as_tensor(image, multiple_of=16)
        if image_tensor.dim() == 3:
            image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

        vae_info = context.models.load(self.vae.vae)
        context.util.signal_progress("VAE-encoding style reference")
        # Reuse the Qwen-Image encoder: Krea-2 shares its VAE, and this already applies the per-channel
        # latents_mean/latents_std normalization the transformer expects.
        latents = QwenImageImageToLatentsInvocation.vae_encode(vae_info=vae_info, image_tensor=image_tensor)

        latents = latents.detach().to("cpu")
        # Release the encode intermediates before the denoise node partial-loads the transformer.
        TorchDevice.empty_cache()
        name = context.tensors.save(tensor=latents)

        return Krea2StyleReferenceOutput.build(
            Krea2StyleReferenceField(
                reference_latents_name=name,
                width=self.width,
                height=self.height,
                style_strength=self.style_strength,
                blocks=self.blocks,
                ref_k_strength=self.ref_k_strength,
                adain_strength=self.adain_strength,
                value_mode=self.value_mode,
                value_adain_strength=self.value_adain_strength,
                ref_value_mix=self.ref_value_mix,
                high_scale_start=self.high_scale_start,
                high_scale_end=self.high_scale_end,
                low_scale_start=self.low_scale_start,
                low_scale_end=self.low_scale_end,
                beta=self.beta,
            )
        )
