from typing import Any, Literal, Optional, Sequence, Union

import torch
from pydantic import BaseModel, ConfigDict, Field

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    InputField,
    LatentsField,
    MetadataField,
    OutputField,
    UIType,
    VideoField,
)
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.primitives import LatentsOutput, StringOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.controlnet_utils import CONTROLNET_MODE_VALUES, CONTROLNET_RESIZE_VALUES
from invokeai.version.invokeai_version import __version__


class MetadataItemField(BaseModel):
    label: str = Field(description=FieldDescriptions.metadata_item_label)
    value: Any = Field(description=FieldDescriptions.metadata_item_value)


class LoRAMetadataField(BaseModel):
    """LoRA Metadata Field"""

    model: ModelIdentifierField = Field(description=FieldDescriptions.lora_model)
    weight: float = Field(description=FieldDescriptions.lora_weight)


class IPAdapterMetadataField(BaseModel):
    """IP Adapter Field, minus the CLIP Vision Encoder model"""

    image: ImageField = Field(description="The IP-Adapter image prompt.")
    ip_adapter_model: ModelIdentifierField = Field(description="The IP-Adapter model.")
    clip_vision_model: Literal["ViT-L", "ViT-H", "ViT-G"] = Field(description="The CLIP Vision model")
    method: Literal["full", "style", "composition", "style_strong", "style_precise"] = Field(
        description="Method to apply IP Weights with"
    )
    weight: Union[float, list[float]] = Field(description="The weight given to the IP-Adapter")
    begin_step_percent: float = Field(description="When the IP-Adapter is first applied (% of total steps)")
    end_step_percent: float = Field(description="When the IP-Adapter is last applied (% of total steps)")


class T2IAdapterMetadataField(BaseModel):
    image: ImageField = Field(description="The control image.")
    processed_image: Optional[ImageField] = Field(default=None, description="The control image, after processing.")
    t2i_adapter_model: ModelIdentifierField = Field(description="The T2I-Adapter model to use.")
    weight: Union[float, list[float]] = Field(default=1, description="The weight given to the T2I-Adapter")
    begin_step_percent: float = Field(
        default=0, ge=0, le=1, description="When the T2I-Adapter is first applied (% of total steps)"
    )
    end_step_percent: float = Field(
        default=1, ge=0, le=1, description="When the T2I-Adapter is last applied (% of total steps)"
    )
    resize_mode: CONTROLNET_RESIZE_VALUES = Field(default="just_resize", description="The resize mode to use")


class ControlNetMetadataField(BaseModel):
    image: ImageField = Field(description="The control image")
    processed_image: Optional[ImageField] = Field(default=None, description="The control image, after processing.")
    control_model: ModelIdentifierField = Field(description="The ControlNet model to use")
    control_weight: Union[float, list[float]] = Field(default=1, description="The weight given to the ControlNet")
    begin_step_percent: float = Field(
        default=0, ge=0, le=1, description="When the ControlNet is first applied (% of total steps)"
    )
    end_step_percent: float = Field(
        default=1, ge=0, le=1, description="When the ControlNet is last applied (% of total steps)"
    )
    control_mode: CONTROLNET_MODE_VALUES = Field(default="balanced", description="The control mode to use")
    resize_mode: CONTROLNET_RESIZE_VALUES = Field(default="just_resize", description="The resize mode to use")


@invocation_output("metadata_item_output")
class MetadataItemOutput(BaseInvocationOutput):
    """Metadata Item Output"""

    item: MetadataItemField = OutputField(description="Metadata Item")


@invocation("metadata_item", title="Metadata Item", tags=["metadata"], category="metadata", version="1.0.1")
class MetadataItemInvocation(BaseInvocation):
    """Used to create an arbitrary metadata item. Provide "label" and make a connection to "value" to store that data as the value."""

    label: str = InputField(description=FieldDescriptions.metadata_item_label)
    value: Any = InputField(description=FieldDescriptions.metadata_item_value, ui_type=UIType.Any)

    def invoke(self, context: InvocationContext) -> MetadataItemOutput:
        return MetadataItemOutput(item=MetadataItemField(label=self.label, value=self.value))


@invocation_output("metadata_output")
class MetadataOutput(BaseInvocationOutput):
    metadata: MetadataField = OutputField(description="Metadata Dict")


@invocation_output("latents_meta_output")
class LatentsMetaOutput(LatentsOutput, MetadataOutput):
    """Base class for invocations that output latents with metadata"""

    @classmethod
    def build(
        cls,
        latents_name: str,
        latents: torch.Tensor,
        seed: Optional[int] = None,
        metadata: Optional[MetadataField] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> "LatentsMetaOutput":
        return cls(
            latents=LatentsField(latents_name=latents_name, seed=seed),
            width=width if width is not None else latents.shape[-1] * LATENT_SCALE_FACTOR,
            height=height if height is not None else latents.shape[-2] * LATENT_SCALE_FACTOR,
            metadata=metadata or MetadataField.model_validate({}),
        )


@invocation("metadata", title="Metadata", tags=["metadata"], category="metadata", version="1.0.1")
class MetadataInvocation(BaseInvocation):
    """Takes a MetadataItem or collection of MetadataItems and outputs a MetadataDict."""

    items: Union[list[MetadataItemField], MetadataItemField] = InputField(
        description=FieldDescriptions.metadata_item_polymorphic
    )

    def invoke(self, context: InvocationContext) -> MetadataOutput:
        if isinstance(self.items, MetadataItemField):
            # single metadata item
            data = {self.items.label: self.items.value}
        else:
            # collection of metadata items
            data = {item.label: item.value for item in self.items}

        # add app version
        data.update({"app_version": __version__, "metadata_version": CORE_METADATA_VERSION})
        return MetadataOutput(metadata=MetadataField.model_validate(data))


@invocation("merge_metadata", title="Metadata Merge", tags=["metadata"], category="metadata", version="1.0.1")
class MergeMetadataInvocation(BaseInvocation):
    """Merged a collection of MetadataDict into a single MetadataDict."""

    collection: list[MetadataField] = InputField(description=FieldDescriptions.metadata_collection)

    def invoke(self, context: InvocationContext) -> MetadataOutput:
        data = {}
        for item in self.collection:
            data.update(item.model_dump())

        return MetadataOutput(metadata=MetadataField.model_validate(data))


GENERATION_MODES = Literal[
    "txt2img",
    "img2img",
    "inpaint",
    "outpaint",
    "sdxl_txt2img",
    "sdxl_img2img",
    "sdxl_inpaint",
    "sdxl_outpaint",
    "flux_txt2img",
    "flux_img2img",
    "flux_inpaint",
    "flux_outpaint",
    "flux2_txt2img",
    "flux2_img2img",
    "flux2_inpaint",
    "flux2_outpaint",
    "sd3_txt2img",
    "sd3_img2img",
    "sd3_inpaint",
    "sd3_outpaint",
    "cogview4_txt2img",
    "cogview4_img2img",
    "cogview4_inpaint",
    "cogview4_outpaint",
    "z_image_txt2img",
    "z_image_img2img",
    "z_image_inpaint",
    "z_image_outpaint",
    "ernie_image_txt2img",
    "ideogram4_txt2img",
    "qwen_image_txt2img",
    "qwen_image_img2img",
    "qwen_image_inpaint",
    "qwen_image_outpaint",
    "anima_txt2img",
    "anima_img2img",
    "anima_inpaint",
    "anima_outpaint",
    "krea2_txt2img",
    "krea2_img2img",
    "krea2_inpaint",
    "krea2_outpaint",
    "qwen_image_2_1_txt2img",
    "qwen_image_2_1_img2img",
    "qwen_image_2_1_inpaint",
    "qwen_image_2_1_outpaint",
    "moss_image_txt2img",
    "moss_image_img2img",
    "moss_image_inpaint",
    "moss_image_outpaint",
    "wan_txt2img",
    "wan_img2img",
    "wan_inpaint",
    "wan_outpaint",
    "wan_t2v",
    "wan_i2v",
    "wan_interpolate",
    "wan_extend_video",
    "minimax_h3_t2v",
    "minimax_h3_i2v",
    "ltx2_t2v",
    "ltx2_i2v",
    "ltx2_a2v",
    "ltx2_v2a",
    "ltx2_lf2v",
    "ltx2_flf2v",
    "ltx2_extend_video",
    "minimax_h3_lf2v",
    "minimax_h3_flf2v",
    "minimax_h3_extend_video",
    "minimax_h3_ref2v",
    "minimax_h3_txt2img",
]


class MiniMaxH3ReferenceMetadataField(BaseModel):
    """One recorded Ref2VA reference: names and options only, in conditioning order."""

    kind: Literal["image", "video"] = Field(description="Whether the reference was an image or a video.")
    image_name: Optional[str] = Field(default=None, description="The reference image's name.")
    video_name: Optional[str] = Field(default=None, description="The reference video's name.")
    conditioning: Optional[str] = Field(
        default=None, description="A video reference's conditioning choice (video_audio / video / audio)."
    )
    detail: Optional[str] = Field(default=None, description="An image reference's sizing choice (max / match).")
    start_frame: Optional[int] = Field(default=None, description="A video reference's trim start (inclusive).")
    end_frame: Optional[int] = Field(default=None, description="A video reference's trim end (inclusive).")


# The version of the metadata record `core_metadata` emits, stamped as `metadata_version`. It is
# semver over the *record*, not this node: a minor bump adds keys (any 1.x reader parses any
# other 1.x, ignoring unknown keys); a major bump renames, removes or retypes one. A record
# without the key predates versioning and is read with the same rules. Documented, with the
# field table, in docs/development/Architecture/media-metadata.
CORE_METADATA_VERSION = "1.1.0"


def serialize_lora_metadata(*lora_lists: Optional[Sequence[Any]]) -> list[dict[str, Any]]:
    """Serialize and deduplicate LoRA field sequences into metadata dictionaries."""
    seen: set[tuple[Any, float]] = set()
    result: list[dict[str, Any]] = []
    for lora_list in lora_lists:
        if not lora_list:
            continue
        for item in lora_list:
            key = getattr(item.lora, "key", item.lora)
            ident = (key, item.weight)
            if ident in seen:
                continue
            seen.add(ident)
            if isinstance(item.lora, ModelIdentifierField):
                result.append(
                    LoRAMetadataField(model=item.lora, weight=item.weight).model_dump(
                        exclude_none=True, exclude={"id", "type", "is_intermediate", "use_cache"}
                    )
                )
            else:
                result.append({"model": item.lora, "weight": item.weight})
    return result


def serialize_adapter_metadata(item_or_items: Any) -> list[Any]:
    """Normalize a single adapter field or list of adapter fields into serialized metadata dicts."""
    if item_or_items is None:
        return []
    items = item_or_items if isinstance(item_or_items, list) else [item_or_items]
    return [
        item.model_dump(exclude_none=True, exclude={"id", "type", "is_intermediate", "use_cache"})
        if hasattr(item, "model_dump")
        else item
        for item in items
    ]


def serialize_ip_adapter_ref_images(
    ip_adapter: Any,
    context: Optional[Any] = None,
    default_width: int = 1024,
    default_height: int = 1024,
) -> list[dict[str, Any]]:
    """Serialize IP-Adapter inputs into v7 GenerateReferenceImage metadata entries (`ref_images`)."""
    if ip_adapter is None:
        return []
    items = ip_adapter if isinstance(ip_adapter, list) else [ip_adapter]
    ref_images: list[dict[str, Any]] = []
    idx = 0
    for item in items:
        raw_image = item.get("image") if isinstance(item, dict) else getattr(item, "image", None)
        if raw_image is None:
            raw_image_name = item.get("image_name") if isinstance(item, dict) else getattr(item, "image_name", None)
            if raw_image_name:
                raw_image = {"image_name": raw_image_name}
            else:
                continue
        img_list = raw_image if isinstance(raw_image, list) else [raw_image]
        ip_adapter_model = (
            item.get("ip_adapter_model") if isinstance(item, dict) else getattr(item, "ip_adapter_model", None)
        )
        image_encoder_model = (
            item.get("image_encoder_model") if isinstance(item, dict) else getattr(item, "image_encoder_model", None)
        )
        if hasattr(ip_adapter_model, "model_dump"):
            model_dict = ip_adapter_model.model_dump(exclude_none=True, exclude={"id", "is_intermediate", "use_cache"})
        elif isinstance(ip_adapter_model, dict):
            model_dict = dict(ip_adapter_model)
            model_dict.setdefault("type", "ip_adapter")
        else:
            model_dict = {}

        explicit_cvm = (
            item.get("clip_vision_model") if isinstance(item, dict) else getattr(item, "clip_vision_model", None)
        )
        if explicit_cvm in ("ViT-H", "ViT-G", "ViT-L"):
            clip_vision_model = explicit_cvm
        else:
            enc_name = (
                image_encoder_model.get("name", "")
                if isinstance(image_encoder_model, dict)
                else getattr(image_encoder_model, "name", "")
            ) or ""
            enc_lower = str(enc_name).lower()
            if "vit-l" in enc_lower or "large" in enc_lower:
                clip_vision_model = "ViT-L"
            elif "sdxl" in enc_lower or "vit-g" in enc_lower:
                clip_vision_model = "ViT-G"
            else:
                clip_vision_model = "ViT-H"

        method = (item.get("method") if isinstance(item, dict) else getattr(item, "method", None)) or "full"
        has_weight = ("weight" in item) if isinstance(item, dict) else hasattr(item, "weight")
        has_strength = ("strength" in item) if isinstance(item, dict) else hasattr(item, "strength")
        if has_weight:
            raw_weight = item["weight"] if isinstance(item, dict) else item.weight
        elif has_strength:
            raw_weight = item["strength"] if isinstance(item, dict) else item.strength
            if raw_weight is not None and float(raw_weight) <= 0.0:
                continue
        else:
            raw_weight = 1.0
        weight = float(raw_weight[0] if isinstance(raw_weight, list) and raw_weight else raw_weight)
        begin_step_percent = float(
            item.get("begin_step_percent", 0.0) if isinstance(item, dict) else getattr(item, "begin_step_percent", 0.0)
        )
        end_step_percent = float(
            item.get("end_step_percent", 1.0) if isinstance(item, dict) else getattr(item, "end_step_percent", 1.0)
        )

        for img in img_list:
            img_name = img.get("image_name") if isinstance(img, dict) else getattr(img, "image_name", None)
            if not img_name:
                continue
            img_width, img_height = int(default_width), int(default_height)
            if context is not None:
                try:
                    dto = context.images.get_dto(img_name)
                    dto_w = getattr(dto, "width", None)
                    dto_h = getattr(dto, "height", None)
                    if isinstance(dto_w, int) and dto_w > 0 and isinstance(dto_h, int) and dto_h > 0:
                        img_width, img_height = int(dto_w), int(dto_h)
                except Exception:
                    pass
            idx += 1
            ref_images.append(
                {
                    "id": f"ip_adapter_{idx}",
                    "isEnabled": True,
                    "config": {
                        "type": "ip_adapter",
                        "image": {
                            "original": {
                                "image": {
                                    "image_name": img_name,
                                    "width": img_width,
                                    "height": img_height,
                                }
                            }
                        },
                        "model": model_dict,
                        "clipVisionModel": clip_vision_model,
                        "method": method,
                        "weight": weight,
                        "beginEndStepPct": [begin_step_percent, end_step_percent],
                    },
                }
            )
    return ref_images


def _ref_image_dedup_key(entry: Any) -> Optional[tuple[Any, Any, Any]]:
    if not isinstance(entry, dict):
        return None
    cfg = entry.get("config")
    if not isinstance(cfg, dict):
        return None
    cfg_type = cfg.get("type")
    img = cfg.get("image")
    image_name = None
    if isinstance(img, dict):
        crop = img.get("crop")
        orig = img.get("original")
        if isinstance(crop, dict) and isinstance(crop.get("image"), dict):
            image_name = crop["image"].get("image_name")
        if not image_name and isinstance(orig, dict) and isinstance(orig.get("image"), dict):
            image_name = orig["image"].get("image_name")
        if not image_name:
            image_name = img.get("image_name")
    if not image_name:
        return None
    model = cfg.get("model")
    model_key = model.get("key") if isinstance(model, dict) else getattr(model, "key", None)
    return (cfg_type, image_name, model_key)


def build_denoise_metadata(existing: Optional[MetadataField], updates: dict[str, Any]) -> MetadataField:
    """Build a new MetadataField by copying existing metadata and merging denoise updates without mutating input."""
    md: dict[str, Any] = {} if existing is None else existing.root.copy()
    existing_gen_mode = md.get("generation_mode")
    existing_ref_images = md.get("ref_images")
    md.update(updates)
    if existing_gen_mode is not None and "generation_mode" in updates:
        md["generation_mode"] = existing_gen_mode
    if "ref_images" in updates and isinstance(existing_ref_images, list) and isinstance(updates["ref_images"], list):
        merged_refs: list[Any] = []
        seen_keys: set[tuple[Any, Any, Any]] = set()
        for entry in [*existing_ref_images, *updates["ref_images"]]:
            dedup_key = _ref_image_dedup_key(entry)
            if dedup_key is not None:
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
            merged_refs.append(entry)
        md["ref_images"] = merged_refs
    md["app_version"] = __version__
    md["metadata_version"] = CORE_METADATA_VERSION
    return MetadataField.model_validate(md)


@invocation(
    "core_metadata",
    title="Core Metadata",
    tags=["metadata"],
    category="metadata",
    version="2.8.0",
    classification=Classification.Internal,
)
class CoreMetadataInvocation(BaseInvocation):
    """Used internally by Invoke to collect metadata for generations."""

    generation_mode: Optional[GENERATION_MODES] = InputField(
        default=None,
        description="The generation mode that output this image",
    )
    positive_prompt: Optional[str] = InputField(default=None, description="The positive prompt parameter")
    negative_prompt: Optional[str] = InputField(default=None, description="The negative prompt parameter")
    width: Optional[int] = InputField(default=None, description="The width parameter")
    height: Optional[int] = InputField(default=None, description="The height parameter")
    seed: Optional[int] = InputField(default=None, description="The seed used for noise generation")
    rand_device: Optional[str] = InputField(default=None, description="The device used for random number generation")
    cfg_scale: Optional[float] = InputField(default=None, description="The classifier-free guidance scale parameter")
    cfg_rescale_multiplier: Optional[float] = InputField(
        default=None, description=FieldDescriptions.cfg_rescale_multiplier
    )
    steps: Optional[int] = InputField(default=None, description="The number of steps used for inference")
    scheduler: Optional[str] = InputField(default=None, description="The scheduler used for inference")
    seamless_x: Optional[bool] = InputField(default=None, description="Whether seamless tiling was used on the X axis")
    seamless_y: Optional[bool] = InputField(default=None, description="Whether seamless tiling was used on the Y axis")
    clip_skip: Optional[int] = InputField(
        default=None,
        description="The number of skipped CLIP layers",
    )
    model: Optional[ModelIdentifierField] = InputField(default=None, description="The main model used for inference")
    controlnets: Optional[list[ControlNetMetadataField]] = InputField(
        default=None, description="The ControlNets used for inference"
    )
    ipAdapters: Optional[list[IPAdapterMetadataField]] = InputField(
        default=None, description="The IP Adapters used for inference"
    )
    t2iAdapters: Optional[list[T2IAdapterMetadataField]] = InputField(
        default=None, description="The IP Adapters used for inference"
    )
    loras: Optional[list[LoRAMetadataField]] = InputField(default=None, description="The LoRAs used for inference")
    strength: Optional[float] = InputField(
        default=None,
        description="The strength used for latents-to-latents",
    )
    init_image: Optional[str] = InputField(
        default=None,
        description="The name of the initial image",
    )
    vae: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="The VAE used for decoding, if the main model's default was not used",
    )
    qwen3_encoder: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="The Qwen3 text encoder model used for Z-Image inference",
    )
    # FLUX.2 [dev] and ERNIE-Image use Mistral-family text encoders where FLUX.2 Klein uses Qwen3, so they
    # need their own slot rather than reusing `qwen3_encoder` - the two are never both present on one image.
    mistral_encoder: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="The Mistral text encoder model used for FLUX.2 [dev] or ERNIE-Image inference",
    )
    # Ideogram 4 assembles its structured JSON caption at generation time (ideogram4_caption_builder),
    # so this is a declared field rather than a static extra: the graph wires the builder's output to it
    # via an edge, capturing the exact caption encoded for each (possibly batched) image.
    ideogram4_caption: Optional[str] = InputField(
        default=None,
        description="The structured JSON caption encoded for Ideogram 4 inference",
    )
    # MOSS-Image places canvas regions on the prompt at generation time (moss_image_layout_prompt), so, as for
    # `ideogram4_caption`, the graph records the layout prompt actually encoded through an edge.
    moss_image_layout_prompt: Optional[str] = InputField(
        default=None,
        description="The JSON layout prompt encoded for MOSS-Image inference",
    )

    # Video generation. Frame count and keyframes have no home among the image fields, and
    # the two MiniMax H3 single-file overrides are separate models from the main install, so
    # `model` alone would not identify what actually ran.
    num_frames: Optional[int] = InputField(
        default=None,
        description="The number of video frames generated",
    )
    first_frame_image: Optional[ImageField] = InputField(
        default=None,
        description="The image used as the video's first frame",
    )
    last_frame_image: Optional[ImageField] = InputField(
        default=None,
        description="The image used as the video's last frame",
    )
    source_video: Optional[VideoField] = InputField(
        default=None,
        description="The video this generation was derived from, e.g. the clip an extend workflow continues",
    )
    source_video_start_frame: Optional[int] = InputField(
        default=None,
        description="The first frame (inclusive) of the source video that was kept",
    )
    source_video_end_frame: Optional[int] = InputField(
        default=None,
        description="The last frame (inclusive) of the source video that was kept",
    )
    ltx2_context_frames: Optional[int] = InputField(
        default=None,
        description="Frames of the source an LTX-2 continuation opened with, which the join then crossfaded "
        "out of both halves",
    )
    fps: Optional[int] = InputField(
        default=None,
        description="The frame rate of the generated video",
    )
    wan_guidance_scale_low_noise: Optional[float] = InputField(
        default=None,
        description="The classifier-free guidance scale used by the Wan low-noise expert, when it differed",
    )
    wan_t5_encoder_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="The standalone UMT5-XXL encoder used with a single-file Wan main model",
    )
    wan_transformer_low_noise: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="The standalone low-noise expert used with a single-file Wan main model",
    )
    wan_component_source: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="The Wan Diffusers install whose VAE and encoder served a single-file main model",
    )
    minimax_h3_transformer_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="The single-file MiniMax H3 transformer used in place of the main model's transformer",
    )
    minimax_h3_text_encoder_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="The single-file MiniMax H3 Qwen3-VL text encoder used in place of the main model's",
    )
    minimax_h3_component_source: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="The MiniMax H3 Diffusers install whose VAEs and encoder served a single-file transformer",
    )
    minimax_h3_hybrid_base_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="The FL2VA transformer whose later blocks overlaid the Ref2VA transformer (the hybrid)",
    )
    minimax_h3_hybrid_start_block: Optional[int] = InputField(
        default=None,
        description="The first transformer block taken from the hybrid base",
    )
    minimax_h3_references: Optional[list[MiniMaxH3ReferenceMetadataField]] = InputField(
        default=None,
        description="The ordered Ref2VA references this generation was conditioned on (names and options only; "
        "the media is re-resolved from the gallery at recall time)",
    )

    # High resolution fix metadata.
    hrf_enabled: Optional[bool] = InputField(
        default=None,
        description="Whether or not high resolution fix was enabled.",
    )
    # TODO: should this be stricter or do we just let the UI handle it?
    hrf_method: Optional[str] = InputField(
        default=None,
        description="The high resolution fix upscale method.",
    )
    hrf_strength: Optional[float] = InputField(
        default=None,
        description="The high resolution fix img2img strength used in the upscale pass.",
    )

    # SDXL
    positive_style_prompt: Optional[str] = InputField(
        default=None,
        description="The positive style prompt parameter",
    )
    negative_style_prompt: Optional[str] = InputField(
        default=None,
        description="The negative style prompt parameter",
    )

    # SDXL Refiner
    refiner_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="The SDXL Refiner model used",
    )
    refiner_cfg_scale: Optional[float] = InputField(
        default=None,
        description="The classifier-free guidance scale parameter used for the refiner",
    )
    refiner_steps: Optional[int] = InputField(
        default=None,
        description="The number of steps used for the refiner",
    )
    refiner_scheduler: Optional[str] = InputField(
        default=None,
        description="The scheduler used for the refiner",
    )
    refiner_positive_aesthetic_score: Optional[float] = InputField(
        default=None,
        description="The aesthetic score used for the refiner",
    )
    refiner_negative_aesthetic_score: Optional[float] = InputField(
        default=None,
        description="The aesthetic score used for the refiner",
    )
    refiner_start: Optional[float] = InputField(
        default=None,
        description="The start value used for refiner denoising",
    )

    def invoke(self, context: InvocationContext) -> MetadataOutput:
        """Collects and outputs a CoreMetadata object"""

        as_dict = self.model_dump(exclude_none=True, exclude={"id", "type", "is_intermediate", "use_cache"})
        as_dict["app_version"] = __version__
        as_dict["metadata_version"] = CORE_METADATA_VERSION

        return MetadataOutput(metadata=MetadataField.model_validate(as_dict))

    model_config = ConfigDict(extra="allow")


@invocation(
    "metadata_field_extractor",
    title="Metadata Field Extractor",
    tags=["metadata"],
    category="metadata",
    version="1.0.0",
    classification=Classification.Deprecated,
)
class MetadataFieldExtractorInvocation(BaseInvocation):
    """Extracts the text value from an image's metadata given a key.
    Raises an error if the image has no metadata or if the value is not a string (nesting not permitted)."""

    image: ImageField = InputField(description="The image to extract metadata from")
    key: str = InputField(description="The key in the image's metadata to extract the value from")

    def invoke(self, context: InvocationContext) -> StringOutput:
        image_name = self.image.image_name

        metadata = context.images.get_metadata(image_name=image_name)
        if not metadata:
            raise ValueError(f"No metadata found on image {image_name}")

        try:
            val = metadata.root[self.key]
            if not isinstance(val, str):
                raise ValueError(f"Metadata at key '{self.key}' must be a string")
            return StringOutput(value=val)
        except KeyError as e:
            raise ValueError(f"No key '{self.key}' found in the metadata for {image_name}") from e
