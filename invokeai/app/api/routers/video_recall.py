"""Router for pushing video generation parameters and media into a user's Video panel.

The video counterpart of `recall_parameters.py`. Parameters are validated and resolved here, then
delivered to the owner's open frontends as one `video_recall_requested` event; the frontend owns
how they land in the panel.
"""

from typing import Annotated, Any, Literal, Optional, Union

from fastapi import Body, HTTPException, Path, Query, Request
from fastapi.openapi.models import Example
from fastapi.routing import APIRouter
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers._access import assert_image_read_access, assert_video_read_access
from invokeai.app.api.routers.image_move_maintenance import assert_image_move_maintenance_inactive
from invokeai.app.api.routers.videos import VIDEO_UPLOAD_OPENAPI_EXTRA, ingest_uploaded_video
from invokeai.app.services.events.events_common import VideoRecallAction, VideoRecallMode, VideoRecallVideo
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.model_records.model_records_base import UnknownModelException
from invokeai.app.services.video_records.video_records_common import VideoRecordNotFoundException
from invokeai.app.services.videos.videos_common import VideoDTO
from invokeai.app.util.misc import SEED_MAX
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType

video_recall_router = APIRouter(prefix="/v1/recall/video", tags=["recall"])

VIDEO_BASES = frozenset({BaseModelType.Wan, BaseModelType.MiniMaxH3, BaseModelType.LTX2})

# Each model slot's type and the bases the Video panel accepts in it (None: any base, for encoders
# installed as base-agnostic). Keys are the video metadata record's own names, so the frontend reads
# the resolved payload exactly as it reads a recorded one. The frontend applies recalled components
# without re-checking them, so a slot must never be handed a model from another family.
MODEL_SLOTS: dict[str, tuple[ModelType, Optional[frozenset[BaseModelType]]]] = {
    "model": (ModelType.Main, VIDEO_BASES),
    "vae": (ModelType.VAE, frozenset({BaseModelType.Wan})),
    "wan_t5_encoder_model": (ModelType.WanT5Encoder, None),
    "wan_transformer_low_noise": (ModelType.Main, frozenset({BaseModelType.Wan})),
    "wan_component_source": (ModelType.Main, frozenset({BaseModelType.Wan})),
    "minimax_h3_transformer_model": (ModelType.Main, frozenset({BaseModelType.MiniMaxH3})),
    "minimax_h3_component_source": (ModelType.Main, frozenset({BaseModelType.MiniMaxH3})),
    "minimax_h3_text_encoder_model": (ModelType.Qwen3VLEncoder, frozenset({BaseModelType.MiniMaxH3})),
    "minimax_h3_hybrid_base_model": (ModelType.Main, frozenset({BaseModelType.MiniMaxH3})),
    "ltx2_component_source": (ModelType.Main, frozenset({BaseModelType.LTX2})),
    "ltx2_text_encoder_model": (ModelType.Gemma4Encoder, frozenset({BaseModelType.LTX2})),
}

# The frontend's caps on Ref2VA references.
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_IMAGES = 9
MAX_LORAS = 32

ModelName = Annotated[str, StringConstraints(min_length=1, max_length=255)]
MediaName = Annotated[str, StringConstraints(min_length=1, max_length=255)]


def _check_trim(start: Optional[int], end: Optional[int], what: str) -> None:
    if (start is None) != (end is None):
        raise ValueError(f"{what}: give both the start and end frame, or neither")
    if start is not None and end is not None and start > end:
        raise ValueError(f"{what}: the start frame is after the end frame")


class VideoLoRARecallParameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: ModelName = Field(description="The name (or key) of the LoRA model")
    weight: float = Field(default=1.0, ge=-10, le=10, description="The weight for the LoRA")


class ImageRefParameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_name: MediaName = Field(description="The name of a gallery image")


class VideoRefParameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_name: MediaName = Field(description="The name of a gallery video")


class ImageReferenceParameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["image"]
    image_name: MediaName = Field(description="The name of a gallery image")
    detail: Optional[Literal["max", "match"]] = Field(default=None, description="The reference image's detail mode")


class VideoReferenceParameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["video"]
    video_name: MediaName = Field(description="The name of a gallery video")
    conditioning: Optional[Literal["video_audio", "video", "audio"]] = Field(
        default=None, description="Which of the clip's streams condition the generation"
    )
    start_frame: Optional[int] = Field(default=None, ge=0, description="First frame of the clip to use (inclusive)")
    end_frame: Optional[int] = Field(default=None, ge=0, description="Last frame of the clip to use (inclusive)")

    @model_validator(mode="after")
    def _trim_is_whole(self) -> "VideoReferenceParameter":
        _check_trim(self.start_frame, self.end_frame, "reference video trim")
        return self


ReferenceParameter = Annotated[Union[ImageReferenceParameter, VideoReferenceParameter], Field(discriminator="kind")]


class VideoRecallParameter(BaseModel):
    """Video generation parameters to apply to the Video panel.

    Field names follow the video metadata record (see the Media Metadata architecture doc). Models
    are given by name (or key) and resolved to installed models; media by gallery name. Bounds are
    the ones the Video panel accepts, so a value it would ignore is refused here instead.
    """

    model_config = ConfigDict(extra="forbid")

    positive_prompt: Optional[str] = Field(default=None, description="Positive prompt text")
    negative_prompt: Optional[str] = Field(
        default=None, description="Negative prompt text; an explicit null turns the negative prompt off"
    )
    seed: Optional[int] = Field(default=None, ge=0, le=SEED_MAX, description="Random seed; ignored when remixing")
    num_frames: Optional[int] = Field(default=None, ge=1, description="Number of frames to generate")
    fps: Optional[int] = Field(default=None, ge=1, le=120, description="Output frame rate")
    width: Optional[int] = Field(default=None, ge=1, description="Output width in pixels")
    height: Optional[int] = Field(default=None, ge=1, description="Output height in pixels")
    steps: Optional[int] = Field(default=None, ge=1, description="Number of denoising steps")
    cfg_scale: Optional[float] = Field(default=None, ge=1, description="CFG scale")
    wan_guidance_scale_low_noise: Optional[float] = Field(
        default=None, ge=1, description="Wan A14B: the low-noise expert's CFG scale"
    )
    ltx2_audio_cfg_scale: Optional[float] = Field(default=None, ge=1, description="LTX-2: audio CFG scale")
    ltx2_stg_scale: Optional[float] = Field(default=None, ge=0, description="LTX-2: spatiotemporal guidance scale")
    ltx2_modality_scale: Optional[float] = Field(default=None, ge=1, description="LTX-2: modality guidance scale")
    ltx2_context_frames: Optional[int] = Field(
        default=None, ge=1, description="LTX-2: frames of the initial video an extension continues from"
    )
    minimax_h3_hybrid_start_block: Optional[int] = Field(
        default=None, ge=0, description="MiniMax H3 Ref2VA: first block taken from the hybrid FL2VA base"
    )

    model: Optional[ModelName] = Field(default=None, description="The main video model's name (or key)")
    vae: Optional[ModelName] = Field(default=None, description="Wan: a standalone VAE's name (or key)")
    wan_t5_encoder_model: Optional[ModelName] = Field(default=None, description="Wan: standalone UMT5 encoder")
    wan_transformer_low_noise: Optional[ModelName] = Field(default=None, description="Wan A14B: low-noise expert")
    wan_component_source: Optional[ModelName] = Field(default=None, description="Wan: Diffusers component source")
    minimax_h3_transformer_model: Optional[ModelName] = Field(
        default=None, description="MiniMax H3: transformer override"
    )
    minimax_h3_component_source: Optional[ModelName] = Field(
        default=None, description="MiniMax H3: Diffusers component source"
    )
    minimax_h3_text_encoder_model: Optional[ModelName] = Field(default=None, description="MiniMax H3: Qwen3-VL encoder")
    minimax_h3_hybrid_base_model: Optional[ModelName] = Field(default=None, description="MiniMax H3 Ref2VA: FL2VA base")
    ltx2_component_source: Optional[ModelName] = Field(default=None, description="LTX-2: component source")
    ltx2_text_encoder_model: Optional[ModelName] = Field(default=None, description="LTX-2: Gemma-4 encoder")
    loras: Optional[list[VideoLoRARecallParameter]] = Field(
        default=None, max_length=MAX_LORAS, description="LoRAs with their weights; an empty list asks for none"
    )

    first_frame_image: Optional[ImageRefParameter] = Field(default=None, description="First-frame image")
    last_frame_image: Optional[ImageRefParameter] = Field(default=None, description="Last-frame image")
    source_video: Optional[VideoRefParameter] = Field(default=None, description="Initial video to extend")
    source_video_start_frame: Optional[int] = Field(
        default=None, ge=0, description="First frame of the initial video to use (inclusive)"
    )
    source_video_end_frame: Optional[int] = Field(
        default=None, ge=0, description="Last frame of the initial video to use (inclusive)"
    )
    ltx2_conditioning_video: Optional[VideoRefParameter] = Field(
        default=None, description="LTX-2: a clip whose audio (or picture) conditions the whole generation"
    )
    ltx2_conditioning_role: Optional[Literal["audio", "video"]] = Field(
        default=None, description="LTX-2: which stream of `ltx2_conditioning_video` is the condition; required with it"
    )
    minimax_h3_references: Optional[list[ReferenceParameter]] = Field(
        default=None,
        max_length=MAX_REFERENCE_VIDEOS + MAX_REFERENCE_IMAGES,
        description=(
            f"MiniMax H3 Ref2VA references in conditioning order: at most {MAX_REFERENCE_VIDEOS} videos and "
            f"{MAX_REFERENCE_IMAGES} images; an empty list asks for none"
        ),
    )

    @model_validator(mode="after")
    def _qualifiers_are_consistent(self) -> "VideoRecallParameter":
        _check_trim(self.source_video_start_frame, self.source_video_end_frame, "source video trim")
        if (self.ltx2_conditioning_video is None) != (self.ltx2_conditioning_role is None):
            raise ValueError("ltx2_conditioning_video and ltx2_conditioning_role must be given together")
        if self.minimax_h3_references:
            videos = sum(1 for reference in self.minimax_h3_references if reference.kind == "video")
            if videos > MAX_REFERENCE_VIDEOS or len(self.minimax_h3_references) - videos > MAX_REFERENCE_IMAGES:
                raise ValueError(
                    f"at most {MAX_REFERENCE_VIDEOS} reference videos and {MAX_REFERENCE_IMAGES} reference images"
                )
        return self

    def provided(self) -> dict[str, Any]:
        """The fields the caller sent. Null means "not sent", except that a null negative prompt turns it off."""
        sent = self.model_dump(exclude_none=True)
        if "negative_prompt" in self.model_fields_set and self.negative_prompt is None:
            sent["negative_prompt"] = None
        return sent


class VideoRecallParametersResponse(BaseModel):
    status: Literal["success", "no_parameters_provided", "nothing_resolved"]
    queue_id: str
    parameters: dict[str, Any] = Field(description="The resolved parameters sent to the frontend")
    skipped: list[str] = Field(description="Requested fields dropped because their model or media could not be used")
    overridden: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Requested media fields dropped because another medium in the same request takes precedence over them, "
            "mapped to that field (e.g. first_frame_image -> source_video)"
        ),
    )


class VideoRecallMediaResponse(BaseModel):
    status: Literal["success"]
    queue_id: str
    action: VideoRecallAction
    video: VideoDTO
    uploaded: bool = Field(description="Whether the video was uploaded into the gallery by this request")


def _model_identifier(config: AnyModelConfig) -> dict[str, Any]:
    return {
        "key": config.key,
        "hash": config.hash,
        "name": config.name,
        "base": config.base.value,
        "type": config.type.value,
    }


def resolve_video_model(
    name_or_key: str, model_type: ModelType, bases: Optional[frozenset[BaseModelType]]
) -> Optional[dict[str, Any]]:
    """Resolve a model key, else name, of the given type and bases to a metadata `ModelIdentifier`."""
    store = ApiDependencies.invoker.services.model_manager.store

    def usable(config: AnyModelConfig) -> bool:
        return config.type == model_type and (bases is None or config.base in bases)

    try:
        by_key = store.get_model(name_or_key)
        if usable(by_key):
            return _model_identifier(by_key)
    except UnknownModelException:
        pass

    for config in store.search_by_attr(model_name=name_or_key, model_type=model_type):
        if usable(config):
            return _model_identifier(config)
    return None


def _check_image(image_name: str, current_user: CurrentUserOrDefault) -> bool:
    """False for an image that does not exist; raise if it exists but the caller may not read it."""
    if not ApiDependencies.invoker.services.image_records.exists(image_name):
        return False
    assert_image_read_access(image_name, current_user)
    return True


def _check_video(video_name: str, current_user: CurrentUserOrDefault) -> bool:
    """False for a video that does not exist; raise if it exists but the caller may not read it."""
    if not ApiDependencies.invoker.services.video_records.exists(video_name):
        return False
    assert_video_read_access(video_name, current_user)
    return True


def resolve_video_recall_parameters(
    parameters: VideoRecallParameter, mode: VideoRecallMode, current_user: CurrentUserOrDefault
) -> tuple[dict[str, Any], list[str]]:
    """Resolve models and media to their metadata shapes. Returns (resolved, skipped field names)."""
    logger = ApiDependencies.invoker.services.logger
    provided = parameters.provided()
    if mode == "remix":
        provided.pop("seed", None)

    resolved: dict[str, Any] = {}
    skipped: list[str] = []

    def skip(field: str, reason: str) -> None:
        logger.warning(f"Video recall: skipping {field}: {reason}")
        skipped.append(field)

    for key, value in provided.items():
        if key in MODEL_SLOTS:
            model_type, bases = MODEL_SLOTS[key]
            identifier = resolve_video_model(value, model_type, bases)
            if identifier is None:
                skip(key, f"no usable {model_type.value} model {value!r}")
            else:
                resolved[key] = identifier
        elif key == "loras":
            loras: list[dict[str, Any]] = []
            for index, lora in enumerate(value):
                identifier = resolve_video_model(lora["model_name"], ModelType.LoRA, VIDEO_BASES)
                if identifier is None:
                    skip(f"loras[{index}]", f"no video LoRA {lora['model_name']!r}")
                else:
                    loras.append({"model": identifier, "weight": lora["weight"]})
            # An empty request asks for no LoRAs; one whose every entry was dropped asks for nothing.
            if loras or not value:
                resolved["loras"] = loras
        elif key in ("first_frame_image", "last_frame_image"):
            if _check_image(value["image_name"], current_user):
                resolved[key] = value
            else:
                skip(key, f"image {value['image_name']!r} not found")
        elif key in ("source_video", "ltx2_conditioning_video"):
            if _check_video(value["video_name"], current_user):
                resolved[key] = value
            else:
                skip(key, f"video {value['video_name']!r} not found")
        elif key == "minimax_h3_references":
            references: list[dict[str, Any]] = []
            for index, reference in enumerate(value):
                found = (
                    _check_image(reference["image_name"], current_user)
                    if reference["kind"] == "image"
                    else _check_video(reference["video_name"], current_user)
                )
                if found:
                    references.append(reference)
                else:
                    skip(f"minimax_h3_references[{index}]", "media not found")
            if references or not value:
                resolved["minimax_h3_references"] = references
        else:
            resolved[key] = value

    # Trim bounds and the conditioning role qualify their medium; without it they mean nothing.
    for medium, qualifiers in MEDIA_QUALIFIERS.items():
        if medium not in resolved:
            for qualifier in qualifiers:
                if resolved.pop(qualifier, None) is not None:
                    skip(qualifier, f"{medium} is not being recalled")

    return resolved, skipped


# Media keys and the keys that only qualify them.
MEDIA_QUALIFIERS: dict[str, tuple[str, ...]] = {
    "source_video": ("source_video_start_frame", "source_video_end_frame"),
    "ltx2_conditioning_video": ("ltx2_conditioning_role",),
}


def apply_media_precedence(resolved: dict[str, Any]) -> dict[str, str]:
    """Drop media a higher-precedence medium in the same request excludes; returns {dropped field: winner}.

    Mirrors the Video panel's reading of a record: a whole-generation conditioning clip excludes every other slot;
    Ref2VA references replace the frame slots (an initial video rides alongside them); an initial video replaces
    the first frame, which extend mode extracts from the clip itself.
    """
    rules: list[tuple[str, bool, tuple[str, ...]]] = [
        (
            "ltx2_conditioning_video",
            "ltx2_conditioning_video" in resolved,
            ("minimax_h3_references", "source_video", "first_frame_image", "last_frame_image"),
        ),
        (
            "minimax_h3_references",
            bool(resolved.get("minimax_h3_references")),
            ("first_frame_image", "last_frame_image"),
        ),
        ("source_video", "source_video" in resolved, ("first_frame_image",)),
    ]
    overridden: dict[str, str] = {}
    for winner, present, losers in rules:
        if not present or winner not in resolved:
            continue
        for loser in losers:
            # An explicitly empty reference list asks for no references; the winning clip clears them too, so it
            # loses nothing and stays.
            if resolved.get(loser) == []:
                continue
            if resolved.pop(loser, None) is not None:
                overridden[loser] = winner
                for qualifier in MEDIA_QUALIFIERS.get(loser, ()):
                    if resolved.pop(qualifier, None) is not None:
                        overridden[qualifier] = winner
    return overridden


# Swagger pre-fills the first example. Without these it offers every field with placeholder values, and the
# placeholder media then take precedence over the one field a caller meant to send.
VIDEO_RECALL_EXAMPLES: dict[str, Example] = {
    "prompt_and_model": Example(
        summary="Prompt and model",
        value={"positive_prompt": "a heron takes flight over a misty lake", "model": "Wan 2.2 I2V A14B (Diffusers)"},
    ),
    "first_frame": Example(
        summary="Start from a gallery image",
        value={
            "model": "Wan 2.2 I2V A14B (Diffusers)",
            "first_frame_image": {"image_name": "<gallery image name>.png"},
        },
    ),
    "extend_video": Example(
        summary="Extend a gallery video",
        value={
            "positive_prompt": "the camera keeps panning left",
            "source_video": {"video_name": "<gallery video name>.mp4"},
            "source_video_start_frame": 0,
            "source_video_end_frame": 80,
        },
    ),
    "ref2va_references": Example(
        summary="MiniMax H3 Ref2VA references",
        value={
            "model": "MiniMax H3 Ref2VA Transformer (int8, pruned)",
            "minimax_h3_references": [
                {"kind": "image", "image_name": "<gallery image name>.png"},
                {"kind": "video", "video_name": "<gallery video name>.mp4", "conditioning": "video_audio"},
            ],
        },
    ),
    "every_field": Example(
        summary="Every field (a reference, not a request to send as-is)",
        description=(
            "Every accepted field with a placeholder value. Delete the ones you do not mean to set: media fields "
            "override one another (conditioning video, then references, then source video, then frames), so as-is "
            "only the conditioning video would apply, and the placeholder names would be skipped as not found."
        ),
        value={
            "positive_prompt": "a heron takes flight over a misty lake",
            "negative_prompt": "blurry, low quality",
            "seed": 1234,
            "num_frames": 81,
            "fps": 16,
            "width": 832,
            "height": 480,
            "steps": 30,
            "cfg_scale": 5.0,
            "wan_guidance_scale_low_noise": 4.0,
            "ltx2_audio_cfg_scale": 7.0,
            "ltx2_stg_scale": 1.0,
            "ltx2_modality_scale": 3.0,
            "ltx2_context_frames": 49,
            "minimax_h3_hybrid_start_block": 25,
            "model": "<video model name or key>",
            "vae": "<Wan VAE name or key>",
            "wan_t5_encoder_model": "<Wan UMT5 encoder name or key>",
            "wan_transformer_low_noise": "<Wan low-noise expert name or key>",
            "wan_component_source": "<Wan Diffusers install name or key>",
            "minimax_h3_transformer_model": "<MiniMax H3 transformer name or key>",
            "minimax_h3_component_source": "<MiniMax H3 Diffusers install name or key>",
            "minimax_h3_text_encoder_model": "<MiniMax H3 Qwen3-VL encoder name or key>",
            "minimax_h3_hybrid_base_model": "<MiniMax H3 FL2VA transformer name or key>",
            "ltx2_component_source": "<LTX-2 components name or key>",
            "ltx2_text_encoder_model": "<LTX-2 Gemma-4 encoder name or key>",
            "loras": [{"model_name": "<LoRA name or key>", "weight": 1.0}],
            "first_frame_image": {"image_name": "<gallery image name>.png"},
            "last_frame_image": {"image_name": "<gallery image name>.png"},
            "source_video": {"video_name": "<gallery video name>.mp4"},
            "source_video_start_frame": 0,
            "source_video_end_frame": 80,
            "ltx2_conditioning_video": {"video_name": "<gallery video name>.mp4"},
            "ltx2_conditioning_role": "audio",
            "minimax_h3_references": [
                {"kind": "image", "image_name": "<gallery image name>.png", "detail": "max"},
                {
                    "kind": "video",
                    "video_name": "<gallery video name>.mp4",
                    "conditioning": "video_audio",
                    "start_frame": 0,
                    "end_frame": 48,
                },
            ],
        },
    ),
}


@video_recall_router.post(
    "/{queue_id}",
    operation_id="recall_video_parameters",
    response_model=VideoRecallParametersResponse,
)
def recall_video_parameters(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(..., description="The queue id to perform this operation on"),
    parameters: VideoRecallParameter = Body(
        ...,
        description="Video parameters to recall. Every field is optional: send only the ones to change.",
        openapi_examples=VIDEO_RECALL_EXAMPLES,
    ),
    mode: VideoRecallMode = Query(
        default="recall", description="`recall` applies every field; `remix` applies everything except the seed"
    ),
    strict: bool = Query(
        default=False,
        description=(
            "When true, the parameters are treated as a whole generation record: LoRAs and media not "
            "included are cleared from the Video panel. When false, only the included fields change."
        ),
    ),
) -> VideoRecallParametersResponse:
    """Apply video generation parameters to the current user's Video panel.

    Models and media that cannot be used are dropped and listed in `skipped`; the rest are applied.
    """
    if not parameters.provided():
        return VideoRecallParametersResponse(
            status="no_parameters_provided", queue_id=queue_id, parameters={}, skipped=[]
        )

    assert_image_move_maintenance_inactive()
    resolved, skipped = resolve_video_recall_parameters(parameters, mode, current_user)
    overridden = apply_media_precedence(resolved)
    if not resolved:
        return VideoRecallParametersResponse(
            status="nothing_resolved", queue_id=queue_id, parameters={}, skipped=skipped, overridden=overridden
        )

    ApiDependencies.invoker.services.events.emit_video_recall_requested(
        queue_id, current_user.user_id, "parameters", mode=mode, strict=strict, parameters=resolved
    )
    return VideoRecallParametersResponse(
        status="success", queue_id=queue_id, parameters=resolved, skipped=skipped, overridden=overridden
    )


def _emit_video_placement(
    queue_id: str, user_id: str, action: VideoRecallAction, video: VideoDTO, uploaded: bool
) -> VideoRecallMediaResponse:
    ApiDependencies.invoker.services.events.emit_video_recall_requested(
        queue_id,
        user_id,
        action,
        video=VideoRecallVideo(
            video_name=video.video_name,
            width=video.width,
            height=video.height,
            duration=video.duration,
            fps=video.fps,
            media_origin=video.media_origin,
        ),
    )
    return VideoRecallMediaResponse(status="success", queue_id=queue_id, action=action, video=video, uploaded=uploaded)


def _place_gallery_video(
    queue_id: str, action: VideoRecallAction, video_name: str, current_user: CurrentUserOrDefault
) -> VideoRecallMediaResponse:
    assert_video_read_access(video_name, current_user)
    try:
        video = ApiDependencies.invoker.services.videos.get_dto(video_name)
    except VideoRecordNotFoundException:
        raise HTTPException(status_code=404, detail="Video not found")
    return _emit_video_placement(queue_id, current_user.user_id, action, video, uploaded=False)


async def _place_uploaded_video(
    request: Request,
    queue_id: str,
    action: VideoRecallAction,
    board_id: Optional[str],
    current_user: CurrentUserOrDefault,
) -> VideoRecallMediaResponse:
    video = await ingest_uploaded_video(
        request, current_user, video_category=ImageCategory.GENERAL, is_intermediate=False, board_id=board_id
    )
    return _emit_video_placement(queue_id, current_user.user_id, action, video, uploaded=True)


_VIDEO_NAME_QUERY = Query(..., min_length=1, max_length=255, description="The name of the gallery video")
_BOARD_ID_QUERY = Query(default=None, description="The board to upload the video to; Uncategorized when omitted")
_UPLOAD_RESPONSES: dict[int | str, dict[str, Any]] = {
    413: {"description": "The video exceeds the upload size limit"},
    415: {"description": "The file is not a supported video or audio file"},
    429: {"description": "Too many concurrent video uploads"},
}


@video_recall_router.post(
    "/{queue_id}/initial-video",
    operation_id="recall_initial_video",
    response_model=VideoRecallMediaResponse,
)
def recall_initial_video(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(..., description="The queue id to perform this operation on"),
    video_name: str = _VIDEO_NAME_QUERY,
) -> VideoRecallMediaResponse:
    """Place a gallery video in the current user's Video panel as the Initial Video."""
    return _place_gallery_video(queue_id, "initial_video", video_name, current_user)


@video_recall_router.post(
    "/{queue_id}/initial-video/upload",
    operation_id="recall_initial_video_upload",
    response_model=VideoRecallMediaResponse,
    responses=_UPLOAD_RESPONSES,
    openapi_extra=VIDEO_UPLOAD_OPENAPI_EXTRA,
)
async def recall_initial_video_upload(
    current_user: CurrentUserOrDefault,
    request: Request,
    queue_id: str = Path(..., description="The queue id to perform this operation on"),
    board_id: Optional[str] = _BOARD_ID_QUERY,
) -> VideoRecallMediaResponse:
    """Upload a video into the gallery and place it in the current user's Video panel as the Initial Video."""
    return await _place_uploaded_video(request, queue_id, "initial_video", board_id, current_user)


@video_recall_router.post(
    "/{queue_id}/reference-video",
    operation_id="recall_reference_video",
    response_model=VideoRecallMediaResponse,
)
def recall_reference_video(
    current_user: CurrentUserOrDefault,
    queue_id: str = Path(..., description="The queue id to perform this operation on"),
    video_name: str = _VIDEO_NAME_QUERY,
) -> VideoRecallMediaResponse:
    """Append a gallery video to the current user's reference videos (models that take them, e.g. MiniMax H3 Ref2VA)."""
    return _place_gallery_video(queue_id, "reference_video", video_name, current_user)


@video_recall_router.post(
    "/{queue_id}/reference-video/upload",
    operation_id="recall_reference_video_upload",
    response_model=VideoRecallMediaResponse,
    responses=_UPLOAD_RESPONSES,
    openapi_extra=VIDEO_UPLOAD_OPENAPI_EXTRA,
)
async def recall_reference_video_upload(
    current_user: CurrentUserOrDefault,
    request: Request,
    queue_id: str = Path(..., description="The queue id to perform this operation on"),
    board_id: Optional[str] = _BOARD_ID_QUERY,
) -> VideoRecallMediaResponse:
    """Upload a video into the gallery and append it to the current user's reference videos."""
    return await _place_uploaded_video(request, queue_id, "reference_video", board_id, current_user)
