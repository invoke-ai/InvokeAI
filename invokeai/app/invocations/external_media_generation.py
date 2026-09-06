import json
from typing import Any

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, VideoField
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation(
    "fal_generic_media_native",
    title="fal.ai Generic Media",
    tags=["external", "generation", "fal", "fal.ai", "generic"],
    category="media",
    version="1.0.0",
)
class FalGenericMediaInvocation(BaseInvocation):
    """Submit arbitrary JSON to any fal.ai endpoint and return its raw JSON result."""

    model_id: str = InputField(description="fal.ai endpoint ID, for example fal-ai/kling-video/v3/pro/text-to-video")
    input_json: str = InputField(default="{}", description="JSON object sent to the fal.ai endpoint")
    image: ImageField | None = InputField(
        default=None, description="Optional local image for ${image_url} placeholders"
    )
    mask: ImageField | None = InputField(default=None, description="Optional local mask for ${mask_url} placeholders")
    reference_images: list[ImageField] = InputField(
        default=[], description="Optional local images for ${image_urls} or ${reference_image_urls} placeholders"
    )
    video: VideoField | None = InputField(
        default=None, description="Optional local video for ${video_url} placeholders"
    )

    def invoke(self, context: InvocationContext) -> StringOutput:
        try:
            payload = json.loads(self.input_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"fal.ai input_json is not valid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError("fal.ai input_json must be a JSON object")

        media_kwargs: dict[str, Any] = {}
        if self.image is not None:
            media_kwargs["image"] = context.images.get_pil(self.image.image_name, mode="RGB")
        if self.mask is not None:
            media_kwargs["mask_image"] = context.images.get_pil(self.mask.image_name, mode="L")
        if self.reference_images:
            media_kwargs["reference_images"] = [
                context.images.get_pil(field.image_name, mode="RGB") for field in self.reference_images
            ]
        if self.video is not None:
            media_kwargs["video_path"] = context.videos.get_path(self.video.video_name)

        result: dict[str, Any] = context._services.external_generation.generate_generic(
            provider_id="fal",
            model_id=self.model_id,
            payload=payload,
            **media_kwargs,
        )
        return StringOutput(value=json.dumps(result, ensure_ascii=False, sort_keys=True))
