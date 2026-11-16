import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FluxKontextConditioningField, Input, InputField, UIComponent
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux3.layout import Flux3LayoutRegion, flux3_locked_rows
from invokeai.backend.flux3.prompt_upsampler import (
    FLUX3_UPSAMPLER_MAX_NEW_TOKENS,
    PromptUpsamplerCanceled,
    prepare_upsampler_references,
    upsample_flux3_prompt,
)
from invokeai.backend.flux3.ref_images import FLUX3_MAX_REFERENCE_IMAGES
from invokeai.backend.model_manager.taxonomy import ModelType
from invokeai.backend.text_llm_pipeline import load_text_llm_tokenizer_and_processor


@invocation(
    "flux3_prompt_upsampler",
    title="Prompt Upsampler - FLUX.3",
    tags=["prompt", "upsampler", "layout", "flux3", "flux"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
    # Deliberately NOT idle_gpu_offloadable, for the reason the ERNIE-Image prompt enhancer gives: this runs an
    # autoregressive `generate()` of up to `max_new_tokens` on every execution, a cost that never amortizes into a
    # borrowed device's model cache the way an encoder forward does.
)
class Flux3PromptUpsamplerInvocation(BaseInvocation):
    """Writes the caption FLUX.3 encodes with Black Forest Labs' prompt upsampler.

    The upsampler sees what the generation sees: the prompt, the reference images of an edit and the canvas regions of
    a layout. The mode follows the inputs (text-to-image, edit, and each with a layout). Regions replace the Layout
    Prompt node's: they and the rows of a box table that ends the prompt are kept exactly in the answer's box table.
    Decoding is greedy, so the same inputs give the same caption.
    """

    prompt: str = InputField(
        default="",
        description="The request to expand into the caption, optionally ending with a box table of its own.",
        ui_component=UIComponent.Textarea,
    )
    upsampler_model: ModelIdentifierField = InputField(
        title="Prompt Upsampler",
        description="The FLUX.3 prompt upsampler: a Text LLM that can read images.",
        ui_model_type=ModelType.TextLLM,
    )
    reference_images: FluxKontextConditioningField | list[FluxKontextConditioningField] | None = InputField(
        default=None,
        description="Reference images of an edit, in attachment order (up to 10), the ones the denoiser receives.",
        input=Input.Connection,
        title="Reference Images",
    )
    regions: list[Flux3LayoutRegion] = InputField(
        default=[],
        description="Boxes on the 0-1000 grid across the image, assembled from Canvas Regional Guidance layers. "
        "Object rows, or with reference images rows that place new content in the output.",
    )
    max_new_tokens: int = InputField(
        default=FLUX3_UPSAMPLER_MAX_NEW_TOKENS,
        ge=64,
        le=FLUX3_UPSAMPLER_MAX_NEW_TOKENS,
        description="The most tokens the caption may take.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> StringOutput:
        references = self._reference_fields()
        self._require_vision_text_llm(context)
        if len(references) > FLUX3_MAX_REFERENCE_IMAGES:
            raise ValueError(f"FLUX.3 takes at most {FLUX3_MAX_REFERENCE_IMAGES} reference images.")

        request, locked_rows = flux3_locked_rows(self.prompt, self.regions, edit_targets=bool(references))
        images = prepare_upsampler_references([context.images.get_pil(field.image.image_name) for field in references])
        mode = ("i2i" if images else "t2i") + ("_bbox" if locked_rows else "")

        # Loaded before the model so reading them does not hold it on the device. A processor also for a request
        # without images: one path serves every mode.
        _, processor = load_text_llm_tokenizer_and_processor(
            lambda submodel: context.models.load(self.upsampler_model, submodel), with_processor=True
        )
        assert processor is not None

        context.util.signal_progress("Loading the prompt upsampler")
        try:
            loaded = context.models.load(self.upsampler_model)
            with loaded.model_on_device() as (_, model):
                try:
                    caption = upsample_flux3_prompt(
                        model,
                        processor,
                        request,
                        images=images,
                        locked_rows=locked_rows,
                        max_new_tokens=self.max_new_tokens,
                        # compute_device honours cpu_only and is stable under partial loading, where the first parameter
                        # may sit on the CPU.
                        device=loaded.compute_device,
                        is_canceled=context.util.is_canceled,
                        on_progress=lambda tokens: context.util.signal_progress(
                            f"Upsampling the prompt ({tokens} tokens)"
                        ),
                    )
                except PromptUpsamplerCanceled as e:
                    raise CanceledException from e
        finally:
            # Outside the `with`, which holds the model locked. The cache would make room for the text encoder and the
            # transformer by itself, but by splitting the largest unlocked model against each model that loads next;
            # moving the upsampler out whole leaves them their VRAM. The cost is a re-stream of the upsampler into
            # VRAM on its next run (about 2.5 s on an RTX 4090).
            context.models.offload_from_vram(self.upsampler_model)

        # The caption is a user's prompt, and can run to 15 KB.
        context.logger.info(f"FLUX.3 prompt upsampler ({mode}) wrote a caption of {len(caption)} characters")
        context.logger.debug(f"FLUX.3 prompt upsampler caption: {caption!r}")
        return StringOutput(value=caption)

    def _reference_fields(self) -> list[FluxKontextConditioningField]:
        if self.reference_images is None:
            return []
        if isinstance(self.reference_images, list):
            return self.reference_images
        return [self.reference_images]

    def _require_vision_text_llm(self, context: InvocationContext) -> None:
        config = context.models.get_config(self.upsampler_model)
        if config.type is not ModelType.TextLLM or not getattr(config, "supports_images", False):
            raise ValueError(
                f"'{config.name}' cannot be the FLUX.3 prompt upsampler: choose an installed Text LLM that reads "
                "images, such as the black-forest-labs flux-3-image-klein-pup model."
            )
