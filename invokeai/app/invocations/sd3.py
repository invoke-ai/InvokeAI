from contextlib import ExitStack
from typing import Optional, cast

import torch
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from pydantic import field_validator
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Input,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR, SCHEDULER_NAME_VALUES
from invokeai.app.invocations.denoise_latents import get_scheduler
from invokeai.app.invocations.fields import FieldDescriptions, InputField, LatentsField, OutputField, UIType
from invokeai.app.invocations.model import ModelIdentifierField, SD3CLIPField, TransformerField, VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.misc import SEED_MAX
from invokeai.backend.model_manager.config import SubModelType

sd3_pipeline: Optional[StableDiffusion3Pipeline] = None


class FakeVae:
    class FakeVaeConfig:
        def __init__(self) -> None:
            self.block_out_channels = [0]

    def __init__(self) -> None:
        self.config = FakeVae.FakeVaeConfig()


@invocation_output("sd3_model_loader_output")
class SD3ModelLoaderOutput(BaseInvocationOutput):
    """Stable Diffuion 3 base model loader output"""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    clip: SD3CLIPField = OutputField(description=FieldDescriptions.clip, title="CLIP")
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation("sd3_model_loader", title="SD3 Main Model", tags=["model", "sd3"], category="model", version="1.0.0")
class SD3ModelLoaderInvocation(BaseInvocation):
    """Loads an SD3 base model, outputting its submodels."""

    model: ModelIdentifierField = InputField(description=FieldDescriptions.sd3_main_model, ui_type=UIType.SD3MainModel)

    def invoke(self, context: InvocationContext) -> SD3ModelLoaderOutput:
        model_key = self.model.key

        if not context.models.exists(model_key):
            raise Exception(f"Unknown model: {model_key}")

        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        scheduler = self.model.model_copy(update={"submodel_type": SubModelType.Scheduler})
        tokenizer_1 = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        text_encoder_1 = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        tokenizer_2 = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer2})
        text_encoder_2 = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder2})
        try:
            tokenizer_3 = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer3})
            text_encoder_3 = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder3})
        except Exception:
            tokenizer_3 = None
            text_encoder_3 = None
        vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})

        return SD3ModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, scheduler=scheduler),
            clip=SD3CLIPField(
                tokenizer_1=tokenizer_1,
                text_encoder_1=text_encoder_1,
                tokenizer_2=tokenizer_2,
                text_encoder_2=text_encoder_2,
                tokenizer_3=tokenizer_3,
                text_encoder_3=text_encoder_3,
            ),
            vae=VAEField(vae=vae),
        )


@invocation(
    "sd3_image_generator", title="Stable Diffusion 3", tags=["latent", "sd3"], category="latents", version="1.0.0"
)
class StableDiffusion3Invocation(BaseInvocation):
    """Generates an image using Stable Diffusion 3."""

    transformer: TransformerField = InputField(
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
        ui_order=0,
    )
    clip: SD3CLIPField = InputField(
        description=FieldDescriptions.clip,
        input=Input.Connection,
        title="CLIP",
        ui_order=1,
    )
    noise: Optional[LatentsField] = InputField(
        default=None,
        description=FieldDescriptions.noise,
        input=Input.Connection,
        ui_order=2,
    )
    scheduler: SCHEDULER_NAME_VALUES = InputField(
        default="euler_f",
        description=FieldDescriptions.scheduler,
        ui_type=UIType.Scheduler,
    )
    positive_prompt: str = InputField(default="", title="Positive Prompt")
    negative_prompt: str = InputField(default="", title="Negative Prompt")
    steps: int = InputField(default=20, gt=0, description=FieldDescriptions.steps)
    guidance_scale: float = InputField(default=7.0, description=FieldDescriptions.cfg_scale, title="CFG Scale")
    use_clip_3: bool = InputField(default=True, description="Use TE5 Encoder of SD3", title="Use TE5 Encoder")

    seed: int = InputField(
        default=0,
        ge=0,
        le=SEED_MAX,
        description=FieldDescriptions.seed,
    )
    width: int = InputField(
        default=1024,
        multiple_of=LATENT_SCALE_FACTOR,
        gt=0,
        description=FieldDescriptions.width,
    )
    height: int = InputField(
        default=1024,
        multiple_of=LATENT_SCALE_FACTOR,
        gt=0,
        description=FieldDescriptions.height,
    )

    @field_validator("seed", mode="before")
    def modulo_seed(cls, v: int):
        """Return the seed modulo (SEED_MAX + 1) to ensure it is within the valid range."""
        return v % (SEED_MAX + 1)

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        with ExitStack() as stack:
            tokenizer_1 = stack.enter_context(context.models.load(self.clip.tokenizer_1))
            tokenizer_2 = stack.enter_context(context.models.load(self.clip.tokenizer_2))
            text_encoder_1 = stack.enter_context(context.models.load(self.clip.text_encoder_1))
            text_encoder_2 = stack.enter_context(context.models.load(self.clip.text_encoder_2))
            transformer = stack.enter_context(context.models.load(self.transformer.transformer))

            assert isinstance(transformer, SD3Transformer2DModel)
            assert isinstance(text_encoder_1, CLIPTextModelWithProjection)
            assert isinstance(text_encoder_2, CLIPTextModelWithProjection)
            assert isinstance(tokenizer_1, CLIPTokenizer)
            assert isinstance(tokenizer_2, CLIPTokenizer)

            if self.use_clip_3 and self.clip.tokenizer_3 and self.clip.text_encoder_3:
                tokenizer_3 = stack.enter_context(context.models.load(self.clip.tokenizer_3))
                text_encoder_3 = stack.enter_context(context.models.load(self.clip.text_encoder_3))
                assert isinstance(text_encoder_3, T5EncoderModel)
                assert isinstance(tokenizer_3, T5TokenizerFast)
            else:
                tokenizer_3 = None
                text_encoder_3 = None

            scheduler = get_scheduler(
                context=context,
                scheduler_info=self.transformer.scheduler,
                scheduler_name=self.scheduler,
                seed=self.seed,
            )

            sd3_pipeline = StableDiffusion3Pipeline(
                transformer=transformer,
                vae=FakeVae(),
                text_encoder=text_encoder_1,
                text_encoder_2=text_encoder_2,
                text_encoder_3=text_encoder_3,
                tokenizer=tokenizer_1,
                tokenizer_2=tokenizer_2,
                tokenizer_3=tokenizer_3,
                scheduler=scheduler,
            )

            results = sd3_pipeline(
                self.positive_prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance_scale,
                output_type="latent",
            )

            latents = cast(torch.Tensor, results.images[0])
            latents = latents.unsqueeze(0)

        latents_name = context.tensors.save(latents)
        return LatentsOutput.build(latents_name, latents=latents, seed=self.seed)
