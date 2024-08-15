from pathlib import Path
from typing import Literal

import accelerate
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from PIL import Image
from safetensors.torch import load_file
from transformers.models.auto import AutoModelForTextEncoding

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    ConditioningField,
    FieldDescriptions,
    Input,
    InputField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.quantization.bnb_llm_int8 import quantize_model_llm_int8
from invokeai.backend.quantization.bnb_nf4 import quantize_model_nf4
from invokeai.backend.quantization.fast_quantized_diffusion_model import FastQuantizedDiffusersModel
from invokeai.backend.quantization.fast_quantized_transformers_model import FastQuantizedTransformersModel
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import FLUXConditioningInfo

TFluxModelKeys = Literal["flux-schnell"]
FLUX_MODELS: dict[TFluxModelKeys, str] = {"flux-schnell": "black-forest-labs/FLUX.1-schnell"}


class QuantizedFluxTransformer2DModel(FastQuantizedDiffusersModel):
    base_class = FluxTransformer2DModel


class QuantizedModelForTextEncoding(FastQuantizedTransformersModel):
    auto_class = AutoModelForTextEncoding


@invocation(
    "flux_text_to_image",
    title="FLUX Text to Image",
    tags=["image"],
    category="image",
    version="1.0.0",
)
class FluxTextToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Text-to-image generation using a FLUX model."""

    model: TFluxModelKeys = InputField(description="The FLUX model to use for text-to-image generation.")
    quantization_type: Literal["raw", "NF4", "llm_int8"] = InputField(
        default="raw", description="The type of quantization to use for the transformer model."
    )
    use_8bit: bool = InputField(
        default=False, description="Whether to quantize the transformer model to 8-bit precision."
    )
    positive_text_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")
    num_steps: int = InputField(default=4, description="Number of diffusion steps.")
    guidance: float = InputField(
        default=4.0,
        description="The guidance strength. Higher values adhere more strictly to the prompt, and will produce less diverse images.",
    )
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        model_path = context.models.download_and_cache_model(FLUX_MODELS[self.model])

        # Load the conditioning data.
        cond_data = context.conditioning.load(self.positive_text_conditioning.conditioning_name)
        assert len(cond_data.conditionings) == 1
        flux_conditioning = cond_data.conditionings[0]
        assert isinstance(flux_conditioning, FLUXConditioningInfo)

        latents = self._run_diffusion(context, model_path, flux_conditioning.clip_embeds, flux_conditioning.t5_embeds)
        image = self._run_vae_decoding(context, model_path, latents)
        image_dto = context.images.save(image=image)
        return ImageOutput.build(image_dto)

    def _run_diffusion(
        self,
        context: InvocationContext,
        flux_model_dir: Path,
        clip_embeddings: torch.Tensor,
        t5_embeddings: torch.Tensor,
    ):
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(flux_model_dir / "scheduler", local_files_only=True)

        # HACK(ryand): Manually empty the cache. Currently we don't check the size of the model before loading it from
        # disk. Since the transformer model is large (24GB), there's a good chance that it will OOM on 32GB RAM systems
        # if the cache is not empty.
        context.models._services.model_manager.load.ram_cache.make_room(24 * 2**30)

        transformer_path = flux_model_dir / "transformer"
        with context.models.load_local_model(
            model_path=transformer_path, loader=self._load_flux_transformer
        ) as transformer:
            assert isinstance(transformer, FluxTransformer2DModel)

            flux_pipeline_with_transformer = FluxPipeline(
                scheduler=scheduler,
                vae=None,
                text_encoder=None,
                tokenizer=None,
                text_encoder_2=None,
                tokenizer_2=None,
                transformer=transformer,
            )

            dtype = torch.bfloat16
            t5_embeddings = t5_embeddings.to(dtype=dtype)
            clip_embeddings = clip_embeddings.to(dtype=dtype)

            latents = flux_pipeline_with_transformer(
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_steps,
                guidance_scale=self.guidance,
                generator=torch.Generator().manual_seed(self.seed),
                prompt_embeds=t5_embeddings,
                pooled_prompt_embeds=clip_embeddings,
                output_type="latent",
                return_dict=False,
            )[0]

        assert isinstance(latents, torch.Tensor)
        return latents

    def _run_vae_decoding(
        self,
        context: InvocationContext,
        flux_model_dir: Path,
        latents: torch.Tensor,
    ) -> Image.Image:
        vae_path = flux_model_dir / "vae"
        with context.models.load_local_model(model_path=vae_path, loader=self._load_flux_vae) as vae:
            assert isinstance(vae, AutoencoderKL)

            flux_pipeline_with_vae = FluxPipeline(
                scheduler=None,
                vae=vae,
                text_encoder=None,
                tokenizer=None,
                text_encoder_2=None,
                tokenizer_2=None,
                transformer=None,
            )

            latents = flux_pipeline_with_vae._unpack_latents(
                latents, self.height, self.width, flux_pipeline_with_vae.vae_scale_factor
            )
            latents = (
                latents / flux_pipeline_with_vae.vae.config.scaling_factor
            ) + flux_pipeline_with_vae.vae.config.shift_factor
            latents = latents.to(dtype=vae.dtype)
            image = flux_pipeline_with_vae.vae.decode(latents, return_dict=False)[0]
            image = flux_pipeline_with_vae.image_processor.postprocess(image, output_type="pil")[0]

        assert isinstance(image, Image.Image)
        return image

    def _load_flux_transformer(self, path: Path) -> FluxTransformer2DModel:
        if self.quantization_type == "raw":
            model = FluxTransformer2DModel.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16)
        elif self.quantization_type == "NF4":
            model_config = FluxTransformer2DModel.load_config(path, local_files_only=True)
            with accelerate.init_empty_weights():
                empty_model = FluxTransformer2DModel.from_config(model_config)
            assert isinstance(empty_model, FluxTransformer2DModel)

            model_nf4_path = path / "bnb_nf4"
            assert model_nf4_path.exists()
            with accelerate.init_empty_weights():
                model = quantize_model_nf4(empty_model, modules_to_not_convert=set(), compute_dtype=torch.bfloat16)

            # TODO(ryand): Right now, some of the weights are loaded in bfloat16. Think about how best to handle
            # this on GPUs without bfloat16 support.
            sd = load_file(model_nf4_path / "model.safetensors")
            model.load_state_dict(sd, strict=True, assign=True)
        elif self.quantization_type == "llm_int8":
            model_config = FluxTransformer2DModel.load_config(path, local_files_only=True)
            with accelerate.init_empty_weights():
                empty_model = FluxTransformer2DModel.from_config(model_config)
            assert isinstance(empty_model, FluxTransformer2DModel)
            model_int8_path = path / "bnb_llm_int8"
            assert model_int8_path.exists()
            with accelerate.init_empty_weights():
                model = quantize_model_llm_int8(empty_model, modules_to_not_convert=set())

            sd = load_file(model_int8_path / "model.safetensors")
            model.load_state_dict(sd, strict=True, assign=True)
        else:
            raise ValueError(f"Unsupported quantization type: {self.quantization_type}")

        assert isinstance(model, FluxTransformer2DModel)
        return model

    @staticmethod
    def _load_flux_vae(path: Path) -> AutoencoderKL:
        model = AutoencoderKL.from_pretrained(path, local_files_only=True)
        assert isinstance(model, AutoencoderKL)
        return model
