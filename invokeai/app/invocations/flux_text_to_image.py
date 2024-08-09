from pathlib import Path
from typing import Literal

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from optimum.quanto import qfloat8
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from transformers.models.auto import AutoModelForTextEncoding

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.quantization.fast_quantized_diffusion_model import FastQuantizedDiffusersModel
from invokeai.backend.quantization.fast_quantized_transformers_model import FastQuantizedTransformersModel
from invokeai.backend.util.devices import TorchDevice

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
    use_8bit: bool = InputField(
        default=False, description="Whether to quantize the transformer model to 8-bit precision."
    )
    positive_prompt: str = InputField(description="Positive prompt for text-to-image generation.")
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

        t5_embeddings, clip_embeddings = self._encode_prompt(context, model_path)
        latents = self._run_diffusion(context, model_path, clip_embeddings, t5_embeddings)
        image = self._run_vae_decoding(context, model_path, latents)
        image_dto = context.images.save(image=image)
        return ImageOutput.build(image_dto)

    def _encode_prompt(self, context: InvocationContext, flux_model_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
        # Determine the T5 max sequence length based on the model.
        if self.model == "flux-schnell":
            max_seq_len = 256
        # elif self.model == "flux-dev":
        #     max_seq_len = 512
        else:
            raise ValueError(f"Unknown model: {self.model}")

        # Load the CLIP tokenizer.
        clip_tokenizer_path = flux_model_dir / "tokenizer"
        clip_tokenizer = CLIPTokenizer.from_pretrained(clip_tokenizer_path, local_files_only=True)
        assert isinstance(clip_tokenizer, CLIPTokenizer)

        # Load the T5 tokenizer.
        t5_tokenizer_path = flux_model_dir / "tokenizer_2"
        t5_tokenizer = T5TokenizerFast.from_pretrained(t5_tokenizer_path, local_files_only=True)
        assert isinstance(t5_tokenizer, T5TokenizerFast)

        clip_text_encoder_path = flux_model_dir / "text_encoder"
        t5_text_encoder_path = flux_model_dir / "text_encoder_2"
        with (
            context.models.load_local_model(
                model_path=clip_text_encoder_path, loader=self._load_flux_text_encoder
            ) as clip_text_encoder,
            context.models.load_local_model(
                model_path=t5_text_encoder_path, loader=self._load_flux_text_encoder_2
            ) as t5_text_encoder,
        ):
            assert isinstance(clip_text_encoder, CLIPTextModel)
            assert isinstance(t5_text_encoder, T5EncoderModel)
            pipeline = FluxPipeline(
                scheduler=None,
                vae=None,
                text_encoder=clip_text_encoder,
                tokenizer=clip_tokenizer,
                text_encoder_2=t5_text_encoder,
                tokenizer_2=t5_tokenizer,
                transformer=None,
            )

            # prompt_embeds: T5 embeddings
            # pooled_prompt_embeds: CLIP embeddings
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt=self.positive_prompt,
                prompt_2=self.positive_prompt,
                device=TorchDevice.choose_torch_device(),
                max_sequence_length=max_seq_len,
            )

        assert isinstance(prompt_embeds, torch.Tensor)
        assert isinstance(pooled_prompt_embeds, torch.Tensor)
        return prompt_embeds, pooled_prompt_embeds

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

            t5_embeddings = t5_embeddings.to(dtype=transformer.dtype)
            clip_embeddings = clip_embeddings.to(dtype=transformer.dtype)

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

    @staticmethod
    def _load_flux_text_encoder(path: Path) -> CLIPTextModel:
        model = CLIPTextModel.from_pretrained(path, local_files_only=True)
        assert isinstance(model, CLIPTextModel)
        return model

    def _load_flux_text_encoder_2(self, path: Path) -> T5EncoderModel:
        if self.use_8bit:
            model_8bit_path = path / "quantized"
            if model_8bit_path.exists():
                # The quantized model exists, load it.
                # TODO(ryand): The requantize(...) operation in from_pretrained(...) is very slow. This seems like
                # something that we should be able to make much faster.
                q_model = QuantizedModelForTextEncoding.from_pretrained(model_8bit_path)

                # Access the underlying wrapped model.
                # We access the wrapped model, even though it is private, because it simplifies the type checking by
                # always returning a T5EncoderModel from this function.
                model = q_model._wrapped
            else:
                # The quantized model does not exist yet, quantize and save it.
                # TODO(ryand): dtype?
                model = T5EncoderModel.from_pretrained(path, local_files_only=True)
                assert isinstance(model, T5EncoderModel)

                q_model = QuantizedModelForTextEncoding.quantize(model, weights=qfloat8)

                model_8bit_path.mkdir(parents=True, exist_ok=True)
                q_model.save_pretrained(model_8bit_path)

                # (See earlier comment about accessing the wrapped model.)
                model = q_model._wrapped
        else:
            model = T5EncoderModel.from_pretrained(path, local_files_only=True)

        assert isinstance(model, T5EncoderModel)
        return model

    def _load_flux_transformer(self, path: Path) -> FluxTransformer2DModel:
        if self.use_8bit:
            model_8bit_path = path / "quantized"
            if model_8bit_path.exists():
                # The quantized model exists, load it.
                # TODO(ryand): The requantize(...) operation in from_pretrained(...) is very slow. This seems like
                # something that we should be able to make much faster.
                q_model = QuantizedFluxTransformer2DModel.from_pretrained(model_8bit_path)

                # Access the underlying wrapped model.
                # We access the wrapped model, even though it is private, because it simplifies the type checking by
                # always returning a FluxTransformer2DModel from this function.
                model = q_model._wrapped
            else:
                # The quantized model does not exist yet, quantize and save it.
                # TODO(ryand): Loading in float16 and then quantizing seems to result in NaNs. In order to run this on
                # GPUs that don't support bfloat16, we would need to host the quantized model instead of generating it
                # here.
                model = FluxTransformer2DModel.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16)
                assert isinstance(model, FluxTransformer2DModel)

                q_model = QuantizedFluxTransformer2DModel.quantize(model, weights=qfloat8)

                model_8bit_path.mkdir(parents=True, exist_ok=True)
                q_model.save_pretrained(model_8bit_path)

                # (See earlier comment about accessing the wrapped model.)
                model = q_model._wrapped
        else:
            model = FluxTransformer2DModel.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16)

        assert isinstance(model, FluxTransformer2DModel)
        return model

    @staticmethod
    def _load_flux_vae(path: Path) -> AutoencoderKL:
        model = AutoencoderKL.from_pretrained(path, local_files_only=True)
        assert isinstance(model, AutoencoderKL)
        return model
