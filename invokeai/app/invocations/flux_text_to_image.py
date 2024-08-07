from pathlib import Path
from typing import Literal

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux import FluxPipeline
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.util.devices import TorchDevice

TFluxModelKeys = Literal["flux-schnell"]
FLUX_MODELS: dict[TFluxModelKeys, str] = {"flux-schnell": "black-forest-labs/FLUX.1-schnell"}


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
        # Determine the T5 max sequence lenght based on the model.
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
        scheduler = FlowMatchEulerDiscreteScheduler()

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

            return flux_pipeline_with_transformer(
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

    def _run_vae_decoding(
        self,
        context: InvocationContext,
        flux_model_dir: Path,
        latent: torch.Tensor,
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
            image = flux_pipeline_with_vae.vae.decode(latents, return_dict=False)[0]
            image = flux_pipeline_with_vae.image_processor.postprocess(image, output_type="pil")

        assert isinstance(image, Image.Image)
        return image

    @staticmethod
    def _load_flux_text_encoder(path: Path) -> CLIPTextModel:
        model = CLIPTextModel.from_pretrained(path, local_files_only=True)
        assert isinstance(model, CLIPTextModel)
        return model

    @staticmethod
    def _load_flux_text_encoder_2(path: Path) -> T5EncoderModel:
        model = T5EncoderModel.from_pretrained(path, local_files_only=True)
        assert isinstance(model, T5EncoderModel)
        return model

    @staticmethod
    def _load_flux_transformer(path: Path) -> FluxTransformer2DModel:
        model = FluxTransformer2DModel.from_pretrained(path, local_files_only=True)
        assert isinstance(model, FluxTransformer2DModel)
        return model

    @staticmethod
    def _load_flux_vae(path: Path) -> AutoencoderKL:
        model = AutoencoderKL.from_pretrained(path, local_files_only=True)
        assert isinstance(model, AutoencoderKL)
        return model
