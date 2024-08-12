from pathlib import Path
from typing import Literal
from pydantic import Field

import accelerate
import torch
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from invokeai.app.invocations.model import ModelIdentifierField
from optimum.quanto import qfloat8
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
    UIType,
)
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.quantization.bnb_nf4 import quantize_model_nf4
from invokeai.backend.quantization.fast_quantized_diffusion_model import FastQuantizedDiffusersModel
from invokeai.backend.quantization.fast_quantized_transformers_model import FastQuantizedTransformersModel
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import FLUXConditioningInfo
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

    flux_model: ModelIdentifierField = InputField(
        description="The Flux model",
        input=Input.Any,
        ui_type=UIType.FluxMainModel
    )
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
        # model_path = context.models.download_and_cache_model(FLUX_MODELS[self.model])
        flux_transformer_path = context.models.download_and_cache_model(
            "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors"
        )
        flux_ae_path = context.models.download_and_cache_model(
            "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors"
        )

        # Load the conditioning data.
        cond_data = context.conditioning.load(self.positive_text_conditioning.conditioning_name)
        assert len(cond_data.conditionings) == 1
        flux_conditioning = cond_data.conditionings[0]
        assert isinstance(flux_conditioning, FLUXConditioningInfo)

        latents = self._run_diffusion(
            context, flux_transformer_path, flux_conditioning.clip_embeds, flux_conditioning.t5_embeds
        )
        image = self._run_vae_decoding(context, flux_ae_path, latents)
        image_dto = context.images.save(image=image)
        return ImageOutput.build(image_dto)

    def _run_diffusion(
        self,
        context: InvocationContext,
        flux_transformer_path: Path,
        clip_embeddings: torch.Tensor,
        t5_embeddings: torch.Tensor,
    ):
        inference_dtype = TorchDevice.choose_torch_dtype()

        # Prepare input noise.
        # TODO(ryand): Does the seed behave the same on different devices? Should we re-implement this to always use a
        # CPU RNG?
        x = get_noise(
            num_samples=1,
            height=self.height,
            width=self.width,
            device=TorchDevice.choose_torch_device(),
            dtype=inference_dtype,
            seed=self.seed,
        )

        img, img_ids = self._prepare_latent_img_patches(x)

        # HACK(ryand): Find a better way to determine if this is a schnell model or not.
        is_schnell = "shnell" in str(flux_transformer_path)
        timesteps = get_schedule(
            num_steps=self.num_steps,
            image_seq_len=img.shape[1],
            shift=not is_schnell,
        )

        bs, t5_seq_len, _ = t5_embeddings.shape
        txt_ids = torch.zeros(bs, t5_seq_len, 3, dtype=inference_dtype, device=TorchDevice.choose_torch_device())

        # HACK(ryand): Manually empty the cache. Currently we don't check the size of the model before loading it from
        # disk. Since the transformer model is large (24GB), there's a good chance that it will OOM on 32GB RAM systems
        # if the cache is not empty.
        context.models._services.model_manager.load.ram_cache.make_room(24 * 2**30)

        with context.models.load_local_model(
            model_path=flux_transformer_path, loader=self._load_flux_transformer
        ) as transformer:
            assert isinstance(transformer, Flux)

            x = denoise(
                model=transformer,
                img=img,
                img_ids=img_ids,
                txt=t5_embeddings,
                txt_ids=txt_ids,
                vec=clip_embeddings,
                timesteps=timesteps,
                guidance=self.guidance,
            )

        x = unpack(x.float(), self.height, self.width)

        return x

    def _prepare_latent_img_patches(self, latent_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert an input image in latent space to patches for diffusion.

        This implementation was extracted from:
        https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/sampling.py#L32

        Returns:
            tuple[Tensor, Tensor]: (img, img_ids), as defined in the original flux repo.
        """
        bs, c, h, w = latent_img.shape

        # Pixel unshuffle with a scale of 2, and flatten the height/width dimensions to get an array of patches.
        img = rearrange(latent_img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if img.shape[0] == 1 and bs > 1:
            img = repeat(img, "1 ... -> bs ...", bs=bs)

        # Generate patch position ids.
        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        return img, img_ids

    def _run_vae_decoding(
        self,
        context: InvocationContext,
        flux_ae_path: Path,
        latents: torch.Tensor,
    ) -> Image.Image:
        with context.models.load_local_model(model_path=flux_ae_path, loader=self._load_flux_vae) as vae:
            assert isinstance(vae, AutoEncoder)
            # TODO(ryand): Test that this works with both float16 and bfloat16.
            with torch.autocast(device_type=latents.device.type, dtype=TorchDevice.choose_torch_dtype()):
                img = vae.decode(latents)

        img.clamp(-1, 1)
        img = rearrange(img[0], "c h w -> h w c")
        img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())

        return img_pil

    def _load_flux_transformer(self, path: Path) -> FluxTransformer2DModel:
        inference_dtype = TorchDevice.choose_torch_dtype()
        if self.quantization_type == "raw":
            # TODO(ryand): Determine if this is a schnell model or a dev model and load the appropriate config.
            params = flux_configs["flux-schnell"].params

            # Initialize the model on the "meta" device.
            with accelerate.init_empty_weights():
                model = Flux(params).to(inference_dtype)

            state_dict = load_file(path)
            # TODO(ryand): Cast the state_dict to the appropriate dtype?
            model.load_state_dict(state_dict, strict=True, assign=True)
        elif self.quantization_type == "NF4":
            model_path = path.parent / "bnb_nf4.safetensors"

            # TODO(ryand): Determine if this is a schnell model or a dev model and load the appropriate config.
            params = flux_configs["flux-schnell"].params
            # Initialize the model on the "meta" device.
            with accelerate.init_empty_weights():
                model = Flux(params)
                model = quantize_model_nf4(model, modules_to_not_convert=set(), compute_dtype=torch.bfloat16)

            # TODO(ryand): Right now, some of the weights are loaded in bfloat16. Think about how best to handle
            # this on GPUs without bfloat16 support.
            state_dict = load_file(model_path)
            model.load_state_dict(state_dict, strict=True, assign=True)

        elif self.quantization_type == "llm_int8":
            raise NotImplementedError("LLM int8 quantization is not yet supported.")
            # model_config = FluxTransformer2DModel.load_config(path, local_files_only=True)
            # with accelerate.init_empty_weights():
            #     empty_model = FluxTransformer2DModel.from_config(model_config)
            # assert isinstance(empty_model, FluxTransformer2DModel)
            # model_int8_path = path / "bnb_llm_int8"
            # assert model_int8_path.exists()
            # with accelerate.init_empty_weights():
            #     model = quantize_model_llm_int8(empty_model, modules_to_not_convert=set())

            # sd = load_file(model_int8_path / "model.safetensors")
            # model.load_state_dict(sd, strict=True, assign=True)
        else:
            raise ValueError(f"Unsupported quantization type: {self.quantization_type}")

        assert isinstance(model, FluxTransformer2DModel)
        return model

    @staticmethod
    def _load_flux_vae(path: Path) -> AutoEncoder:
        # TODO(ryand): Determine if this is a schnell model or a dev model and load the appropriate config.
        ae_params = flux_configs["flux1-schnell"].ae_params
        with accelerate.init_empty_weights():
            ae = AutoEncoder(ae_params)

        state_dict = load_file(path)
        ae.load_state_dict(state_dict, strict=True, assign=True)
        return ae
