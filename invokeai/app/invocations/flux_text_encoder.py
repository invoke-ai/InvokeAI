from pathlib import Path

import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from optimum.quanto import qfloat8
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import InputField
from invokeai.app.invocations.flux_text_to_image import FLUX_MODELS, QuantizedModelForTextEncoding, TFluxModelKeys
from invokeai.app.invocations.primitives import ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, FLUXConditioningInfo
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "flux_text_encoder",
    title="FLUX Text Encoding",
    tags=["image"],
    category="image",
    version="1.0.0",
)
class FluxTextEncoderInvocation(BaseInvocation):
    model: TFluxModelKeys = InputField(description="The FLUX model to use for text-to-image generation.")
    use_8bit: bool = InputField(
        default=False, description="Whether to quantize the transformer model to 8-bit precision."
    )
    positive_prompt: str = InputField(description="Positive prompt for text-to-image generation.")

    # TODO(ryand): Should we create a new return type for this invocation? This ConditioningOutput is clearly not
    # compatible with other ConditioningOutputs.
    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        model_path = context.models.download_and_cache_model(FLUX_MODELS[self.model])

        t5_embeddings, clip_embeddings = self._encode_prompt(context, model_path)
        conditioning_data = ConditioningFieldData(
            conditionings=[FLUXConditioningInfo(clip_embeds=clip_embeddings, t5_embeds=t5_embeddings)]
        )

        conditioning_name = context.conditioning.save(conditioning_data)
        return ConditioningOutput.build(conditioning_name)

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
