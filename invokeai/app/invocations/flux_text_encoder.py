import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField
from invokeai.app.invocations.model import CLIPField, T5EncoderField
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
    clip: CLIPField = InputField(
        title="CLIP",
        description=FieldDescriptions.clip,
        input=Input.Connection,
    )
    t5Encoder: T5EncoderField = InputField(
        title="T5Encoder",
        description=FieldDescriptions.t5Encoder,
        input=Input.Connection,
    )
    positive_prompt: str = InputField(description="Positive prompt for text-to-image generation.")

    # TODO(ryand): Should we create a new return type for this invocation? This ConditioningOutput is clearly not
    # compatible with other ConditioningOutputs.
    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        t5_embeddings, clip_embeddings = self._encode_prompt(context)
        conditioning_data = ConditioningFieldData(
            conditionings=[FLUXConditioningInfo(clip_embeds=clip_embeddings, t5_embeds=t5_embeddings)]
        )

        conditioning_name = context.conditioning.save(conditioning_data)
        return ConditioningOutput.build(conditioning_name)

    def _encode_prompt(self, context: InvocationContext) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Determine the T5 max sequence length based on the model.
        # if self.model == "flux-schnell":
        max_seq_len = 256
        # # elif self.model == "flux-dev":
        # #     max_seq_len = 512
        # else:
        #     raise ValueError(f"Unknown model: {self.model}")

        # Load CLIP.
        clip_tokenizer_info = context.models.load(self.clip.tokenizer)
        clip_text_encoder_info = context.models.load(self.clip.text_encoder)

        # Load T5.
        t5_tokenizer_info = context.models.load(self.t5Encoder.tokenizer)
        t5_text_encoder_info = context.models.load(self.t5Encoder.text_encoder)

        with (
            clip_text_encoder_info as clip_text_encoder,
            t5_text_encoder_info as t5_text_encoder,
            clip_tokenizer_info as clip_tokenizer,
            t5_tokenizer_info as t5_tokenizer,
        ):
            assert isinstance(clip_text_encoder, CLIPTextModel)
            assert isinstance(t5_text_encoder, T5EncoderModel)
            assert isinstance(clip_tokenizer, CLIPTokenizer)
            assert isinstance(t5_tokenizer, T5TokenizerFast)

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
            prompt_embeds, pooled_prompt_embeds, _ = pipeline.encode_prompt(
                prompt=self.positive_prompt,
                prompt_2=self.positive_prompt,
                device=TorchDevice.choose_torch_device(),
                max_sequence_length=max_seq_len,
            )

        assert isinstance(prompt_embeds, torch.Tensor)
        assert isinstance(pooled_prompt_embeds, torch.Tensor)
        return prompt_embeds, pooled_prompt_embeds
