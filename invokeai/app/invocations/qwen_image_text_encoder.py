# Copyright (c) 2024, Brandon W. Rising and the InvokeAI Development Team
"""Qwen-Image text encoding invocation.

Encodes the prompt using Qwen2.5-VL and returns embeddings and attention mask
following the Qwen-Image pipeline's prompt handling template.
"""

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, UIComponent
from invokeai.app.invocations.model import Qwen2_5VLField
from invokeai.app.invocations.primitives import QwenImageConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData


@invocation(
    "qwen_image_text_encoder",
    title="Prompt - Qwen-Image",
    tags=["prompt", "conditioning", "qwen"],
    category="conditioning",
    version="1.0.0",
)
class QwenImageTextEncoderInvocation(BaseInvocation):
    """Encodes a text prompt for Qwen-Image generation."""

    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)
    qwen2_5_vl: Qwen2_5VLField = InputField(
        title="Qwen2.5-VL",
        description="Qwen2.5-VL vision-language model for text encoding",
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> QwenImageConditioningOutput:
        """Encode the prompt using Qwen-Image's text encoder."""
        
        # Load the text encoder info first to get the model
        text_encoder_info = context.models.load(self.qwen2_5_vl.text_encoder)

        # Load the Qwen2.5-VL tokenizer and text encoder with proper device management
        with text_encoder_info.model_on_device() as (cached_weights, text_encoder), \
            context.models.load(self.qwen2_5_vl.tokenizer) as tokenizer:

            try:
                # Build prompt template and tokenize
                template = (
                    "<|im_start|>system\n"
                    "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
                    "<|im_start|>user\n{}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                drop_idx = 34  # number of special tokens before the user content in template
                txt = template.format(self.prompt)

                tok = tokenizer(
                    txt,
                    max_length=1024 + drop_idx,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                tok = {k: v.to(text_encoder.device) for k, v in tok.items()}

                # Encode with hidden states
                enc = text_encoder(
                    input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], output_hidden_states=True
                )
                hidden_states = enc.hidden_states[-1]  # [B, seq, hidden]

                # Extract masked hidden states per-sample
                mask = tok["attention_mask"].bool()
                valid_lengths = mask.sum(dim=1)
                selected = hidden_states[mask]
                split = torch.split(selected, valid_lengths.tolist(), dim=0)
                split = [s[drop_idx:] for s in split]

                # Build attention masks aligned to truncated sequences
                attn_masks = [torch.ones(s.size(0), dtype=torch.long, device=text_encoder.device) for s in split]
                max_seq_len = max(s.size(0) for s in split) if split else 0
                if max_seq_len == 0:
                    raise ValueError("Empty encoded sequence after applying template and mask")

                # Pad to max sequence length and stack
                embeds = torch.stack(
                    [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split]
                )
                embeds_mask = torch.stack(
                    [torch.cat([m, m.new_zeros(max_seq_len - m.size(0))]) for m in attn_masks]
                )

                embeds = embeds.to(dtype=text_encoder.dtype).contiguous()

                # Save conditioning (move to cpu for storage)
                class QwenImageConditioningInfo:
                    def __init__(self, text_embeds: torch.Tensor, text_embeds_mask: torch.Tensor, prompt: str):
                        self.text_embeds = text_embeds
                        self.text_embeds_mask = text_embeds_mask
                        self.prompt = prompt

                conditioning_info = QwenImageConditioningInfo(embeds.cpu(), embeds_mask.cpu(), self.prompt)
                conditioning_data = ConditioningFieldData(conditionings=[conditioning_info])
                conditioning_name = context.conditioning.save(conditioning_data)
                return QwenImageConditioningOutput.build(conditioning_name)

            except Exception as e:
                context.logger.error(f"Error encoding Qwen-Image text: {e}")
                class QwenImageConditioningInfo:
                    def __init__(self, prompt: str):
                        self.prompt = prompt
                conditioning_info = QwenImageConditioningInfo(self.prompt)
                conditioning_data = ConditioningFieldData(conditionings=[conditioning_info])
                conditioning_name = context.conditioning.save(conditioning_data)
                return QwenImageConditioningOutput.build(conditioning_name)
