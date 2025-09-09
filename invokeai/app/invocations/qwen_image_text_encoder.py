# Copyright (c) 2024, Brandon W. Rising and the InvokeAI Development Team
"""Qwen-Image text encoding invocation."""

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, UIComponent
from invokeai.app.invocations.model import Qwen2_5VLField
from invokeai.app.invocations.primitives import QwenImageConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData
from invokeai.backend.util.devices import TorchDevice


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
        
        device = TorchDevice.choose_torch_device()
        
        # Load the Qwen2.5-VL tokenizer and text encoder
        with context.models.load(self.qwen2_5_vl.tokenizer) as tokenizer_info, \
             context.models.load(self.qwen2_5_vl.text_encoder) as text_encoder_info:
            
            tokenizer = tokenizer_info.model
            text_encoder = text_encoder_info.model.to(device)
            
            try:
                # Tokenize the prompt
                # Qwen2.5-VL supports much longer sequences than CLIP
                text_inputs = tokenizer(
                    self.prompt,
                    padding="max_length",
                    max_length=1024,  # Qwen2.5-VL supports much longer sequences
                    truncation=True,
                    return_tensors="pt",
                )
                
                # Encode the text
                text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
                
                # Create a simple conditioning info that stores the embeddings
                # For now, we'll create a simple class to hold the data
                class QwenImageConditioningInfo:
                    def __init__(self, text_embeds: torch.Tensor, prompt: str):
                        self.text_embeds = text_embeds
                        self.prompt = prompt
                
                conditioning_info = QwenImageConditioningInfo(text_embeddings, self.prompt)
                conditioning_data = ConditioningFieldData(conditionings=[conditioning_info])
                
                conditioning_name = context.conditioning.save(conditioning_data)
                return QwenImageConditioningOutput.build(conditioning_name)
                
            except Exception as e:
                context.logger.error(f"Error encoding Qwen-Image text: {e}")
                # Fallback to simple text storage
                class QwenImageConditioningInfo:
                    def __init__(self, prompt: str):
                        self.prompt = prompt
                
                conditioning_info = QwenImageConditioningInfo(self.prompt)
                conditioning_data = ConditioningFieldData(conditionings=[conditioning_info])
                conditioning_name = context.conditioning.save(conditioning_data)
                return QwenImageConditioningOutput.build(conditioning_name)