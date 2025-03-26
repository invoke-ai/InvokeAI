from pathlib import Path
from typing import Optional

import torch
from PIL.Image import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor

from invokeai.backend.raw_model import RawModel


class LlavaOnevisionModel(RawModel):
    def __init__(self, vllm_model: LlavaOnevisionForConditionalGeneration, processor: LlavaOnevisionProcessor):
        self._vllm_model = vllm_model
        self._processor = processor

    @classmethod
    def load_from_path(cls, path: str | Path):
        vllm_model = LlavaOnevisionForConditionalGeneration.from_pretrained(path, local_files_only=True)
        assert isinstance(vllm_model, LlavaOnevisionForConditionalGeneration)
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        assert isinstance(processor, LlavaOnevisionProcessor)
        return cls(vllm_model, processor)

    def run(self, prompt: str, images: list[Image], device: torch.device, dtype: torch.dtype) -> str:
        # TODO(ryand): Tune the max number of images that are useful for the model.
        if len(images) > 3:
            raise ValueError(
                f"{len(images)} images were provided as input to the LLaVA OneVision model. "
                "Pass <=3 images for good performance."
            )

        # Define a chat history and use `apply_chat_template` to get correctly formatted prompt.
        # "content" is a list of dicts with types "text" or "image".
        content = [{"type": "text", "text": prompt}]
        # Add the correct number of images.
        for _ in images:
            content.append({"type": "image"})

        conversation = [{"role": "user", "content": content}]
        prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self._processor(images=images or None, text=prompt, return_tensors="pt").to(device=device, dtype=dtype)
        output = self._vllm_model.generate(**inputs, max_new_tokens=400, do_sample=False)
        output_str: str = self._processor.decode(output[0][2:], skip_special_tokens=True)
        # The output_str will include the prompt, so we extract the response.
        response = output_str.split("assistant\n", 1)[1].strip()
        return response

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        self._vllm_model.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        """Get size of the model in memory in bytes."""
        # HACK(ryand): Fix this issue with circular imports.
        from invokeai.backend.model_manager.load.model_util import calc_module_size

        return calc_module_size(self._vllm_model)
