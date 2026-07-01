import torch
from PIL.Image import Image
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor


class LlavaOnevisionPipeline:
    """A wrapper for a LLaVA Onevision model + processor."""

    def __init__(self, vllm_model: LlavaOnevisionForConditionalGeneration, processor: LlavaOnevisionProcessor):
        self._vllm_model = vllm_model
        self._processor = processor

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
