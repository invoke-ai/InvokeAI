import threading
from typing import Callable

import torch
from PIL.Image import Image
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor, TextIteratorStreamer

ProgressCallback = Callable[[int, int], None]


class LlavaOnevisionPipeline:
    """A wrapper for a LLaVA Onevision model + processor."""

    def __init__(self, vllm_model: LlavaOnevisionForConditionalGeneration, processor: LlavaOnevisionProcessor):
        self._vllm_model = vllm_model
        self._processor = processor

    def run(
        self,
        prompt: str,
        images: list[Image],
        device: torch.device,
        dtype: torch.dtype,
        max_new_tokens: int = 400,
        progress_callback: ProgressCallback | None = None,
    ) -> str:
        # TODO(ryand): Tune the max number of images that are useful for the model.
        if len(images) > 3:
            raise ValueError(
                f"{len(images)} images were provided as input to the LLaVA OneVision model. "
                "Pass <=3 images for good performance."
            )

        # Define a chat history and use `apply_chat_template` to get correctly formatted prompt.
        # "content" is a list of dicts with types "text" or "image".
        content = [{"type": "text", "text": prompt}]
        for _ in images:
            content.append({"type": "image"})

        conversation = [{"role": "user", "content": content}]
        formatted_prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self._processor(images=images or None, text=formatted_prompt, return_tensors="pt").to(
            device=device, dtype=dtype
        )

        tokenizer = self._processor.tokenizer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=streamer,
        )

        generation_error: list[BaseException] = []

        def _generate() -> None:
            try:
                self._vllm_model.generate(**generation_kwargs)
            except BaseException as e:
                generation_error.append(e)

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        chunks: list[str] = []
        for chunk in streamer:
            if not chunk:
                continue
            chunks.append(chunk)
            if progress_callback is not None:
                token_count = len(tokenizer.encode("".join(chunks), add_special_tokens=False))
                progress_callback(min(token_count, max_new_tokens), max_new_tokens)

        thread.join()
        if generation_error:
            raise generation_error[0]

        return "".join(chunks).strip()
