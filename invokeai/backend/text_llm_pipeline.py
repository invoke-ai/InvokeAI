import threading
from typing import Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TextIteratorStreamer

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert prompt writer for AI image generation. "
    "Given a brief description, expand it into a detailed, vivid prompt suitable for generating high-quality images. "
    "Only output the expanded prompt, nothing else."
)


ProgressCallback = Callable[[int, int], None]


class TextLLMPipeline:
    """A wrapper for a causal language model + tokenizer for text generation."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self._model = model
        self._tokenizer = tokenizer

    def run(
        self,
        prompt: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 300,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float16,
        progress_callback: ProgressCallback | None = None,
    ) -> str:
        # Build messages for chat template if supported, otherwise use raw prompt.
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template is not None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            formatted_prompt: str = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            if system_prompt:
                formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            else:
                formatted_prompt = prompt

        inputs = self._tokenizer(formatted_prompt, return_tensors="pt").to(device=device)

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer,
        )

        # model.generate blocks until done; run it in a thread so we can consume the
        # streamer iteratively and emit progress.
        generation_error: list[BaseException] = []

        def _generate() -> None:
            try:
                self._model.generate(**generation_kwargs)
            except BaseException as e:
                generation_error.append(e)

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        chunks: list[str] = []
        token_count = 0
        for chunk in streamer:
            if not chunk:
                continue
            chunks.append(chunk)
            # The streamer yields decoded text chunks rather than individual tokens.
            # Re-tokenizing each chunk to count tokens is expensive; instead approximate
            # by re-tokenizing the accumulated text. This is exact enough for a progress bar.
            token_count = len(self._tokenizer.encode("".join(chunks), add_special_tokens=False))
            if progress_callback is not None:
                progress_callback(min(token_count, max_new_tokens), max_new_tokens)

        thread.join()
        if generation_error:
            raise generation_error[0]

        return "".join(chunks).strip()
