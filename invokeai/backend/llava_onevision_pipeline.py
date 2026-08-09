import queue
import threading
import time
from typing import Callable

import torch
from PIL.Image import Image
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor, TextIteratorStreamer

ProgressCallback = Callable[[int, int], None]

# Backstop timeout (seconds) for the streamer's blocking queue.get() between tokens.
# The common failure mode — generate() raising — is handled explicitly by calling
# streamer.end() in the worker's except block, so this only guards the rarer case
# where generate() hangs without raising and never signals end(). It is deliberately
# generous because first-token latency on large models can be several seconds.
STREAM_TIMEOUT = 120.0

# Minimum interval (seconds) between progress emissions. Each emission re-encodes the
# full accumulated text (O(n^2) overall) and pushes a socket event, so throttling keeps
# the cost bounded when max_new_tokens is large. A final emission after the loop ensures
# the reported token count is exact regardless of throttling.
PROGRESS_EMIT_INTERVAL = 0.1


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
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=STREAM_TIMEOUT)
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
                # transformers only calls streamer.end() on the normal exit of the
                # generation loop, so on failure we must signal it ourselves or the
                # consumer below blocks forever on the streamer's queue.
                streamer.end()

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        chunks: list[str] = []
        last_emit = 0.0

        def _emit_progress() -> None:
            token_count = len(tokenizer.encode("".join(chunks), add_special_tokens=False))
            if progress_callback is not None:
                progress_callback(min(token_count, max_new_tokens), max_new_tokens)

        try:
            for chunk in streamer:
                if not chunk:
                    continue
                chunks.append(chunk)
                now = time.monotonic()
                if progress_callback is not None and now - last_emit >= PROGRESS_EMIT_INTERVAL:
                    _emit_progress()
                    last_emit = now
        except queue.Empty as e:
            # The streamer timed out waiting for the next token: generate() stalled
            # without raising and without signalling end(). Surface any captured error,
            # otherwise raise a timeout rather than block on thread.join() below.
            if generation_error:
                raise generation_error[0] from e
            raise RuntimeError(f"Image-to-prompt generation stalled (no output for {STREAM_TIMEOUT}s)") from e

        # Guarantee a final emission so the reported token count is exact even if the
        # last increment was throttled.
        if progress_callback is not None and chunks:
            _emit_progress()

        thread.join()
        if generation_error:
            raise generation_error[0]

        return "".join(chunks).strip()
