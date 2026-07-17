import queue
import threading
import time
from typing import Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TextIteratorStreamer

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert prompt writer for AI image generation. "
    "Given a brief description, expand it into a detailed, vivid prompt suitable for generating high-quality images. "
    "Only output the expanded prompt, nothing else."
)


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

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=STREAM_TIMEOUT
        )
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
                # transformers only calls streamer.end() on the normal exit of the
                # generation loop, so on failure we must signal it ourselves or the
                # consumer below blocks forever on the streamer's queue.
                streamer.end()

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        chunks: list[str] = []
        token_count = 0
        last_emit = 0.0

        def _emit_progress() -> None:
            nonlocal token_count
            # The streamer yields decoded text chunks rather than individual tokens.
            # Re-tokenizing each chunk to count tokens is expensive; instead approximate
            # by re-tokenizing the accumulated text. This is exact enough for a progress bar.
            token_count = len(self._tokenizer.encode("".join(chunks), add_special_tokens=False))
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
            raise RuntimeError(f"Text generation stalled (no output for {STREAM_TIMEOUT}s)") from e

        # Guarantee a final emission so the reported token count is exact even if the
        # last increment was throttled.
        if progress_callback is not None and chunks:
            _emit_progress()

        thread.join()
        if generation_error:
            raise generation_error[0]

        return "".join(chunks).strip()
