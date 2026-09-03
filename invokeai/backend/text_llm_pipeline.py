import queue
import threading
import time
from typing import Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TextIteratorStreamer
from transformers.generation.logits_process import LogitsProcessor

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


_seeded_multinomial_state = threading.local()


def _get_original_multinomial(multinomial):
    """Unwrap a prior InvokeAI patch so module reloads do not stack wrappers."""
    return getattr(multinomial, "_invokeai_original_multinomial", multinomial)


_original_torch_multinomial = _get_original_multinomial(torch.multinomial)
_original_tensor_multinomial = _get_original_multinomial(torch.Tensor.multinomial)


class _SeededMultinomialMode:
    """Use an invocation-local generator for multinomial sampling in this thread."""

    def __init__(self, seed: int):
        self._seed = seed
        self._generators: dict[torch.device, torch.Generator] = {}
        self._previous: "_SeededMultinomialMode | None" = None

    def __enter__(self) -> "_SeededMultinomialMode":
        self._previous = getattr(_seeded_multinomial_state, "active", None)
        _seeded_multinomial_state.active = self
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._previous is None:
            del _seeded_multinomial_state.active
        else:
            _seeded_multinomial_state.active = self._previous

    def _get_generator(self, device: torch.device) -> torch.Generator:
        generator_device = torch.device("cpu") if device.type == "mps" else device
        if generator_device not in self._generators:
            self._generators[generator_device] = torch.Generator(device=generator_device).manual_seed(self._seed)
        return self._generators[generator_device]


def _sample_with_seed(input: torch.Tensor, args: tuple, kwargs: dict, original_multinomial) -> torch.Tensor:
    mode: _SeededMultinomialMode | None = getattr(_seeded_multinomial_state, "active", None)
    one_shot = False
    if mode is None:
        mode = getattr(_seeded_multinomial_state, "next_mode", None)
        one_shot = mode is not None

    try:
        if mode is None or (kwargs.get("generator") is not None and not one_shot):
            return original_multinomial(input, *args, **kwargs)

        kwargs = dict(kwargs)
        kwargs["generator"] = mode._get_generator(input.device)
        if input.device.type != "mps":
            return original_multinomial(input, *args, **kwargs)

        # MPS has no usable device generator. Sampling on CPU keeps the invocation-local seed,
        # then returns the selected token to the model device. Generation already synchronizes
        # on each sampled token, so this is a correctness fallback rather than a fast path.
        output = original_multinomial(input.cpu(), *args, **kwargs)
        return output.to(input.device)
    finally:
        if one_shot:
            _seeded_multinomial_state.__dict__.pop("next_mode", None)


def _seeded_torch_multinomial(input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return _sample_with_seed(input, args, kwargs, _original_torch_multinomial)


def _seeded_tensor_multinomial(input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return _sample_with_seed(input, args, kwargs, _original_tensor_multinomial)


_seeded_torch_multinomial._invokeai_original_multinomial = _original_torch_multinomial
_seeded_tensor_multinomial._invokeai_original_multinomial = _original_tensor_multinomial


class _SeededMultinomialProcessor(LogitsProcessor):
    """Arm seeded sampling for the multinomial call immediately after logits processing."""

    def __init__(self, mode: _SeededMultinomialMode):
        self._mode = mode

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        _seeded_multinomial_state.next_mode = self._mode
        return scores


# Transformers calls torch.multinomial directly today, while Tensor.multinomial is a supported
# equivalent used by other callers. Install wrappers once so ordinary calls pay only a small
# thread-local check; model forward operators are untouched.
if torch.multinomial is not _seeded_torch_multinomial:
    torch.multinomial = _seeded_torch_multinomial
if torch.Tensor.multinomial is not _seeded_tensor_multinomial:
    torch.Tensor.multinomial = _seeded_tensor_multinomial


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
        seed: int = 0,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float16,
        progress_callback: ProgressCallback | None = None,
    ) -> str:
        # Build messages for chat template if supported, otherwise use raw prompt.
        used_chat_template = (
            hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template is not None
        )
        if used_chat_template:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            try:
                # enable_thinking=False makes reasoning models (Qwen3, DeepSeek-R1 distills and
                # other templates that honour the flag) emit an empty thinking block instead of a
                # chain of thought, so prompt expansion returns the prompt rather than the model's
                # reasoning. Templates that do not know the flag ignore it.
                formatted_prompt: str = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except Exception as e:  # noqa: BLE001 - jinja2 TemplateError is not importable here
                # Some chat templates (notably Gemma) reject a dedicated "system" role. Fold the
                # system prompt into the first user turn and retry instead of failing the expansion.
                if system_prompt and "system role" in str(e).lower():
                    merged = [{"role": "user", "content": f"{system_prompt}\n\n{prompt}"}]
                    formatted_prompt = self._tokenizer.apply_chat_template(
                        merged, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                else:
                    raise
        else:
            if system_prompt:
                formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            else:
                formatted_prompt = prompt

        # A rendered chat template already carries the model's control tokens, so adding special
        # tokens again duplicates BOS for the families that use one (Gemma, Llama). The raw
        # fallback prompt above has no control tokens and still needs them.
        inputs = self._tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=not used_chat_template).to(
            device=device
        )

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
            # Arm the generator only for Transformers' sampling call. Wrapping all of
            # generate() in TorchFunctionMode intercepts every model-forward operator.
            logits_processor=[_SeededMultinomialProcessor(_SeededMultinomialMode(seed))],
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
