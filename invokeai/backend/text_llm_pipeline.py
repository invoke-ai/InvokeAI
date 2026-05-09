import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert prompt writer for AI image generation. "
    "Given a brief description, expand it into a detailed, vivid prompt suitable for generating high-quality images. "
    "Only output the expanded prompt, nothing else."
)


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
            # Fallback for models without chat template
            if system_prompt:
                formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            else:
                formatted_prompt = prompt

        inputs = self._tokenizer(formatted_prompt, return_tensors="pt").to(device=device)
        output = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        # Decode only the newly generated tokens (exclude the input prompt tokens).
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = output[0][input_length:]
        response = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return response
