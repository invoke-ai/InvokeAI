# Initially pulled from https://github.com/black-forest-labs/flux

from torch import Tensor, nn
from transformers import PreTrainedModel, PreTrainedTokenizer


class HFEncoder(nn.Module):
    def __init__(self, encoder: PreTrainedModel, tokenizer: PreTrainedTokenizer, is_clip: bool, max_length: int):
        super().__init__()
        self.max_length = max_length
        self.is_clip = is_clip
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"
        self.tokenizer = tokenizer
        self.hf_module = encoder
        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
