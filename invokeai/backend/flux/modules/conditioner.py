# Initially pulled from https://github.com/black-forest-labs/flux


from torch import Tensor, nn
from transformers import PreTrainedModel, PreTrainedTokenizer


class HFEncoder(nn.Module):
    def __init__(self, encoder: PreTrainedModel, tokenizer: PreTrainedTokenizer, is_clip: bool):
        super().__init__()
        self.is_clip = is_clip
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"
        self.tokenizer = tokenizer
        self.hf_module = encoder
        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str], valid_seq_lens: list[int]) -> Tensor:
        """Encode text into a tensor.

        Args:
            text: A list of text prompts to encode.
            valid_seq_lens: A list of valid sequence lengths. The shortest valid sequence length that can contain the
                text will be used. If the largest valid sequence length cannot contain the text, the encoding will be
                truncated.
        """
        valid_seq_lens = sorted(valid_seq_lens)

        # Perform initial encoding with the maximum valid sequence length.
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=max(valid_seq_lens),
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        # Find selected_seq_len, the minimum valid sequence length that can contain all of the input tokens.
        seq_len: int = batch_encoding["length"][0].item()
        selected_seq_len = valid_seq_lens[-1]
        for len in valid_seq_lens:
            if len >= seq_len:
                selected_seq_len = len
                break

        input_ids = batch_encoding["input_ids"][..., :selected_seq_len]

        outputs = self.hf_module(
            input_ids=input_ids.to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
