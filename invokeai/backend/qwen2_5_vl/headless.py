"""The Qwen2.5-VL encoder without its LM head, as every Qwen-Image encoder loader builds it.

Conditioning reads the inner model's hidden states, so nothing runs the head. Its untied weight (152064 x 3584 on the
7B, about 1.09 GB in bf16) would only be held in RAM and moved to VRAM.
"""

import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration


class OmittedLMHead(torch.nn.Module):
    """Stands in for the LM head of an encoder loaded without it.

    The encoder's consumers take hidden states from the inner model, so nothing runs the head. A caller that runs
    the full forward gets this error rather than logits from an uninitialized weight.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "This Qwen2.5-VL encoder was loaded without its LM head; call `.model(...)` for hidden states."
        )


class HeadlessQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """`Qwen2_5_VLForConditionalGeneration` whose `from_pretrained` leaves `lm_head.weight` unread."""

    # A config that ties the head to the embeddings has nothing to tie it to.
    _tied_weights_keys: dict[str, str] = {}
    _keys_to_ignore_on_load_unexpected = [r"^lm_head\."]

    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.lm_head = OmittedLMHead()


# transformers picks the key conversion for the legacy checkpoint layout (`model.layers.*`, `visual.*`) by class name,
# so under its own name this class would load none of those weights.
HeadlessQwen2_5_VLForConditionalGeneration.__name__ = Qwen2_5_VLForConditionalGeneration.__name__
