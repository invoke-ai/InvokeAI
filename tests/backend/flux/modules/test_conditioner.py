import torch

from invokeai.backend.flux.modules.conditioner import HFEncoder


class FakeTokenizer:
    def __call__(
        self,
        text,
        truncation,
        max_length,
        return_length,
        return_overflowing_tokens,
        padding,
        return_tensors,
    ):
        del text, truncation, max_length, return_length, return_overflowing_tokens, padding, return_tensors
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}


class FakeEncoderOutput(dict):
    pass


class FakePartiallyLoadedEncoder(torch.nn.Module):
    def __init__(self, effective_device: torch.device):
        super().__init__()
        self.register_parameter("cpu_param", torch.nn.Parameter(torch.ones(1)))
        self.register_buffer("active_buffer", torch.ones(1, device=effective_device))
        self.forward_input_device: torch.device | None = None

    def forward(self, input_ids: torch.Tensor, attention_mask=None, output_hidden_states: bool = False):
        del attention_mask, output_hidden_states
        self.forward_input_device = input_ids.device
        return FakeEncoderOutput(pooler_output=torch.ones((1, 4), dtype=torch.float32))


def test_hf_encoder_uses_effective_device_for_partially_loaded_models():
    effective_device = torch.device("meta")
    encoder = FakePartiallyLoadedEncoder(effective_device=effective_device)
    hf_encoder = HFEncoder(encoder=encoder, tokenizer=FakeTokenizer(), is_clip=True, max_length=77)

    hf_encoder(["test prompt"])

    assert encoder.forward_input_device == effective_device
