"""LTX-2 prompt conditioning: the Gemma-4 tower, then the text connectors.

LTX-2 does not condition on a text encoder's last hidden state. It stacks the hidden states of
*every* Gemma layer (48 plus the embedding output) into one 188160-wide per-token feature, and a
separate connector model -- a per-token RMS norm, a per-modality projection and two small
transformers with learnable registers -- turns that into the two streams the transformer's prompt
cross-attention reads.

The two stages are split across two functions because they are two models: the caller runs the
12B tower under its own lock, releases it, and only then locks the connectors, so the largest
prompt encode holds one of them on the device at a time.
"""

import torch

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import LTX2ConditioningInfo

# Fixed costs of each stage's own activations, beside the per-layer states both of them carry.
# Measured on a W7900 (gfx1100, released int8 Gemma-4 and bf16 connectors) at 1024 tokens, as peak
# *reserved* minus the model: the tower needs 1.92 GiB against 0.72 GiB of states, the connectors
# 1.74 GiB against 0.36 GiB. The multipliers below count the copies each stage holds at once -- the
# tower keeps every layer's output and then the stacked tensor, the connectors normalize, scale and
# project the stack per modality -- and the bases cover the rest, with 16-26% margin.
_TOWER_STATE_COPIES = 2
_TOWER_BASE_BYTES = 1536 * 2**20
_CONNECTOR_STATE_COPIES = 4
_CONNECTOR_BASE_BYTES = 768 * 2**20


def estimate_tower_working_memory(text_encoder, max_sequence_length: int) -> int:
    """Estimate the working memory for one Gemma-4 prompt encode.

    Unlike the encoders every other architecture here runs, this one keeps *every* layer's hidden
    state and then stacks them: at 48 layers, 3840 wide and 1024 tokens that is 368 MiB twice over,
    which the cache would otherwise pack VRAM straight through.
    """
    config = getattr(text_encoder, "config", None)
    layers = int(getattr(config, "num_hidden_layers", 48)) + 1
    hidden = int(getattr(config, "hidden_size", 3840))
    dtype = getattr(text_encoder, "dtype", torch.bfloat16)

    return _TOWER_BASE_BYTES + _TOWER_STATE_COPIES * layers * max_sequence_length * hidden * dtype.itemsize


def estimate_connector_working_memory(hidden_states: torch.Tensor) -> int:
    """Estimate the working memory to project one prompt's stacked states into the two streams."""
    return _CONNECTOR_BASE_BYTES + _CONNECTOR_STATE_COPIES * hidden_states.numel() * hidden_states.element_size()


@torch.no_grad()
def encode_hidden_states(
    text_encoder,
    tokenizer,
    prompt: str,
    *,
    max_sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize and run the Gemma tower; returns the packed per-layer states and the token mask.

    The states are ``(1, max_sequence_length, hidden * (layers + 1))`` and the mask
    ``(1, max_sequence_length)``. Every prompt is padded to the full length -- that is the shape the
    connectors' registers are defined against -- so a short prompt costs the same as a long one.
    """
    # Gemma is trained with left padding, and the released pipelines set this on the tokenizer
    # rather than per call. Both attributes are set to the same values on every call, so the shared
    # cached tokenizer is left in the state any other caller would also put it in.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        [prompt.strip()],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    attention_mask = inputs.attention_mask.to(device)
    outputs = text_encoder(
        input_ids=inputs.input_ids.to(device),
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    return torch.stack(outputs.hidden_states, dim=-1).flatten(2, 3), attention_mask


@torch.no_grad()
def apply_connectors(connectors, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> LTX2ConditioningInfo:
    """Project the packed per-layer states into the video and audio prompt streams."""
    device = next(connectors.parameters()).device
    dtype = next(connectors.parameters()).dtype
    video_embeds, audio_embeds, mask = connectors(
        hidden_states.to(device=device, dtype=dtype),
        attention_mask.to(device),
    )
    return LTX2ConditioningInfo(
        video_embeds=video_embeds.detach().cpu(),
        audio_embeds=audio_embeds.detach().cpu(),
        attention_mask=mask.detach().cpu(),
    )
