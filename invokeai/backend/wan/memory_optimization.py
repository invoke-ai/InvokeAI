from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput

WAN_ACTIVATION_CHUNK_SIZE = 1024


@dataclass
class _CompactTimestepConditioning:
    timestep_embeddings: torch.Tensor
    modulation: torch.Tensor
    indices: torch.Tensor


def _get_modulation(
    block: torch.nn.Module,
    temb: torch.Tensor | _CompactTimestepConditioning,
    start: int,
    end: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(temb, _CompactTimestepConditioning):
        modulation = temb.modulation[temb.indices[:, start:end]].to(device=device, dtype=torch.float32)
        modulation = block.scale_shift_table.unsqueeze(0).to(device) + modulation  # type: ignore[attr-defined]
        chunks = modulation.chunk(6, dim=2)
        return tuple(chunk.squeeze(2) for chunk in chunks)  # type: ignore[return-value]
    if temb.ndim == 4:
        modulation = block.scale_shift_table.unsqueeze(0).to(device) + temb[:, start:end].to(  # type: ignore[attr-defined]
            device=device, dtype=torch.float32
        )
        chunks = modulation.chunk(6, dim=2)
        return tuple(chunk.squeeze(2) for chunk in chunks)  # type: ignore[return-value]

    modulation = block.scale_shift_table.to(device) + temb.to(device=device, dtype=torch.float32)  # type: ignore[attr-defined]
    return modulation.chunk(6, dim=1)  # type: ignore[return-value]


def _optimized_wan_transformer_forward(
    transformer: torch.nn.Module,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: torch.Tensor | None = None,
    return_dict: bool = True,
    attention_kwargs: dict[str, Any] | None = None,
) -> Any:
    original_forward = transformer._invokeai_original_forward  # type: ignore[attr-defined]
    # The custom transformer path is only needed to compact TI2V's per-token
    # timesteps. Keep Diffusers' decorated forward for scalar timesteps and
    # non-default attention kwargs.
    if torch.is_grad_enabled() or timestep.ndim != 2 or attention_kwargs:
        return original_forward(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
        )

    batch_size, _, num_frames, height, width = hidden_states.shape
    patch_frames, patch_height, patch_width = transformer.config.patch_size  # type: ignore[attr-defined]
    post_patch_num_frames = num_frames // patch_frames
    post_patch_height = height // patch_height
    post_patch_width = width // patch_width

    rotary_emb = transformer.rope(hidden_states)  # type: ignore[attr-defined]
    hidden_states = transformer.patch_embedding(hidden_states).flatten(2).transpose(1, 2)  # type: ignore[attr-defined]

    unique_timesteps, inverse_indices = torch.unique(timestep, sorted=False, return_inverse=True)
    temb, timestep_projection, encoder_hidden_states, encoder_hidden_states_image = transformer.condition_embedder(  # type: ignore[attr-defined]
        unique_timesteps,
        encoder_hidden_states,
        encoder_hidden_states_image,
        timestep_seq_len=None,
    )
    compact_timestep = _CompactTimestepConditioning(
        timestep_embeddings=temb,
        modulation=timestep_projection.unflatten(1, (6, -1)),
        indices=inverse_indices.view_as(timestep),
    )
    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    for block in transformer.blocks:  # type: ignore[attr-defined]
        hidden_states = block(hidden_states, encoder_hidden_states, compact_timestep, rotary_emb)

    sequence_length = hidden_states.shape[1]
    chunk_size: int = transformer._invokeai_activation_chunk_size  # type: ignore[attr-defined]
    projected = hidden_states.new_empty(
        (batch_size, sequence_length, transformer.proj_out.out_features)  # type: ignore[attr-defined]
    )
    for start in range(0, sequence_length, chunk_size):
        end = min(start + chunk_size, sequence_length)
        timestep_chunk = compact_timestep.timestep_embeddings[compact_timestep.indices[:, start:end]].to(
            hidden_states.device
        )
        shift, scale = (
            transformer.scale_shift_table.unsqueeze(0).to(hidden_states.device) + timestep_chunk.unsqueeze(2)  # type: ignore[attr-defined]
        ).chunk(2, dim=2)
        normalized_chunk = (
            transformer.norm_out(hidden_states[:, start:end].float()) * (1 + scale.squeeze(2))  # type: ignore[attr-defined]
            + shift.squeeze(2)
        ).type_as(hidden_states)
        projected[:, start:end].copy_(transformer.proj_out(normalized_chunk))  # type: ignore[attr-defined]
    hidden_states = projected

    hidden_states = hidden_states.reshape(
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        patch_frames,
        patch_height,
        patch_width,
        -1,
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def _optimized_wan_block_forward(
    block: torch.nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor | _CompactTimestepConditioning,
    rotary_emb: torch.Tensor,
) -> torch.Tensor:
    original_forward = block._invokeai_original_forward  # type: ignore[attr-defined]
    chunk_size: int = block._invokeai_activation_chunk_size  # type: ignore[attr-defined]
    sequence_length = hidden_states.shape[1]

    if torch.is_grad_enabled() or (
        sequence_length <= chunk_size and not isinstance(temb, _CompactTimestepConditioning)
    ):
        return original_forward(hidden_states, encoder_hidden_states, temb, rotary_emb)

    normalized = torch.empty_like(hidden_states)
    for start in range(0, sequence_length, chunk_size):
        end = min(start + chunk_size, sequence_length)
        shift_msa, scale_msa, _, _, _, _ = _get_modulation(block, temb, start, end, hidden_states.device)
        normalized_chunk = (
            block.norm1(hidden_states[:, start:end].float()) * (1 + scale_msa) + shift_msa  # type: ignore[attr-defined]
        ).type_as(hidden_states)
        normalized[:, start:end].copy_(normalized_chunk)

    attention_output = block.attn1(normalized, None, None, rotary_emb)  # type: ignore[attr-defined]
    del normalized

    updated = torch.empty_like(hidden_states)
    for start in range(0, sequence_length, chunk_size):
        end = min(start + chunk_size, sequence_length)
        _, _, gate_msa, _, _, _ = _get_modulation(block, temb, start, end, hidden_states.device)
        updated_chunk = (hidden_states[:, start:end].float() + attention_output[:, start:end] * gate_msa).type_as(
            hidden_states
        )
        updated[:, start:end].copy_(updated_chunk)
    hidden_states = updated
    del attention_output

    normalized = torch.empty_like(hidden_states)
    for start in range(0, sequence_length, chunk_size):
        end = min(start + chunk_size, sequence_length)
        normalized[:, start:end].copy_(
            block.norm2(hidden_states[:, start:end].float()).type_as(hidden_states)  # type: ignore[attr-defined]
        )
    attention_output = block.attn2(normalized, encoder_hidden_states, None, None)  # type: ignore[attr-defined]
    hidden_states = hidden_states + attention_output
    del normalized, attention_output

    output = torch.empty_like(hidden_states)
    for start in range(0, sequence_length, chunk_size):
        end = min(start + chunk_size, sequence_length)
        _, _, _, shift_mlp, scale_mlp, gate_mlp = _get_modulation(block, temb, start, end, hidden_states.device)
        normalized_chunk = (
            block.norm3(hidden_states[:, start:end].float()) * (1 + scale_mlp) + shift_mlp  # type: ignore[attr-defined]
        ).type_as(hidden_states)
        feed_forward_output = block.ffn(normalized_chunk)  # type: ignore[attr-defined]
        output_chunk = (hidden_states[:, start:end].float() + feed_forward_output.float() * gate_mlp).type_as(
            hidden_states
        )
        output[:, start:end].copy_(output_chunk)

    return output


@contextmanager
def wan_memory_optimization(
    transformer: torch.nn.Module,
    *,
    enabled: bool,
    activation_chunk_size: int = WAN_ACTIVATION_CHUNK_SIZE,
) -> Iterator[None]:
    """Temporarily chunk Wan transformer pointwise activations during inference."""
    if not enabled:
        yield
        return
    if activation_chunk_size <= 0:
        raise ValueError("activation_chunk_size must be positive")

    blocks: Any = getattr(transformer, "blocks", None)
    if blocks is None:
        raise TypeError(f"Expected a Wan transformer with blocks, got {type(transformer).__name__}.")
    blocks = list(blocks)
    if hasattr(transformer, "_invokeai_original_forward") or any(
        hasattr(block, "_invokeai_original_forward") for block in blocks
    ):
        raise RuntimeError("Wan memory optimization context cannot be nested.")

    patched_blocks: list[tuple[torch.nn.Module, Any, bool]] = []
    original_transformer_forward = transformer.forward
    transformer_had_instance_forward = "forward" in transformer.__dict__
    patch_transformer_forward = all(
        hasattr(transformer, name)
        for name in ("condition_embedder", "patch_embedding", "proj_out", "rope", "scale_shift_table")
    )
    try:
        if patch_transformer_forward:
            transformer._invokeai_original_forward = original_transformer_forward
            transformer._invokeai_activation_chunk_size = activation_chunk_size
            transformer.forward = MethodType(_optimized_wan_transformer_forward, transformer)
        for block in blocks:
            original_forward = block.forward
            had_instance_forward = "forward" in block.__dict__
            block._invokeai_original_forward = original_forward
            block._invokeai_activation_chunk_size = activation_chunk_size
            block.forward = MethodType(_optimized_wan_block_forward, block)
            patched_blocks.append((block, original_forward, had_instance_forward))
        yield
    finally:
        for block, original_forward, had_instance_forward in patched_blocks:
            if had_instance_forward:
                block.forward = original_forward
            else:
                del block.forward
            del block._invokeai_original_forward
            del block._invokeai_activation_chunk_size
        if patch_transformer_forward:
            if transformer_had_instance_forward:
                transformer.forward = original_transformer_forward
            else:
                del transformer.forward
            del transformer._invokeai_original_forward
            del transformer._invokeai_activation_chunk_size
