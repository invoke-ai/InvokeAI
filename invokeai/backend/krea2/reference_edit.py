"""Reference-image ("edit") conditioning for the Krea-2 transformer.

Krea-2 edit LoRAs (ai-toolkit ``krea2`` trained with ``edit: true``, e.g. pose or depth "controlnet"
LoRAs) condition on clean reference latents that ride along in the image sequence. This is ComfyUI's
"index_timestep_zero" reference method, and it is what those LoRAs were trained with:

- every reference is packed like the target and appended after it: ``[text | target | ref 1 | ...]``,
  all through ``img_in`` together;
- reference ``i`` gets rotary coordinates ``(i, h, w)``
  (``sampling_utils.prepare_position_ids_with_references``);
- text and target are modulated with the real timestep, the reference span with timestep 0 -- the
  references are clean, so they are presented as fully denoised;
- attention is full joint attention over the whole sequence, and only the target tokens are returned.

Some edit LoRAs are trained with ai-toolkit's ``kv_cache`` option ("isolate_refs") instead: reference
queries attend only to reference keys, everyone else sees everything. The reference hidden states then
depend on neither the text nor the timestep, so ``precompute_reference_kv`` runs the references alone
through the blocks once, at t=0, and ``Krea2ReferenceKVCache`` hands every block's post-RoPE K/V to its
attention processor, which appends them to the keys/values of the ordinary ``[text | target]`` pass. This
is exactly the joint pass with the isolating mask, at a fraction of the cost (ostris
``_precompute_ref_kv`` / ``_forward_with_cached_refs``). Nothing in a LoRA file says which mode it was
trained with.

The diffusers forward has no seam for a second modulation, so ``forward_with_references`` restates it
(``Krea2Transformer2DModel.forward``, diffusers 0.41) from the model's own modules. Every submodule is
still called as a module, so InvokeAI's attention processors, LoRA patches and layer-wise casting
apply unchanged. Without references it reproduces the stock forward exactly; the tests pin both.

References: ComfyUI ``comfy/ldm/krea2/model.py`` (``_forward``, ``timestep_zero_index``), ostris
``ComfyUI-Krea2-Ostris-Edit`` (``_forward_with_refs``), ai-toolkit
``extensions_built_in/diffusion_models/krea2/src/mmdit.py``.
"""

from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn.functional as F


def block_forward_split(
    block: torch.nn.Module,
    hidden_states: torch.Tensor,
    temb_mod: torch.Tensor,
    temb_mod_ref: torch.Tensor,
    split: int,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``Krea2TransformerBlock.forward`` with two modulations: ``temb_mod`` for tokens ``[:split]``
    (text + target) and ``temb_mod_ref`` for tokens ``[split:]`` (the references).

    Both halves are modulated and gated separately and joined again, so attention and the MLP still see
    the whole sequence. The arithmetic per token is the stock block's, out of place, so mixed parameter
    dtypes promote exactly as they do there.
    """
    modulation = temb_mod.unflatten(-1, (6, -1)) + block.scale_shift_table
    ref_modulation = temb_mod_ref.unflatten(-1, (6, -1)) + block.scale_shift_table
    prescale, preshift, pregate, postscale, postshift, postgate = modulation.unbind(-2)
    ref_prescale, ref_preshift, ref_pregate, ref_postscale, ref_postshift, ref_postgate = ref_modulation.unbind(-2)

    def modulate(
        normed: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        ref_scale: torch.Tensor,
        ref_shift: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [(1.0 + scale) * normed[:, :split] + shift, (1.0 + ref_scale) * normed[:, split:] + ref_shift], dim=1
        )

    def gate(output: torch.Tensor, gate: torch.Tensor, ref_gate: torch.Tensor) -> torch.Tensor:
        return torch.cat([gate * output[:, :split], ref_gate * output[:, split:]], dim=1)

    attn_out = block.attn(
        modulate(block.norm1(hidden_states), prescale, preshift, ref_prescale, ref_preshift),
        attention_mask=attention_mask,
        image_rotary_emb=image_rotary_emb,
    )
    hidden_states = hidden_states + gate(attn_out, pregate, ref_pregate)
    ff_out = block.ff(modulate(block.norm2(hidden_states), postscale, postshift, ref_postscale, ref_postshift))
    hidden_states = hidden_states + gate(ff_out, postgate, ref_postgate)
    return hidden_states


def forward_with_references(
    transformer: torch.nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    position_ids: torch.Tensor,
    reference_hidden_states: torch.Tensor | None,
    encoder_attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Predict the velocity for the target tokens with clean reference tokens in the sequence.

    Args:
        transformer: a diffusers ``Krea2Transformer2DModel``.
        hidden_states: packed noisy target latents, ``(B, target_len, in_channels)``.
        encoder_hidden_states: tapped text-encoder states, ``(B, text_len, num_text_layers, text_dim)``.
        timestep: flow time in ``[0, 1]``, ``(B,)``.
        position_ids: ``(text_len + target_len + reference_len, 3)``, from
            ``prepare_position_ids_with_references``.
        reference_hidden_states: all packed reference latents concatenated along the token axis,
            ``(B or 1, reference_len, in_channels)``. ``None`` runs this restated forward with the stock blocks;
            it exists so the restatement can be pinned against ``Krea2Transformer2DModel.forward`` -- the denoise
            node itself calls the stock transformer when there are no references.
        encoder_attention_mask: optional ``(B, text_len)`` boolean mask of valid text tokens.

    Returns:
        The velocity for the target tokens only, ``(B, target_len, in_channels)``.
    """
    batch_size, image_seq_len, _ = hidden_states.shape
    text_seq_len = encoder_hidden_states.shape[1]
    reference_seq_len = 0 if reference_hidden_states is None else reference_hidden_states.shape[1]
    expected_positions = text_seq_len + image_seq_len + reference_seq_len
    if position_ids.shape != (expected_positions, 3):
        raise ValueError(
            f"Krea-2 position_ids have shape {tuple(position_ids.shape)}, but text, target and reference tokens "
            f"need ({expected_positions}, 3)."
        )

    temb = transformer.time_embed(timestep, dtype=hidden_states.dtype)
    temb_mod = transformer.time_mod_proj(F.gelu(temb, approximate="tanh"))

    text_attention_mask = None
    attention_mask = None
    if encoder_attention_mask is not None:
        text_attention_mask = encoder_attention_mask[:, None, None, :]
        image_mask = encoder_attention_mask.new_ones((batch_size, image_seq_len + reference_seq_len))
        attention_mask = torch.cat([encoder_attention_mask, image_mask], dim=1)[:, None, None, :]

    encoder_hidden_states = transformer.text_fusion(encoder_hidden_states, attention_mask=text_attention_mask)
    encoder_hidden_states = transformer.txt_in(encoder_hidden_states)

    image_tokens = hidden_states
    if reference_hidden_states is not None:
        references = reference_hidden_states.to(dtype=hidden_states.dtype).expand(batch_size, -1, -1)
        image_tokens = torch.cat([hidden_states, references], dim=1)
    hidden_states = torch.cat([encoder_hidden_states, transformer.img_in(image_tokens)], dim=1)

    image_rotary_emb = transformer.rotary_emb(position_ids)

    target_end = text_seq_len + image_seq_len
    if reference_hidden_states is None:
        for block in transformer.transformer_blocks:
            hidden_states = block(hidden_states, temb_mod, image_rotary_emb, attention_mask)
    else:
        # The references are clean latents, so they are modulated as at t=0 on every step.
        temb_ref = transformer.time_embed(torch.zeros_like(timestep), dtype=hidden_states.dtype)
        temb_mod_ref = transformer.time_mod_proj(F.gelu(temb_ref, approximate="tanh"))
        for block in transformer.transformer_blocks:
            hidden_states = block_forward_split(
                block, hidden_states, temb_mod, temb_mod_ref, target_end, image_rotary_emb, attention_mask
            )

    # The final layer is modulated with the real timestep: it only ever sees the target tokens.
    return transformer.final_layer(hidden_states[:, text_seq_len:target_end], temb)


class Krea2ReferenceKVMode(Enum):
    OFF = "off"
    CAPTURE = "capture"
    APPEND = "append"


@dataclass
class Krea2ReferenceKVCache:
    """Per-block reference K/V shared between ``precompute_reference_kv`` and the blocks' attention processors.

    Tensors are ``[B, kv_heads, reference_len, head_dim]``, post-RoPE and pre GQA expansion. The processors
    stay installed on the cached transformer after the invocation, so :meth:`clear` belongs on the denoise
    node's exit stack.
    """

    mode: Krea2ReferenceKVMode = Krea2ReferenceKVMode.OFF
    _cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self._cache)

    def store(self, block_index: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self._cache[block_index] = (key, value)

    def extend(
        self, block_index: int, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this block's cached reference K/V to the live keys/values along the token axis."""
        if attention_mask is not None:
            # A mask sized for the live keys would not cover the appended ones. The denoise node never passes
            # one with references (no text padding mask; regional prompts are refused).
            raise ValueError("The Krea-2 reference K/V cache does not support attention masks.")
        try:
            cached_key, cached_value = self._cache[block_index]
        except KeyError as e:
            raise RuntimeError(f"Krea-2 reference K/V cache holds nothing for block {block_index}.") from e
        batch_size = key.shape[0]
        return (
            torch.cat([key, cached_key.expand(batch_size, -1, -1, -1)], dim=2),
            torch.cat([value, cached_value.expand(batch_size, -1, -1, -1)], dim=2),
        )

    def clear(self) -> None:
        """Drop every retained tensor. Wired to the denoise node's exit stack."""
        self.mode = Krea2ReferenceKVMode.OFF
        self._cache.clear()


def precompute_reference_kv(
    transformer: torch.nn.Module,
    reference_hidden_states: torch.Tensor,
    reference_position_ids: torch.Tensor,
    cache: Krea2ReferenceKVCache,
) -> None:
    """Fill ``cache`` from one pass of the references alone, then switch it to appending.

    The references are modulated at t=0 and positioned by their own ids ``(i, h, w)``; each block's attention
    processor records its K/V, and the reference hidden states continue into the next block. Run it once per
    generation, with the LoRA patches already applied: they change the reference K/V.
    """
    hidden_states = transformer.img_in(reference_hidden_states)
    zero_timestep = torch.zeros(hidden_states.shape[0], device=hidden_states.device, dtype=hidden_states.dtype)
    temb_ref = transformer.time_embed(zero_timestep, dtype=hidden_states.dtype)
    temb_mod_ref = transformer.time_mod_proj(F.gelu(temb_ref, approximate="tanh"))
    image_rotary_emb = transformer.rotary_emb(reference_position_ids)

    cache.clear()
    cache.mode = Krea2ReferenceKVMode.CAPTURE
    try:
        for block in transformer.transformer_blocks:
            hidden_states = block(hidden_states, temb_mod_ref, image_rotary_emb, None)
    except BaseException:
        cache.clear()
        raise
    if len(cache) != len(transformer.transformer_blocks):
        cache.clear()
        raise RuntimeError(
            "The Krea-2 attention processors did not record the reference K/V of every block; the reference K/V "
            "cache needs the processors built with it."
        )
    cache.mode = Krea2ReferenceKVMode.APPEND


def reference_kv_cache_bytes(
    *, num_layers: int, num_heads: int, num_kv_heads: int, head_dim: int, reference_seq_len: int, element_size: int
) -> int:
    """Memory the reference K/V cache adds to a denoise step.

    The retained cache (one key and one value per block, at the KV head count) plus one block's transient
    reference keys/values as attention sees them: concatenated to the live ones and, on builds without a
    fused grouped-query kernel, expanded to the query head count.
    """
    retained = num_layers * 2 * num_kv_heads * head_dim * reference_seq_len * element_size
    transient = 2 * num_heads * head_dim * reference_seq_len * element_size
    return retained + transient
