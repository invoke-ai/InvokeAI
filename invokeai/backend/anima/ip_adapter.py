"""The SigLIP2 IP-Adapter for Anima (LuciferTC/Anima-IP-Adapter).

Reference implementation: https://github.com/LuciferTC9527/ComfyUI-Anima_IP-Adapter (`nodes.py`, Apache-2.0).

The adapter adds a second, image-conditioned attention to every DiT block. Its query is the block's own
cross-attention query (`cross_attn.q_norm(cross_attn.q_proj(x))` on the adaLN-modulated input, so a LoRA or
ControlNet-LLLite on `q_proj` reaches it too); its keys and values are per-block projections of the SigLIP2 image
tokens, with no key norm and no output projection; a per-block gate computed from the timestep embedding scales it,
and it is added after the whole block. The checkpoint also carries a PEFT LoRA trained with the adapter. The
reference applies only its cross-attention part, at scale 1.0, and so does `lora_patch`.

Several reference images may share one adapter: their scaled tokens are concatenated along the token axis into one
key/value set, so each block runs one attention per adapter rather than one per image. The reference takes one image
per adapter; with one image this is exactly its computation.
"""

import re
from functools import partial
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Sequence

import accelerate
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.transforms.functional import to_tensor

from invokeai.backend.anima.anima_transformer import CosmosAttention, IPAdapterBlockFn
from invokeai.backend.anima.block_layout import adapter_block_positions
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

if TYPE_CHECKING:
    from transformers import SiglipVisionModel

_IP_KEY_RE = re.compile(r"^blocks\.(\d+)\.(ip_k_proj|ip_v_proj|adaln_ip\.1)\.(weight|bias)$")
# The reference loader's own filter: only the cross-attention projections of the bundled LoRA are applied.
_CROSS_ATTN_LORA_KEY_RE = re.compile(
    r"^lora\.base_model\.model\.blocks\.(\d+)\.cross_attn\.(q_proj|k_proj|v_proj|output_proj)\.lora_([AB])"
    r"(?:\.default)?\.weight$"
)
_LORA_KEY_PREFIX = "lora."

# Metadata switches of the reference's later training variants. This checkpoint has both off; a file with either on
# computes something else, so it is refused rather than run as if it were this one.
_UNSUPPORTED_METADATA_FLAGS = ("ip_norm_keys", "ip_inject_before_mlp")


# google/siglip2-base-patch16-512, the encoder the adapter was trained with: 512x512 input, 1024 tokens of 768.
SIGLIP_IMAGE_SIZE = 512


def siglip_pixel_values(image: Image.Image, size: int = SIGLIP_IMAGE_SIZE) -> torch.Tensor:
    """The reference's preprocessing: the longest side to `size` (bilinear), centred on a black square, mean/std 0.5.

    Not `SiglipImageProcessor`, which stretches to the square: the adapter was trained on letterboxed images.
    Returns (1, 3, size, size) float32.
    """
    image = image.convert("RGB")
    width, height = image.size
    ratio = size / max(width, height)
    new_width, new_height = max(1, round(width * ratio)), max(1, round(height * ratio))
    square = Image.new("RGB", (size, size), (0, 0, 0))
    square.paste(
        image.resize((new_width, new_height), Image.Resampling.BILINEAR),
        ((size - new_width) // 2, (size - new_height) // 2),
    )
    return ((to_tensor(square) - 0.5) / 0.5).unsqueeze(0)


def siglip_fp32_working_bytes(encoder: "SiglipVisionModel") -> int:
    """What `encode_image_tokens` allocates beside the cached encoder: a float32 copy of its weights."""
    return sum(p.numel() for p in encoder.parameters()) * 4


@torch.no_grad()
def encode_image_tokens(encoder: "SiglipVisionModel", pixel_values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """SigLIP2 `last_hidden_state` for `pixel_values`, with float32 activations.

    Known deviation from the reference, which loads the encoder in float32: the shared SigLIP loader (FLUX Redux uses it
    too) holds the encoder at the session's inference dtype, so these float32 copies, made on `device` rather than cast
    in place, upcast weights already rounded to that dtype. With bf16 weights that moves the tokens by about 0.4%
    relative L2 (measured on random weights).
    """
    params = {name: p.to(device=device, dtype=torch.float32) for name, p in encoder.named_parameters()}
    outputs = torch.func.functional_call(
        encoder,
        params,
        (pixel_values.to(device=device, dtype=torch.float32),),
        {"interpolate_pos_encoding": True},
    )
    return outputs.last_hidden_state


def validate_anima_ip_adapter(keys: Iterable[str], metadata: Optional[Mapping[str, str]]) -> None:
    """Raise `ValueError` for a checkpoint of a reference variant `AnimaIPAdapter` does not compute.

    Shared by model identification and the loader, so such a file is refused at install rather than at generation.
    """
    for flag in _UNSUPPORTED_METADATA_FLAGS:
        if (metadata or {}).get(flag, "False").lower() == "true":
            raise ValueError(f"Unsupported Anima IP-Adapter: it was trained with `{flag}`, which is not supported.")
    unsupported = [key for key in keys if not _IP_KEY_RE.match(key) and not key.startswith(_LORA_KEY_PREFIX)]
    if unsupported:
        raise ValueError(
            "Unsupported Anima IP-Adapter: it has weights this adapter does not use, e.g. "
            f"{', '.join(sorted(unsupported)[:3])}."
        )


class AnimaIPAdapterBlock(nn.Module):
    def __init__(self, token_dim: int, inner_dim: int, x_dim: int):
        super().__init__()
        self.ip_k_proj = nn.Linear(token_dim, inner_dim, bias=True)
        self.ip_v_proj = nn.Linear(token_dim, inner_dim, bias=True)
        self.adaln_ip = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, x_dim, bias=True))


class AnimaIPAdapter(nn.Module):
    """The adapter's per-block weights, plus the cross-attention LoRA trained with them."""

    def __init__(self, num_blocks: int, token_dim: int, inner_dim: int, x_dim: int, lora: ModelPatchRaw):
        super().__init__()
        self.blocks = nn.ModuleList([AnimaIPAdapterBlock(token_dim, inner_dim, x_dim) for _ in range(num_blocks)])
        self._lora = lora

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def lora_patch(self) -> ModelPatchRaw:
        """The bundled LoRA's cross-attention q/k/v/output layers, keyed for `ANIMA_LORA_TRANSFORMER_PREFIX`."""
        return self._lora

    @classmethod
    def from_state_dict(
        cls, sd: Mapping[str, torch.Tensor], metadata: Optional[Mapping[str, str]] = None
    ) -> "AnimaIPAdapter":
        """Build the adapter from its checkpoint, adopting the checkpoint's tensors rather than copying them."""
        # Imported here: importing the LoRA conversion module before `model_manager.load` trips an import cycle.
        from invokeai.backend.patches.lora_conversions.anima_lora_conversion_utils import (
            lora_model_from_anima_state_dict,
        )

        validate_anima_ip_adapter(sd.keys(), metadata)

        ip_sd: dict[str, torch.Tensor] = {}
        lora_pairs: dict[str, dict[str, torch.Tensor]] = {}
        for key, tensor in sd.items():
            if _IP_KEY_RE.match(key):
                ip_sd[key] = tensor
            elif match := _CROSS_ATTN_LORA_KEY_RE.match(key):
                block, projection, side = match.groups()
                lora_pairs.setdefault(f"blocks.{block}.cross_attn.{projection}", {})[side] = tensor

        block_indices = sorted({int(_IP_KEY_RE.match(key).group(1)) for key in ip_sd})  # type: ignore[union-attr]
        if not block_indices:
            raise ValueError("Not an Anima IP-Adapter: no `blocks.N.ip_k_proj` weights.")
        num_blocks = block_indices[-1] + 1
        inner_dim, token_dim = ip_sd[f"blocks.{block_indices[0]}.ip_k_proj.weight"].shape
        x_dim = ip_sd[f"blocks.{block_indices[0]}.adaln_ip.1.weight"].shape[0]

        # As the reference does, a LoRA layer missing one of its two halves is skipped. No alpha key, so the PEFT
        # conversion scales by 1.0 -- the reference's alpha / rank with alpha == rank.
        lora_sd = {
            f"{layer}.lora_{side}.weight": pair[side] for layer, pair in lora_pairs.items() if len(pair) == 2
            for side in ("A", "B")
        }  # fmt: skip
        lora = lora_model_from_anima_state_dict(lora_sd)

        with accelerate.init_empty_weights():
            model = cls(num_blocks, token_dim, inner_dim, x_dim, lora)
        # Outside the empty-weights scope: it patches `register_parameter`, through which `assign=True` installs
        # each tensor, so loading inside it would put every weight back on the meta device. Strict: every block from
        # 0 to the highest must be present, and nothing else.
        model.load_state_dict(ip_sd, strict=True, assign=True)
        return model


class AnimaIPAdapterExtension:
    """The reference images driving one `AnimaIPAdapter` for one denoise run."""

    def __init__(
        self,
        adapter: AnimaIPAdapter,
        images: Sequence[tuple[torch.Tensor, float]],
        transformer_depth: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Args:
            images: Per reference image, its SigLIP2 `last_hidden_state`, (1, num_tokens, token_dim), and its
                strength, which scales the tokens before the (biased) key/value projections, as the reference does.
        """
        if not images:
            raise ValueError("An Anima IP-Adapter extension needs at least one reference image.")
        self._adapter = adapter
        self._tokens = [
            (tokens.to(device=device, dtype=torch.float32) * strength).to(dtype=dtype) for tokens, strength in images
        ]
        self._adapter_block_for: dict[int, int] = self._map_blocks(adapter.num_blocks, transformer_depth)

    @staticmethod
    def _map_blocks(adapter_depth: int, transformer_depth: int) -> dict[int, int]:
        """Transformer block index -> adapter block index.

        A depth-expanded Anima keeps the blocks the adapter was trained on at the positions `block_layout` records,
        the same mapping its LoRAs and LLLite adapters use; the new blocks get no IP term. Untested on 2.9B/3.8B.
        """
        positions = adapter_block_positions(adapter_depth - 1, transformer_depth)
        if positions is None:
            return {i: i for i in range(min(adapter_depth, transformer_depth))}
        return {target: source for source, target in enumerate(positions)}

    def block_fns(
        self, transformer_blocks: Sequence[nn.Module], active: Optional[Sequence[int]] = None
    ) -> list[Optional[IPAdapterBlockFn]]:
        """This adapter's term for each block of the transformer, or None for a block it does not reach.

        Their tokens are concatenated here, once per call: a caller whose set of images repeats across steps keeps
        the result rather than calling again.

        Args:
            active: Indices into the constructor's `images` taking part, in that order; all of them by default.
        """
        indices = range(len(self._tokens)) if active is None else active
        if not indices:
            raise ValueError("No active reference image; the caller should add no term instead.")
        tokens = self._tokens[indices[0]] if len(indices) == 1 else torch.cat([self._tokens[i] for i in indices], 1)
        fns: list[Optional[IPAdapterBlockFn]] = []
        for index, block in enumerate(transformer_blocks):
            adapter_index = self._adapter_block_for.get(index)
            if adapter_index is None:
                fns.append(None)
                continue
            # `q_proj` is looked up at call time, as the block's own cross-attention does, so the LoRA and LLLite
            # patches applied after this is built reach the IP query too, as in the reference.
            fns.append(partial(self._block_term, self._adapter.blocks[adapter_index], block.cross_attn, tokens))
        return fns

    @staticmethod
    def _block_term(
        ip_block: AnimaIPAdapterBlock,
        cross_attn: CosmosAttention,
        image_tokens: torch.Tensor,
        x: torch.Tensor,
        emb_B_T_D: torch.Tensor,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        T = emb_B_T_D.shape[1]
        n_heads, head_dim = cross_attn.n_heads, cross_attn.head_dim
        tokens = image_tokens.to(dtype=x.dtype).expand(B, -1, -1)

        q = cross_attn.q_norm(cross_attn.q_proj(x).reshape(B, L, n_heads, head_dim).transpose(1, 2))
        k = ip_block.ip_k_proj(tokens).reshape(B, -1, n_heads, head_dim).transpose(1, 2)
        v = ip_block.ip_v_proj(tokens).reshape(B, -1, n_heads, head_dim).transpose(1, 2)
        ip = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, L // T, n_heads * head_dim)

        gate = ip_block.adaln_ip(emb_B_T_D)  # (B, T, D)
        return (gate.unsqueeze(2) * ip).reshape(B, L, -1)


def combine_block_fns(
    per_adapter: Sequence[Sequence[Optional[IPAdapterBlockFn]]],
) -> Optional[list[Optional[IPAdapterBlockFn]]]:
    """Sum several adapters' per-block terms. None when there is nothing to add, so the transformer runs unchanged."""
    if not per_adapter:
        return None
    combined: list[Optional[IPAdapterBlockFn]] = []
    for fns in zip(*per_adapter, strict=True):
        present = [fn for fn in fns if fn is not None]
        combined.append(None if not present else present[0] if len(present) == 1 else partial(_summed, present))
    return combined


def _summed(fns: list[IPAdapterBlockFn], x: torch.Tensor, emb_B_T_D: torch.Tensor) -> torch.Tensor:
    total = fns[0](x, emb_B_T_D)
    for fn in fns[1:]:
        total = total + fn(x, emb_B_T_D)
    return total
