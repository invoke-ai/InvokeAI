"""Where an Anima single file keeps its transformer, and the shape of the transformer built from it.

Shared by the loaders, which build the transformer, and by identification, which has to refuse a file the
loaders could not fill (see `anima_transformer_shape_contract`). Only `anima_transformer_shape_contract` imports
the network, and only when it runs: identification imports this module for every file it probes.
"""

from collections.abc import Mapping
from typing import Any

import torch

from invokeai.backend.model_manager.checkpoint_prefix import CheckpointPrefix

#: Anima's own wrapper namespaces. `net.` is the official packaging; the ComfyUI bundle uses the
#: usual `model.diffusion_model.`. `diffusion_model.` is deliberately absent -- no Anima
#: redistribution measured uses it, and `net.` is not stripped anywhere else because it is also the
#: root module name of a PiD checkpoint.
ANIMA_KEY_PREFIXES = ("model.diffusion_model.", "net.")


def strip_anima_bundle_prefix(sd: dict) -> dict:
    """Strip the transformer-key prefix from an Anima single-file checkpoint.

    Handles both packaging formats:
      - Official format: keys prefixed with `net.` (e.g. `net.blocks.0...`)
      - ComfyUI bundled format: transformer keys prefixed with `model.diffusion_model.`
        alongside `first_stage_model.*` (VAE) and `cond_stage_model.*` (text encoder).

    Only keys under the detected prefix are kept; unrelated keys from bundled
    checkpoints (VAE, text encoder) are dropped. If no known prefix is present, the
    state dict is returned unchanged.
    """
    return CheckpointPrefix.detect(sd, ANIMA_KEY_PREFIXES).strip(sd, drop_foreign=True)


# Checkpoint tensors that are not part of the transformer's in-memory state. Suffixes match
# derived buffers that the model regenerates at runtime (registered as non-persistent or
# recomputed locally); prefixes match metadata that export tools serialize alongside the
# weights (e.g. sampling schedules). Extend these tuples as new checkpoint variants surface.
_NON_MODEL_KEY_SUFFIXES = (
    ".inv_freq",
    "pos_embedder.dim_spatial_range",
    "pos_embedder.dim_temporal_range",
    "pos_embedder.seq",
)
_NON_MODEL_KEY_PREFIXES = ("model_sampling.",)


def filter_non_model_keys(sd: dict) -> dict:
    """Drop checkpoint keys that don't belong to the transformer module's state dict."""
    return {
        k: v
        for k, v in sd.items()
        if not (k.endswith(_NON_MODEL_KEY_SUFFIXES) or k.startswith(_NON_MODEL_KEY_PREFIXES))
    }


# Anima's transformer architecture at the depth of the official release. Kept at module level so tests
# can instantiate the real module graph (e.g. to pin `_skip_layerwise_casting_patterns` to actual
# dotted module paths) without duplicating these values. `num_blocks` is the only value that differs
# between redistributions -- see `anima_transformer_config`.
ANIMA_TRANSFORMER_CONFIG = {
    "max_img_h": 240,
    "max_img_w": 240,
    "max_frames": 1,
    "in_channels": 16,
    "out_channels": 16,
    "patch_spatial": 2,
    "patch_temporal": 1,
    "concat_padding_mask": True,
    "model_channels": 2048,
    "num_blocks": 28,
    "num_heads": 16,
    "mlp_ratio": 4.0,
    "crossattn_emb_channels": 1024,
    "pos_emb_cls": "rope3d",
    # Anima reuses the Cosmos-Predict2 2B Text2Image DiT, which trains with
    # rope_scale=(t=1.0, h=4.0, w=4.0). The NTK-scaled spatial RoPE base is mandatory; omitting it
    # (theta=10000 on all axes) shifts every step's velocity ~7% off and compounds into degraded
    # images. Matches diffusers CosmosTransformer3DModel rope_scale via *_extrapolation_ratio.
    "rope_h_extrapolation_ratio": 4.0,
    "rope_w_extrapolation_ratio": 4.0,
    "rope_t_extrapolation_ratio": 1.0,
    "use_adaln_lora": True,
    "adaln_lora_dim": 256,
    "extra_per_block_abs_pos_emb": False,
    "image_model": "anima",
}


def count_anima_dit_blocks(sd: Mapping[str, Any]) -> int:
    """Number of DiT blocks in a prefix-stripped Anima state dict.

    Depth-expanded finetunes keep every other dimension of the official 28-block model: Anima-2.9B
    has 40 blocks and Anima-3.8B 52, both grown by LLaMA-Pro style interleaved insertion. A model
    built at the official depth would load the first 28 of them and drop the rest as unexpected
    keys -- reported only at DEBUG, with the surviving blocks in the wrong order, so the result is
    a degraded image rather than an error.

    Raises:
        ValueError: if the state dict has no DiT blocks or the block indices have gaps.
    """
    indices = {
        int(parts[1])
        for key in sd
        if isinstance(key, str) and key.startswith("blocks.")
        for parts in [key.split(".", 2)]
        if parts[1].isdigit()
    }
    if not indices:
        raise ValueError("Anima checkpoint has no DiT blocks (no `blocks.<n>.` keys after the prefix strip).")
    count = max(indices) + 1
    if len(indices) != count:
        missing = sorted(set(range(count)) - indices)
        raise ValueError(f"Anima checkpoint has gaps in its DiT blocks: {missing[:10]} missing of 0-{count - 1}.")
    return count


def anima_transformer_config(sd: Mapping[str, Any]) -> dict[str, Any]:
    """`ANIMA_TRANSFORMER_CONFIG` at the depth of the checkpoint in `sd` (prefix-stripped)."""
    return {**ANIMA_TRANSFORMER_CONFIG, "num_blocks": count_anima_dit_blocks(sd)}


Shapes = dict[str, tuple[int, ...]]


def anima_transformer_shape_contract(sd: dict[str | int, Any]) -> tuple[Shapes, Shapes]:
    """The shapes of an Anima file's transformer tensors, and the shapes of the transformer built for it.

    Both are keyed in the model's namespace: the file's after the prefix strip and filter the loaders run, the
    model's as `AnimaTransformer` at the file's depth. Identification compares the two, so that a file the
    loader could not fill is refused at install rather than at its first render. Derived from the module itself
    rather than a list, so it cannot drift from what the loader builds. Built on the meta device, which
    allocates nothing and draws no random numbers.

    Raises:
        ValueError: The file has no DiT blocks, or gaps in them.
    """
    from invokeai.backend.anima.anima_transformer import AnimaTransformer

    present = filter_non_model_keys(strip_anima_bundle_prefix(sd))
    with torch.device("meta"):
        model = AnimaTransformer(**anima_transformer_config(present))
    return (
        {key: tuple(value.shape) for key, value in present.items()},
        {key: tuple(value.shape) for key, value in model.state_dict().items()},
    )
