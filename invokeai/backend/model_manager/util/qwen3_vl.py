"""Shared helpers for loading Qwen3-VL encoder checkpoints.

Used by Krea-2, Ideogram 4 and MiniMax H3, and the tower drop also by Qwen-Image's Qwen2.5-VL GGUF encoder,
whose module tree keeps the tower under the same attribute. The visual-tower helpers live here rather than beside
one family's loader because three loaders in two files drop the tower, and because what counts as
the tower has to agree with the key mapping that sits next to it -- two definitions would drift.
MiniMax H3 deliberately does not use them: it runs the tower for its reference images.

The same predicate also serves the opposite direction: Krea-2's edit mode loads the tower on its own, as a
separate model-cache entry, and lends it to the text encoder for the forwards that read reference images
(`read_qwen3vl_visual_tower_tensors`, `attach_qwen3vl_visual_tower`). Keeping both directions here is what
guarantees the tower that is lent back is exactly the one that was dropped.
"""

import json
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from invokeai.backend.quantization.fp8_scaled import FP8_WEIGHT_DTYPES, MXFP8_FORMAT, parse_quantization_metadata
from invokeai.backend.quantization.int8_convrot import read_comfy_quant_markers


def normalize_qwen3vl_rope_config(config: Any) -> Any:
    """Mirror Qwen3-VL rope_parameters into rope_scaling for Transformers compatibility.

    Some Qwen3-VL checkpoints store rope settings under ``rope_parameters``, but the installed
    transformers' Qwen3VL rotary embedding reads ``rope_scaling`` (None there) and crashes.
    """
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        rope_params = getattr(text_config, "rope_parameters", None)
        if getattr(text_config, "rope_scaling", None) is None and rope_params is not None:
            text_config.rope_scaling = rope_params
    return config


def qwen3vl_target_key(key: str) -> str:
    """Map one ComfyUI single-file Qwen3-VL key (or module path) to the transformers layout.

    ComfyUI/native layout uses a single ``model.`` prefix for both towers; transformers splits them:
    ``model.visual.*`` -> ``visual.*`` and ``model.<rest>`` (layers/embed_tokens/norm) -> ``language_model.<rest>``.

    Shared by the state-dict remap and the fp8 layer-hint remap so the two cannot drift apart: a hint
    keyed by a path the model does not have matches nothing and is silently ignored.
    """
    # Strip a leading "model." (some checkpoints prefix everything with it), then route by tower.
    key = key[len("model.") :] if key.startswith("model.") else key
    if key.startswith("visual.") or key.startswith("language_model."):
        # Already the transformers layout (e.g. "model.language_model.*" / "model.visual.*").
        return key
    # Bare language-model keys (layers.* / embed_tokens / norm) belong under language_model.
    return "language_model." + key


def drop_qwen3vl_visual_tower(model: torch.nn.Module, *, required: bool = True) -> bool:
    """Replace the Qwen3-VL visual tower with a parameter-free stub. Returns whether one was found.

    Krea-2 and Ideogram 4 condition on the language tower's hidden states alone.
    ``Qwen3VLModel.forward`` reaches ``self.visual`` only for ``pixel_values``/``pixel_values_videos``,
    which neither invocation passes, and Ideogram 4 does not go through ``forward`` at all -- it drives
    ``model.language_model`` directly. Removing the tower was measured to leave Krea-2's conditioning
    bit-identical (max abs diff 0.0). Measured resident saving on the 4B: 0.774 GiB from the bf16
    checkpoint (8.266 -> 7.492) and 0.390 GiB from the fp8_scaled one (4.499 -> 4.109), the smaller
    figure because that path was already casting the tower to fp8 storage to blunt its cost.

    With ``required`` (the default), raises when there is no tower to replace rather than letting
    ``setattr`` invent the attribute. On the single-file path a renamed tower would be caught anyway
    -- its weights would stay on meta and the completeness sweep would refuse the load -- but on the
    ``from_pretrained`` paths nothing else would notice: the renamed tower would load, a dead
    ``visual`` attribute would be added beside it, and the encoder would keep every byte this is
    meant to save, silently.

    Ideogram 4 passes ``required=False``, and that is not the same hazard: it builds whatever the
    folder's ``config.json`` declares, so an encoder without a vision tower is a shape this loader
    legitimately meets rather than a sign the architecture moved underneath us. Such a build fails
    later and on its own terms, when the caption encoder reaches for ``language_model``.

    Qwen-Image's GGUF encoder drops the Qwen2.5-VL tower through this too (pass ``model.model``): transformers keeps
    it under the same ``visual`` attribute, and llama.cpp keeps its weights in a separate ``mmproj`` file.

    MiniMax H3 genuinely runs this tower for its reference images, and loads through its own loaders in
    ``model_loaders/minimax_h3.py``; nothing here is reachable from them.

    A LoRA layer addressed at ``visual.*`` is skipped instead of applied -- possible in principle, and
    in fact produced by the converter (see ``test_krea2_lora_conversion_utils``). It could not have
    changed an image before either, for the same reason the tower can go, so the callers suppress the
    per-layer warning rather than letting it read as an adapter failure.
    """
    existing = getattr(model, "visual", None)
    if not isinstance(existing, torch.nn.Module) or isinstance(existing, torch.nn.Identity):
        if required:
            raise RuntimeError(
                f"{type(model).__name__} has no Qwen-VL visual tower to drop. The architecture has "
                "changed shape; the tower would otherwise stay resident with nothing reporting it."
            )
        return False
    model.visual = torch.nn.Identity()
    return True


def drop_qwen3vl_visual_tower_keys(sd: dict[str, Any]) -> dict[str, Any]:
    """Drop the visual tower's tensors from a single-file Qwen3-VL state dict.

    Dropping them here rather than after the load is what makes the saving a *peak* one and not only a
    resident one. ``safetensors.torch.load_file`` is mmap-backed, so on arrival these tensors are
    file-backed views costing no committed memory; what used to commit them was the work downstream --
    on the fp8 branch ``_apply_fp8_to_nn_module`` allocated a fresh fp8 copy of the whole tower, and on
    the bf16 branch the pages were faulted in at the device move. Neither happens to tensors that are
    no longer in the dict.

    The predicate routes through ``qwen3vl_target_key`` so "what counts as the visual tower" is
    defined in one place, and it matches on the module path prefix rather than a list of suffixes --
    which is what makes it take *every* sidecar a visual weight carries, including spellings this file
    does not enumerate (``scale_weight``/``scale_input`` as well as ``weight_scale``/``input_scale``,
    and ``comfy_quant``). Enumerating suffixes instead would silently start leaving orphans behind.
    """
    return {k: v for k, v in sd.items() if not is_qwen3vl_visual_key(k)}


def is_qwen3vl_visual_key(key: Any) -> bool:
    """Whether a checkpoint key (in either spelling) belongs to the visual tower, sidecars included."""
    return isinstance(key, str) and qwen3vl_target_key(key).startswith("visual.")


def qwen3vl_visual_tower_files(folder: Path) -> list[Path]:
    """The safetensors files of a transformers-layout Qwen3-VL folder that hold visual-tower tensors.

    A sharded folder names, per key, the shard holding it, so only the shards with tower keys are
    returned; an unsharded folder returns its weight files.
    """
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        weight_map: dict[str, str] = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return sorted({folder / name for key, name in weight_map.items() if is_qwen3vl_visual_key(key)})
    # Assumes one weight set per folder: no Krea-2/Qwen3-VL folder ships variant duplicates (``*.fp16.safetensors``).
    return sorted(folder.glob("*.safetensors"))


def qwen3vl_visual_tower_nbytes(files: Iterable[Path]) -> int:
    """Bytes the visual tower's tensors occupy in `files`, read from the headers alone.

    What the model cache should reserve before loading the tower on its own: sizing that load by the whole
    file would evict ten times what the tower needs.
    """
    from invokeai.backend.model_manager.model_on_disk import read_safetensors_header

    total = 0
    for path in files:
        for key, entry in read_safetensors_header(path).items():
            if is_qwen3vl_visual_key(key):
                start, end = entry["data_offsets"]
                total += end - start
    return total


def read_qwen3vl_visual_tower_tensors(files: Iterable[Path]) -> dict[str, torch.Tensor]:
    """Read only the visual tower's tensors from `files`, keyed as the checkpoint spells them.

    Filtered by key before anything is read, so the peak is the tower's size and not the encoder's: the
    language tower of a 4B single file is ten times larger and is never touched.
    """
    from safetensors import safe_open

    from invokeai.backend.quantization.fp8_scaled import safetensors_backend

    tensors: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu", backend=safetensors_backend(path)) as handle:
            for key in handle.keys():
                if is_qwen3vl_visual_key(key):
                    tensors[key] = handle.get_tensor(key)
    return tensors


@contextmanager
def attach_qwen3vl_visual_tower(model: torch.nn.Module, tower: torch.nn.Module) -> Iterator[torch.nn.Module]:
    """Lend `tower` to a `Qwen3VLModel` whose own tower `drop_qwen3vl_visual_tower` replaced, for one forward.

    Restores the parameter-free stub on exit, on success and on an exception alike, together with the 3-D rope
    deltas the multimodal forward caches on the model, so the encoder leaves exactly as it came.

    Invariant for the caller: both `model` and `tower` are separate model-cache entries, and both must be
    locked on the device *before* entering and unlocked only *after* exiting. The cache sizes, moves and
    offloads a model by walking its parameters; while attached, the tower's parameters are also the
    encoder's, so an offload or VRAM accounting of either entry inside this window would count or move the
    other's weights.

    Refuses a model that still has a real tower (attaching would discard it, silently) and a tower whose
    output width is not the language model's hidden size (its embeddings could not be scattered into the
    token sequence).
    """
    existing = getattr(model, "visual", None)
    if not isinstance(existing, torch.nn.Identity):
        raise RuntimeError(
            f"{type(model).__name__} has no dropped visual tower to replace (found {type(existing).__name__}); "
            "attaching another would discard it."
        )
    text_hidden = getattr(getattr(model.config, "text_config", None), "hidden_size", None)
    tower_out = getattr(getattr(tower, "config", None), "out_hidden_size", None)
    if text_hidden is None or tower_out != text_hidden:
        raise ValueError(
            f"This Qwen3-VL vision tower emits {tower_out}-wide embeddings, but the text encoder's hidden size is "
            f"{text_hidden}; the tower belongs to a different Qwen3-VL."
        )
    prior_rope_deltas = getattr(model, "rope_deltas", None)
    model.visual = tower
    try:
        yield model
    finally:
        model.visual = existing
        model.rope_deltas = prior_rope_deltas


def drop_qwen3vl_unloaded_keys(sd: dict[str, Any]) -> dict[str, Any]:
    """Keep only what the single-file encoder loader reads: drop the visual tower and the LM head.

    ``Qwen3VLModel`` has no LM head, so a non-strict load would discard it anyway -- but only after the
    side-channel handling had folded and reserved for it: on the 8B fp8 builds a 151936x4096 bf16 copy.
    Identification judges a file's quantization by the same view, so it refuses nothing the loader would
    never read.
    """
    return {k: v for k, v in sd.items() if not _is_qwen3vl_unloaded_key(k)}


def _is_qwen3vl_unloaded_key(key: Any) -> bool:
    """Whether a checkpoint key belongs to the visual tower or the LM head, sidecars included.

    The head is matched where the remap sends it, like the tower: a top-level ``lm_head.*`` and a ``model.lm_head.*``
    both land on ``language_model.lm_head``, which names no module. Left in, an nvfp4 head fails the install that
    way, and a scaled fp8 one is folded and reserved for, then discarded.
    """
    return is_qwen3vl_visual_key(key) or qwen3vl_lm_head_suffix(key) is not None


_LM_HEAD_TARGET = "language_model.lm_head."


def qwen3vl_lm_head_suffix(key: Any) -> str | None:
    """What follows the LM head's path in a checkpoint key (``weight``, ``weight_scale``, ``comfy_quant``, ...), in
    either spelling (``lm_head.*``, ``model.lm_head.*``); None for a key of anything else."""
    if not isinstance(key, str):
        return None
    target = qwen3vl_target_key(key)
    return target[len(_LM_HEAD_TARGET) :] if target.startswith(_LM_HEAD_TARGET) else None


def read_qwen3vl_lm_head_tensors(path: Path) -> dict[str, torch.Tensor]:
    """Read only the LM head's tensors from a single file, keyed ``lm_head.<suffix>`` whichever spelling it uses.

    For the Text LLM view of an encoder file, whose language-model load drops the head (`drop_qwen3vl_unloaded_keys`).
    Filtered by key before anything is read, like `read_qwen3vl_visual_tower_tensors`.
    """
    from safetensors import safe_open

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            suffix = qwen3vl_lm_head_suffix(key)
            if suffix is not None:
                tensors[f"lm_head.{suffix}"] = handle.get_tensor(key)
    return tensors


def qwen3vl_lm_head_entries(state_dict: Mapping[Any, Any]) -> dict[str, Any]:
    """The LM head's entries of a single file's state dict (`meta` or real), keyed by what follows the head's path."""
    return {suffix: value for key, value in state_dict.items() if (suffix := qwen3vl_lm_head_suffix(key)) is not None}


def qwen3vl_token_embedding(state_dict: Mapping[Any, Any]) -> Any:
    """The token embedding's weight in a single file's state dict, in either spelling, or None."""
    return next(
        (
            value
            for key, value in state_dict.items()
            if qwen3vl_target_key(str(key)) == "language_model.embed_tokens.weight"
        ),
        None,
    )


def qwen3vl_lm_head_hints(path: Path, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """The LM head's quantization hints: its ``.comfy_quant`` marker, overlaid by its ``_quantization_metadata`` entry.

    The same two sources, in the same order, as the encoder loader reads for every other layer. The marker is read from
    the file rather than from tensors, because identification holds `meta` tensors with no bytes to parse.
    """
    hints: dict[str, Any] = {}
    for layers in (read_comfy_quant_markers(path), parse_quantization_metadata(metadata)):
        for layer, layer_hints in layers.items():
            if qwen3vl_lm_head_suffix(f"{layer}.weight") == "weight" and isinstance(layer_hints, Mapping):
                hints.update(layer_hints)
    return hints


# Read by the Text LLM loader's fp8 extraction: the scales, and the marker it pops.
_LM_HEAD_SIDECARS = frozenset({"weight_scale", "scale_weight", "input_scale", "scale_input", "comfy_quant"})
_PLAIN_SCALE_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def qwen3vl_lm_head_refusal(head: Mapping[str, Any], hints: Mapping[str, Any], embedding: Any) -> str | None:
    """Why the Text LLM view of a Qwen3-VL file cannot widen its untied LM head, or None when it can.

    Identification withholds the Text LLM role and the loader refuses for exactly these reasons, so a file serves as
    a Text LLM only when its head loads. Reads dtypes and shapes, so a header's `meta` tensors answer it.

    Args:
        head: `qwen3vl_lm_head_entries` of the file.
        hints: `qwen3vl_lm_head_hints` of the file.
        embedding: The file's token embedding, whose shape an untied head shares.
    """
    weight = head.get("weight")
    if not isinstance(weight, torch.Tensor):
        return "it has no lm_head.weight to generate text with"
    if not weight.dtype.is_floating_point:
        return (
            f"its LM head is stored as {weight.dtype}; only a float LM head, scaled fp8 included, loads as a Text LLM"
        )
    embedding_shape = tuple(getattr(embedding, "shape", ()))
    if tuple(weight.shape) != embedding_shape:
        return f"its LM head is {tuple(weight.shape)}, but its token embedding is {embedding_shape}"
    unread = sorted(set(head) - {"weight"} - _LM_HEAD_SIDECARS)
    if unread:
        return f"its LM head carries {', '.join(f'lm_head.{suffix}' for suffix in unread)}, which is not read"
    scale = head.get("weight_scale", head.get("scale_weight"))
    if (
        weight.dtype in FP8_WEIGHT_DTYPES
        and isinstance(scale, torch.Tensor)
        and scale.dtype not in _PLAIN_SCALE_DTYPES
        and hints.get("format") != MXFP8_FORMAT
    ):
        return (
            f"its fp8 LM head has a {scale.dtype} scale, an MX exponent grid that neither its comfy_quant marker nor "
            f"the _quantization_metadata header names '{MXFP8_FORMAT}'"
        )
    return None
