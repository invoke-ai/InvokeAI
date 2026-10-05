"""Shared helpers for loading Qwen3-VL encoder checkpoints.

Used by Krea-2, Ideogram 4 and MiniMax H3. The visual-tower helpers live here rather than beside
one family's loader because three loaders in two files drop the tower, and because what counts as
the tower has to agree with the key mapping that sits next to it -- two definitions would drift.
MiniMax H3 deliberately does not use them: it runs the tower for its reference images.
"""

from typing import Any

import torch


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
                f"{type(model).__name__} has no Qwen3-VL visual tower to drop. The architecture has "
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
    return {k: v for k, v in sd.items() if not (isinstance(k, str) and qwen3vl_target_key(k).startswith("visual."))}
