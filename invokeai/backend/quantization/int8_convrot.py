"""Runtime support for Comfy "int8_tensorwise + convrot" quantized linears.

``int8_tensorwise`` is a ComfyUI-wide scheme, not one architecture's format: the same
spelling appears on MiniMax H3 (the first consumer here), Krea-2, and whatever Comfy-Org
publishes next. It therefore lives beside the other schemes in ``backend/quantization``
rather than in an architecture package, so a second architecture costs a call site and not
a second copy of the mathematics.

A quantized linear stores its weight as symmetric per-output-channel int8:

- ``<layer>.weight``: int8 ``[out, in]``
- ``<layer>.weight_scale``: float32 ``[out, 1]``
- ``<layer>.comfy_quant``: a uint8 JSON marker, e.g.
  ``{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}``

With ``convrot``, the weights were rotated along the input dimension before
quantization: ``W_rot = grouped_256(W) @ H^T`` where ``H`` is the normalized
REGULAR Hadamard matrix built from the 4x4 seed ``[[1,1,1,-1],[1,1,-1,1],
[1,-1,1,1],[-1,1,1,1]]`` by Kronecker powers (sizes are powers of 4) and scaled
by ``1/sqrt(size)``. This ``H`` is symmetric and orthonormal (``H == H.T``,
``H @ H == I``), so recovering the un-rotated weight is the SAME grouped matmul:
``W = grouped_256(W_rot) @ H``. (Semantics verified against
Comfy-Org/comfy-quants and comfy-kitchen ``tensor/int8_utils.py`` — reimplemented
here, not copied.)

Comfy's W8A8 kernels instead rotate activations at runtime and run int8 GEMMs.
We target bf16 compute with int8 *storage*: ``Int8ConvrotLinear`` keeps the int8
weight and scale resident (4.6x smaller than bf16) and materializes the
dequantized, derotated bf16 weight per forward call. The derotation is a
``[out, in/256, 256] @ [256, 256]`` matmul — a rounding error next to the
transformer forward itself — and the transient bf16 weight (<= ~310 MB for H3's
largest layer) has to fit inside the calling node's working-memory reservation:
``dequantizing_linear.peak_dequant_transient_bytes`` is what a denoise node adds to
its estimate for that, since the model's resident size does not account for it.
"""

import json
import struct
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from invokeai.backend.quantization.dequantizing_linear import DequantizingLinear
from invokeai.backend.quantization.fp8_scaled import (
    COMFY_QUANT_SUFFIX,
    FP8_WEIGHT_DTYPES,
    INPUT_SCALE_SUFFIXES,
    WEIGHT_SCALE_SUFFIXES,
    is_castable_float,
    iter_weight_scale_pairs,
)

CONVROT_GROUP_SIZE = 256

_HADAMARD_SEED = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))

INT8_TENSORWISE_FORMAT = "int8_tensorwise"

# Per-layer side-channel entries the int8 path reads elsewhere or cannot use at all; see
# `drop_unconsumed_quantization_sidecars`.
_UNCONSUMED_SIDECAR_SUFFIXES = (COMFY_QUANT_SUFFIX, *INPUT_SCALE_SUFFIXES)


def check_hadamard_size(size: int) -> None:
    """Refuse a rotation width this scheme has no Hadamard for.

    Called at *construction* as well as at build time. The matrix itself is now minted lazily on
    the first forward, so without this a checkpoint declaring, say, ``convrot_groupsize: 512`` --
    a power of 2, not of 4, and divisible into plenty of real layer widths -- would load, cache and
    reach VRAM, then raise out of the middle of generation with no layer or file named.
    """
    if size < 4 or (size & (size - 1)) != 0 or (size.bit_length() - 1) % 2 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")


def build_regular_hadamard(size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Normalized regular Hadamard matrix of a power-of-4 size (CPU tensor)."""
    check_hadamard_size(size)
    h4 = torch.tensor(_HADAMARD_SEED, dtype=torch.float64)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return (h / (size**0.5)).to(dtype)


# One derotation matrix per (size, device, dtype), shared by every layer that needs it.
#
# The matrix is a constant of the scheme, so a per-module copy was pure duplication: Krea-2's
# transformer has 264 Linears and the Qwen3-VL encoder several hundred more, at a 256x256 fp32
# buffer (262144 bytes) each. That cost was paid three times over:
#
#  - 69 MB / ~92 MB of byte-identical RAM, charged to the cache entry by `calc_module_size`, which
#    sums `model.buffers()`;
#  - the same again on the compute device and NOT charged, because `CachedModelWithPartialLoad`
#    sizes itself from `model.state_dict()` and a non-persistent buffer is not in it -- while
#    `_move_non_persistent_buffers_to_device` copies every one of them to the device on each lock;
#  - a fresh `.to(dtype)` copy per layer per forward, since the buffer was fp32 and compute is
#    bf16: tens of thousands of identical 128 KB allocations over a 50-step generation.
#
# Lifetime: entries are minted on demand and never evicted, and nothing here is tied to a model's
# lifetime -- an entry outliving the model that minted it is reused by the next load on the same
# device. The bound is the set of group sizes checkpoints actually declare (256 in every build
# seen; 64 is expressible) times the compute device and dtype, i.e. a handful of <=256 KB tensors
# for the life of the process. A checkpoint declaring an absurd group size would pin more, but it
# runs into `build_regular_hadamard` first, whose fp64 Kronecker build of the same matrix is the
# larger cost by far.
_HADAMARD_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def shared_regular_hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """The derotation matrix for ``size``, shared across every layer using it on ``(device, dtype)``.

    Returned tensors are shared and must be treated as read-only. :func:`build_regular_hadamard`
    stays pure and is what callers building their own (e.g. a quantizer's forward rotation) use.
    """
    key = (size, device, dtype)
    matrix = _HADAMARD_CACHE.get(key)
    if matrix is None:
        matrix = build_regular_hadamard(size, dtype).to(device=device)
        _HADAMARD_CACHE[key] = matrix
    return matrix


def parse_comfy_quant_bytes(raw: bytes) -> dict:
    """Decode the raw bytes of a ``<layer>.comfy_quant`` blob into its JSON dict, or ``{}``.

    Tolerant on purpose, and it has to be: this runs over *every* marker in the file before the
    loader knows which format it is dealing with, so a blob belonging to some other scheme decides
    whether an fp8 checkpoint loads at all. Comfy pads these to a fixed width with NUL bytes -- the
    fp8 reader documents that and strips it -- and a strict parse turns a padded marker into
    `JSONDecodeError: Extra data` out of the middle of a load, naming neither the file nor the key.

    A malformed marker is a lost hint, never a failed load: an int8 weight whose marker did not
    parse is then caught by the orphan check in the loaders, which says what is wrong.

    Takes bytes rather than a tensor so the safetensors *header* reader -- which seeks to a blob's
    offsets and never materializes a tensor -- reads markers the same way the state-dict readers do.
    """
    try:
        parsed = json.loads(raw.decode("utf-8", errors="replace").rstrip("\x00"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_comfy_quant_marker(blob: torch.Tensor) -> dict:
    """Decode a ``<layer>.comfy_quant`` uint8 tensor into its JSON dict, or ``{}``."""
    try:
        raw = blob.cpu().numpy().tobytes()
    except Exception:
        return {}
    return parse_comfy_quant_bytes(raw)


def read_comfy_quant_markers(path: Path) -> dict[str, dict[str, Any]]:
    """Read every ``<layer>.comfy_quant`` marker from a safetensors file WITHOUT loading tensor
    data - header parse plus a seek per marker blob. Keys are the raw (un-renamed) layer names.

    Lets a loader reject unsupported quantization formats (e.g. the fp8_scaled repacks, which
    share this key layout) before committing to a ~20 GiB read, and model identification -- which
    sees only a header's dtypes and shapes -- check what a marker declares.

    Marker bytes go through the same tolerant parser the state-dict readers use. This reader runs
    FIRST, so a strict parse here is what a NUL-padded marker -- which Comfy writes, and which that
    parser exists to absorb -- would actually hit: a `JSONDecodeError` out of the middle of a load,
    naming neither the file nor the key.
    """
    markers: dict[str, dict[str, Any]] = {}
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        header.pop("__metadata__", None)
        for key, entry in header.items():
            if not key.endswith(COMFY_QUANT_SUFFIX):
                continue
            start, end = entry["data_offsets"]
            f.seek(8 + header_len + start)
            markers[key[: -len(COMFY_QUANT_SUFFIX)]] = parse_comfy_quant_bytes(f.read(end - start))
    return markers


def as_column_scale(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """A per-output-channel scale shaped `[out, 1]`, so it multiplies down the rows.

    `check_int8_scale_layout` accepts `[out]` as well as `[out, 1]`, but broadcasting aligns
    *trailing* dimensions: `weight[out, in] * scale[out]` scales along the input axis instead. For a
    square weight that is silent -- measured at 0.82 correlation against the intended result, so the
    model loads and generates subtly wrong images -- and for every other shape it raises with a size
    mismatch that names neither the layer nor the cause.

    A scalar, `[1]` or `[1, 1]` is per-tensor and broadcasts correctly as it is.

    Only for a 2-D weight. On a 1-D one -- the quantized norms a "quantize everything" repack ships
    -- a `[N]` scale is already elementwise, and making it `[N, 1]` broadcasts `[N] * [N, 1]` into an
    `[N, N]` matrix: a weight the model cannot load and, before it gets there, an allocation the
    load never reserved for (a 3072-wide norm becomes 9.4M elements).
    """
    if weight.dim() == 2 and scale.dim() == 1 and scale.shape[0] == weight.shape[0]:
        return scale.unsqueeze(1)
    return scale


def dequantize_convrot_weight(
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    convrot: bool,
    dtype: torch.dtype,
    group_size: int = CONVROT_GROUP_SIZE,
) -> torch.Tensor:
    """Recover the bf16/fp16 weight from int8 storage (and undo convrot if applied)."""
    w = weight_q.to(torch.float32) * as_column_scale(weight_q, weight_scale).to(torch.float32)
    if convrot:
        out_features, in_features = w.shape
        if in_features % group_size != 0:
            raise ValueError(f"convrot weight in_features {in_features} not divisible by {group_size}")
        h = shared_regular_hadamard(group_size, w.device, w.dtype)
        w = (w.view(out_features, in_features // group_size, group_size) @ h).view(out_features, in_features)
    return w.to(dtype)


class Int8ConvrotLinear(DequantizingLinear):
    """A linear layer storing Comfy int8_tensorwise(+convrot) weights, dequantized per forward.

    The int8 weight and fp32 scale are registered as PERSISTENT buffers named ``weight`` and
    ``weight_scale`` — exactly the converted checkpoint's key names — so ``load_state_dict``
    consumes the quantized tensors directly and the model cache moves them between devices
    like any other weight. The Hadamard matrix is neither loaded nor held per module: it is a
    constant of the scheme, so the dequantization takes it from :func:`shared_regular_hadamard`,
    which keeps one per ``(size, device, dtype)`` for every layer in the model.

    The model cache wraps this module as ``CustomInt8ConvrotLinear`` (see
    ``AUTOCAST_MODULE_TYPE_MAPPING``), which enables sidecar LoRA patches and lets a partial
    load leave some int8 buffers on the CPU — the per-call ``.to(device)`` then streams them
    (at half the bf16 byte count) instead of failing outright. Fully-resident operation remains
    the intended regime (~20 GiB free VRAM for H3's pruned transformer); streamed layers pay a
    per-forward PCIe cost, and an unquantized model is still the better citizen on small cards.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        convrot: bool,
        bias: torch.Tensor | None = None,
        group_size: int = CONVROT_GROUP_SIZE,
    ) -> None:
        super().__init__()
        if weight.dtype != torch.int8:
            raise ValueError(f"expected int8 weight, got {weight.dtype}")
        self.out_features, self.in_features = weight.shape
        self.convrot = convrot
        self.group_size = group_size
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", weight_scale.to(torch.float32))
        if convrot:
            check_hadamard_size(group_size)
            if self.in_features % group_size != 0:
                raise ValueError(f"convrot weight in_features {self.in_features} not divisible by {group_size}")
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def _dequantized_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # `.to(device)` is a no-op in the intended fully-resident regime; under partial load
        # the model cache may leave these buffers on the CPU, in which case this call streams
        # the int8 weight to the compute device per forward (see the class docstring).
        #
        # Dequant + derotation run directly in the compute dtype: int8 values are exact in
        # bf16/fp16, the scale multiply adds ~0.2% relative rounding vs the ~0.4-0.8% int8
        # quantization floor, and the matmul accumulates fp32 internally. This keeps the
        # per-call transient at ~two weight-sized tensors (~620 MB peak for the fused-SwiGLU
        # fc1 in bf16) instead of tripling through an fp32 intermediate (~1.5 GiB). An fp32
        # compute dtype still gets the exact fp32 path for free.
        scale = as_column_scale(self.weight, self.weight_scale).to(device=device, dtype=dtype)
        w = self.weight.to(device=device, dtype=dtype) * scale
        if self.convrot:
            w = (
                w.view(self.out_features, self.in_features // self.group_size, self.group_size)
                @ shared_regular_hadamard(self.group_size, device, dtype)
            ).view(self.out_features, self.in_features)
        return w

    def dequant_transient_bytes(self, compute_dtype: torch.dtype) -> int:
        """Two weight-sized tensors in the compute dtype.

        Both halves of ``_dequantized_weight`` peak at two: the dtype cast of the int8 weight is alive
        alongside the product it is multiplied into, and that product is then alive alongside the
        derotation matmul's output. Each pair is freed before the next allocates.

        Under partial load the int8 weight is additionally streamed to the device per call, adding up to
        half a weight again; the resident-model regime this scheme targets pays nothing for that. Verified
        against the real `z_image_turbo_int8_convrot` checkpoint: exact for most shapes, and 2 MiB under on
        the two that make cuBLAS take a workspace (10240x3840 and 3840x10240), against a 3 GiB reservation
        floor.
        """
        return 2 * self.out_features * self.in_features * compute_dtype.itemsize

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, convrot={self.convrot}"


def extract_int8_convrot_markers(sd: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Pop every ``int8_tensorwise`` marker out of ``sd``, keyed by the layer path it names.

    Markers for other formats are left in place, together with their weights and scales: ComfyUI's
    fp8_scaled repacks share this key layout and belong to the fp8 path. Deciding per marker rather
    than per file is also what keeps mixed-precision checkpoints correct -- one Krea-2 build leaves
    40 weights in bf16 with no marker at all, and both Qwen3-VL encoders leave over 200.

    Use this when the loader intends to keep the weights int8 and swap in
    :class:`Int8ConvrotLinear`, which is what every loader here does.
    """
    markers = {}
    for key in [k for k in sd if isinstance(k, str) and k.endswith(COMFY_QUANT_SUFFIX)]:
        marker = parse_comfy_quant_marker(sd[key])
        if marker.get("format") != INT8_TENSORWISE_FORMAT:
            continue
        markers[key[: -len(COMFY_QUANT_SUFFIX)]] = marker
        del sd[key]
    return markers


def reject_int8_layers_a_plain_fold_cannot_decode(sd: Mapping[str, Any], what: str = "This checkpoint") -> None:
    """Refuse an ``int8_tensorwise`` layer to a fold that only multiplies the scale in.

    ``int8_tensorwise`` and ComfyUI's scaled fp8 share a key layout — a ``.weight`` beside a
    ``.weight_scale`` — so a loader with no int8 branch folds one as if it were the other. The scale
    even goes down the right axis, which is what makes this quiet: the result has the weight's
    shape, its dtype and roughly its magnitude. What is missing is the inverse Hadamard rotation
    ``convrot`` applies along the input dim. Measured on a 64x256 layer, the folded weight
    correlates with the one the checkpoint encodes at **0.07**, against 0.9999 for the real decode.

    Only the rotated case is refused. With ``convrot`` off, ``codes * scale`` *is* the whole decode
    — bit-identical to :class:`Int8ConvrotLinear` with ``convrot=False``, checked — so where a fold
    actually runs, such a build folds to the right weight and must keep doing so. (It does not
    follow that it loads: the Mistral encoder keeps fp8 rather than folding on any CUDA device, and
    there an unrotated int8 weight is left untouched and dies later in ``load_state_dict``. That is
    how it behaved before this check and is not what this check is about.) An int8 weight with no
    marker is refused too: nothing says whether it was rotated, and every published build marks
    every one of its int8 weights.

    Read from the *per-tensor* marker only. The safetensors header names formats as well, but no
    observed header entry carries ``convrot`` or the group size (see ``flux.py``, where the
    per-layer marker wins for that reason), so a header entry is not evidence that a layer is safe
    to fold — consulting one
    would let a header-named rotated build straight through. Every caller here still has the markers
    in ``sd``: the two folds run before anything pops them, and the Mistral seam checks ahead of
    ``extract_fp8_scaled_layers`` for the same reason.

    Narrower than :func:`reject_unmarked_int8_weights` in two ways, because it runs before a model
    exists: it sees only int8 weights that have a scale beside them, and it cannot skip weights the
    model will discard. A merged file bundling an int8 submodel a non-strict load would drop is
    therefore refused rather than ignored.
    """
    rotated: list[str] = []
    unmarked: list[str] = []
    for weight_key, _scale_key in iter_weight_scale_pairs(sd):
        weight = sd[weight_key]
        if getattr(weight, "dtype", None) is not torch.int8:
            continue
        path = weight_key[: -len(".weight")]
        blob = sd.get(f"{path}{COMFY_QUANT_SUFFIX}")
        marker = parse_comfy_quant_marker(blob) if blob is not None else None
        if not marker or marker.get("format") != INT8_TENSORWISE_FORMAT:
            unmarked.append(path)
        elif marker.get("convrot"):
            rotated.append(path)

    if rotated:
        raise ValueError(
            f"{what} carries {len(rotated)} int8_tensorwise layer(s) quantized with convrot (e.g. "
            f"{', '.join(sorted(rotated)[:3])}), and this loader has no int8 branch. Folding the scale in "
            "without the inverse rotation produces a weight of the right shape that bears no relation to the "
            "one stored. Use the fp8 or bf16 build of this checkpoint."
        )
    if unmarked:
        raise ValueError(
            f"{what} carries {len(unmarked)} int8 weight(s) with no int8_tensorwise marker (e.g. "
            f"{', '.join(sorted(unmarked)[:3])}), so nothing says whether they were rotated. Refusing rather "
            "than guessing: read unrotated, a rotated weight loads cleanly and generates noise."
        )


def check_int8_scale_layout(path: str, weight: torch.Tensor, scale: torch.Tensor) -> None:
    """Refuse a scale granularity this decode does not implement.

    Two layouts are supported, because they are the two this dequantization is correct for:
    per-output-channel (``[out, 1]`` or ``[out]``) and per-tensor (a scalar). Some repacks emit a
    blockwise grid instead - a 6144x6144 weight with a ``[48, 48]`` scale is a 128x128 block grid -
    which needs a different multiply. Left to broadcasting that either raises somewhere less
    informative or, for an unlucky shape, silently scales the wrong axis.

    What counts as per-output-channel depends on the weight's rank, and getting that wrong is the
    same silent failure by another route. On a 1-D weight only ``[N]`` is elementwise; ``[N, 1]``
    broadcasts to ``[N, N]``. Above rank 2 there is no output-channel axis this decode can name, so
    only a per-tensor scale is accepted rather than guessing which axis the producer meant.
    """
    per_channel: set[tuple[int, ...]] = set()
    if weight.dim() == 2:
        per_channel = {(weight.shape[0],), (weight.shape[0], 1)}
    elif weight.dim() == 1:
        per_channel = {(weight.shape[0],)}
    if scale.dim() == 0 or tuple(scale.shape) in {(1,), (1, 1)} or tuple(scale.shape) in per_channel:
        return
    raise ValueError(
        f"'{path}' has a {tuple(scale.shape)} scale for a {tuple(weight.shape)} weight, which is "
        "neither per-output-channel nor per-tensor. Blockwise scale grids are not implemented."
    )


def reject_unmarked_int8_weights(
    sd: dict[str, Any], markers: Mapping[str, Any], architecture: str, model: torch.nn.Module | None = None
) -> None:
    """Refuse a checkpoint holding an int8 weight that no ``int8_tensorwise`` marker claims.

    Call this *outside* the "did we find markers" branch. An int8 weight whose marker is missing --
    or, since :func:`parse_comfy_quant_marker` is tolerant, one whose marker did not parse -- leaves
    `markers` without an entry for it, so a check living inside the int8 path would never see the
    very case it exists for. Such a weight then takes the unquantized path: cast to the compute
    dtype as raw int8 codes, unscaled and un-derotated, into a model that loads clean and generates
    noise.

    That tolerance is deliberate and this is its other half: a malformed marker is a lost hint, and
    this is what stops the loss from being silent.

    Only a ``<path>.weight`` a marker names is exempt. Slicing seven characters off every int8 key
    regardless of its suffix truncates unrelated ones: ``foo.qkv.scales`` becomes ``foo.qkv``, so a
    marker on the *weight* at ``foo.qkv`` exempted a packed sidecar the decode never places, and a
    key shorter than the suffix collapses to the empty string. An int8 tensor under any name but a
    marked ``.weight`` is a payload this decode cannot place, and saying so is the whole point.

    Pass ``model`` where the caller has already built it, and the check narrows to weights this
    model actually consumes -- the same filter :func:`reject_foreign_quantization_scales` applies,
    for the same reason. A merged single file may bundle a quantized *submodel* beside the
    transformer; a non-strict load discards those keys rather than casting them, so they cannot
    become noise and are not this loader's business. Without ``model`` every int8 key in ``sd`` is
    in scope, which is right for a caller that has not built the module tree yet.
    """
    consumed = None if model is None else {name for name, _ in model.named_modules()}
    orphans = sorted(
        k
        for k, v in sd.items()
        if v.dtype is torch.int8
        and not (isinstance(k, str) and k.endswith(".weight") and k[: -len(".weight")] in markers)
        and (consumed is None or (isinstance(k, str) and k.rsplit(".", 1)[0] in consumed))
    )
    if orphans:
        raise ValueError(
            f"{architecture} checkpoint has {len(orphans)} int8 weight(s) with no `comfy_quant` marker, "
            f"e.g. {orphans[:3]}. Loading them would produce a model that runs and generates noise."
        )


def drop_unconsumed_quantization_sidecars(sd: dict[str, Any]) -> dict[str, Any]:
    """Remove quantization metadata no loader here consumes.

    - ``.comfy_quant`` markers whose format was handled elsewhere, or not at all.
    - ``.input_scale`` / ``.scale_input``: activation scales for W8A8 inference. This code
      dequantizes the weight and computes in bf16, so there is nothing to apply them to. (Both
      spellings appear in the wild; one Qwen3-VL repack ships 337 of the former.)

    `load_state_dict(strict=False)` would ignore them, but they are still cast and still counted
    against the RAM reservation - and a loader that later switches to strict would fail on them.

    Matched as whole suffixes, not substrings: a module legitimately named ``...input_scaler`` owns
    a real ``.weight``, and dropping it surfaces under Z-Image's strict load as a missing key.
    """
    return {k: v for k, v in sd.items() if not (isinstance(k, str) and k.endswith(_UNCONSUMED_SIDECAR_SUFFIXES))}


def resolve_quantized_module_paths(
    markers: dict[str, dict[str, Any]], key_map: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Re-key markers from the checkpoint's names to the built model's names.

    The markers are read in the checkpoint's key space, because that is the only place where a
    marker and its weight are reliably paired: the key conversions rename `.weight` (and, by
    substring, `.weight_scale`) but leave `.comfy_quant` behind on the old name. Following the
    weight's own rename is therefore the only mapping that cannot drift from the conversion.
    """
    resolved: dict[str, dict[str, Any]] = {}
    for path, marker in markers.items():
        weight_key = key_map.get(f"{path}.weight", f"{path}.weight")
        resolved[weight_key[: -len(".weight")]] = marker
    return resolved


def _resolve_int8_target(model: torch.nn.Module, path: str) -> tuple[torch.nn.Module, str, torch.nn.Linear]:
    """The owner, attribute name and module a marker names, as ``setattr`` and the swap need them.

    Refuses anything an ``Int8ConvrotLinear`` cannot stand in for. Without this, a marker naming a
    module the built model lacks leaves ``get_submodule`` to raise a bare ``AttributeError`` out of
    the middle of a load, and a marker on a non-Linear (an ``nn.Embedding``, say) installs a linear
    in its place and only fails much later inside that module's forward.
    """
    parent_path, _, attribute = path.rpartition(".")
    try:
        parent = model.get_submodule(parent_path) if parent_path else model
        target = getattr(parent, attribute)
    except AttributeError:
        raise ValueError(
            f"'{path}' is marked int8_tensorwise but the built model has no such module. The "
            "checkpoint is misidentified, or its keys need a conversion this loader did not apply."
        ) from None
    if not isinstance(target, torch.nn.Linear):
        raise ValueError(
            f"'{path}' is marked int8_tensorwise but is a {type(target).__name__}, not an nn.Linear. "
            "Only a Linear weight can be kept in int8 storage."
        )
    return parent, attribute, target


def _can_stay_int8(path: str, weight: Any, model: torch.nn.Module | None, skip_patterns: Iterable[str] = ()) -> bool:
    """Whether a marked layer may keep its int8 storage instead of being dequantized on load.

    Single source of truth for that decision, so :func:`split_int8_convrot_layers` and
    :func:`predict_int8_cast_size` cannot drift: the loaders reserve RAM against the prediction and
    then run the split, and a layer the split widens but the prediction charged 1 byte/element for
    is a reservation that is short by exactly that layer.

    Deliberately not shared with :func:`fp8_scaled._is_fp8_matmul_weight`, which asks a similar
    question: that one admits any weight of rank >= 2 (an fp8 Conv weight is usable), keys off the
    state-dict key rather than the module path, and treats a marker naming a module the model lacks
    as "cannot stay quantized". All three differ here, so sharing it would mean two policy
    parameters used once each.

    Two filters, both of which the fp8 path applies for the same reasons:

    - Only a 2-D ``nn.Linear`` weight. Repacks that "quantize everything" ship 1-D quantized norms;
      there is no linear for those to become, and an int8 norm weight would flow into the
      activations as raw codes.
    - Never a module the model declares precision-sensitive (``skip_patterns``, i.e. diffusers'
      ``_skip_layerwise_casting_patterns`` / ``_keep_in_fp32_modules``). Z-Image's
      ``ZImageTimestepEmbedder.forward`` reads ``self.mlp[0].weight.dtype`` to pick the dtype it
      casts its activations to; on an ``Int8ConvrotLinear`` that reads ``torch.int8``, the forward
      falls through to a ``compute_dtype`` attribute these modules do not have, and the timestep
      branch silently runs in float32 and hands a float32 embedding to a bf16 model.
    """
    if getattr(weight, "dim", None) is None or weight.dim() != 2:
        return False
    if any(pattern in path for pattern in skip_patterns):
        return False
    if model is None:
        # No model to resolve against: the 2-D test above is the safe subset. A marker naming a
        # module the model lacks is not silently dropped here — `swap_in_int8_linears` names it.
        return True
    try:
        return isinstance(model.get_submodule(path), torch.nn.Linear)
    except AttributeError:
        return True


def split_int8_convrot_layers(
    sd: dict[str, Any],
    quantized: dict[str, dict[str, Any]],
    dtype: torch.dtype,
    *,
    model: torch.nn.Module | None = None,
    skip_patterns: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    """Dequantize in place the marked layers that cannot stay int8; return the ones that can.

    Mirrors :func:`split_fp8_scaled_layers`. Every layer this widens is one
    :func:`swap_in_int8_linears` must not be handed: it would either install an
    ``Int8ConvrotLinear`` where the model wants something else, or leave a precision-sensitive
    module reporting ``torch.int8`` as its weight dtype to a forward that branches on it.

    Dequantizing rather than refusing is deliberate, and is what the fp8 path does: the scale and
    rotation are both present, so the decode is unambiguous, and the affected tensors are the small
    ones (norms, timestep MLPs). Refusing would reject a checkpoint that loads correctly for the
    sake of a saving measured in megabytes.
    """
    surviving = _surviving_int8_layers(sd, quantized, model, skip_patterns)
    for path, marker in quantized.items():
        if path in surviving:
            continue
        weight, scale = sd[f"{path}.weight"], sd[f"{path}.weight_scale"]
        check_int8_scale_layout(path, weight, scale)
        convrot = bool(marker.get("convrot", False))
        if convrot and weight.dim() != 2:
            raise ValueError(
                f"'{path}' is marked convrot but its weight is {tuple(weight.shape)}. The rotation is "
                "defined over a weight's input dimension, so there is nothing to un-rotate here."
            )
        sd[f"{path}.weight"] = dequantize_convrot_weight(
            weight, scale, convrot, dtype, int(marker.get("convrot_groupsize", CONVROT_GROUP_SIZE))
        )
        del sd[f"{path}.weight_scale"]
    return surviving


def predict_int8_cast_size(
    sd: Mapping[str, Any],
    dtype: torch.dtype,
    quantized: dict[str, dict[str, Any]],
    *,
    model: torch.nn.Module | None = None,
    skip_patterns: Iterable[str] = (),
) -> int:
    """Bytes the state dict will occupy once the split and :func:`cast_unquantized` have run.

    Loaders call this to size their ``make_room()`` reservation, and call it *before* the split so
    that the tensors the split widens are already reserved for when they appear. (The split's own
    per-layer fp32 transient is absorbed by that reservation's slack rather than modelled: it runs
    one widened layer at a time, and the widened set is by construction the small one -- skip-pattern
    and non-Linear layers.)

    Three widths are in play and only one of them is ``dtype.itemsize``. A payload that stays int8
    occupies one byte, and charging it two asks the cache to free ~12 GB a Krea-2 build never uses.
    A pinned ``weight_scale`` stays fp32, and charging it two under-counts it by half. A payload the
    split is about to dequantize arrives at ``dtype.itemsize`` while its scale disappears into it,
    and charging that one byte is the direction that actually hurts: the reservation comes up short
    by exactly the layers the split widened.
    """
    surviving = _surviving_int8_layers(sd, quantized, model, skip_patterns)
    pinned = _pinned_quantized_keys(surviving)
    widening = {path: marker for path, marker in quantized.items() if path not in surviving}
    dequantized_weights = {f"{path}.weight" for path in widening}
    consumed_scales = {f"{path}.weight_scale" for path in widening}

    total = 0
    for key, tensor in sd.items():
        if key in consumed_scales:
            # Folded into its weight by the split and dropped from the dict.
            continue
        if key in pinned or (key not in dequantized_weights and not is_castable_float(tensor)):
            total += tensor.nelement() * tensor.element_size()
        else:
            total += tensor.nelement() * dtype.itemsize
    return total


def _surviving_int8_layers(
    sd: Mapping[str, Any],
    quantized: dict[str, dict[str, Any]],
    model: torch.nn.Module | None,
    skip_patterns: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """The marked layers the split will leave int8. Shared so the split and the prediction of what
    it costs are the same decision, evaluated twice."""
    patterns = tuple(skip_patterns)
    surviving: dict[str, dict[str, Any]] = {}
    for path, marker in quantized.items():
        weight, scale = sd.get(f"{path}.weight"), sd.get(f"{path}.weight_scale")
        # A missing half is not this decision to make: `swap_in_int8_linears` names which one.
        if weight is None or scale is None or _can_stay_int8(path, weight, model, patterns):
            surviving[path] = marker
    return surviving


def _pinned_quantized_keys(quantized: Mapping[str, Any]) -> set[str]:
    """The state-dict keys that must reach ``load_state_dict`` exactly as stored."""
    return {key for path in quantized for key in (f"{path}.weight", f"{path}.weight_scale")}


def swap_in_int8_linears(model: torch.nn.Module, sd: dict[str, Any], quantized: dict[str, dict[str, Any]]) -> None:
    """Replace each quantized ``nn.Linear`` with an ``Int8ConvrotLinear`` sized from the state dict.

    The weights stay int8 and rotated as stored; the layer dequantizes and derotates per forward.
    That keeps a 12 GB checkpoint at 12 GB resident instead of the ~24 GB a dense decode would
    produce, on every platform and without the fp8-storage opt-in (which is off by default and
    unavailable outside CUDA/XPU).

    Its persistent buffers are named ``weight``/``weight_scale`` -- the checkpoint's own spelling --
    so the ``load_state_dict`` that follows assigns the quantized tensors straight into them.

    Pass only what :func:`split_int8_convrot_layers` returned: every layer refused below is one that
    split would have dequantized instead.
    """
    for path, marker in quantized.items():
        weight, scale = sd.get(f"{path}.weight"), sd.get(f"{path}.weight_scale")
        if weight is None or scale is None:
            raise ValueError(
                f"'{path}' is marked int8_tensorwise but is missing its "
                f"{'weight' if weight is None else 'weight_scale'}."
            )
        if weight.dim() != 2:
            raise ValueError(
                f"'{path}' is marked int8_tensorwise but its weight is {tuple(weight.shape)}, not 2-D. "
                "Only a Linear weight can be kept in int8 storage; dequantize this one instead."
            )
        check_int8_scale_layout(path, weight, scale)
        parent, attribute, target = _resolve_int8_target(model, path)
        bias = sd.get(f"{path}.bias")
        if (bias is None) is not (target.bias is None):
            # The replacement's buffers are whatever the checkpoint supplied, so from here on the
            # module's key set mirrors the file rather than the architecture -- and the strict
            # `load_state_dict` the loaders run afterwards can no longer tell the two apart. A
            # repack that drops all-zero biases would otherwise load, cache and render with every
            # quantized layer silently missing its offset.
            missing, extra = ("checkpoint", "model") if bias is None else ("model", "checkpoint")
            raise ValueError(
                f"'{path}' is marked int8_tensorwise and the {extra}'s Linear has a bias, but the "
                f"{missing} has none. The checkpoint does not match this architecture."
            )
        setattr(
            parent,
            attribute,
            Int8ConvrotLinear(
                weight=weight,
                weight_scale=scale,
                convrot=bool(marker.get("convrot", False)),
                bias=bias,
                group_size=int(marker.get("convrot_groupsize", CONVROT_GROUP_SIZE)),
            ),
        )


def cast_unquantized(sd: dict[str, Any], dtype: torch.dtype, quantized: dict[str, dict[str, Any]]) -> None:
    """Cast the dense float tensors to the compute dtype, leaving the quantized payloads alone.

    An int8 weight cast to bf16 is no longer int8, and its float32 scale is what
    ``Int8ConvrotLinear`` multiplies by -- both have to reach ``load_state_dict`` as stored.

    Integer payloads (embedding indices, packed buffers, bool masks) are not weights and keep their
    dtype, exactly as :func:`cast_state_dict` leaves them: an index buffer installed as a float
    raises ``IndexError: tensors used as indices must be long, byte or bool`` inside the forward,
    and a bool mask cast to bf16 changes what the mask means without raising at all.
    """
    pinned = _pinned_quantized_keys(quantized)
    for key in sd:
        if key not in pinned and is_castable_float(sd[key]):
            sd[key] = sd[key].to(dtype)


def install_int8_convrot_layers(
    model: torch.nn.Module,
    sd: dict[str, Any],
    quantized: dict[str, dict[str, Any]],
    dtype: torch.dtype,
    *,
    architecture: str,
    reserve: Callable[[int], None],
    skip_patterns: Iterable[str] = (),
    extra_reserved_bytes: int = 0,
) -> dict[str, dict[str, Any]]:
    """Take a marked int8 checkpoint from state dict to installed ``Int8ConvrotLinear`` modules.

    Five steps whose order is the point, and which every loader that keeps int8 weights needs in
    full. Each step depends on the one before it:

    1. :func:`reject_foreign_quantization_scales` -- before anything is cast, because a scale from
       another scheme means its weight is about to be cast without one, off by ``1/weight_scale``,
       while the orphaned scale disappears into ``strict=False``. After the model exists, because
       only a weight this model consumes can be corrupted that way.
    2. ``reserve`` -- before the split, which dequantizes the layers it cannot keep. A reservation
       made afterwards lets that transient land on an unreserved cache, and a locked model cannot be
       evicted to make room for it.
    3. :func:`split_int8_convrot_layers` -- widens what cannot stay int8 and returns what can.
    4. :func:`cast_unquantized` -- the dense tensors only; the surviving payloads reach
       ``load_state_dict`` exactly as stored.
    5. :func:`swap_in_int8_linears` -- the surviving layers, never the full marker set: a marker on
       a module the split widened would install an ``Int8ConvrotLinear`` where the model wants
       something else.

    Written down once because getting it partially right is silent: Z-Image and the PiD decoder each
    skipped step 1 and loaded a mixed checkpoint's fp8 weights unscaled, with nothing in the log.

    ``reserve`` is the cache's ``make_room``; ``extra_reserved_bytes`` is for a loader whose file
    also holds layers of another scheme -- Krea-2 and Z-Image add what their nvfp4 layers will cost,
    so the one reservation covers the whole load. Returns the layers that stayed int8.
    """
    reject_foreign_quantization_scales(sd, quantized, architecture, model)
    reserve(
        predict_int8_cast_size(sd, dtype, quantized, model=model, skip_patterns=skip_patterns) + extra_reserved_bytes
    )
    surviving = split_int8_convrot_layers(sd, quantized, dtype, model=model, skip_patterns=skip_patterns)
    cast_unquantized(sd, dtype, surviving)
    swap_in_int8_linears(model, sd, surviving)
    return surviving


def reject_foreign_quantization_scales(
    sd: Mapping[str, Any], quantized: Mapping[str, Any], architecture: str, model: torch.nn.Module
) -> None:
    """Refuse a weight scale belonging to a quantization scheme this branch does not run.

    The mirror image of :func:`reject_unmarked_int8_weights`, and the enforcement of the premise
    these loaders decide on once per file: ``int8_tensorwise`` and ComfyUI's scaled fp8 share this
    key layout, and a checkpoint carries one or the other. Inside the int8 branch the whole fp8
    pipeline is skipped, so a layer of the other scheme is cast to the compute dtype *without* its
    scale -- off by ``1/weight_scale`` -- while the orphaned scale is dropped by ``strict=False``
    and says nothing. The premise was documented and never checked.

    Checked here rather than against ``load_state_dict``'s unexpected keys, which would also reject
    the benign extras ``strict=False`` exists to tolerate. Two filters keep it to weights that are
    genuinely about to load wrong, because the failure is fatal and a false positive costs a user a
    checkpoint that worked:

    - The scale's weight must *actually* be float8. These loaders do not prefix-filter the state
      dict, so a merged single file's bundled text encoder is in ``sd`` too, and a dtype test is
      what tells a foreign quantized layer from a foreign dense one.
    - ``model`` must actually have the module. A merged file may bundle a scaled-fp8 *submodel*
      beside the int8 transformer -- an all-in-one ComfyUI export is exactly that -- and those keys
      are discarded by ``strict=False`` rather than loaded, so they cannot load unscaled and are not
      this loader's business. Only a weight this model consumes can be corrupted by the cast.

    A raw fp8 weight with no scale at all is not an orphan and needs none -- e4m3fn is a subset of
    bf16, so casting it is value-exact.

    Call with ``sd`` and ``quantized`` in the same key space as ``model``, i.e. after any key
    conversion and before the cast.
    """
    consumed = {name for name, _ in model.named_modules()}
    orphans = sorted(
        key
        for key in sd
        if isinstance(key, str)
        and key.endswith(WEIGHT_SCALE_SUFFIXES)
        and key.rsplit(".", 1)[0] not in quantized
        and key.rsplit(".", 1)[0] in consumed
        and getattr(sd.get(f"{key.rsplit('.', 1)[0]}.weight"), "dtype", None) in FP8_WEIGHT_DTYPES
    )
    if orphans:
        raise ValueError(
            f"{architecture} checkpoint has {len(orphans)} float8 weight scale(s) that no "
            f"`int8_tensorwise` marker claims, e.g. {orphans[:3]}. A file mixing int8_tensorwise "
            "with scaled fp8 is not supported: the fp8 weights would load unscaled."
        )
