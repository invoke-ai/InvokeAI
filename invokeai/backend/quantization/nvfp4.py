"""Comfy "nvfp4" quantized linears: packed storage decoded per forward.

``nvfp4`` is a ComfyUI-wide scheme, not one architecture's format: Comfy-Org ships it for Z-Image,
Krea-2, Qwen-Image, the FLUX.2 Mistral encoder and more, often beside scaled fp8 and plain bf16 in the
same file. It therefore lives next to ``fp8_scaled`` and ``int8_convrot`` rather than in an
architecture package.

A quantized linear ``[out, in]`` is stored as:

- ``<layer>.weight``: uint8 ``[out, in/2]``, two 4-bit E2M1 codes per byte. Element ``2j`` is the
  *upper* nibble; reading the lower one first decodes to noise.
- ``<layer>.weight_scale``: float8_e4m3fn ``[out, in/16]``, one scale per 16-element block, in NVIDIA's
  cuBLAS block-scale layout rather than row by row (see :func:`unblock_scale_grid`).
- ``<layer>.weight_scale_2``: a float32 scalar, multiplied in:
  ``W = E2M1[codes] * block_scale * weight_scale_2``.
- optionally ``<layer>.input_scale``, an activation scale for fp4 compute, and a ``.comfy_quant``
  marker. Some producers name their layers in the ``_quantization_metadata`` header instead.

These are ComfyUI's conventions, established against real checkpoints rather than a specification:
decoding every nvfp4 tensor of the Mistral FLUX.2 encoder and Z-Image Turbo builds reaches cosine 0.995
against the bf16 build in the same repository, the floor of 4-bit block quantization. Reading the scale
grid row by row instead loads without an error and reaches 0.68-0.95 -- a model that runs and follows
its conditioning slightly worse -- which is why tests pin the layout against the measured index formula
and a slice of a real checkpoint. Nothing guarantees that another producer writing the same key names
shares these conventions, so a layer is only read when a Comfy marker or header entry names it nvfp4.

Loaders keep the layers packed: :func:`pop_nvfp4_layers` takes them out of the state dict before any fp8
handling, and :func:`install_nvfp4_layers` swaps :class:`NVFP4Linear` modules in before the load. A layer
then occupies about 0.56 bytes per weight element against bf16's 2 and dequantizes per forward.

AWQ builds (``pre_quant_scale``) are refused: their input channels were rescaled before quantization, so a
plain decode would produce wrong weights.
"""

import functools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from invokeai.backend.quantization.block_scale_tiles import TILE_ROWS, check_tile_layout, unblock_scale_grid
from invokeai.backend.quantization.dequantizing_linear import DequantizingLinear
from invokeai.backend.quantization.fp8_scaled import COMFY_QUANT_SUFFIX, INPUT_SCALE_SUFFIXES, iter_weight_scale_pairs
from invokeai.backend.quantization.int8_convrot import parse_comfy_quant_marker

NVFP4_FORMAT = "nvfp4"
NVFP4_BLOCK_SIZE = 16

WEIGHT_SCALE_2_SUFFIX = ".weight_scale_2"
PRE_QUANT_SCALE_SUFFIX = ".pre_quant_scale"

# Everything a layer carries besides its weight.
_SIDE_CHANNEL_SUFFIXES = (".weight_scale", WEIGHT_SCALE_2_SUFFIX, COMFY_QUANT_SUFFIX, *INPUT_SCALE_SUFFIXES)

_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
# Both decoded elements of every possible byte, upper nibble first: one lookup per byte.
_E2M1_PAIRS_BY_BYTE = torch.stack([_E2M1_VALUES[torch.arange(256) >> 4], _E2M1_VALUES[torch.arange(256) & 0x0F]], dim=1)
# Every float8_e4m3fn value by its bit pattern. Block scales are widened through this rather than cast, so the
# decode needs no float8 kernel, and a packed layer can hold its scales as plain bytes: MPS has no float8 tensors.
_E4M3_BY_BYTE = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32).unsqueeze(1)


# The tables on each device, shared by every layer and read-only. Held per module they would be copied to the
# device on every lock and cast on every forward -- the lesson of int8_convrot's Hadamard matrix. Bounded by the
# devices and compute dtypes in use, at 512 values each.
@functools.cache
def _e2m1_pairs_on(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _E2M1_PAIRS_BY_BYTE.to(device=device, dtype=dtype)


@functools.cache
def _e4m3_values_on(device: torch.device) -> torch.Tensor:
    return _E4M3_BY_BYTE.to(device)


def _check_layout(weight: torch.Tensor | None, weight_scale: torch.Tensor | None, weight_scale_2: torch.Tensor) -> None:
    """Refuse checkpoint tensors this decode would misread. Reads dtypes and shapes only."""
    if weight is None or weight_scale is None:
        raise ValueError(f"has a weight_scale_2 but no {'weight' if weight is None else 'weight_scale'}")
    if weight.dtype is not torch.uint8 or weight.dim() != 2:
        raise ValueError(f"expected a packed uint8 [out, in/2] weight, got {weight.dtype} {tuple(weight.shape)}")
    if weight_scale.dtype is not torch.float8_e4m3fn or weight_scale.dim() != 2:
        raise ValueError(
            f"expected a float8_e4m3fn [out, in/16] block scale, got {weight_scale.dtype} {tuple(weight_scale.shape)}"
        )
    if weight_scale_2.numel() != 1:
        raise ValueError(f"expected a scalar weight_scale_2, got shape {tuple(weight_scale_2.shape)}")
    out_features, packed_width = weight.shape
    rows, blocks = weight_scale.shape
    if rows != out_features or packed_width * 2 != blocks * NVFP4_BLOCK_SIZE:
        raise ValueError(
            f"a {tuple(weight_scale.shape)} block scale does not describe a {tuple(weight.shape)} packed weight "
            f"({NVFP4_BLOCK_SIZE}-element blocks, two elements per byte)"
        )
    check_tile_layout(rows, blocks)


def dequantize_nvfp4_weight(
    weight: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Decode one nvfp4 weight, as the checkpoint stores it, to ``[out, in]`` in ``dtype`` on the weight's device."""
    _check_layout(weight, weight_scale, weight_scale_2)
    return _decode(weight, weight_scale.view(torch.uint8), weight_scale_2, dtype)


def _decode(
    weight: torch.Tensor, scale_bytes: torch.Tensor, weight_scale_2: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """The decode behind :func:`dequantize_nvfp4_weight` and :class:`NVFP4Linear`, over the block scales' bytes.

    Works in ``dtype`` directly. The E2M1 values and the e4m3 block scales are exact in bf16, and the rounding
    the float32 global scale introduces is far below the 4-bit quantization error. The scale grid is reordered
    while still one byte per entry and widened before the weight is, so the weight's decode allocates only its
    result and an int32 byte index beside that grid (see :meth:`NVFP4Linear.dequant_transient_bytes`).
    """
    device = weight.device
    out_features, blocks = scale_bytes.shape
    grid = F.embedding(unblock_scale_grid(scale_bytes.to(device)).to(torch.int32), _e4m3_values_on(device))
    grid = grid.mul_(weight_scale_2.to(device=device, dtype=torch.float32).reshape(())).to(dtype)
    values = F.embedding(weight.to(torch.int32), _e2m1_pairs_on(device, dtype))
    values = values.view(out_features, blocks, NVFP4_BLOCK_SIZE).mul_(grid)
    return values.view(out_features, blocks * NVFP4_BLOCK_SIZE)


class NVFP4Linear(DequantizingLinear):
    """A linear layer storing Comfy nvfp4 weights, dequantized per forward.

    Built from the checkpoint's tensors and registered as persistent buffers under their names, so
    ``load_state_dict`` assigns them as stored. The block scales keep their tiled layout but are held as their
    bytes: MPS cannot hold a float8 tensor, and the decode reads them through a table either way. Resident, a
    layer costs about 0.56 bytes per weight element against bf16's 2.

    The model cache wraps this module as ``CustomNVFP4Linear`` (see ``AUTOCAST_MODULE_TYPE_MAPPING``) for
    sidecar LoRA patches, and a partial load that leaves its buffers on the CPU streams them per forward.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        _check_layout(weight, weight_scale, weight_scale_2)
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1] * 2
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", weight_scale.view(torch.uint8))
        self.register_buffer("weight_scale_2", weight_scale_2)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def _dequantized_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # `.to(device)` is a no-op when the module is resident; under partial load it streams the packed payload
        # to the compute device for this call.
        return _decode(self.weight.to(device), self.weight_scale, self.weight_scale_2, dtype)

    def dequant_transient_bytes(self, compute_dtype: torch.dtype) -> int:
        """The weight in the compute dtype, the int32 byte index it is gathered through, and the scale grid.

        The grid's own widening happens first and peaks lower, at half a byte per weight element. Under partial
        load the packed payload is additionally streamed per call; the resident regime pays nothing for that.
        """
        elements = self.out_features * self.in_features
        return elements * (2 + compute_dtype.itemsize) + elements // NVFP4_BLOCK_SIZE * compute_dtype.itemsize

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, nvfp4"


@dataclass(frozen=True)
class NVFP4Payload:
    """One layer's packed tensors, as the checkpoint stores them."""

    weight: torch.Tensor
    weight_scale: torch.Tensor
    weight_scale_2: torch.Tensor

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    @property
    def in_features(self) -> int:
        return self.weight.shape[1] * 2

    def nbytes(self) -> int:
        return sum(t.nelement() * t.element_size() for t in (self.weight, self.weight_scale, self.weight_scale_2))


def _find_nvfp4_layers(sd: Mapping[str, Any], header_layers: Mapping[str, Any] | None) -> list[str]:
    """Every nvfp4 layer in ``sd``, after refusing what this module would read wrong. Mutates nothing.

    Detection is structural -- a ``weight_scale_2`` beside the weight -- and decided per layer, never per
    file. Refused, naming the layers: AWQ layers; markers that contradict their tensors; packed weights
    with a ``weight_scale`` but no ``weight_scale_2``; malformed layers; and layers that neither a
    ``.comfy_quant`` marker nor a ``_quantization_metadata`` entry (``header_layers``, in ``sd``'s key
    space) names as nvfp4.
    """
    layers = sorted(
        k[: -len(WEIGHT_SCALE_2_SUFFIX)] for k in sd if isinstance(k, str) and k.endswith(WEIGHT_SCALE_2_SUFFIX)
    )
    claimed = set(layers)

    awq = [path for path in layers if f"{path}{PRE_QUANT_SCALE_SUFFIX}" in sd]
    if awq:
        raise ValueError(
            f"Checkpoint has {len(awq)} nvfp4 layer(s) with an AWQ pre_quant_scale, e.g. {awq[:3]}. AWQ rescales "
            "input channels before quantization and is not supported; decoding these as plain nvfp4 would load "
            "wrong weights."
        )

    marked: dict[str, Any] = {}
    for key in [k for k in sd if isinstance(k, str) and k.endswith(COMFY_QUANT_SUFFIX)]:
        path = key[: -len(COMFY_QUANT_SUFFIX)]
        marked[path] = parse_comfy_quant_marker(sd[key]).get("format")
        if marked[path] == NVFP4_FORMAT and path not in claimed:
            raise ValueError(f"nvfp4 layer '{path}' has no weight_scale_2, so it cannot be decoded.")
        if path in claimed and marked[path] not in (None, NVFP4_FORMAT):
            raise ValueError(f"nvfp4 layer '{path}' is marked {marked[path]!r}, which contradicts its tensors.")

    half = sorted(
        weight_key
        for weight_key, _ in iter_weight_scale_pairs(sd)
        if weight_key[: -len(".weight")] not in claimed and getattr(sd[weight_key], "dtype", None) is torch.uint8
    )
    if half:
        raise ValueError(
            f"Checkpoint has {len(half)} packed uint8 weight(s) with a weight_scale but no weight_scale_2, "
            f"e.g. {half[:3]}. They look like nvfp4 missing its global scale; loading them would produce a "
            "model that runs and generates noise."
        )

    for path in layers:
        try:
            _check_layout(
                sd.get(f"{path}.weight"), sd.get(f"{path}.weight_scale"), sd[f"{path}{WEIGHT_SCALE_2_SUFFIX}"]
            )
        except ValueError as e:
            raise ValueError(f"nvfp4 layer '{path}': {e}") from None

    headers = header_layers or {}
    unnamed = [
        path
        for path in layers
        if marked.get(path) != NVFP4_FORMAT
        and not (isinstance(headers.get(path), Mapping) and headers[path].get("format") == NVFP4_FORMAT)
    ]
    if unnamed:
        raise ValueError(
            f"Checkpoint has {len(unnamed)} layer(s) laid out like nvfp4 that no ComfyUI `comfy_quant` marker or "
            f"`_quantization_metadata` entry names, e.g. {unnamed[:3]}. Only ComfyUI's conventions for this layout "
            "are known here; reading another producer's file by them could load a model that runs and generates "
            "wrong images. Use a ComfyUI nvfp4 build or the unquantized model. If this file does name its layers, "
            "look for a warning that its safetensors header could not be read."
        )
    return layers


def reject_nvfp4_layers_a_plain_fold_cannot_decode(sd: Mapping[str, Any], what: str = "This checkpoint") -> None:
    """Refuse an nvfp4 layer to a fold that only multiplies a scale into a weight.

    **Wan** (`wan.py:511`) is the seam this exists for: it reads scaled fp8 and never calls
    :func:`pop_nvfp4_layers`. The other caller of the shared fold, the Qwen2.5-VL encoder, *does*
    pop first (`qwen_image.py:473`, forty-odd lines before its fold) and ships the packed layers, so
    the guard is unreachable there — do not read it as evidence that encoder lacks nvfp4 support.

    The fold has no dtype gate, by design, so it takes the layer: the block-scale grid carries one
    entry per 16 logical elements and the weight packs two 4-bit codes per byte, so eight packed
    columns fall to each grid entry and the shapes line up by accident. What comes out is the packed
    bytes multiplied by the grid, at the compute dtype and half the logical width, reported in the
    log as a dequantized weight.

    Half the width is usually caught at ``load_state_dict`` — but not always, and that is the reason
    to refuse rather than rely on it. Wan infers ``text_dim`` from ``shape[1]`` of
    ``condition_embedder.text_embedder.linear_1`` (`wan.py:356`), which is the dimension the fold
    halves: with that layer packed, the *architecture* is built to match the mangled weight and
    nothing mismatches. The load then fails on the leftover ``weight_scale_2`` with a message about
    Wan variants with extra conditioning branches — a diagnosis pointing somewhere else entirely.

    Detection is the union of two structural tests: a ``weight_scale_2`` beside the weight, which is
    what the decode keys on, and a packed ``uint8`` weight carrying a block scale without one, which
    is the half-state :func:`_find_nvfp4_layers` refuses by name. Keying on the first alone would
    miss the second, and the fold takes it just as readily.

    Narrower than the decode in one way worth knowing: at the Wan seam a bundled nvfp4 text encoder
    in an all-in-one file is removed by ``_drop_benign_extra_keys`` before the fold, so it is dropped
    rather than refused.
    """
    named = {
        key[: -len(WEIGHT_SCALE_2_SUFFIX)]
        for key in sd
        if isinstance(key, str)
        and key.endswith(WEIGHT_SCALE_2_SUFFIX)
        and f"{key[: -len(WEIGHT_SCALE_2_SUFFIX)]}.weight" in sd
    }
    # A packed weight with a block scale and no global one is the half-state `_find_nvfp4_layers`
    # refuses by name. Keying on `weight_scale_2` alone would miss it, and the fold takes it just as
    # readily -- so the detection here is the union, not the narrower test.
    half = {
        weight_key[: -len(".weight")]
        for weight_key, _scale_key in iter_weight_scale_pairs(sd)
        if getattr(sd.get(weight_key), "dtype", None) is torch.uint8
    }
    packed = sorted(named | half)
    if packed:
        raise ValueError(
            f"{what} carries {len(packed)} nvfp4 layer(s) (e.g. {', '.join(packed[:3])}) and this loader does "
            "not support nvfp4. Their weights are two 4-bit codes per byte, so folding a scale into them "
            "produces a tensor of half the width the model needs. Use the fp8 or bf16 build of this checkpoint."
        )


def pop_nvfp4_layers(sd: dict[str, Any], header_layers: Mapping[str, Any] | None = None) -> dict[str, NVFP4Payload]:
    """Take every nvfp4 layer's packed tensors out of ``sd`` and drop the rest of its side channel.

    For loaders that keep the layers packed. Call it before any fp8 handling or cast: the scaled-fp8
    extraction pops every ``weight_scale`` and discards those whose weight is not float8 -- nvfp4's block
    scales included -- and a cast would widen the packed payload. Out of ``sd`` they survive both, and
    :func:`install_nvfp4_layers` puts them back. Whatever a decode would refuse is refused before ``sd``
    is touched.
    """
    payloads: dict[str, NVFP4Payload] = {}
    for path in _find_nvfp4_layers(sd, header_layers):
        payloads[path] = NVFP4Payload(
            weight=sd.pop(f"{path}.weight"),
            weight_scale=sd.pop(f"{path}.weight_scale"),
            weight_scale_2=sd.pop(f"{path}{WEIGHT_SCALE_2_SUFFIX}"),
        )
        for suffix in _SIDE_CHANNEL_SUFFIXES:
            sd.pop(f"{path}{suffix}", None)
    return payloads


def split_nvfp4_rows(path: str, payload: NVFP4Payload, parts: int) -> list[NVFP4Payload]:
    """Split a fused projection's packed tensors into ``parts`` equal row ranges, in order -- a QKV into Q, K, V.

    The block scales split by rows too, but only on whole tile rows (see :func:`unblock_scale_grid`): a part
    made of whole tile rows is a valid grid of its own, while any other cut hands one part another's scales.
    That is refused rather than guessed. The global scale belongs to every part.
    """
    rows = payload.out_features
    if rows % parts or (rows // parts) % TILE_ROWS:
        raise ValueError(
            f"nvfp4 layer '{path}': its {rows} rows do not split into {parts} parts of whole {TILE_ROWS}-row "
            "scale tiles."
        )
    step = rows // parts
    return [
        NVFP4Payload(
            payload.weight[start : start + step], payload.weight_scale[start : start + step], payload.weight_scale_2
        )
        for start in range(0, rows, step)
    ]


def predict_nvfp4_install_size(
    model: torch.nn.Module,
    payloads: Mapping[str, NVFP4Payload],
    dtype: torch.dtype,
    skip_patterns: Iterable[str] = (),
) -> int:
    """Bytes the payloads occupy once :func:`install_nvfp4_layers` has run over them with the same arguments.

    The packed tensors for layers that stay packed, the compute dtype for the ones it decodes. Loaders add
    this to the reservation they make for the rest of the state dict, which no longer holds these layers.
    """
    patterns = tuple(skip_patterns)
    return sum(
        payload.nbytes()
        if _stays_packed(model, path, payload, patterns)
        else payload.out_features * payload.in_features * dtype.itemsize
        for path, payload in payloads.items()
    )


def install_nvfp4_layers(
    model: torch.nn.Module,
    sd: dict[str, Any],
    payloads: Mapping[str, NVFP4Payload],
    dtype: torch.dtype,
    skip_patterns: Iterable[str] = (),
) -> int:
    """Put the nvfp4 layers into the built model and ``sd``; return how many stay packed.

    A layer stays packed as an :class:`NVFP4Linear` where the model has an ``nn.Linear`` outside its
    precision-sensitive modules (``skip_patterns``). Anything else gets its weight decoded to ``dtype``:
    Z-Image's timestep embedder reads its own weight's dtype to pick the dtype of its activations, and a
    packed module there would hand it ``torch.uint8``.

    Call it after the state dict's casts and right before a strict ``load_state_dict(sd, assign=True)``,
    with ``sd`` and ``payloads`` in the model's key space. A packed layer's ``bias`` is taken from ``sd``.
    """
    patterns = tuple(skip_patterns)
    packed = 0
    for path, payload in payloads.items():
        if not _stays_packed(model, path, payload, patterns):
            sd[f"{path}.weight"] = dequantize_nvfp4_weight(
                payload.weight, payload.weight_scale, payload.weight_scale_2, dtype
            )
            continue
        module = NVFP4Linear(payload.weight, payload.weight_scale, payload.weight_scale_2, bias=sd.get(f"{path}.bias"))
        # The strict load that follows checks keys against the new module, not the model's Linear, so a bias one of
        # them has and the other lacks would pass it unnoticed.
        if (module.bias is None) != (model.get_submodule(path).bias is None):
            raise ValueError(
                f"nvfp4 layer '{path}' has {'no' if module.bias is None else 'a'} bias, but the built model's Linear "
                f"there has {'one' if module.bias is None else 'none'}. The checkpoint is misidentified."
            )
        parent_path, _, attribute = path.rpartition(".")
        setattr(model.get_submodule(parent_path) if parent_path else model, attribute, module)
        # The strict load assigns every buffer from `sd`, so it has to find them as the module holds them.
        sd.update(module.state_dict(prefix=f"{path}."))
        packed += 1
    return packed


def _stays_packed(model: torch.nn.Module, path: str, payload: NVFP4Payload, skip_patterns: tuple[str, ...]) -> bool:
    """Whether a layer stays packed: the one decision the install and its size prediction share."""
    try:
        module = model.get_submodule(path)
    except AttributeError:
        raise ValueError(
            f"nvfp4 layer '{path}' names no module in the built model. The checkpoint is misidentified, or its "
            "keys need a conversion this loader did not apply."
        ) from None
    if not isinstance(module, torch.nn.Linear) or any(pattern in path for pattern in skip_patterns):
        return False
    # The strict load checks a decoded weight's shape against its parameter, but assigns packed buffers into a new
    # module unchecked, so a mismatch would only surface inside the forward.
    if (module.out_features, module.in_features) != (payload.out_features, payload.in_features):
        raise ValueError(
            f"nvfp4 layer '{path}' holds a {payload.out_features}x{payload.in_features} weight, but the built model's "
            f"Linear there is {module.out_features}x{module.in_features}. The checkpoint is misidentified."
        )
    return True
