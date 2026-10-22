import gc
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Literal

import gguf
import numpy as np
import torch

from invokeai.backend.quantization.fp8_scaled import COMFY_QUANT_SUFFIX
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.utils import TORCH_COMPATIBLE_QTYPES
from invokeai.backend.quantization.int8_convrot import (
    CONVROT_GROUP_SIZE,
    INT4_TENSORWISE_FORMAT,
    INT8_TENSORWISE_FORMAT,
    marker_layout,
)
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger()


class WrappedGGUFReader:
    """Wrapper around GGUFReader that adds a close() method."""

    def __init__(self, path: Path):
        self.reader = gguf.GGUFReader(path)

    def __enter__(self):
        return self.reader

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Explicitly close the memory-mapped file."""
        if hasattr(self.reader, "data"):
            try:
                self.reader.data.flush()
                del self.reader.data
            except (AttributeError, OSError, ValueError) as e:
                logger.warning(f"Wasn't able to close GGUF memory map: {e}")
        del self.reader
        gc.collect()


ORIG_SHAPE_KEY_PREFIX = "comfy.gguf.orig_shape."


def _coerce_dim(value: Any) -> int | None:
    """Coerce a single value from a GGUF metadata array to a positive dimension, or None if it isn't one.

    Metadata is untrusted input, so anything that is not a finite, integral, positive number is rejected rather
    than silently truncated (``int(2.5) == 2``) or allowed to raise (``int(float("inf"))``).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        dim = int(value)
    elif isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value) or not value.is_integer():
            return None
        dim = int(value)
    else:
        return None
    return dim if dim > 0 else None


def _read_comfy_orig_shapes(reader: gguf.GGUFReader) -> dict[str, torch.Size]:
    """Read ComfyUI's ``comfy.gguf.orig_shape.<tensor name>`` metadata.

    ComfyUI's GGUF converter can only quantize 2-D tensors, so it reshapes any tensor whose native
    rank/shape the quantizer rejects (e.g. Krea-2's ``first.weight`` of (6144, 64)) into a workable
    2-D shape and records the native shape under this key. Without honoring it, the tensor loads with
    the reshaped shape and ``load_state_dict`` fails with a size mismatch.
    """
    orig_shapes: dict[str, torch.Size] = {}
    for key, field in reader.fields.items():
        if not key.startswith(ORIG_SHAPE_KEY_PREFIX):
            continue
        tensor_name = key[len(ORIG_SHAPE_KEY_PREFIX) :]
        try:
            contents = field.contents()
        except Exception as e:
            logger.warning(f"Ignoring malformed GGUF metadata key {key!r}: {e}")
            continue
        if not isinstance(contents, (list, tuple)):
            logger.warning(f"Ignoring malformed GGUF metadata key {key!r}: expected an array, got {contents!r}")
            continue
        dims = tuple(_coerce_dim(v) for v in contents)
        if not dims or any(d is None for d in dims):
            logger.warning(f"Ignoring malformed GGUF metadata key {key!r}: {tuple(contents)!r}")
            continue
        orig_shapes[tensor_name] = torch.Size(dims)
    return orig_shapes


COMFY_QUANT_KEY_PREFIX = "comfy.gguf.quant."

GGUFConvrotMode = Literal["refuse", "decode", "ignore"]

# ComfyUI-GGUF's name for its packed int4 convrot layers (`Q4_CR`), and the one backing InvokeAI decodes.
INT4_CR_FORMAT = "int4_cr"
INT4_CR_BACKING = "w4a4"


def parse_gguf_convrot_markers(fields: Mapping[str, Any], source: str) -> dict[str, dict[str, Any]]:
    """Parse ComfyUI-GGUF's convrot layers from a GGUF's ``comfy.gguf.quant.<weight name>`` metadata.

    ComfyUI-GGUF (github.com/molbal/ComfyUI-GGUF) stores ComfyUI's integer convrot schemes in a GGUF
    container, under GGML type ``I8`` -- GGUF has no other integer type -- with a per-output-row scale beside
    them as ``<weight name>_scale``. The JSON a safetensors export keeps in ``<layer>.comfy_quant`` here lives
    in the metadata, as a string. ``I8`` is not a GGML quantization -- ``gguf`` has no dequantizer for it --
    so these layers cannot ride the GGMLTensor path at all. Two formats are read:

    - ``Q8_CR``, ``int8_tensorwise``: int8 codes ``[out, in]`` and an F32 scale.
    - ``Q4_CR``, ``int4_cr`` with ``"backing": "w4a4"``: comfy-kitchen's packed int4 layout, codes ``[out, in/2]``
      two per byte and an F16 scale, always rotated. The JSON's ``orig_shape`` gives the unpacked shape. It
      comes back as an ``int4_tensorwise`` marker, which is what :mod:`int8_convrot` decodes it as.

    ``fields`` maps metadata keys to their string values; other keys are ignored. Returns the markers keyed by
    weight name, in the form :mod:`int8_convrot` reads. Raises ``ValueError`` for anything else the converter
    writes: ``int4_cr`` without the w4a4 backing (its retired W4A16 form, with group-wise scales) and the retired
    ``int4_pytorch``. Those tensors are ``I8`` too, and nothing here decodes them. Model identification turns
    that into the installer's refusal, so a file is refused for the same reason at install and at load.
    """
    layers: dict[str, dict[str, Any]] = {}
    for key, value in fields.items():
        if not key.startswith(COMFY_QUANT_KEY_PREFIX):
            continue
        weight_name = key[len(COMFY_QUANT_KEY_PREFIX) :]
        try:
            config = json.loads(value)
        except Exception as e:
            raise ValueError(f"{source}: GGUF metadata {key!r} is not valid quantization JSON: {e}") from None
        if not isinstance(config, dict):
            raise ValueError(f"{source}: GGUF metadata {key!r} is not a quantization object: {config!r}")
        fmt = config.get("format")
        if fmt == INT8_TENSORWISE_FORMAT:
            marker = dict(config)
            # The converter's own loader treats `convrot` without `weight_rotated` as a file from an older
            # converter that stored the codes unrotated, and runs it without the rotation. Un-rotating
            # weights that were never rotated would scramble them, so follow it.
            if marker.get("convrot") and not marker.get("weight_rotated", False):
                marker["convrot"] = False
                marker.pop("convrot_groupsize", None)
        elif fmt == INT4_CR_FORMAT and config.get("backing") == INT4_CR_BACKING:
            orig_shape = config.get("orig_shape")
            if not (
                isinstance(orig_shape, list)
                and len(orig_shape) == 2
                and all(isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in orig_shape)
                and orig_shape[1] % 2 == 0
            ):
                raise ValueError(f"{source}: Q4_CR tensor {weight_name!r} has no usable orig_shape ({orig_shape!r}).")
            group_size = config.get("convrot_groupsize", CONVROT_GROUP_SIZE)
            if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
                raise ValueError(
                    f"{source}: Q4_CR tensor {weight_name!r} has no usable convrot_groupsize ({group_size!r})."
                )
            # The w4a4 backing is symmetric and pre-rotated by definition -- its kernel rotates the activations to
            # match -- so the decode assumes both. A file that says otherwise would decode to a weight of the right
            # shape and the wrong values (without the rotation a real Q4_CR layer correlates with its weights at
            # 0.06), so it is refused instead.
            if config.get("sym", True) is not True or config.get("weight_rotated", True) is not True:
                raise ValueError(
                    f"{source}: Q4_CR tensor {weight_name!r} declares sym={config.get('sym')!r}, "
                    f"weight_rotated={config.get('weight_rotated')!r}; only symmetric, pre-rotated codes are decoded."
                )
            marker = {
                "format": INT4_TENSORWISE_FORMAT,
                "convrot": True,
                "convrot_groupsize": group_size,
                "orig_shape": list(orig_shape),
            }
        else:
            backing = ""
            if fmt == INT4_CR_FORMAT:
                backing = f" with backing {config['backing']!r}" if "backing" in config else " without a backing"
            raise ValueError(
                f"{source}: tensor {weight_name!r} uses ComfyUI-GGUF quantization {fmt!r}{backing}, which is not "
                f"supported. Only {INT8_TENSORWISE_FORMAT!r} (Q8_CR) and {INT4_CR_FORMAT!r} with the "
                f"{INT4_CR_BACKING!r} backing (Q4_CR) layers can be loaded from GGUF."
            )
        if not weight_name.endswith(".weight"):
            raise ValueError(f"{source}: {fmt} metadata names {weight_name!r}, not a weight.")
        layers[weight_name] = marker
    return layers


def gguf_sd_loader(
    path: Path, compute_dtype: torch.dtype, *, convrot: GGUFConvrotMode = "refuse"
) -> dict[str, torch.Tensor]:
    """Read a GGUF file into a state dict of ``GGMLTensor``s, dequantized on the fly at use.

    ``convrot`` says what to do with ComfyUI-GGUF's ``Q8_CR``/``Q4_CR`` layers (see
    :func:`parse_gguf_convrot_markers`):

    - ``"decode"``: their weight, scale and bias come back as plain tensors with a ``<layer>.comfy_quant``
      marker beside them -- exactly a safetensors ``int8_tensorwise``/``int4_tensorwise`` export, so the loader
      decodes both containers with the same :mod:`int8_convrot` code. Only a loader that does so may ask for this.
    - ``"refuse"``: any other loader. Raises before the tensors are read; without it the codes reach a plain
      ``nn.Linear`` and the load fails on a dtype error that names neither the file nor the format.
      Identification refuses such files at install already, so this guards records made before it did.
    - ``"ignore"``: model identification, which needs shapes and never builds a module. The file reads as
      it always has; whether its quantization is supported is the installer's question, answered from
      the metadata (see ``ModelConfigFactory``), so it is not raised here once per candidate config.
    """
    with WrappedGGUFReader(path) as reader:
        sd: dict[str, torch.Tensor] = {}
        orig_shapes = _read_comfy_orig_shapes(reader)
        layers: dict[str, dict[str, Any]] = {}
        if convrot != "ignore":
            quant_fields = {k: f.contents() for k, f in reader.fields.items() if k.startswith(COMFY_QUANT_KEY_PREFIX)}
            layers = parse_gguf_convrot_markers(quant_fields, path.name)
        if layers and convrot == "refuse":
            raise ValueError(
                f"{path.name} holds {len(layers)} ComfyUI-GGUF convrot (Q8_CR or Q4_CR) layer(s), e.g. "
                f"{next(iter(layers))!r}. Loading those from GGUF is not supported for this model type; "
                "use a GGML-quantized (Q8_0, Q4_K, ...) or safetensors build instead."
            )
        packed = {name for name, marker in layers.items() if marker_layout(marker).packed_int4}
        scales = {f"{name}_scale": name for name in layers}
        biases = {f"{name[: -len('.weight')]}.bias" for name in layers}
        missing = sorted((layers.keys() | scales.keys()) - {tensor.name for tensor in reader.tensors})
        if missing:
            raise ValueError(f"{path.name}: convrot metadata names layers the file does not hold, e.g. {missing[0]!r}.")

        for tensor in reader.tensors:
            # Use .copy() to create a true copy of the data, not a view.
            # This is critical on Windows where the memory-mapped file cannot be deleted
            # while tensors still hold references to the mapped memory.
            torch_tensor = torch.from_numpy(tensor.data.copy())

            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            if tensor.name in layers or tensor.name in scales:
                # Stored as written: a packed int4 weight holds half its elements, so it is checked against the
                # quant JSON's own unpacked shape below, never reshaped to a `comfy.gguf.orig_shape` entry.
                name = scales.get(tensor.name, tensor.name)
                is_scale = tensor.name in scales
                expected = (
                    gguf.GGMLQuantizationType.I8
                    if not is_scale
                    else gguf.GGMLQuantizationType.F16
                    if name in packed
                    else gguf.GGMLQuantizationType.F32
                )
                if tensor.tensor_type != expected:
                    raise ValueError(
                        f"{path.name}: convrot tensor {tensor.name!r} is stored as {tensor.tensor_type.name}, "
                        f"expected {expected.name}."
                    )
                if name in packed and not is_scale:
                    rows, cols = layers[name]["orig_shape"]
                    if tuple(shape) != (rows, cols // 2):
                        raise ValueError(
                            f"{path.name}: Q4_CR tensor {tensor.name!r} is stored as {tuple(shape)}, which does not hold "
                            f"the packed codes of its declared ({rows}, {cols}) weight."
                        )
                # The int4 scale widens to fp32, the dtype `Int4ConvrotLinear` holds it in.
                sd[tensor.name] = (
                    torch_tensor.reshape(shape).to(torch.float32) if is_scale else torch_tensor.reshape(shape)
                )
                continue

            orig_shape = orig_shapes.get(tensor.name)
            if orig_shape is not None:
                if orig_shape.numel() != shape.numel():
                    raise ValueError(
                        f"GGUF tensor {tensor.name!r} declares original shape {tuple(orig_shape)}, which has a "
                        f"different element count than its stored shape {tuple(shape)}."
                    )
                shape = orig_shape

            if tensor.tensor_type in TORCH_COMPATIBLE_QTYPES:
                torch_tensor = torch_tensor.view(*shape)
            ggml_tensor = GGMLTensor(
                torch_tensor,
                ggml_quantization_type=tensor.tensor_type,
                tensor_shape=shape,
                compute_dtype=compute_dtype,
            )
            # A convrot layer's bias becomes a buffer of its quantized module, whose forward casts it to the
            # activation dtype -- a dtype change a GGMLTensor refuses.
            sd[tensor.name] = ggml_tensor.get_dequantized_tensor() if tensor.name in biases else ggml_tensor

        for weight_name, marker in layers.items():
            raw = json.dumps(marker).encode("utf-8")
            marker_key = f"{weight_name[: -len('.weight')]}{COMFY_QUANT_SUFFIX}"
            sd[marker_key] = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        return sd


# GGUF's unquantized storage types. `gguf_sd_loader` wraps BF16 like a quantized type (torch has no
# GGML-side view for it), so here it is as packed as Q4_0 -- but dequantizing it changes no byte count.
_UNQUANTIZED_GGML_TYPES = frozenset(
    {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16}
)


def unpack_ggml_at_load(
    sd: dict[str, torch.Tensor],
    model: torch.nn.Module,
    skip_patterns: tuple[str, ...],
    reserve: Callable[[int], None],
) -> int:
    """Replace, in `sd`, the GGUF tensors that must not -- or need not -- stay packed. Returns the count.

    Must not: `GGMLTensor` only works where torch dispatches the op to it (a Linear's matmul, `mul`,
    `add`). An embedding lookup is not dispatched at all, `F.rms_norm` and `F.layer_norm` see the
    packed buffer, a convolution is not a Linear, and a model that reads its compute dtype off a
    weight reads the *storage* dtype -- `uint8` -- and casts its activations to it. So everything
    that is not a Linear weight or bias, plus `skip_patterns` (the model's own precision-sensitive
    modules, see `_model_declared_skip_patterns`). Releases differ in which of these they quantize,
    hence a rule by module rather than by quantization type.

    Need not: an unquantized tensor (F32/F16/BF16) is no larger unpacked, but kept packed it is
    dequantized on every forward -- for a 53248-wide BF16 projection, a measured 1.87 GB transient
    per call.

    The reservation is absolute, because `make_room` makes that much room rather than adding to the
    file-size reservation the framework made before the loader ran: everything that stays packed at
    its packed size, everything replaced at its unpacked size, and one replacement in flight. BF16 is
    reinterpreted rather than run through the GGML kernel, which widens through int32 and float32 on
    the way -- about 8 bytes per element.
    """
    linear_params = {
        f"{module_name}.{param_name}"
        for module_name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        for param_name, _ in module.named_parameters(recurse=False)
    }
    keys = [
        key
        for key, value in sd.items()
        if isinstance(value, GGMLTensor)
        and (
            value._ggml_quantization_type in _UNQUANTIZED_GGML_TYPES
            or key not in linear_params
            or any(pattern in key for pattern in skip_patterns)
        )
    ]
    unpacked = {key: sd[key].tensor_shape.numel() * sd[key].compute_dtype.itemsize for key in keys}
    kept = sum(
        value.quantized_data.nbytes if isinstance(value, GGMLTensor) else value.nbytes
        for key, value in sd.items()
        if key not in unpacked
    )
    reserve(kept + sum(unpacked.values()) + max(unpacked.values(), default=0))
    # One at a time, so each packed original is released as its replacement lands.
    for key in keys:
        sd[key] = _unpack(sd[key])
    return len(keys)


def _unpack(value: GGMLTensor) -> torch.Tensor:
    """`value` as a plain tensor in its compute dtype; BF16 as a view of its bytes, not a decode."""
    if value._ggml_quantization_type is gguf.GGMLQuantizationType.BF16:
        return value.quantized_data.view(torch.bfloat16).reshape(value.tensor_shape).to(value.compute_dtype)
    return value.get_dequantized_tensor()
