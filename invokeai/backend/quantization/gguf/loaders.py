import gc
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Literal

import gguf
import numpy as np
import torch

from invokeai.backend.quantization.fp8_scaled import COMFY_QUANT_SUFFIX
from invokeai.backend.quantization.gguf.ggml_tensor import HOST_DEQUANT_PEAK_COPIES, GGMLTensor
from invokeai.backend.quantization.gguf.utils import TORCH_COMPATIBLE_QTYPES
from invokeai.backend.quantization.int8_convrot import INT8_TENSORWISE_FORMAT
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger()

#: The names `GGUFReader` gives the file header's own values, in the same namespace as the metadata.
_HEADER_FIELD_NAMES = frozenset({"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"})
#: How `GGUFReader` refuses a metadata entry named like one of those, as the `KeyError`'s message.
_REPEATED_HEADER_FIELD = re.compile(rf"^Duplicate ({'|'.join(map(re.escape, sorted(_HEADER_FIELD_NAMES)))}) ")


class _HeaderTolerantGGUFReader(gguf.GGUFReader):
    """`GGUFReader` for a file that repeats the header fields as metadata; see `_open_gguf_reader`.

    Some converters copy a reader's fields back into the file they write, the three header fields among
    them: every file of vanes430's Anima-2.9B GGUFs carries them. The header is read first and wins; the
    copies are dropped, and they may be stale (that file's copied `GGUF.kv_count` is 3, its header's 6).
    """

    def _push_field(self, field: gguf.ReaderField, *args: Any, **kwargs: Any) -> int:
        # The reader pushes the header before any metadata, so a second field of one of its names is a copy.
        # Arguments are forwarded as given: `_push_field` is gguf-py's private method, and this override only
        # runs for the files above.
        if field.name in _HEADER_FIELD_NAMES and field.name in self.fields:
            return 0
        return super()._push_field(field, *args, **kwargs)


def _open_gguf_reader(path: Path) -> gguf.GGUFReader:
    """gguf-py's own reader; for a file that repeats the header as metadata, which it refuses, the tolerant one.

    The retry is narrow on purpose: every other file never meets the subclass, so a gguf-py release that
    changes the private method it overrides can break only the files gguf-py itself cannot read.
    """
    try:
        return gguf.GGUFReader(path)
    except KeyError as e:
        if not (e.args and isinstance(e.args[0], str) and _REPEATED_HEADER_FIELD.match(e.args[0])):
            raise
    return _HeaderTolerantGGUFReader(path)


class WrappedGGUFReader:
    """Wrapper around GGUFReader that adds a close() method."""

    def __init__(self, path: Path):
        self.reader = _open_gguf_reader(path)

    def __enter__(self):
        return self.reader

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Drop this wrapper's references to the reader and its memory map, then collect.

        The map closes once nothing references it. A caller's `with WrappedGGUFReader(path) as reader:` keeps the
        reader, and through its tensor views the map, until that name goes out of scope -- for the callers here,
        when they return -- so the collection releases only what reference cycles would otherwise hold.
        """
        if hasattr(self.reader, "data"):
            try:
                self.reader.data.flush()
                del self.reader.data
            except (AttributeError, OSError, ValueError) as e:
                logger.warning(f"Wasn't able to close GGUF memory map: {e}")
        del self.reader
        gc.collect()


def gguf_tensor_bytes(path: Path) -> dict[str, int]:
    """The stored size of each of a GGUF file's tensors, by name, from its header alone; no tensor data is read.

    For a loader that decides from the names which tensors to read, and sizes what it will read. The reader is
    referenced by this expression only, so the memory map is released as soon as the sizes are read, without the
    full collection `WrappedGGUFReader.close` runs, which costs most of a second on a loaded process (measured on
    Windows, where a file still mapped cannot be deleted: it deletes at once with the collector disabled).
    """
    return {tensor.name: int(tensor.n_bytes) for tensor in _open_gguf_reader(path).tensors}


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

Q8CRMode = Literal["refuse", "decode"]


def parse_q8_cr_markers(fields: Mapping[str, Any], source: str) -> dict[str, dict[str, Any]]:
    """Parse ComfyUI-GGUF's ``Q8_CR`` layers from a GGUF's ``comfy.gguf.quant.<weight name>`` metadata.

    ``Q8_CR`` (github.com/molbal/ComfyUI-GGUF) is ComfyUI's ``int8_tensorwise`` + convrot scheme in a
    GGUF container: the int8 codes are stored under GGML type ``I8`` (GGUF has no other int8 type), a
    per-output-row F32 scale beside them as ``<weight name>_scale``, and the JSON that a safetensors
    export keeps in ``<layer>.comfy_quant`` here lives in the metadata, as a string. ``I8`` is not a
    GGML quantization -- ``gguf`` has no dequantizer for it -- so these layers cannot ride the
    GGMLTensor path at all.

    ``fields`` maps metadata keys to their values; only the ``comfy.gguf.quant.*`` keys, whose values
    are JSON strings, are read. Returns the int8 markers keyed by weight name, in the form
    :mod:`int8_convrot` reads. Raises ``ValueError`` for anything else the converter writes
    (``int4_cr``, the retired ``int4_pytorch``): those tensors are ``I8`` too, and nothing here
    decodes them. Model identification turns that into the installer's refusal, so a file is refused
    for the same reason at install and at load.
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
        if not isinstance(config, dict) or config.get("format") != INT8_TENSORWISE_FORMAT:
            fmt = config.get("format") if isinstance(config, dict) else config
            raise ValueError(
                f"{source}: tensor {weight_name!r} uses ComfyUI-GGUF quantization {fmt!r}, which is not "
                f"supported. Only {INT8_TENSORWISE_FORMAT!r} (Q8_CR) layers can be loaded from GGUF."
            )
        if not weight_name.endswith(".weight"):
            raise ValueError(f"{source}: {INT8_TENSORWISE_FORMAT} metadata names {weight_name!r}, not a weight.")
        marker = dict(config)
        # The converter's own loader treats `convrot` without `weight_rotated` as a file from an older
        # converter that stored the codes unrotated, and runs it without the rotation. Un-rotating
        # weights that were never rotated would scramble them, so follow it.
        if marker.get("convrot") and not marker.get("weight_rotated", False):
            marker["convrot"] = False
            marker.pop("convrot_groupsize", None)
        layers[weight_name] = marker
    return layers


def _logical_shape(tensor: gguf.ReaderTensor, orig_shapes: Mapping[str, torch.Size]) -> torch.Size:
    """The shape a tensor has in the model: GGUF's reversed dims, or the native shape ComfyUI recorded."""
    shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
    orig_shape = orig_shapes.get(tensor.name)
    if orig_shape is None:
        return shape
    if orig_shape.numel() != shape.numel():
        raise ValueError(
            f"GGUF tensor {tensor.name!r} declares original shape {tuple(orig_shape)}, which has a "
            f"different element count than its stored shape {tuple(shape)}."
        )
    return orig_shape


def gguf_header_state_dict(path: Path, compute_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """A GGUF file's tensors as `gguf_sd_loader` returns them, but on the `meta` device: read from the header.

    Model identification needs names, shapes and quantization types, never values, and the header holds them
    all. `gguf_sd_loader` copies every tensor out of the file -- for a 23.6 GB transformer, 22 GiB of RAM per
    probe -- to answer questions the first few MB answer. Each tensor comes back as the `GGMLTensor` the full
    read gives, with the same packed and logical shapes; a ComfyUI-GGUF ``Q8_CR`` layer comes back as its raw
    ``I8`` codes, without the markers a decoding read adds, since whether it is supported is decided from the
    metadata. The reader maps the file but touches only the header.
    """
    with WrappedGGUFReader(path) as reader:
        orig_shapes = _read_comfy_orig_shapes(reader)
        sd: dict[str, torch.Tensor] = {}
        for tensor in reader.tensors:
            shape = _logical_shape(tensor, orig_shapes)
            stored_dtype = torch.from_numpy(np.empty(0, dtype=tensor.data.dtype)).dtype
            stored_shape = shape if tensor.tensor_type in TORCH_COMPATIBLE_QTYPES else tensor.data.shape
            sd[tensor.name] = GGMLTensor(
                torch.empty(stored_shape, dtype=stored_dtype, device="meta"),
                ggml_quantization_type=tensor.tensor_type,
                tensor_shape=shape,
                compute_dtype=compute_dtype,
            )
        return sd


def gguf_sd_loader(
    path: Path,
    compute_dtype: torch.dtype,
    *,
    q8_cr: Q8CRMode = "refuse",
    keep: Callable[[str], bool] | None = None,
) -> dict[str, torch.Tensor]:
    """Read a GGUF file into a state dict of ``GGMLTensor``s, dequantized on the fly at use.

    ``keep`` selects the tensors to read by name. The rest are skipped before their data is copied, so a
    loader that uses only part of a file -- a transformer with its text connectors bundled beside it, an
    encoder with its tokenizer packed in -- does not pay RAM for the part it drops. It selects tensors
    only: ``Q8_CR`` markers are read for the whole file, so ``"refuse"`` refuses a file even if ``keep``
    would skip every marked layer.

    ``q8_cr`` says what to do with ComfyUI-GGUF ``Q8_CR`` layers (see :func:`parse_q8_cr_markers`):

    - ``"decode"``: their weight, scale and bias come back as plain tensors with a ``<layer>.comfy_quant``
      marker beside them -- exactly a safetensors ``int8_tensorwise`` export, so the loader decodes both
      containers with the same :mod:`int8_convrot` code. Only a loader that does so may ask for this.
    - ``"refuse"``: any other loader. Raises before the tensors are read; without it the int8 codes reach
      a plain ``nn.Linear`` and the load fails on a dtype error that names neither the file nor the format.
      Identification refuses such files at install already, so this guards records made before it did.

    Model identification does not come here: it reads the header (`gguf_header_state_dict`).
    """
    with WrappedGGUFReader(path) as reader:
        sd: dict[str, torch.Tensor] = {}
        orig_shapes = _read_comfy_orig_shapes(reader)
        quant_fields = {k: f.contents() for k, f in reader.fields.items() if k.startswith(COMFY_QUANT_KEY_PREFIX)}
        int8_layers = parse_q8_cr_markers(quant_fields, path.name)
        if int8_layers and q8_cr == "refuse":
            raise ValueError(
                f"{path.name} holds {len(int8_layers)} ComfyUI-GGUF Q8_CR (int8 convrot) layer(s), e.g. "
                f"{next(iter(int8_layers))!r}. Loading Q8_CR from GGUF is not supported for this model type; "
                "use a GGML-quantized (Q8_0, Q4_K, ...) or safetensors build instead."
            )
        int8_scales = {f"{name}_scale" for name in int8_layers}
        int8_biases = {f"{name[: -len('.weight')]}.bias" for name in int8_layers}
        missing = sorted((int8_layers.keys() | int8_scales) - {tensor.name for tensor in reader.tensors})
        if missing:
            raise ValueError(f"{path.name}: Q8_CR metadata names layers the file does not hold, e.g. {missing[0]!r}.")

        for tensor in reader.tensors:
            if keep is not None and not keep(tensor.name):
                continue
            # Use .copy() to create a true copy of the data, not a view.
            # This is critical on Windows where the memory-mapped file cannot be deleted
            # while tensors still hold references to the mapped memory.
            torch_tensor = torch.from_numpy(tensor.data.copy())

            shape = _logical_shape(tensor, orig_shapes)

            if tensor.name in int8_layers or tensor.name in int8_scales:
                expected = gguf.GGMLQuantizationType.I8 if tensor.name in int8_layers else gguf.GGMLQuantizationType.F32
                if tensor.tensor_type != expected:
                    raise ValueError(
                        f"{path.name}: Q8_CR tensor {tensor.name!r} is stored as {tensor.tensor_type.name}, "
                        f"expected {expected.name}."
                    )
                sd[tensor.name] = torch_tensor.reshape(shape)
                continue

            if tensor.tensor_type in TORCH_COMPATIBLE_QTYPES:
                torch_tensor = torch_tensor.view(*shape)
            ggml_tensor = GGMLTensor(
                torch_tensor,
                ggml_quantization_type=tensor.tensor_type,
                tensor_shape=shape,
                compute_dtype=compute_dtype,
            )
            # An int8 layer's bias becomes a buffer of `Int8ConvrotLinear`, whose forward casts it to the
            # activation dtype -- a dtype change a GGMLTensor refuses.
            sd[tensor.name] = ggml_tensor.get_dequantized_tensor() if tensor.name in int8_biases else ggml_tensor

        for weight_name, marker in int8_layers.items():
            raw = json.dumps(marker).encode("utf-8")
            marker_key = f"{weight_name[: -len('.weight')]}{COMFY_QUANT_SUFFIX}"
            sd[marker_key] = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        return sd


# GGUF's unquantized storage types. `gguf_sd_loader` wraps BF16 like a quantized type (torch has no
# GGML-side view for it), so here it is as packed as Q4_0 -- but dequantizing it changes no byte count.
UNQUANTIZED_GGML_TYPES = frozenset(
    {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16}
)

# Output elements a quantized tensor is decoded in at a time. The kernels' temporaries run to several
# times what they decode (`HOST_DEQUANT_PEAK_COPIES`), which for a whole 2 GB embedding is ~8 GB of RAM
# on top of the model; in 32 MB slices of a preallocated result it is a few hundred MB.
_UNPACK_CHUNK_ELEMENTS = 1 << 24


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
    the way -- about 8 bytes per element. What is in flight beside a result is the packed original it
    replaces and, for a quantized tensor, the temporaries of one row slice (see `unpack_ggml_tensor`).
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
            value._ggml_quantization_type in UNQUANTIZED_GGML_TYPES
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
    in_flight = max((_in_flight_bytes(sd[key]) for key in keys), default=0)
    reserve(kept + sum(unpacked.values()) + in_flight)
    # One at a time, so each packed original is released as its replacement lands.
    for key in keys:
        sd[key] = unpack_ggml_tensor(sd[key])
    return len(keys)


def _row_slices(value: GGMLTensor) -> tuple[torch.Tensor, int, int]:
    """A quantized `value`'s packed rows, the elements one row decodes to, and the rows one slice takes."""
    rows = value.quantized_data.reshape(-1, value.quantized_data.shape[-1])
    per_row = value.tensor_shape.numel() // rows.shape[0]
    return rows, per_row, max(1, _UNPACK_CHUNK_ELEMENTS // per_row)


def _in_flight_bytes(value: GGMLTensor) -> int:
    """What replacing `value` holds beside its result: the packed original, released only once the result
    lands, and for a quantized tensor the decode temporaries of one slice."""
    packed = value.quantized_data.nbytes
    if value._ggml_quantization_type in UNQUANTIZED_GGML_TYPES:
        return packed
    rows, per_row, step = _row_slices(value)
    return packed + HOST_DEQUANT_PEAK_COPIES * min(step, rows.shape[0]) * per_row * value.compute_dtype.itemsize


def unpack_ggml_tensor(value: GGMLTensor) -> torch.Tensor:
    """`value` as a plain tensor in its compute dtype; BF16 as a view of its bytes, not a decode.

    A quantized tensor is decoded a slice of rows at a time into its preallocated result. GGML blocks
    never cross a row, so each slice decodes on its own to exactly what the whole would.
    """
    qtype = value._ggml_quantization_type
    if qtype is gguf.GGMLQuantizationType.BF16:
        return value.quantized_data.view(torch.bfloat16).reshape(value.tensor_shape).to(value.compute_dtype)
    if qtype in UNQUANTIZED_GGML_TYPES:
        return value.get_dequantized_tensor()
    rows, per_row, step = _row_slices(value)
    out = torch.empty(value.tensor_shape.numel(), dtype=value.compute_dtype, device=value.quantized_data.device)
    for start in range(0, rows.shape[0], step):
        chunk = rows[start : start + step]
        decoded = GGMLTensor(
            chunk,
            ggml_quantization_type=qtype,
            tensor_shape=torch.Size([chunk.shape[0] * per_row]),
            compute_dtype=value.compute_dtype,
        ).get_dequantized_tensor()
        out[start * per_row : start * per_row + decoded.numel()] = decoded
        # Released before the next slice decodes, which would otherwise run beside this one.
        del decoded
    return out.reshape(value.tensor_shape)
