from collections.abc import Sequence

import gguf
import numpy as np
import pytest
import torch

from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.quantization.gguf import loaders
from invokeai.backend.quantization.gguf.ggml_tensor import HOST_DEQUANT_PEAK_COPIES, GGMLTensor
from invokeai.backend.quantization.gguf.loaders import (
    WrappedGGUFReader,
    gguf_header_state_dict,
    gguf_sd_loader,
    gguf_tensor_bytes,
    unpack_ggml_at_load,
)
from tests.fixtures.quantized_payloads import q8_cr_marker, quantize_convrot, write_gguf


def _write_gguf(path, *, orig_shape: Sequence[float] | None) -> None:
    """Write a tiny GGUF holding one F32 tensor stored 2-D as (256, 1536) i.e. torch (1536, 256)."""
    writer = gguf.GGUFWriter(str(path), "krea2")
    stored = np.arange(1536 * 256, dtype=np.float32).reshape(1536, 256)
    writer.add_tensor("first.weight", stored, raw_dtype=gguf.GGMLQuantizationType.F32)
    if orig_shape is not None:
        writer.add_array("comfy.gguf.orig_shape.first.weight", list(orig_shape))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_gguf_sd_loader_honors_comfy_orig_shape(tmp_path):
    """ComfyUI reshapes non-2-D tensors before quantizing; the recorded native shape must win."""
    path = tmp_path / "model.gguf"
    _write_gguf(path, orig_shape=(6144, 64))

    sd = gguf_sd_loader(path, compute_dtype=torch.bfloat16)

    assert tuple(sd["first.weight"].shape) == (6144, 64)
    assert tuple(sd["first.weight"].get_dequantized_tensor().shape) == (6144, 64)


def test_gguf_sd_loader_without_orig_shape(tmp_path):
    path = tmp_path / "model.gguf"
    _write_gguf(path, orig_shape=None)

    sd = gguf_sd_loader(path, compute_dtype=torch.bfloat16)

    assert tuple(sd["first.weight"].shape) == (1536, 256)


def test_gguf_sd_loader_rejects_orig_shape_with_wrong_element_count(tmp_path):
    path = tmp_path / "model.gguf"
    _write_gguf(path, orig_shape=(6144, 65))

    with pytest.raises(ValueError, match="different element count"):
        gguf_sd_loader(path, compute_dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "orig_shape",
    [
        pytest.param([2.5, 157286.4], id="non-integral"),
        pytest.param([float("inf")], id="infinite"),
        pytest.param([float("nan"), 64.0], id="nan"),
        pytest.param([-6144.0, -64.0], id="negative"),
    ],
)
def test_gguf_sd_loader_ignores_malformed_orig_shape(tmp_path, orig_shape):
    """Malformed metadata must be ignored with a warning - never truncated to a wrong shape, never raised."""
    path = tmp_path / "model.gguf"
    _write_gguf(path, orig_shape=orig_shape)

    sd = gguf_sd_loader(path, compute_dtype=torch.bfloat16)

    assert tuple(sd["first.weight"].shape) == (1536, 256)


def _q8_cr_layer(name: str, out_features: int = 4, group_size: int = 256, seed: int = 0) -> dict[str, torch.Tensor]:
    """One Q8_CR layer's tensors as ComfyUI-GGUF writes them."""

    torch.manual_seed(seed)
    payload = quantize_convrot(torch.randn(out_features, group_size), group_size=group_size)
    return {f"{name}.weight": payload.codes, f"{name}.weight_scale": payload.scale}


def test_a_q8_cr_layer_loads_as_an_int8_tensorwise_export(tmp_path):
    """The codes, scale and marker come back exactly as a safetensors export holds them -- and a
    GGML-quantized neighbour in the same file, as the converter's mixed plans write one, stays the
    GGMLTensor it was. (The decode itself is driven end to end by the Krea-2 loader tests.)"""
    from invokeai.backend.quantization.int8_convrot import extract_int8_convrot_markers

    path = tmp_path / "model.gguf"
    tensors = _q8_cr_layer("blocks.0.attn.wq")
    write_gguf(
        path,
        {**tensors, "blocks.0.prenorm.scale": torch.ones(256)},
        quant={"blocks.0.attn.wq.weight": q8_cr_marker()},
        q8_0={"blocks.0.mlp.up.weight": torch.randn(4, 64)},
    )

    sd = gguf_sd_loader(path, compute_dtype=torch.float32, q8_cr="decode")

    weight, scale = sd["blocks.0.attn.wq.weight"], sd["blocks.0.attn.wq.weight_scale"]
    assert not isinstance(weight, GGMLTensor) and torch.equal(weight, tensors["blocks.0.attn.wq.weight"])
    assert not isinstance(scale, GGMLTensor) and torch.equal(scale, tensors["blocks.0.attn.wq.weight_scale"])
    assert extract_int8_convrot_markers(sd) == {"blocks.0.attn.wq": q8_cr_marker()}
    up = sd["blocks.0.mlp.up.weight"]
    assert isinstance(up, GGMLTensor) and up.get_dequantized_tensor().shape == (4, 64)
    assert isinstance(sd["blocks.0.prenorm.scale"], GGMLTensor)


def test_a_q8_cr_file_is_refused_unless_the_caller_decodes_int8(tmp_path):
    """Every loader but the ones wired for int8 would put the codes into a plain Linear."""

    path = tmp_path / "model.gguf"
    write_gguf(path, _q8_cr_layer("blocks.0.attn.wq"), quant={"blocks.0.attn.wq.weight": q8_cr_marker()})

    with pytest.raises(ValueError, match=r"model\.gguf holds 1 ComfyUI-GGUF Q8_CR"):
        gguf_sd_loader(path, compute_dtype=torch.float32)


def test_the_header_read_reports_q8_cr_codes_as_i8_without_raising(tmp_path):
    """Identification asks only for shapes, once per candidate config. A format it cannot decode must not
    raise there -- the factory refuses it from the metadata, with its reason -- and nothing is translated."""

    path = tmp_path / "model.gguf"
    write_gguf(path, _q8_cr_layer("proj"), quant={"proj.weight": {"format": "int4_cr", "backing": "w4a4"}})

    sd = gguf_header_state_dict(path, compute_dtype=torch.float32)

    assert isinstance(sd["proj.weight"], GGMLTensor)
    assert sd["proj.weight"]._ggml_quantization_type is gguf.GGMLQuantizationType.I8
    assert "proj.comfy_quant" not in sd


@pytest.mark.parametrize("weight_rotated", [pytest.param(None, id="missing"), pytest.param(False, id="false")])
def test_a_convrot_layer_not_marked_weight_rotated_is_read_unrotated(tmp_path, weight_rotated):
    """Older converters stored the codes unrotated under `convrot: true`; ComfyUI-GGUF's own loader
    runs those without the rotation, and un-rotating them would scramble the weight."""
    from invokeai.backend.quantization.int8_convrot import extract_int8_convrot_markers

    path = tmp_path / "model.gguf"
    marker = q8_cr_marker(weight_rotated=weight_rotated)
    if weight_rotated is None:
        del marker["weight_rotated"]
    write_gguf(path, _q8_cr_layer("proj"), quant={"proj.weight": marker})

    markers = extract_int8_convrot_markers(gguf_sd_loader(path, compute_dtype=torch.float32, q8_cr="decode"))

    assert markers["proj"]["convrot"] is False
    assert "convrot_groupsize" not in markers["proj"]


def test_an_int8_layers_bias_is_read_as_a_plain_tensor(tmp_path):
    """`Int8ConvrotLinear` casts its bias to the activation dtype, which a GGMLTensor refuses."""

    path = tmp_path / "model.gguf"
    bias = torch.arange(4, dtype=torch.float32)
    write_gguf(path, {**_q8_cr_layer("proj"), "proj.bias": bias}, quant={"proj.weight": q8_cr_marker()})

    sd = gguf_sd_loader(path, compute_dtype=torch.float32, q8_cr="decode")

    assert not isinstance(sd["proj.bias"], GGMLTensor)
    assert torch.equal(sd["proj.bias"], bias)


@pytest.mark.parametrize(
    ("quant", "tensors", "message"),
    [
        pytest.param(
            {"proj.weight": {"format": "int4_cr", "backing": "w4a4"}},
            {},
            "quantization 'int4_cr', which is not supported",
            id="int4",
        ),
        pytest.param({"proj.weight": "{not json"}, {}, "is not valid quantization JSON", id="malformed-json"),
        pytest.param({"proj": q8_cr_marker()}, {}, "names 'proj', not a weight", id="not-a-weight"),
        pytest.param(
            {"proj.weight": q8_cr_marker()},
            {"proj.weight_scale": None},
            "does not hold.*'proj.weight_scale'",
            id="missing-scale",
        ),
        pytest.param(
            {"proj.weight": q8_cr_marker()},
            {"proj.weight_scale": torch.ones(4, 1, dtype=torch.float16)},
            "'proj.weight_scale' is stored as F16, expected F32",
            id="scale-not-f32",
        ),
    ],
)
def test_a_q8_cr_file_the_decode_cannot_read_is_refused(tmp_path, quant, tensors, message):
    path = tmp_path / "model.gguf"
    layer = {**_q8_cr_layer("proj"), **tensors}
    write_gguf(path, {k: v for k, v in layer.items() if v is not None}, quant=quant)

    with pytest.raises(ValueError, match=message):
        gguf_sd_loader(path, compute_dtype=torch.float32, q8_cr="decode")


def test_keep_reads_only_the_tensors_it_selects(tmp_path):
    """A loader that drops part of a file -- the connectors bundled beside a transformer, a tokenizer packed
    into an encoder -- never holds it: what `keep` refuses is not in the state dict at all."""
    path = tmp_path / "model.gguf"
    write_gguf(
        path,
        {"blocks.0.proj.weight": torch.ones(4, 32), "connector.registers": torch.ones(4, 32)},
        q8_0={"connector.attn.weight": torch.randn(4, 64)},
    )

    sd = gguf_sd_loader(path, compute_dtype=torch.float32, keep=lambda name: not name.startswith("connector."))

    assert set(sd) == {"blocks.0.proj.weight"}


def _write_tagged(path, metadata: dict[str, int]) -> None:
    """One F32 tensor, plus `metadata` as uint64 entries."""
    writer = gguf.GGUFWriter(str(path), "cosmos")
    for key, value in metadata.items():
        writer.add_uint64(key, value)
    writer.add_tensor("blocks.0.proj.weight", np.ones((4, 32), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_mixed_gguf(path) -> None:
    """One tensor of each storage the releases use: F32, F16, BF16, Q8_0, a K-quant, and a tensor stored flat
    with its native shape in ``comfy.gguf.orig_shape``."""
    rng = np.random.default_rng(0)
    writer = gguf.GGUFWriter(str(path), "flux")
    writer.add_tensor("norm.weight", rng.standard_normal(64).astype(np.float32))
    writer.add_tensor("gate.weight", rng.standard_normal((8, 64)).astype(np.float16))
    bf16 = gguf.quants.quantize(rng.standard_normal((8, 64)).astype(np.float32), gguf.GGMLQuantizationType.BF16)
    writer.add_tensor("proj.bias", bf16, raw_shape=bf16.shape, raw_dtype=gguf.GGMLQuantizationType.BF16)
    q8 = gguf.quants.quantize(rng.standard_normal((16, 64)).astype(np.float32), gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor("proj.weight", q8, raw_shape=q8.shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    q4k = rng.integers(0, 256, size=(4, 144), dtype=np.uint8)
    writer.add_tensor("attn.weight", q4k, raw_shape=q4k.shape, raw_dtype=gguf.GGMLQuantizationType.Q4_K)
    writer.add_tensor("embedding", rng.standard_normal(96).astype(np.float32))
    writer.add_array("comfy.gguf.orig_shape.embedding", [1, 96])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_with_stale_header_copies(path) -> None:
    """Laid out as vanes430's Anima-2.9B GGUFs are: the header's values repeated as metadata, and not all of them
    right (that file's copied `GGUF.kv_count` is 3, its header's 6)."""
    _write_tagged(path, {"GGUF.version": 3, "GGUF.tensor_count": 7, "GGUF.kv_count": 99})


def test_a_file_repeating_the_header_as_metadata_is_read_by_its_header(tmp_path):
    """gguf-py refuses such a file over the name clash, which left it unidentifiable: it could be neither
    installed nor refused with a reason."""
    path = tmp_path / "model.gguf"
    _write_with_stale_header_copies(path)

    sd = gguf_sd_loader(path, compute_dtype=torch.float32)

    assert set(sd) == {"blocks.0.proj.weight"}
    assert torch.equal(sd["blocks.0.proj.weight"].get_dequantized_tensor(), torch.ones(4, 32))
    with WrappedGGUFReader(path) as reader:
        assert reader.fields["GGUF.tensor_count"].contents() == 1


class _Unreachable:
    def __init__(self, *_args, **_kwargs) -> None:
        raise AssertionError("the file fell back to the tolerant reader")


def test_any_other_file_is_read_by_gguf_py_alone(tmp_path, monkeypatch):
    """The tolerant reader overrides a private gguf-py method, so only the files it exists for may reach it: a
    gguf-py release that changes the method must not break every GGUF."""
    monkeypatch.setattr(loaders, "_HeaderTolerantGGUFReader", _Unreachable)
    path = tmp_path / "model.gguf"
    _write_tagged(path, {"general.parameter_count": 128})

    assert set(gguf_sd_loader(path, compute_dtype=torch.float32)) == {"blocks.0.proj.weight"}
    assert set(gguf_tensor_bytes(path)) == {"blocks.0.proj.weight"}


def test_another_repeated_metadata_key_is_still_refused(tmp_path, monkeypatch):
    """Only a repeated header field reaches the tolerant reader; a file that names one of its own entries twice
    is malformed, and gguf-py's refusal stands."""
    monkeypatch.setattr(loaders, "_HeaderTolerantGGUFReader", _Unreachable)
    path = tmp_path / "model.gguf"
    _write_tagged(path, {"general.aaaa": 1, "general.bbbb": 2})
    path.write_bytes(path.read_bytes().replace(b"general.bbbb", b"general.aaaa"))

    with pytest.raises(KeyError, match="Duplicate general.aaaa"):
        gguf_sd_loader(path, compute_dtype=torch.float32)


def test_tensor_sizes_come_from_the_header_and_release_the_file(tmp_path):
    """Read without a full collection, so nothing may keep the memory map: on Windows a mapped file cannot be
    deleted."""
    path = tmp_path / "model.gguf"
    _write_with_stale_header_copies(path)

    assert gguf_tensor_bytes(path) == {"blocks.0.proj.weight": 4 * 32 * 4}
    path.unlink()


def test_the_header_state_dict_is_the_full_read_without_the_data(tmp_path):
    """Identification probes the header version. Every candidate config reads names, shapes and quantization
    types off it, so each tensor has to look exactly as the full read's does -- packed and logical shape, storage
    dtype, type -- while holding nothing."""
    path = tmp_path / "model.gguf"
    _write_mixed_gguf(path)

    full = gguf_sd_loader(path, compute_dtype=torch.float32)
    header = gguf_header_state_dict(path, compute_dtype=torch.float32)

    assert header.keys() == full.keys()
    for name, read in full.items():
        probed = header[name]
        assert isinstance(probed, GGMLTensor) and isinstance(read, GGMLTensor), name
        assert probed.quantized_data.is_meta, name
        assert probed._ggml_quantization_type == read._ggml_quantization_type, name
        assert probed.tensor_shape == read.tensor_shape, name
        assert probed.quantized_data.shape == read.quantized_data.shape, name
        assert probed.quantized_data.dtype == read.quantized_data.dtype, name
        assert probed.compute_dtype == read.compute_dtype, name
    assert header["embedding"].shape == (1, 96)


def test_identification_reads_only_the_header(tmp_path):
    """The install probes every file once per candidate config; for a GGUF that must not copy its tensors."""
    path = tmp_path / "model.gguf"
    _write_mixed_gguf(path)

    sd = ModelOnDisk(path).load_state_dict()

    assert sd and all(isinstance(value, GGMLTensor) and value.quantized_data.is_meta for value in sd.values())


def _quantized_embedding(qtype: gguf.GGMLQuantizationType, rows: int = 37, width: int = 512) -> GGMLTensor:
    """Random blocks of `qtype` for a (rows, width) embedding; scales may decode to NaN, compared as such."""
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    raw = np.random.default_rng(0).integers(0, 256, size=(rows, width // block_size * type_size), dtype=np.uint8)
    return GGMLTensor(
        torch.from_numpy(raw),
        ggml_quantization_type=qtype,
        tensor_shape=torch.Size((rows, width)),
        compute_dtype=torch.bfloat16,
    )


@pytest.mark.parametrize(
    "qtype",
    [
        gguf.GGMLQuantizationType.Q4_0,
        gguf.GGMLQuantizationType.Q8_0,
        gguf.GGMLQuantizationType.Q4_K,
        gguf.GGMLQuantizationType.Q5_K,
        gguf.GGMLQuantizationType.Q6_K,
        # No torch kernel: decoded by gguf's numpy reader.
        gguf.GGMLQuantizationType.IQ4_NL,
    ],
    ids=lambda qtype: qtype.name,
)
def test_a_quantized_tensor_unpacked_in_row_slices_holds_what_the_whole_decodes_to(monkeypatch, qtype):
    """An embedding is read by index, so it is unpacked at load -- three rows per slice here, over 37, so the last
    slice is short. No decode is wider than a slice, which is what bounds the kernels' temporaries; and as GGML
    blocks never cross a row, the result is bit for bit the whole tensor's decode."""
    monkeypatch.setattr(loaders, "_UNPACK_CHUNK_ELEMENTS", 3 * 512)
    weight = _quantized_embedding(qtype)
    expected = weight.get_dequantized_tensor()
    decoded: list[int] = []
    real_decode = GGMLTensor.get_dequantized_tensor

    def recording(tensor: GGMLTensor) -> torch.Tensor:
        result = real_decode(tensor)
        decoded.append(result.numel())
        return result

    monkeypatch.setattr(GGMLTensor, "get_dequantized_tensor", recording)
    sd = {"embed.weight": weight}

    unpack_ggml_at_load(sd, torch.nn.Embedding(37, 512, device="meta"), (), lambda _bytes: None)

    assert decoded == [3 * 512] * 12 + [512]
    assert type(sd["embed.weight"]) is torch.Tensor
    torch.testing.assert_close(sd["embed.weight"], expected, rtol=0, atol=0, equal_nan=True)


def test_unpacking_a_quantized_tensor_reserves_one_slice_of_temporaries_not_the_whole(monkeypatch):
    """The decode kernels' temporaries run to several times what they decode. Held for a whole 2 GB
    embedding that is ~8 GB beside the model; held for one slice it is bounded by the slice. The packed
    original stays alive until its result lands, so it is in flight too."""
    monkeypatch.setattr(loaders, "_UNPACK_CHUNK_ELEMENTS", 3 * 512)
    weight = _quantized_embedding(gguf.GGMLQuantizationType.Q4_K)
    packed = weight.quantized_data.nbytes
    reserved: list[int] = []

    unpack_ggml_at_load({"embed.weight": weight}, torch.nn.Embedding(37, 512, device="meta"), (), reserved.append)

    result = 37 * 512 * torch.bfloat16.itemsize
    assert reserved == [result + packed + HOST_DEQUANT_PEAK_COPIES * 3 * 512 * torch.bfloat16.itemsize]
