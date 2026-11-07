from pathlib import Path
from unittest import mock

import pytest
import safetensors.torch
import torch
from safetensors.torch import load_file, save_file

from invokeai.backend.model_manager.model_on_disk import read_safetensors_header
from invokeai.backend.quantization.fp8_scaled import load_safetensors_checkpoint, safetensors_backend

FLOAT8 = [torch.float8_e4m3fn, torch.float8_e5m2]
_NORM = torch.arange(65, dtype=torch.float32).to(torch.bfloat16)
_BIAS = torch.arange(64, dtype=torch.float32).to(torch.bfloat16) + 1


def _write(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    save_file(tensors, str(path), metadata={"format": "pt"})
    return path


def _fp8_checkpoint(path: Path, dtype: torch.dtype = torch.float8_e4m3fn) -> Path:
    return _write(
        path,
        {
            "block.norm.weight": _NORM,
            "block.proj.bias": _BIAS,
            "block.proj.weight": torch.linspace(-4, 4, 512 * 512).reshape(512, 512).to(dtype),
            "block.proj.weight_scale": torch.tensor(0.5),
        },
    )


def _spy_backends() -> tuple[list[str], mock._patch]:
    backends: list[str] = []

    def spy(path, *args, **kwargs):
        backends.append(kwargs.get("backend", "mmap"))
        return load_file(path, *args, **kwargs)

    return backends, mock.patch.object(safetensors.torch, "load_file", spy)


def _tensors_share_one_mapping(sd: dict[str, torch.Tensor], header: dict, keys: list[str]) -> bool:
    """With mmap each tensor sits at its header offset inside one mapping, so addresses differ exactly as offsets do."""
    first, *rest = keys
    return all(
        sd[first].data_ptr() - sd[key].data_ptr() == header[first]["data_offsets"][0] - header[key]["data_offsets"][0]
        for key in rest
    )


@pytest.mark.parametrize("dtype", FLOAT8, ids=["e4m3", "e5m2"])
def test_a_float8_file_is_read_into_owned_storage_so_the_widened_weights_do_not_pin_it(
    tmp_path: Path, dtype: torch.dtype
) -> None:
    path = _fp8_checkpoint(tmp_path / "fp8.safetensors", dtype)
    header = read_safetensors_header(path)
    survivors = ["block.norm.weight", "block.proj.bias"]

    sd = load_safetensors_checkpoint(path)
    # What a loader does: widen the float8 weight (the source is replaced), keep the dtype-matching tensors.
    sd["block.proj.weight"] = sd["block.proj.weight"].to(torch.float32) * sd.pop("block.proj.weight_scale")

    assert not _tensors_share_one_mapping(sd, header, survivors), "the bf16 survivors are views into the file mapping"
    # The behavioural cost of the pin: a mapped file cannot be deleted on Windows while a view lives.
    path.unlink()
    assert sd["block.norm.weight"].tolist() == list(range(65))
    assert sd["block.proj.bias"].tolist() == [float(i + 1) for i in range(64)]


def test_the_mmap_path_pins_the_file_through_its_survivors(tmp_path: Path) -> None:
    """The same probe on a plain `load_file`: it is the one the float8 path exists to avoid."""
    path = _fp8_checkpoint(tmp_path / "fp8.safetensors")
    header = read_safetensors_header(path)

    sd = load_file(path)
    del sd["block.proj.weight"]

    assert _tensors_share_one_mapping(sd, header, ["block.norm.weight", "block.proj.bias"])


@pytest.mark.parametrize(
    "tensors",
    [
        {"w": torch.ones(4, 4, dtype=torch.bfloat16), "b": torch.zeros(4, dtype=torch.bfloat16)},
        {"w": torch.ones(4, 4, dtype=torch.float16), "b": torch.zeros(4, dtype=torch.float32)},
        {"w": torch.ones(4, 4, dtype=torch.int8), "w_scale": torch.tensor(0.25)},
        # nvfp4: packed codes under `.weight`, block scales in float8 under `.weight_scale`. Nothing is widened.
        {
            "layer.weight": torch.zeros(8, 8, dtype=torch.uint8),
            "layer.weight_scale": torch.ones(8, 1).to(torch.float8_e4m3fn),
            "layer.weight_scale_2": torch.tensor(1.0),
        },
        {"scaled_fp8": torch.empty(0, dtype=torch.float8_e4m3fn), "layer.weight": torch.ones(4, 4)},
    ],
    ids=["bf16", "fp16+fp32", "int8+fp32 scale", "nvfp4", "scaled_fp8 marker only"],
)
def test_a_file_without_a_float8_weight_keeps_the_mmap_load(tmp_path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path = _write(tmp_path / "plain.safetensors", tensors)

    backends, patch = _spy_backends()
    with patch:
        sd = load_safetensors_checkpoint(path)

    assert backends == ["mmap"]
    assert _tensors_share_one_mapping(sd, read_safetensors_header(path), [k for k, v in sd.items() if v.numel()])


def test_a_mixed_nvfp4_build_with_real_float8_weights_is_read_into_owned_storage(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "mixed.safetensors",
        {
            "packed.weight": torch.zeros(8, 8, dtype=torch.uint8),
            "packed.weight_scale": torch.ones(8, 1).to(torch.float8_e4m3fn),
            "dense.weight": torch.ones(8, 8).to(torch.float8_e4m3fn),
            "dense.weight_scale": torch.tensor(1.0),
        },
    )

    assert safetensors_backend(path) == "pread"


@pytest.mark.parametrize("dtype", FLOAT8, ids=["e4m3", "e5m2"])
def test_both_paths_return_the_same_tensors(tmp_path: Path, dtype: torch.dtype) -> None:
    path = _fp8_checkpoint(tmp_path / "fp8.safetensors", dtype)

    owned = load_safetensors_checkpoint(path)
    mapped = load_file(path)

    assert owned.keys() == mapped.keys()
    for key, tensor in mapped.items():
        assert owned[key].dtype == tensor.dtype
        assert owned[key].shape == tensor.shape
        # Compared as bytes: float8 has no equality on CPU.
        assert owned[key].reshape(-1).view(torch.uint8).equal(tensor.reshape(-1).view(torch.uint8))
