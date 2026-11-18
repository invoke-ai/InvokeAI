from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from invokeai.backend.model_manager.model_on_disk import read_safetensors_header
from invokeai.backend.quantization.fp8_scaled import load_safetensors_checkpoint, safetensors_backend
from tests.fixtures.safetensors_files import record_reads

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
def test_a_file_without_a_float8_weight_keeps_the_mmap_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tensors: dict[str, torch.Tensor]
) -> None:
    path = _write(tmp_path / "plain.safetensors", tensors)

    log = record_reads(monkeypatch)
    sd = load_safetensors_checkpoint(path)

    assert log.backends == ["mmap"]
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


def test_a_torchao_fp8_build_is_read_into_owned_storage(tmp_path: Path) -> None:
    """torchao stores its codes under `._weight_qdata`, not `.weight`. Mapped, its dense tensors pin the file whether
    the codes are then folded or kept."""
    path = _write(
        tmp_path / "torchao.safetensors",
        {
            "block.norm.weight": _NORM,
            "block.proj.bias": _BIAS,
            "block.proj._weight_qdata": torch.ones(64, 64).to(torch.float8_e4m3fn),
            "block.proj._weight_scale": torch.ones(64, 1),
        },
    )

    sd = load_safetensors_checkpoint(path)

    assert not _tensors_share_one_mapping(sd, read_safetensors_header(path), ["block.norm.weight", "block.proj.bias"])


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


def test_tensors_the_selection_leaves_out_are_never_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Through ``pread`` a tensor that is read costs its size in process memory whether or not the loader keeps it, so
    a bundled component is dropped by name, from the header."""
    transformer = load_file(_fp8_checkpoint(tmp_path / "fp8.safetensors"))
    path = _write(tmp_path / "bundle.safetensors", {"vae.decoder.weight": torch.ones(512, 512), **transformer})
    log = record_reads(monkeypatch)

    sd = load_safetensors_checkpoint(
        path, select=lambda names: {k: v for k, v in names.items() if k.startswith("block.")}
    )

    assert set(sd) == set(transformer)
    assert set(log.tensors) == set(transformer)
    assert log.backends == ["pread"], "the file's float8 weight decides the backend, whatever is selected"


def test_the_selection_sees_every_key_and_what_it_returns_names_the_keys_to_read(tmp_path: Path) -> None:
    """A loader hands its own key rename and drop functions over: they return renamed keys, whose values still name
    the file's tensors, and the result is keyed by the file's own names."""
    path = _fp8_checkpoint(tmp_path / "fp8.safetensors")
    offered: list[dict[str, str]] = []

    def select(names: dict[str, str]) -> dict[str, str]:
        offered.append(dict(names))
        return {f"model.{key}": value for key, value in names.items() if key != "block.proj.bias"}

    sd = load_safetensors_checkpoint(path, select=select)

    assert offered == [{key: key for key in load_file(path)}]
    assert sorted(sd) == sorted(key for key in load_file(path) if key != "block.proj.bias")


def test_a_selection_naming_a_tensor_the_file_lacks_is_refused_by_name(tmp_path: Path) -> None:
    path = _fp8_checkpoint(tmp_path / "fp8.safetensors")

    with pytest.raises(ValueError, match=r"fp8\.safetensors has no tensor 'block\.missing'"):
        load_safetensors_checkpoint(path, select=lambda names: {"x": "block.missing"})


def test_an_owned_read_gives_a_file_without_float8_weights_storage_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bf16 tower's cast to bf16 returns the tensor itself, so whatever it reads would stay a view of the file."""
    path = _write(tmp_path / "bf16.safetensors", {"w": torch.ones(4, 4, dtype=torch.bfloat16), "b": _BIAS})
    log = record_reads(monkeypatch)

    sd = load_safetensors_checkpoint(path, owned=True)

    assert log.backends == ["pread"]
    assert not _tensors_share_one_mapping(sd, read_safetensors_header(path), ["w", "b"])
