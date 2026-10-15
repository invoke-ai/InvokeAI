"""The header gate shared by the single-file loaders: what a checkpoint may declare, and how strictly
each of the two header transports is read."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.load.model_loaders._single_file_guards import (
    reject_float8_weights,
    reject_formats_declared_in_the_header,
)
from tests.fixtures.quantized_payloads import comfy_quant_marker

ACCEPTED = frozenset({"int8_tensorwise"})
NOTE = "Only int8 files are supported."


def _write(path: Path, tensors: dict[str, torch.Tensor], header: dict[str, str] | None = None) -> Path:
    save_file(tensors, path, metadata=header)
    return path


def test_an_accepted_format_declared_either_way_passes(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "ok.safetensors",
        {"a.weight": torch.zeros(2, 2), "a.comfy_quant": comfy_quant_marker({"format": "int8_tensorwise"})},
        {"_quantization_metadata": json.dumps({"layers": {"b": {"format": "int8_tensorwise"}}})},
    )
    reject_formats_declared_in_the_header(path, "checkpoint", None, ACCEPTED, NOTE)


def test_a_header_entry_without_a_format_declares_nothing(tmp_path: Path) -> None:
    """Per-layer flags alone (`full_precision_matrix_mult`) are a well-formed entry that names no
    scheme; refusing it would turn away real int8 builds."""
    path = _write(
        tmp_path / "flags.safetensors",
        {"a.weight": torch.zeros(2, 2)},
        {"_quantization_metadata": json.dumps({"layers": {"a": {"full_precision_matrix_mult": True}}})},
    )
    reject_formats_declared_in_the_header(path, "checkpoint", None, ACCEPTED, NOTE)


def test_a_foreign_format_in_the_header_is_refused_by_name(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "fp8.safetensors",
        {"a.weight": torch.zeros(2, 2)},
        {"_quantization_metadata": json.dumps({"layers": {"a": {"format": "fp8_scaled"}}})},
    )
    with pytest.raises(ValueError, match="fp8_scaled"):
        reject_formats_declared_in_the_header(path, "checkpoint", None, ACCEPTED, NOTE)


def test_a_marker_without_a_format_is_refused_as_unreadable(tmp_path: Path) -> None:
    """A per-tensor marker exists only to declare a scheme, so one that names none is unreadable."""
    path = _write(
        tmp_path / "marker.safetensors",
        {"a.weight": torch.zeros(2, 2), "a.comfy_quant": comfy_quant_marker({"convrot": True})},
    )
    with pytest.raises(ValueError, match="unreadable"):
        reject_formats_declared_in_the_header(path, "checkpoint", None, ACCEPTED, NOTE)


def test_undeclared_float8_tensors_are_refused_after_the_read(tmp_path: Path) -> None:
    sd = {"a.weight": torch.zeros(2, 2, dtype=torch.float8_e4m3fn), "a.weight_scale": torch.ones(2, 1)}
    with pytest.raises(ValueError, match="float8"):
        reject_float8_weights(sd, "checkpoint", tmp_path / "x.safetensors", NOTE)
