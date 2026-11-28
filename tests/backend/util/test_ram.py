"""Tests for host-memory admission accounting helpers."""

import torch

from invokeai.backend.util.ram import state_dict_cast_bytes


def test_state_dict_cast_bytes_same_dtype_has_no_additional_bytes() -> None:
    state_dict = {"a": torch.zeros(2, dtype=torch.float32)}

    final_bytes, additional_bytes = state_dict_cast_bytes(state_dict, torch.float32, in_place=True)

    assert final_bytes == 8
    assert additional_bytes == 0


def test_state_dict_cast_bytes_in_place_downcast_uses_peak_extra_bytes() -> None:
    state_dict = {"a": torch.zeros(4, dtype=torch.float32), "b": torch.zeros(4, dtype=torch.float32)}

    final_bytes, additional_bytes = state_dict_cast_bytes(state_dict, torch.float16, in_place=True)

    assert final_bytes == 16
    assert additional_bytes == 8


def test_state_dict_cast_bytes_in_place_upcast_includes_cumulative_widening() -> None:
    state_dict = {"a": torch.zeros(4, dtype=torch.float16), "b": torch.zeros(4, dtype=torch.float16)}

    final_bytes, additional_bytes = state_dict_cast_bytes(state_dict, torch.float32, in_place=True)

    assert final_bytes == 32
    assert additional_bytes == 24


def test_state_dict_cast_bytes_out_of_place_keeps_source_and_outputs() -> None:
    state_dict = {"a": torch.zeros(4, dtype=torch.float16), "b": torch.zeros(4, dtype=torch.float16)}

    final_bytes, additional_bytes = state_dict_cast_bytes(state_dict, torch.float32, in_place=False)

    assert final_bytes == 32
    assert additional_bytes == 32


def test_state_dict_cast_bytes_mapped_upcast_does_not_credit_reclaimable_source() -> None:
    state_dict = {f"weight_{index}": torch.zeros(1024 * 1024, dtype=torch.bfloat16) for index in range(4)}

    final_bytes, additional_bytes = state_dict_cast_bytes(
        state_dict, torch.float32, in_place=True, source_tensors_mapped=True
    )

    assert final_bytes == 16 * 1024 * 1024
    assert additional_bytes == final_bytes
