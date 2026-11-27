"""Tests that moving an SDNQTensor to a device moves all of its payloads, not just quantized_data.

The model cache moves parameters with .to(target_device). If only quantized_data moved, a
"GPU-resident" SDNQ parameter would keep its scale / zero_point / svd tensors in system RAM,
forcing a host->device copy of all of them on every dequantization / inference step.
"""

import torch

from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor
from invokeai.backend.quantization.sdnq.utils import SDNQQuantizationType

# Prefer a real cross-device move when CUDA is available; otherwise use the meta device, which still
# exercises the device-realignment logic without needing a GPU.
_TARGET = "cuda" if torch.cuda.is_available() else "meta"


def _make_sdnq_tensor_with_svd() -> SDNQTensor:
    out_features, in_features, num_groups = 4, 8, 2
    return SDNQTensor(
        data=torch.randint(0, 256, (out_features, in_features // 2), dtype=torch.uint8),
        quantization_type=SDNQQuantizationType.UINT4_ASYM,
        tensor_shape=torch.Size([out_features, in_features]),
        compute_dtype=torch.bfloat16,
        scale=torch.rand(out_features, num_groups, 1, dtype=torch.float32),
        zero_point=torch.rand(out_features, num_groups, 1, dtype=torch.float32),
        svd_up=torch.rand(out_features, 2, dtype=torch.float32),
        svd_down=torch.rand(2, in_features, dtype=torch.float32),
        group_size=4,
    )


def test_to_device_moves_scale_zero_point_and_svd():
    t = _make_sdnq_tensor_with_svd()
    assert t.quantized_data.device.type == "cpu"

    moved = t.to(torch.device(_TARGET))

    assert isinstance(moved, SDNQTensor)
    assert moved.quantized_data.device.type == _TARGET
    assert moved._scale.device.type == _TARGET
    assert moved._zero_point is not None and moved._zero_point.device.type == _TARGET
    assert moved._svd_up is not None and moved._svd_up.device.type == _TARGET
    assert moved._svd_down is not None and moved._svd_down.device.type == _TARGET
