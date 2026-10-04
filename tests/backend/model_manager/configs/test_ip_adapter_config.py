"""Tests for InvokeAI-format IP-Adapter probing."""

from pathlib import Path

import pytest
import torch

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.ip_adapter import IPAdapter_InvokeAI_SD1_Config


@pytest.mark.parametrize(
    "metadata",
    ["InvokeAI/ip_adapter_sd_image_encoder\n", "  InvokeAI/ip_adapter_sd_image_encoder  \r\n"],
)
def test_invokeai_ip_adapter_probe_normalizes_image_encoder_metadata(tmp_path: Path, metadata: str) -> None:
    torch.save({"ip_adapter": {"1.to_k_ip.weight": torch.empty(1, 768)}}, tmp_path / "ip_adapter.bin")
    (tmp_path / "image_encoder.txt").write_text(metadata, encoding="utf-8")

    result = ModelConfigFactory.from_model_on_disk(tmp_path, allow_unknown=False)

    assert isinstance(result.config, IPAdapter_InvokeAI_SD1_Config)
    assert result.config.image_encoder_model_id == "InvokeAI/ip_adapter_sd_image_encoder"
