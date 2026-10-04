"""Tests for InvokeAI-format IP-Adapter probing."""

from pathlib import Path

import pytest
import torch

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.ip_adapter import IPAdapter_InvokeAI_SD1_Config


@pytest.mark.parametrize(
    "metadata",
    [
        "InvokeAI/ip_adapter_sd_image_encoder\n",
        "  InvokeAI/ip_adapter_sd_image_encoder  \r\n",
        "InvokeAI/ip_adapter_sd_image_encoder\nignored second line\n",
    ],
)
def test_invokeai_ip_adapter_probe_normalizes_image_encoder_metadata(tmp_path: Path, metadata: str) -> None:
    torch.save({"ip_adapter": {"1.to_k_ip.weight": torch.empty(1, 768)}}, tmp_path / "ip_adapter.bin")
    (tmp_path / "image_encoder.txt").write_text(metadata, encoding="utf-8")

    result = ModelConfigFactory.from_model_on_disk(tmp_path, allow_unknown=False)

    assert isinstance(result.config, IPAdapter_InvokeAI_SD1_Config)
    assert result.config.image_encoder_model_id == "InvokeAI/ip_adapter_sd_image_encoder"


def test_invokeai_ip_adapter_probe_rejects_empty_image_encoder_metadata(tmp_path: Path) -> None:
    torch.save({"ip_adapter": {"1.to_k_ip.weight": torch.empty(1, 768)}}, tmp_path / "ip_adapter.bin")
    (tmp_path / "image_encoder.txt").write_text(" \r\n\t", encoding="utf-8")

    result = ModelConfigFactory.from_model_on_disk(tmp_path, allow_unknown=False)

    assert result.config is None
    assert any("empty image_encoder.txt metadata" in str(error) for error in result.details.values())


def test_invokeai_ip_adapter_probe_applies_encoder_override_without_mutating_it(tmp_path: Path) -> None:
    torch.save({"ip_adapter": {"1.to_k_ip.weight": torch.empty(1, 768)}}, tmp_path / "ip_adapter.bin")
    (tmp_path / "image_encoder.txt").write_text("  \n", encoding="utf-8")
    overrides = {"image_encoder_model_id": "custom/image-encoder"}

    result = ModelConfigFactory.from_model_on_disk(tmp_path, override_fields=overrides, allow_unknown=False)

    assert isinstance(result.config, IPAdapter_InvokeAI_SD1_Config)
    assert result.config.image_encoder_model_id == "custom/image-encoder"
    assert overrides == {"image_encoder_model_id": "custom/image-encoder"}


def test_invokeai_ip_adapter_probe_rejects_empty_encoder_override(tmp_path: Path) -> None:
    torch.save({"ip_adapter": {"1.to_k_ip.weight": torch.empty(1, 768)}}, tmp_path / "ip_adapter.bin")
    (tmp_path / "image_encoder.txt").write_text("custom/image-encoder\n", encoding="utf-8")

    result = ModelConfigFactory.from_model_on_disk(
        tmp_path, override_fields={"image_encoder_model_id": "  "}, allow_unknown=False
    )

    assert result.config is None
    assert any("empty image_encoder_model_id override" in str(error) for error in result.details.values())


def test_invokeai_ip_adapter_probe_rejects_missing_image_encoder_metadata(tmp_path: Path) -> None:
    torch.save({"ip_adapter": {"1.to_k_ip.weight": torch.empty(1, 768)}}, tmp_path / "ip_adapter.bin")

    result = ModelConfigFactory.from_model_on_disk(tmp_path, allow_unknown=False)

    assert result.config is None
    assert any("missing image_encoder.txt metadata file" in str(error) for error in result.details.values())
