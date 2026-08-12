"""Identification and loading must reach the same verdict about an SDNQ folder.

They consult the same directory, so a disagreement is not cosmetic: when identification calls a
markerless export "plain diffusers" and hands it to a diffusers config, the loader then runs
`from_pretrained()` over packed SDNQ weights and either fails or misreads them. This used to be four
near-identical implementations, only one of which looked past the `quantization_config.json` marker.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.main import _is_sdnq_folder as configs_detect
from invokeai.backend.model_manager.configs.t5_encoder import _safetensors_dir_has_sdnq_keys
from invokeai.backend.model_manager.load.model_loaders.flux import _is_sdnq_folder as flux_detect
from invokeai.backend.model_manager.load.model_loaders.vae import _is_sdnq_vae_folder as vae_detect
from invokeai.backend.quantization.sdnq.detection import folder_has_sdnq_keys, is_sdnq_folder

# Every place that asks "is this SDNQ?" — they must not drift apart again.
ALL_DETECTORS = (is_sdnq_folder, configs_detect, flux_detect, vae_detect)


def _write_marker(folder: Path, method: str = "sdnq") -> None:
    (folder / "quantization_config.json").write_text(json.dumps({"quant_method": method}), encoding="utf-8")


def _sdnq_pair(folder: Path, *, sharded: bool) -> None:
    """An SDNQ weight and its scale, optionally split across shards as real exports do."""
    weight = {"blocks.0.attn.to_q.weight": torch.zeros(8, 4, dtype=torch.uint8)}
    scale = {"blocks.0.attn.to_q.scale": torch.zeros(8, 1, dtype=torch.float32)}
    if sharded:
        save_file(weight, str(folder / "model-00001-of-00002.safetensors"))
        save_file(scale, str(folder / "model-00002-of-00002.safetensors"))
    else:
        save_file({**weight, **scale}, str(folder / "model.safetensors"))


@pytest.mark.parametrize("detect", ALL_DETECTORS)
@pytest.mark.parametrize("sharded", [False, True])
def test_every_detector_recognizes_a_markerless_export(tmp_path: Path, detect, sharded: bool) -> None:
    """The reviewer's case: valid SDNQ weights, no `quantization_config.json`."""
    folder = tmp_path / f"markerless-{sharded}-{detect.__name__}"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"_class_name": "FluxTransformer2DModel"}), encoding="utf-8")
    _sdnq_pair(folder, sharded=sharded)

    assert detect(folder) is True


@pytest.mark.parametrize("detect", ALL_DETECTORS)
def test_every_detector_recognizes_the_marker_without_reading_weights(tmp_path: Path, detect) -> None:
    """The marker is definitive and free, so it must not depend on inspectable weights."""
    folder = tmp_path / f"marker-{detect.__name__}"
    folder.mkdir()
    _write_marker(folder)

    assert detect(folder) is True


@pytest.mark.parametrize("detect", ALL_DETECTORS)
def test_no_detector_claims_a_plain_diffusers_folder(tmp_path: Path, detect) -> None:
    """Ordinary weights carry no `.scale` sibling; claiming them would break normal models."""
    folder = tmp_path / f"plain-{detect.__name__}"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKL"}), encoding="utf-8")
    save_file({"encoder.conv_in.weight": torch.zeros(4, 4)}, str(folder / "model.safetensors"))

    assert detect(folder) is False


@pytest.mark.parametrize("detect", ALL_DETECTORS)
def test_a_marker_naming_another_method_still_falls_through_to_the_keys(tmp_path: Path, detect) -> None:
    """A `quant_method` of something else is not evidence *against* SDNQ keys being present."""
    folder = tmp_path / f"othermarker-{detect.__name__}"
    folder.mkdir()
    _write_marker(folder, method="gguf")
    _sdnq_pair(folder, sharded=False)

    assert detect(folder) is True


@pytest.mark.parametrize("detect", ALL_DETECTORS)
def test_an_empty_or_missing_folder_is_not_sdnq(tmp_path: Path, detect) -> None:
    empty = tmp_path / f"empty-{detect.__name__}"
    empty.mkdir()

    assert detect(empty) is False
    assert detect(tmp_path / "does-not-exist") is False


def test_the_key_check_is_exposed_on_its_own_for_the_t5_config(tmp_path: Path) -> None:
    """`t5_encoder` needs the key half without the marker shortcut; it must be the same code."""
    folder = tmp_path / "t5"
    folder.mkdir()
    _sdnq_pair(folder, sharded=True)

    assert folder_has_sdnq_keys(folder) is True
    assert _safetensors_dir_has_sdnq_keys(folder) is True


def test_a_non_safetensors_sdnq_file_is_not_claimed(tmp_path: Path) -> None:
    """`sdnq_sd_loader` reads only safetensors, so calling a `.bin` SDNQ would just move the failure."""
    folder = tmp_path / "binonly"
    folder.mkdir()
    (folder / "model.bin").write_bytes(b"\x00")

    assert is_sdnq_folder(folder) is False
