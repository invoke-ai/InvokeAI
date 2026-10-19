"""Identification of Comfy-Org's single-file Ideogram 4 transformers.

The two branches are key-for-key and shape-for-shape identical: only the file's `model_type`
metadata says which is which, and a swapped pair produces coherent images that ignore the prompt.

The quantization refusals raise `InvalidMatchError` rather than `NotAMatchError` on purpose, and
the distinction is the whole point of the test below: `NotAMatchError` is "not my kind of model"
and lets the file fall through to `Unknown_Config`, which would register a 9 GiB download as a
model record nothing can load.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Ideogram4_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType

MARKER = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}

_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/ideogram4_fp8_scaled.safetensors",
    "file_size": 1000,
    "name": "ideogram4",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def _state_dict(**overrides: Any) -> dict[str, Any]:
    """The four keys the probe requires, as the released files carry them."""
    sd: dict[str, Any] = {
        "embed_image_indicator.weight": torch.zeros(2, 8, dtype=torch.bfloat16),
        "input_proj.weight": torch.zeros(8, 4, dtype=torch.bfloat16),
        "adaln_proj.weight": torch.zeros(4, 8, dtype=torch.bfloat16),
        "final_layer.linear.weight": torch.zeros(4, 8, dtype=torch.bfloat16),
        "layers.0.attention.qkv.weight": torch.zeros(24, 8, dtype=torch.bfloat16),
    }
    sd.update(overrides)
    return sd


def _int8_layer(path: str, marker: dict | None = MARKER) -> dict[str, Any]:
    """One layer as the `int8_convrot` build stores it: codes, per-output-channel scale, marker."""
    layer: dict[str, Any] = {
        f"{path}.weight": torch.zeros(24, 8, dtype=torch.int8),
        f"{path}.weight_scale": torch.zeros(24, 1, dtype=torch.float32),
    }
    if marker is not None:
        layer[f"{path}.comfy_quant"] = torch.frombuffer(
            bytearray(json.dumps(marker).encode("utf-8")), dtype=torch.uint8
        ).clone()
    return layer


def _mod(state_dict: dict[str, Any], *, metadata: dict[str, str] | None = None, name: str = "ideogram4.safetensors"):
    mod = MagicMock()
    mod.path = Path(f"/fake/{name}")
    mod.load_state_dict.return_value = state_dict
    mod.metadata.return_value = metadata or {}
    return mod


def _mod_on_disk(state_dict: dict[str, Any], tmp_path: Path, *, name: str = "ideogram4.safetensors"):
    """A mod whose `path` is a real file, for the checks that read the safetensors header.

    Marker *contents* are not in the state dict identification sees -- it is on the meta device --
    so the int8 refusal seeks into the file. A `MagicMock` path would make every one of those tests
    pass for the wrong reason.
    """
    path = tmp_path / name
    save_file(state_dict, path)
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = state_dict
    mod.metadata.return_value = {}
    return mod


def _identify(mod: MagicMock, **override_fields: Any) -> Main_Checkpoint_Ideogram4_Config:
    with (
        patch("invokeai.backend.model_manager.configs.main.raise_if_not_file"),
        patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields"),
    ):
        return Main_Checkpoint_Ideogram4_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, **override_fields})


class TestBranchIdentification:
    @pytest.mark.parametrize(
        ("model_type", "expected"),
        [("ideogram4_cond", "conditional"), ("ideogram4_uncond", "unconditional")],
    )
    def test_reads_the_branch_from_file_metadata(self, model_type: str, expected: str) -> None:
        config = _identify(_mod(_state_dict(), metadata={"model_type": model_type}))

        assert config.base is BaseModelType.Ideogram4
        assert config.type is ModelType.Main
        assert config.format is ModelFormat.Checkpoint
        assert config.branch == expected

    def test_metadata_beats_a_misleading_filename(self) -> None:
        config = _identify(
            _mod(
                _state_dict(),
                metadata={"model_type": "ideogram4_cond"},
                name="ideogram4_unconditional_fp8_scaled.safetensors",
            )
        )

        assert config.branch == "conditional"

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("ideogram4_unconditional_fp8_scaled.safetensors", "unconditional"),
            ("Ideogram4_UNCONDITIONAL.safetensors", "unconditional"),
            # The metadata spells it "uncond", and a repack that drops the metadata is exactly the
            # kind of tool that would carry that spelling into the filename.
            ("ideogram4_uncond_fp8.safetensors", "unconditional"),
            ("ideogram4_fp8_scaled.safetensors", "conditional"),
        ],
    )
    def test_falls_back_to_the_filename_when_metadata_was_stripped(self, name: str, expected: str) -> None:
        config = _identify(_mod(_state_dict(), name=name))

        assert config.branch == expected

    def test_a_declaration_this_build_does_not_know_is_refused(self) -> None:
        """Not a fall-through to the filename.

        The loader node treats the recorded branch as authoritative *because* it came from the file.
        A future release tagged `ideogram4_5_cond` would otherwise install as one of these two
        branches and guide against the wrong model, with nothing in the log.
        """
        with pytest.raises(InvalidMatchError, match="ideogram4_5_cond"):
            _identify(_mod(_state_dict(), metadata={"model_type": "ideogram4_5_cond"}))

    def test_an_explicit_override_wins_over_both(self) -> None:
        # The config-level contract, not a user-facing remedy: nothing populates `branch` in
        # `build_common_fields`, so today this is only reachable from code. The error a user sees
        # names renaming and re-installing, which is what actually works.
        config = _identify(
            _mod(_state_dict(), metadata={"model_type": "ideogram4_cond"}),
            branch="unconditional",
        )

        assert config.branch == "unconditional"


class TestRefusals:
    def test_rejects_a_state_dict_from_another_architecture(self) -> None:
        with pytest.raises(NotAMatchError, match="Ideogram 4 transformer"):
            _identify(_mod({"double_blocks.0.img_attn.qkv.weight": torch.zeros(1)}))

    def test_rejects_a_partial_key_set(self) -> None:
        sd = _state_dict()
        del sd["embed_image_indicator.weight"]

        with pytest.raises(NotAMatchError, match="Ideogram 4 transformer"):
            _identify(_mod(sd))

    def test_rejects_the_nvfp4_repack(self) -> None:
        # The released file keeps `input_proj` in fp8 and packs only the block linears, so the
        # refusal cannot key on one tensor's dtype -- and a packed nvfp4 weight is uint8, the same
        # dtype as the `comfy_quant` markers that every repack (including the supported fp8 one)
        # carries. `weight_scale_2` is the signal only nvfp4 writes.
        sd = _state_dict(
            **{
                "layers.0.attention.qkv.weight": torch.zeros(24, 4, dtype=torch.uint8),
                "layers.0.attention.qkv.weight_scale_2": torch.zeros((), dtype=torch.float32),
            }
        )

        with pytest.raises(InvalidMatchError, match="nvfp4"):
            _identify(_mod(sd))

    def test_accepts_the_int8_repack(self, tmp_path: Path) -> None:
        # As released: bf16 input projection, int8 block linears with per-output-channel scales and
        # an `int8_tensorwise` marker each. This build carries no `model_type` metadata at all, so
        # its branch comes from the filename -- which makes that fallback load-bearing here rather
        # than a courtesy.
        sd = _state_dict(**_int8_layer("layers.0.attention.qkv"))

        config = _identify(_mod_on_disk(sd, tmp_path, name="ideogram4_unconditional_int8_convrot.safetensors"))

        assert config.branch == "unconditional"

    @pytest.mark.parametrize(
        ("marker", "case"),
        [
            (None, "no marker at all"),
            ({"format": "int8_dynamic"}, "a scheme this loader cannot build"),
        ],
    )
    def test_rejects_int8_weights_with_no_readable_marker(self, tmp_path: Path, marker: dict | None, case: str) -> None:
        """The loader refuses these, so identification has to as well.

        A rotated weight loaded as if it were not one generates noise, which is why
        `reject_unmarked_int8_weights` exists. Letting the file install anyway would move that
        refusal to the first render -- after a 9 GiB download and three installed dependencies.
        """
        sd = _state_dict(**_int8_layer("layers.0.attention.qkv", marker=marker))

        with pytest.raises(InvalidMatchError, match="no readable 'int8_tensorwise' marker"):
            _identify(_mod_on_disk(sd, tmp_path, name="ideogram4_int8.safetensors"))

    def test_accepts_the_scaled_fp8_release(self) -> None:
        # The other quantized build: fp8 weight plus a per-tensor scale.
        sd = _state_dict(
            **{
                "input_proj.weight": torch.zeros(8, 4, dtype=torch.float8_e4m3fn),
                "input_proj.weight_scale": torch.zeros((), dtype=torch.float32),
            }
        )

        config = _identify(_mod(sd, metadata={"model_type": "ideogram4_cond"}))

        assert config.branch == "conditional"


class TestRefusalReachesTheInstaller:
    """The refusals must not be swallowed by the Unknown fallback.

    `ModelConfigFactory.from_model_on_disk` catches `NotAMatchError` per candidate class and, with
    `allow_unknown_models` on (the default), registers anything nothing matched as `Unknown_Config`.
    An `InvalidMatchError` wins over that: no record is written and the reason is what the installer
    reports.
    """

    def test_an_unsupported_repack_is_not_registered_as_unknown(self, tmp_path: Path) -> None:
        from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
        from invokeai.backend.model_manager.model_on_disk import ModelOnDisk

        sd = _state_dict(
            **{
                "layers.0.attention.qkv.weight": torch.zeros(24, 4, dtype=torch.uint8),
                "layers.0.attention.qkv.weight_scale_2": torch.zeros((), dtype=torch.float32),
            }
        )
        checkpoint = tmp_path / "ideogram4_nvfp4_mixed.safetensors"
        save_file(sd, checkpoint, metadata={"model_type": "ideogram4_cond"})

        result = ModelConfigFactory.from_model_on_disk(ModelOnDisk(checkpoint), {}, allow_unknown=True)

        assert result.config is None, "an unsupported repack must not be registered at all"
        reasons = [str(detail) for detail in result.details.values() if isinstance(detail, InvalidMatchError)]
        assert any("nvfp4" in reason for reason in reasons), reasons

    def test_a_supported_repack_is_registered(self, tmp_path: Path) -> None:
        """The other half of the refusal.

        `InvalidMatchError` suppresses the Unknown fallback for every class, so a detector that
        fired one tensor too wide would take the int8 build down with it — and silently, since the
        symptom is a file that simply stops installing.
        """
        from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
        from invokeai.backend.model_manager.model_on_disk import ModelOnDisk

        sd = _state_dict(**_int8_layer("layers.0.attention.qkv"))
        checkpoint = tmp_path / "ideogram4_int8_convrot.safetensors"
        save_file(sd, checkpoint)

        result = ModelConfigFactory.from_model_on_disk(ModelOnDisk(checkpoint), {}, allow_unknown=True)

        assert result.config is not None
        assert result.config.type is ModelType.Main
        assert getattr(result.config, "branch", None) == "conditional"
