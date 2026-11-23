"""Task-variant identification for MiniMax H3 single-file transformer checkpoints.

The two H3 task transformers are key-for-key (and mostly byte-size) identical, so the probe reads
the variant from the file name. Community hybrid merges (#10354) keep FL2VA's body but take the
AdaLN projections of the last blocks from Ref2VA; those projections carry the reference modality,
so such a file has to register as REF2VA or the reference workflow it was built for stays locked.
"""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.main import Main_Checkpoint_MiniMaxH3_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import MiniMaxH3VariantType


def _probe(tmp_path: Path, filename: str, **overrides: object) -> Main_Checkpoint_MiniMaxH3_Config:
    # The three keys the H3 fingerprint checks; weights are never read, so one element each.
    path = tmp_path / filename
    save_file(
        {
            "audio_patch_proj.weight": torch.zeros(1),
            "video_patch_proj.weight": torch.zeros(1),
            "blocks.0.attn.qkv_proj.weight": torch.zeros(1),
            "adaln_t_table": torch.zeros(5, 3),
        },
        str(path),
    )
    on_disk = ModelOnDisk(path)
    return Main_Checkpoint_MiniMaxH3_Config.from_model_on_disk(
        on_disk,
        {
            "hash": "blake3:fake",
            "file_size": path.stat().st_size,
            "name": path.stem,
            "path": str(path),
            "source": str(path),
            "source_type": "path",
            **overrides,
        },
    )


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("minimax_h3_fl2va_int8_convrot.safetensors", MiniMaxH3VariantType.FL2VA),
        ("minimax_h3_ref2va_int8_convrot.safetensors", MiniMaxH3VariantType.REF2VA),
        ("minimax_h3_hybrid_fl2va_ref2va_b25-49_int8.safetensors", MiniMaxH3VariantType.REF2VA),
        # #10354: a hybrid whose name never says "ref2va" was classified FL2VA, which withheld the
        # reference input stack it was merged for.
        ("ErosMax_v1.0_hybrid.safetensors", MiniMaxH3VariantType.REF2VA),
        ("Singularity-v1.3-Hybrid-layer25to49.safetensors", MiniMaxH3VariantType.REF2VA),
    ],
)
def test_filename_decides_the_task_variant(tmp_path: Path, filename: str, expected: MiniMaxH3VariantType) -> None:
    assert _probe(tmp_path, filename).variant == expected


def test_explicit_variant_override_still_wins(tmp_path: Path) -> None:
    """Renamed files stay user-correctable; the name is a heuristic, not a claim."""
    assert _probe(tmp_path, "my-hybrid-merge.safetensors", variant="fl2va").variant == MiniMaxH3VariantType.FL2VA
