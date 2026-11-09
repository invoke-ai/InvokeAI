"""Loading a llama.cpp Qwen3-VL ``mmproj`` GGUF as the HuggingFace vision tower.

A tiny real ``Qwen3VLVisionModel`` is the reference. Its weights are written to a real GGUF under the tensor names
llama.cpp's converter gives them, spelled out here rather than derived from the loader's mapping, and the loader
must hand the same weights back in the HuggingFace modules. Three blocks with deepstack mergers on blocks 1 and 2
make the block-to-position rename observable, and the patch embedding's two temporal halves are told apart.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from invokeai.backend.model_manager.configs.qwen3_vl_encoder import Qwen3VLVisionTower_GGUF_Config
from invokeai.backend.model_manager.load.model_loaders.qwen3_vl_vision_tower import Qwen3VLVisionTowerGGUFLoader
from invokeai.backend.model_manager.taxonomy import Qwen3VLVariantType
from invokeai.backend.model_manager.util.qwen3_vl_mmproj import mmproj_to_vision_tower_state_dict
from tests.backend.model_manager.load.qwen3vl_gguf_fixture import tiny_qwen3vl_config

_MODULE = "invokeai.backend.model_manager.load.model_loaders.qwen3_vl_vision_tower"
_DEEPSTACK_BLOCKS = [1, 2]


@pytest.fixture
def tiny_config(monkeypatch: pytest.MonkeyPatch):
    from transformers import Qwen3VLConfig

    config = tiny_qwen3vl_config()
    config.vision_config.depth = 3
    config.vision_config.deepstack_visual_indexes = _DEEPSTACK_BLOCKS
    config = Qwen3VLConfig.from_dict(config.to_dict())
    monkeypatch.setattr(f"{_MODULE}.load_bundled_qwen3_vl_config_dict", lambda _variant: config.to_dict())
    monkeypatch.setattr(f"{_MODULE}.TorchDevice.choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(f"{_MODULE}.TorchDevice.choose_bfloat16_safe_dtype", lambda _device: torch.float32)
    return config


def _reference_tower(config) -> torch.nn.Module:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    torch.manual_seed(0)
    tower = Qwen3VLVisionModel(config.vision_config).eval()
    with torch.no_grad():
        for parameter in tower.parameters():
            # Distinct values everywhere, in the range float16 holds exactly enough to compare after a round trip.
            parameter.copy_(torch.randn_like(parameter).half().float())
    return tower


def _llamacpp_tensors(tower: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The tower's weights under llama.cpp's names, in float16, written out by hand."""
    sd = tower.state_dict()
    out = {
        "v.patch_embd.weight": sd["patch_embed.proj.weight"][:, :, 0],
        "v.patch_embd.weight.1": sd["patch_embed.proj.weight"][:, :, 1],
        "v.patch_embd.bias": sd["patch_embed.proj.bias"],
        "v.position_embd.weight": sd["pos_embed.weight"],
        "v.post_ln.weight": sd["merger.norm.weight"],
        "v.post_ln.bias": sd["merger.norm.bias"],
    }
    for parameter in ("weight", "bias"):
        out[f"mm.0.{parameter}"] = sd[f"merger.linear_fc1.{parameter}"]
        out[f"mm.2.{parameter}"] = sd[f"merger.linear_fc2.{parameter}"]
        for block in range(tower.config.depth):
            for theirs, ours in (
                ("attn_qkv", "attn.qkv"),
                ("attn_out", "attn.proj"),
                ("ffn_up", "mlp.linear_fc1"),
                ("ffn_down", "mlp.linear_fc2"),
                ("ln1", "norm1"),
                ("ln2", "norm2"),
            ):
                out[f"v.blk.{block}.{theirs}.{parameter}"] = sd[f"blocks.{block}.{ours}.{parameter}"]
        for position, block in enumerate(_DEEPSTACK_BLOCKS):
            for theirs, ours in (("norm", "norm"), ("fc1", "linear_fc1"), ("fc2", "linear_fc2")):
                out[f"v.deepstack.{block}.{theirs}.{parameter}"] = sd[
                    f"deepstack_merger_list.{position}.{ours}.{parameter}"
                ]
    return out


def _write_mmproj(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    import gguf

    writer = gguf.GGUFWriter(str(path), "clip")
    writer.add_string("general.type", "mmproj")
    writer.add_string("clip.projector_type", "qwen3vl_merger")
    for name, tensor in tensors.items():
        # Weights are F16 and biases and norms F32 in the released files; both read back through the same path.
        dtype = np.float16 if name.endswith(".weight") and tensor.ndim > 1 else np.float32
        writer.add_tensor(name, tensor.contiguous().numpy().astype(dtype))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _load(path: Path) -> torch.nn.Module:
    loader = object.__new__(Qwen3VLVisionTowerGGUFLoader)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    loader._logger = MagicMock()
    config = Qwen3VLVisionTower_GGUF_Config.model_construct(
        path=str(path), variant=Qwen3VLVariantType.Qwen3VL_8B, name="tiny-mmproj"
    )
    return loader._load_model(config, None)


def test_the_mmproj_loads_into_the_huggingface_tower_weight_for_weight(tiny_config, tmp_path: Path) -> None:
    reference = _reference_tower(tiny_config)
    path = _write_mmproj(tmp_path / "mmproj-tiny-F16.gguf", _llamacpp_tensors(reference))

    tower = _load(path)

    assert type(tower) is type(reference)
    loaded, expected = tower.state_dict(), reference.state_dict()
    assert loaded.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(loaded[name], value), name
    assert not any(parameter.is_meta for parameter in tower.parameters())


def test_the_loaded_tower_computes_what_the_reference_does(tiny_config, tmp_path: Path) -> None:
    """The weights land where the module reads them, not only under the right names: the fused QKV and the
    patch embedding's halves would pass a name check and still scramble the output."""
    reference = _reference_tower(tiny_config)
    tower = _load(_write_mmproj(tmp_path / "mmproj-tiny-F16.gguf", _llamacpp_tensors(reference)))

    vision = tiny_config.vision_config
    pixels = torch.randn(16, vision.in_channels * vision.temporal_patch_size * vision.patch_size**2)
    grid = torch.tensor([[1, 4, 4]])
    with torch.no_grad():
        expected = reference(pixels, grid_thw=grid)
        actual = tower(pixels, grid_thw=grid)

    assert torch.equal(actual.pooler_output, expected.pooler_output)
    assert all(torch.equal(a, e) for a, e in zip(actual.deepstack_features, expected.deepstack_features, strict=True))


@pytest.mark.parametrize("cpu_only, expected", [(True, torch.device("cpu")), (None, None), (False, None)])
def test_cpu_only_applies_to_the_whole_tower_model(tmp_path: Path, cpu_only: bool | None, expected) -> None:
    """The tower attaches to the encoder and runs on its device, so a CPU-only encoder needs a CPU-only tower; the
    execution device is read off the whole model, which has no submodel type."""
    config = Qwen3VLVisionTower_GGUF_Config.model_construct(
        path=str(tmp_path / "mmproj.gguf"), variant=Qwen3VLVariantType.Qwen3VL_8B, name="tiny", cpu_only=cpu_only
    )

    device = object.__new__(Qwen3VLVisionTowerGGUFLoader)._get_execution_device(config, None)

    assert device == expected


def test_a_tensor_the_tower_has_no_place_for_is_refused(tiny_config, tmp_path: Path) -> None:
    reference = _reference_tower(tiny_config)
    tensors = _llamacpp_tensors(reference) | {"v.blk.0.attn_q_norm.weight": torch.zeros(4)}

    with pytest.raises(ValueError, match="attn_q_norm"):
        mmproj_to_vision_tower_state_dict(tensors, _DEEPSTACK_BLOCKS)


def test_a_deepstack_merger_on_another_block_is_refused(tiny_config) -> None:
    """A file converted for a tower with other deepstack blocks would otherwise fill the mergers in the wrong order."""
    reference = _reference_tower(tiny_config)

    with pytest.raises(ValueError, match="deepstack merger on block 2"):
        mmproj_to_vision_tower_state_dict(_llamacpp_tensors(reference), [0, 1])


def test_a_file_missing_a_block_fails_the_load_naming_the_tensors(tiny_config, tmp_path: Path) -> None:
    reference = _reference_tower(tiny_config)
    tensors = {k: v for k, v in _llamacpp_tensors(reference).items() if not k.startswith("v.blk.1.")}
    path = _write_mmproj(tmp_path / "mmproj-partial.gguf", tensors)

    with pytest.raises(RuntimeError, match="blocks.1"):
        _load(path)
