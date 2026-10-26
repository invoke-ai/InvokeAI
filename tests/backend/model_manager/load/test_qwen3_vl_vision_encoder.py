"""Serving the Qwen3-VL vision tower and processor as their own submodels, for Krea-2's edit mode.

The encoders drop their tower (see `test_qwen3_vl_visual_tower.py`); `SubModelType.VisionEncoder` loads it
back as a separate cache entry, from the same files, reading only the ``visual.*`` tensors. A real tiny
`Qwen3VLModel` supplies both the files and the reference tower every load is compared against.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.main import Main_Diffusers_Krea2_Config
from invokeai.backend.model_manager.configs.qwen3_vl_encoder import (
    Qwen3VLEncoder_Checkpoint_Config,
    Qwen3VLEncoder_GGUF_Config,
    Qwen3VLEncoder_Qwen3VLEncoder_Config,
)
from invokeai.backend.model_manager.load.model_loaders.krea2 import (
    Krea2DiffusersModel,
    Qwen3VLEncoderCheckpointLoader,
    Qwen3VLEncoderGGUFLoader,
    Qwen3VLEncoderLoader,
)
from invokeai.backend.model_manager.load.model_util import calc_model_size_by_data
from invokeai.backend.model_manager.taxonomy import Qwen3VLVariantType, SubModelType
from invokeai.backend.model_manager.util.qwen3_vl import (
    qwen3vl_visual_tower_files,
    qwen3vl_visual_tower_nbytes,
    read_qwen3vl_visual_tower_tensors,
)
from invokeai.backend.quantization.int8_convrot import CONVROT_GROUP_SIZE, Int8ConvrotLinear, build_regular_hadamard
from invokeai.backend.qwen3_vl.qwen3_vl_assets import load_bundled_qwen3_vl_tokenizer
from tests.backend.model_manager.load.qwen3vl_gguf_fixture import tiny_qwen3vl_config

_LOADERS = "invokeai.backend.model_manager.load.model_loaders.krea2"


@pytest.fixture(autouse=True)
def _cpu_float32(monkeypatch):
    """Load at float32 on the CPU, so a loaded tower can be compared to the reference exactly."""
    monkeypatch.setattr(f"{_LOADERS}.TorchDevice.choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(f"{_LOADERS}.TorchDevice.choose_bfloat16_safe_dtype", lambda _device: torch.float32)


def _tiny_encoder(te_config=None):
    from transformers import Qwen3VLModel

    torch.manual_seed(0)
    return Qwen3VLModel(te_config or tiny_qwen3vl_config()).eval()


def _tower_bytes(encoder) -> int:
    return sum(t.nelement() * t.element_size() for t in encoder.visual.state_dict().values())


def _assert_same_tower(tower, reference) -> None:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    assert isinstance(tower, Qwen3VLVisionModel)
    loaded, expected = tower.state_dict(), reference.visual.state_dict()
    assert loaded.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(loaded[key], value), key
    # The cache sizes an entry by what it holds: the tower alone, not the encoder it came from.
    resident = [*reference.visual.parameters(), *reference.visual.buffers()]
    assert calc_model_size_by_data(MagicMock(), tower) == sum(t.nelement() * t.element_size() for t in resident)


def _loader(cls):
    loader = object.__new__(cls)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    loader._logger = MagicMock()
    loader._torch_device = torch.device("cpu")
    return loader


def _single_file(path: Path, state_dict: dict[str, torch.Tensor], te_config) -> Qwen3VLEncoderCheckpointLoader:
    save_file({k: v.contiguous() for k, v in state_dict.items()}, str(path))
    loader = _loader(Qwen3VLEncoderCheckpointLoader)
    loader._load_te_config = lambda _config: te_config
    return loader


def _checkpoint_config(path: Path) -> Qwen3VLEncoder_Checkpoint_Config:
    return Qwen3VLEncoder_Checkpoint_Config.model_construct(
        path=str(path), variant=Qwen3VLVariantType.Qwen3VL_4B, name="tiny"
    )


# --- single file -----------------------------------------------------------------------------------


@pytest.mark.parametrize("prefixed", [False, True], ids=["transformers_layout", "comfyui_layout"])
def test_the_single_file_serves_the_tower_it_carries(tmp_path: Path, prefixed: bool) -> None:
    reference = _tiny_encoder()
    state_dict = dict(reference.state_dict())
    if prefixed:
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    path = tmp_path / "qwen3vl_tiny.safetensors"
    loader = _single_file(path, state_dict, tiny_qwen3vl_config())

    tower = loader._load_model(_checkpoint_config(path), SubModelType.VisionEncoder)

    _assert_same_tower(tower, reference)


def test_only_the_tower_tensors_are_read(tmp_path: Path) -> None:
    reference = _tiny_encoder()
    state_dict = {f"model.{k}": v for k, v in reference.state_dict().items()}
    path = tmp_path / "qwen3vl_tiny.safetensors"
    save_file({k: v.contiguous() for k, v in state_dict.items()}, str(path))

    tensors = read_qwen3vl_visual_tower_tensors([path])

    assert set(tensors) == {k for k in state_dict if k.startswith("model.visual.")}


def test_the_cache_reserves_the_tower_not_the_file(tmp_path: Path) -> None:
    """`_load_and_cache` reserves `get_size_fs` before the load. Sized by the file, a tower load would evict
    ten times what it needs on the real 4B (0.77 of 8.27 GiB)."""
    reference = _tiny_encoder()
    path = tmp_path / "qwen3vl_tiny.safetensors"
    loader = _single_file(path, dict(reference.state_dict()), tiny_qwen3vl_config())
    config = _checkpoint_config(path)

    assert loader.get_size_fs(config, path, SubModelType.VisionEncoder) == _tower_bytes(reference)
    assert loader.get_size_fs(config, path, SubModelType.Processor) == 0
    assert loader.get_size_fs(config, path, SubModelType.TextEncoder) == path.stat().st_size > _tower_bytes(reference)


def test_an_int8_tower_stays_int8(tmp_path: Path) -> None:
    """ComfyUI's int8_tensorwise repacks of the 4B quantize the tower's Linears as well (105 of them). Those
    have to come back as `Int8ConvrotLinear` -- cast to the compute dtype instead, the raw codes would load
    clean and see noise."""
    te_config = tiny_qwen3vl_config()
    te_config.text_config.hidden_size = CONVROT_GROUP_SIZE
    te_config.vision_config.hidden_size = CONVROT_GROUP_SIZE
    te_config.vision_config.out_hidden_size = CONVROT_GROUP_SIZE
    reference = _tiny_encoder(te_config)
    state_dict = {f"model.{k}": v for k, v in reference.state_dict().items()}

    layer = "model.visual.blocks.0.attn.proj"
    original = state_dict[f"{layer}.weight"].float()
    hadamard = build_regular_hadamard(CONVROT_GROUP_SIZE)
    rows = original.shape[0]
    rotated = (original.view(rows, -1, CONVROT_GROUP_SIZE) @ hadamard.T).view(rows, -1)
    scale = rotated.abs().amax(dim=1, keepdim=True) / 127.0
    marker = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": CONVROT_GROUP_SIZE})
    state_dict[f"{layer}.weight"] = torch.clamp(torch.round(rotated / scale), -128, 127).to(torch.int8)
    state_dict[f"{layer}.weight_scale"] = scale.to(torch.float32)
    state_dict[f"{layer}.comfy_quant"] = torch.frombuffer(bytearray(marker.encode("utf-8")), dtype=torch.uint8)
    path = tmp_path / "qwen3vl_int8.safetensors"
    loader = _single_file(path, state_dict, te_config)

    tower = loader._load_model(_checkpoint_config(path), SubModelType.VisionEncoder)

    assert isinstance(tower.blocks[0].attn.proj, Int8ConvrotLinear)
    x = torch.randn(4, CONVROT_GROUP_SIZE)
    torch.testing.assert_close(tower.blocks[0].attn.proj(x), reference.visual.blocks[0].attn.proj(x), atol=0.05, rtol=0)
    assert torch.equal(tower.blocks[1].attn.proj.weight, reference.visual.blocks[1].attn.proj.weight)


def test_a_file_without_a_tower_is_refused(tmp_path: Path) -> None:
    reference = _tiny_encoder()
    language_only = {k: v for k, v in reference.state_dict().items() if not k.startswith("visual.")}
    path = tmp_path / "language_only.safetensors"
    loader = _single_file(path, language_only, tiny_qwen3vl_config())

    with pytest.raises(RuntimeError, match="no Qwen3-VL vision tower"):
        loader._load_model(_checkpoint_config(path), SubModelType.VisionEncoder)


def test_a_gguf_encoder_refuses_the_tower_by_name(tmp_path: Path) -> None:
    """llama.cpp keeps the tower in a separate mmproj file, which identification rejects."""
    config = Qwen3VLEncoder_GGUF_Config.model_construct(
        path=str(tmp_path / "enc.gguf"), variant=Qwen3VLVariantType.Qwen3VL_4B, name="tiny-gguf"
    )

    with pytest.raises(ValueError, match="'tiny-gguf' has no vision tower"):
        _loader(Qwen3VLEncoderGGUFLoader)._load_model(config, SubModelType.VisionEncoder)


def test_the_single_file_processor_wraps_the_bundled_tokenizer(tmp_path: Path) -> None:
    from transformers import Qwen3VLProcessor

    processor = _loader(Qwen3VLEncoderCheckpointLoader)._load_model(
        _checkpoint_config(tmp_path / "unused.safetensors"), SubModelType.Processor
    )

    assert isinstance(processor, Qwen3VLProcessor)
    assert processor.tokenizer is load_bundled_qwen3_vl_tokenizer()
    assert processor.image_processor.patch_size == 16


# --- folders ---------------------------------------------------------------------------------------


def test_the_folder_encoder_serves_its_tower_and_processor(tmp_path: Path) -> None:
    from transformers import Qwen3VLProcessor

    reference = _tiny_encoder()
    reference.save_pretrained(tmp_path)
    load_bundled_qwen3_vl_tokenizer().save_pretrained(tmp_path)
    config = Qwen3VLEncoder_Qwen3VLEncoder_Config.model_construct(
        path=str(tmp_path), variant=Qwen3VLVariantType.Qwen3VL_4B, name="tiny"
    )
    loader = _loader(Qwen3VLEncoderLoader)

    _assert_same_tower(loader._load_model(config, SubModelType.VisionEncoder), reference)
    assert isinstance(loader._load_model(config, SubModelType.Processor), Qwen3VLProcessor)
    assert loader.get_size_fs(config, tmp_path, SubModelType.VisionEncoder) == _tower_bytes(reference)


def test_the_krea2_pipeline_serves_its_encoders_tower_and_processor(tmp_path: Path) -> None:
    """The path the "Krea-2 Turbo" starter model takes: the tower lives in ``text_encoder/``, which
    model_index.json names, but has no folder or index entry of its own."""
    from transformers import Qwen3VLProcessor

    reference = _tiny_encoder()
    reference.save_pretrained(tmp_path / "text_encoder")
    load_bundled_qwen3_vl_tokenizer().save_pretrained(tmp_path / "tokenizer")
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    config = Main_Diffusers_Krea2_Config.model_construct(path=str(tmp_path), repo_variant=None)
    loader = _loader(Krea2DiffusersModel)

    _assert_same_tower(loader._load_model(config, SubModelType.VisionEncoder), reference)
    processor = loader._load_model(config, SubModelType.Processor)
    assert isinstance(processor, Qwen3VLProcessor)
    assert processor.tokenizer("a cat").input_ids == load_bundled_qwen3_vl_tokenizer()("a cat").input_ids
    assert loader.get_size_fs(config, tmp_path, SubModelType.VisionEncoder) == _tower_bytes(reference)
    assert loader.get_size_fs(config, tmp_path, SubModelType.Processor) == 0


def test_a_sharded_folder_reads_only_the_shards_holding_the_tower(tmp_path: Path) -> None:
    reference = _tiny_encoder()
    reference.save_pretrained(tmp_path, max_shard_size="100KB")
    weight_map = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    tower_shards = {tmp_path / name for key, name in weight_map.items() if key.startswith("visual.")}

    files = qwen3vl_visual_tower_files(tmp_path)

    assert set(files) == tower_shards
    # The fixture has to put the language tower in shards of its own, or this shows nothing.
    assert tower_shards < {tmp_path / name for name in weight_map.values()}
    assert qwen3vl_visual_tower_nbytes(files) == _tower_bytes(reference)


def test_a_loaded_tower_lent_to_a_towerless_encoder_matches_the_intact_model(tmp_path: Path) -> None:
    """Pins what a state-dict comparison cannot: the tower is built under `init_empty_weights`, so its
    non-persistent buffers (the rotary `inv_freq`) come from construction, not from the file."""
    import copy

    from invokeai.backend.model_manager.util.qwen3_vl import attach_qwen3vl_visual_tower, drop_qwen3vl_visual_tower
    from invokeai.backend.qwen3_vl.qwen3_vl_assets import build_qwen3_vl_processor

    te_config = tiny_qwen3vl_config()
    # The tiny vocabulary cannot hold the release's image-pad id; any in-vocabulary id serves.
    te_config.image_token_id = 5
    reference = _tiny_encoder(te_config)
    towerless = copy.deepcopy(reference)
    drop_qwen3vl_visual_tower(towerless)
    path = tmp_path / "qwen3vl_tiny.safetensors"
    tower = _single_file(path, dict(reference.state_dict()), te_config)._load_model(
        _checkpoint_config(path), SubModelType.VisionEncoder
    )

    image = build_qwen3_vl_processor(load_bundled_qwen3_vl_tokenizer()).image_processor(
        images=[torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(0))],
        return_tensors="pt",
        do_rescale=False,
    )
    pads = int(image["image_grid_thw"].prod()) // 4
    input_ids = torch.tensor([[1, 2, *[5] * pads, 3, 4]])
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
        "mm_token_type_ids": (input_ids == 5).long(),
        "pixel_values": image["pixel_values"],
        "image_grid_thw": image["image_grid_thw"],
    }

    with torch.no_grad():
        expected = reference(**inputs, output_hidden_states=True, use_cache=False).hidden_states
        with attach_qwen3vl_visual_tower(towerless, tower):
            actual = towerless(**inputs, output_hidden_states=True, use_cache=False).hidden_states

    assert len(actual) == len(expected)
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        assert torch.equal(got, want), f"hidden state {index} differs"
