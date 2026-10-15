"""Dropping the Qwen3-VL visual tower on the safetensors load paths.

Krea-2 and Ideogram 4 condition on the language tower's hidden states; `Qwen3VLModel.forward`
reaches `self.visual` only for `pixel_values`, which neither invocation passes, and Ideogram 4 does
not go through `forward` at all. The tower is therefore never executed, and dropping it takes
0.39-0.77 GiB off the 4B encoder depending on the container.

These tests use a real (tiny) `Qwen3VLModel`, so what disappears is the architecture's actual vision
tower rather than a stand-in. MiniMax H3 does run this tower and loads through
`model_loaders/minimax_h3.py`; nothing here is reachable from it.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.configs.qwen3_vl_encoder import (
    Qwen3VLEncoder_Checkpoint_Config,
    Qwen3VLEncoder_Qwen3VLEncoder_Config,
)
from invokeai.backend.model_manager.load.model_loaders.krea2 import (
    Qwen3VLEncoderCheckpointLoader,
    Qwen3VLEncoderLoader,
)
from invokeai.backend.model_manager.taxonomy import Qwen3VLVariantType, SubModelType
from invokeai.backend.model_manager.util.qwen3_vl import (
    drop_qwen3vl_visual_tower,
    drop_qwen3vl_visual_tower_keys,
)
from invokeai.backend.quantization.int8_convrot import CONVROT_GROUP_SIZE, build_regular_hadamard
from tests.backend.model_manager.load.qwen3vl_gguf_fixture import tiny_qwen3vl_config

# --- the key filter -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "visual.blocks.0.attn.qkv.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "model.visual.pos_embed.weight",
        # A sidecar has to leave with its weight. The sharp case is the int8 branch, which is selected
        # purely by surviving `.comfy_quant` markers: an orphaned visual marker would route the whole
        # checkpoint into int8 handling for a module path the model no longer has.
        "model.visual.blocks.0.attn.qkv.comfy_quant",
        "model.visual.blocks.0.attn.qkv.weight_scale",
        # ComfyUI writes the reversed spelling too, which is why the filter matches on the module
        # path prefix rather than on a list of known suffixes.
        "model.visual.blocks.0.attn.qkv.scale_weight",
    ],
)
def test_visual_tower_keys_are_dropped(key: str) -> None:
    assert drop_qwen3vl_visual_tower_keys({key: "t"}) == {}


@pytest.mark.parametrize(
    "key",
    [
        "model.layers.0.self_attn.q_proj.weight",
        "language_model.layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
        # Not the vision tower: a language-model module that merely contains the substring.
        "model.layers.0.visualization.weight",
    ],
)
def test_language_tower_keys_survive(key: str) -> None:
    assert drop_qwen3vl_visual_tower_keys({key: "t"}) == {key: "t"}


def test_non_string_keys_survive() -> None:
    assert drop_qwen3vl_visual_tower_keys({0: "t"}) == {0: "t"}


def test_a_model_without_a_tower_is_refused_rather_than_given_a_dead_attribute(tmp_path: Path) -> None:
    """`setattr` would happily invent `visual` on anything. On the `from_pretrained` paths nothing
    else would notice a renamed tower: it would load, a dead stub would sit beside it, and the
    encoder would keep every byte this is meant to save."""
    with pytest.raises(RuntimeError, match="no Qwen3-VL visual tower"):
        drop_qwen3vl_visual_tower(torch.nn.Module())


def test_a_model_without_a_tower_is_tolerated_when_the_caller_says_so() -> None:
    """Ideogram 4 builds whatever its folder's config declares, so a tower-less encoder is a shape it
    may legitimately meet rather than a sign the architecture moved."""
    assert drop_qwen3vl_visual_tower(torch.nn.Module(), required=False) is False


# --- the claim the whole change rests on ----------------------------------------------------------


def test_the_conditioning_is_unchanged_without_the_tower() -> None:
    """The justification for dropping the tower is a property of `Qwen3VLModel.forward` that a
    transformers upgrade can quietly take away. This is the only path that can hold both towers, so
    it is the only place the before/after comparison can be made at all.

    Run through the exact call the Krea-2 invocation makes, including the 3-way expanded mRoPE
    position ids.
    """
    model = _tiny_encoder().eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0).unsqueeze(0).expand(3, -1, -1)

    def encode():
        with torch.no_grad():
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            ).hidden_states

    with_tower = encode()
    drop_qwen3vl_visual_tower(model)
    without_tower = encode()

    assert len(with_tower) == len(without_tower)
    for index, (before, after) in enumerate(zip(with_tower, without_tower, strict=True)):
        assert torch.equal(before, after), f"hidden state {index} changed when the tower was dropped"


# --- the load paths -------------------------------------------------------------------------------


def _tiny_encoder(**config_kwargs):
    """A materialized tiny Qwen3-VL, small enough to save and reload in a test."""
    from transformers import Qwen3VLModel

    torch.manual_seed(0)
    return Qwen3VLModel(tiny_qwen3vl_config(**config_kwargs))


def _assert_tower_gone(model) -> None:
    assert isinstance(model.visual, torch.nn.Identity)
    assert list(model.visual.parameters()) == []
    # The language tower is untouched and complete -- the point is to drop the tower, not to lose
    # weights with it.
    assert [name for name, p in model.named_parameters() if p.is_meta] == []
    assert any(name.startswith("language_model.layers.") for name, _ in model.named_parameters())


def _checkpoint_loader(monkeypatch, te_config) -> Qwen3VLEncoderCheckpointLoader:
    loader = object.__new__(Qwen3VLEncoderCheckpointLoader)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    loader._torch_device = torch.device("cpu")
    loader._logger = MagicMock()
    loader._load_te_config = lambda _config: te_config
    loader._apply_fp8_to_nn_module = lambda *a, **k: None
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.krea2.TorchDevice.choose_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.krea2.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.float32,
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.krea2._device_supports_fp8_storage",
        lambda _device, _logger: False,
    )
    return loader


def _write_checkpoint(path: Path, state_dict: dict[str, torch.Tensor]) -> Qwen3VLEncoder_Checkpoint_Config:
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in state_dict.items()}, str(path))
    return Qwen3VLEncoder_Checkpoint_Config.model_construct(
        path=str(path), variant=Qwen3VLVariantType.Qwen3VL_4B, name="tiny"
    )


@pytest.mark.parametrize("prefixed", [False, True], ids=["transformers_layout", "comfyui_layout"])
def test_the_single_file_loader_drops_the_tower(monkeypatch, tmp_path: Path, prefixed: bool) -> None:
    """ComfyUI single-file checkpoints bundle both towers under one ``model.`` prefix; the
    transformers spelling keeps them apart. Both spellings must lose the tower."""
    reference = _tiny_encoder()
    state_dict = dict(reference.state_dict())
    if prefixed:
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    config = _write_checkpoint(tmp_path / "qwen3vl_tiny.safetensors", state_dict)

    model = _checkpoint_loader(monkeypatch, tiny_qwen3vl_config())._load_text_encoder(config)

    _assert_tower_gone(model)
    # The language weights are the ones the file carried, not re-initialised ones.
    torch.testing.assert_close(model.language_model.embed_tokens.weight, reference.language_model.embed_tokens.weight)


def test_the_single_file_loader_reserves_only_for_what_it_keeps(monkeypatch, tmp_path: Path) -> None:
    """Asserted on the bytes the loader asks the cache to free, which is the observable contract --
    the cache evicts other resident models to honour it.

    Note this is the loader's own reservation. `ModelLoader._load_and_cache` separately reserves the
    file's on-disk size before `_load_model` runs, and that one still counts the tower; over-
    reserving is safe and pre-existing, but it means this saving is not the whole story.
    """
    reference = _tiny_encoder()
    state_dict = dict(reference.state_dict())
    config = _write_checkpoint(tmp_path / "qwen3vl_tiny.safetensors", state_dict)
    language_bytes = sum(v.nelement() * v.element_size() for k, v in state_dict.items() if not k.startswith("visual."))
    total_bytes = sum(v.nelement() * v.element_size() for v in state_dict.values())

    loader = _checkpoint_loader(monkeypatch, tiny_qwen3vl_config())
    loader._load_text_encoder(config)

    reserved = [call.args[0] for call in loader._ram_cache.make_room.call_args_list]
    assert reserved, "the loader no longer reserves; this test no longer covers what it claims"
    assert max(reserved) <= language_bytes
    # The fixture has to make the difference visible, or the assertion above proves nothing.
    assert language_bytes < total_bytes


def test_the_int8_branch_drops_the_tower_too(monkeypatch, tmp_path: Path) -> None:
    """The drop sits before the int8/dense split, and it has to stay there. Moved into the dense
    branch, an int8-convrot encoder would refuse to load at all: its visual tensors are gone from the
    state dict, so the tower would stay on the meta device and the completeness sweep would reject
    it -- a hard load failure on a supported container, with the dense tests still green.
    """
    from transformers import Qwen3VLModel

    te_config = tiny_qwen3vl_config()
    # `install_int8_convrot_layers` rotates in groups, so the quantized Linear's input width has to be
    # the group size. Everything else stays at the fixture's dimensions.
    te_config.text_config.hidden_size = CONVROT_GROUP_SIZE
    te_config.vision_config.hidden_size = CONVROT_GROUP_SIZE
    te_config.vision_config.out_hidden_size = CONVROT_GROUP_SIZE

    torch.manual_seed(0)
    state_dict = dict(Qwen3VLModel(te_config).state_dict())

    # One language Linear carries the int8-convrot payload; that is all it takes to select the branch.
    layer = "language_model.layers.0.self_attn.q_proj"
    original = state_dict[f"{layer}.weight"].float()
    hadamard = build_regular_hadamard(CONVROT_GROUP_SIZE)
    rows = original.shape[0]
    rotated = (original.view(rows, -1, CONVROT_GROUP_SIZE) @ hadamard.T).view(rows, -1)
    scale = rotated.abs().amax(dim=1, keepdim=True) / 127.0
    marker = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": CONVROT_GROUP_SIZE})
    state_dict[f"{layer}.weight"] = torch.clamp(torch.round(rotated / scale), -128, 127).to(torch.int8)
    state_dict[f"{layer}.weight_scale"] = scale.to(torch.float32)
    state_dict[f"{layer}.comfy_quant"] = torch.frombuffer(bytearray(marker.encode("utf-8")), dtype=torch.uint8)

    config = _write_checkpoint(tmp_path / "qwen3vl_int8.safetensors", state_dict)

    model = _checkpoint_loader(monkeypatch, te_config)._load_text_encoder(config)

    _assert_tower_gone(model)


def test_the_bundled_pipeline_encoder_drops_the_tower(monkeypatch, tmp_path: Path) -> None:
    """The path a starter model actually takes. Both loader nodes fall back to the pipeline's own
    encoder when no standalone one is wired up, so without this the headline saving never reaches
    the users who install "Krea-2 Turbo" and press Invoke.
    """
    from invokeai.backend.model_manager.configs.main import Main_Diffusers_Krea2_Config
    from invokeai.backend.model_manager.load.model_loaders.krea2 import Krea2DiffusersModel

    encoder = _tiny_encoder()
    config = Main_Diffusers_Krea2_Config.model_construct(path=str(tmp_path), repo_variant=None)
    loader = object.__new__(Krea2DiffusersModel)
    loader.get_hf_load_class = lambda _path, _submodel: SimpleNamespace(from_pretrained=MagicMock(return_value=encoder))
    loader._apply_fp8_layerwise_casting = lambda model, _config, _submodel: model
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.krea2.AutoConfig.from_pretrained",
        lambda *_a, **_k: encoder.config,
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.krea2.TorchDevice.choose_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.krea2.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.float32,
    )

    model = loader._load_model(config, SubModelType.TextEncoder)

    _assert_tower_gone(model)


def test_the_folder_loader_drops_the_tower(tmp_path: Path) -> None:
    """The diffusers-directory path loads through `from_pretrained`, which builds the module tree
    itself, so the tower is read and then freed rather than skipped."""
    _tiny_encoder().save_pretrained(tmp_path)
    config = Qwen3VLEncoder_Qwen3VLEncoder_Config.model_construct(
        path=str(tmp_path), variant=Qwen3VLVariantType.Qwen3VL_4B, name="tiny"
    )

    loader = object.__new__(Qwen3VLEncoderLoader)
    loader._logger = MagicMock()

    model = loader._load_model(config, SubModelType.TextEncoder)

    _assert_tower_gone(model)
