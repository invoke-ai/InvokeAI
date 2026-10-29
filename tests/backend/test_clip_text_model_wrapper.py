"""CLIP text encoder 1 under transformers >=5.6, which removed `CLIPTextModel`'s `text_model` wrapper.

LoRA formats and SDNQ checkpoints still address its modules under `text_model.`, and `CLIPTextModelWithProjection`
(SDXL text encoder 2) still has the wrapper, so every site that walks module paths has to handle both layouts.
These run against real tiny models so a future layout change fails here rather than at generation time.
"""

import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

from invokeai.backend.model_manager.load.model_loaders import flux
from invokeai.backend.model_patcher import ModelPatcher
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor
from invokeai.backend.quantization.sdnq.utils import SDNQQuantizationType

NUM_LAYERS = 3


def _config() -> CLIPTextConfig:
    return CLIPTextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=2,
        vocab_size=64,
        projection_dim=8,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
    )


@pytest.mark.parametrize("model_class", [CLIPTextModel, CLIPTextModelWithProjection])
def test_clip_skip_drops_and_restores_the_last_layers(model_class: type[torch.nn.Module]) -> None:
    model = model_class(_config()).eval()
    ids = torch.tensor([[0, 5, 6, 2]])
    # transformers records hidden states with hooks installed on the layers present at the first forward, so a first
    # forward under clip skip would leave the restored layers unrecorded (#9722, also on 5.5). Warm up unskipped.
    assert len(model(ids, output_hidden_states=True).hidden_states) == NUM_LAYERS + 1

    with ModelPatcher.apply_clip_skip(model, 2):
        assert len(model(ids, output_hidden_states=True).hidden_states) == NUM_LAYERS - 2 + 1

    assert len(model(ids, output_hidden_states=True).hidden_states) == NUM_LAYERS + 1


@pytest.mark.parametrize(
    ("model_class", "fc1"),
    [
        (CLIPTextModel, lambda m: m.encoder.layers[0].mlp.fc1),
        (CLIPTextModelWithProjection, lambda m: m.text_model.encoder.layers[0].mlp.fc1),
    ],
)
@pytest.mark.parametrize(
    "layer_key", ["text_model_encoder_layers_0_mlp_fc1", "text_model.encoder.layers.0.mlp.fc1"], ids=["kohya", "dotted"]
)
def test_text_model_addressed_lora_patches_the_clip_encoder(model_class, fc1, layer_key: str) -> None:
    model = model_class(_config()).eval()
    target = fc1(model)
    rank = 2
    lora = ModelPatchRaw(
        {
            layer_key: LoRALayer.from_state_dict_values(
                values={
                    "lora_down.weight": torch.ones((rank, target.in_features)),
                    "lora_up.weight": torch.ones((target.out_features, rank)),
                }
            )
        }
    )
    original = target.weight.detach().clone()

    with LayerPatcher.apply_smart_model_patches(
        model=model, patches=[(lora, 1.0)], prefix="", dtype=torch.float32, force_direct_patching=True
    ):
        torch.testing.assert_close(target.weight, original + rank)

    torch.testing.assert_close(target.weight, original)


def test_sdnq_clip_loads_a_text_model_prefixed_checkpoint(tmp_path, monkeypatch) -> None:
    reference = CLIPTextModel(_config()).eval()
    reference.config.save_pretrained(tmp_path)
    checkpoint = {f"text_model.{k}": v.clone() for k, v in reference.state_dict().items()}
    monkeypatch.setattr(flux, "sdnq_sd_loader", lambda path, compute_dtype: dict(checkpoint))

    loaded = object.__new__(flux.FluxSDNQDiffusersModel)._load_sdnq_clip(tmp_path)

    for key, value in reference.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_sdnq_clip_dequantizes_the_token_embedding(tmp_path, monkeypatch) -> None:
    reference = CLIPTextModel(_config()).eval()
    reference.config.save_pretrained(tmp_path)
    checkpoint = {f"text_model.{k}": v.clone() for k, v in reference.state_dict().items()}
    shape = reference.embeddings.token_embedding.weight.shape
    checkpoint["text_model.embeddings.token_embedding.weight"] = SDNQTensor(
        data=torch.full(shape, 10, dtype=torch.int8),
        quantization_type=SDNQQuantizationType.INT8_SYM,
        tensor_shape=shape,
        compute_dtype=torch.float32,
        scale=torch.full((shape[0], 1), 0.1),
    )
    monkeypatch.setattr(flux, "sdnq_sd_loader", lambda path, compute_dtype: dict(checkpoint))

    loaded = object.__new__(flux.FluxSDNQDiffusersModel)._load_sdnq_clip(tmp_path)

    weight = loaded.embeddings.token_embedding.weight
    assert not isinstance(weight, SDNQTensor)
    torch.testing.assert_close(weight, torch.full(shape, 1.0))
