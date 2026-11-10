"""An ERNIE-Image LoRA on a GGUF build: the sidecar has to compute what merging the LoRA into dense weights does.

GGUF Linears keep their weights packed, so the denoise node forces sidecar patching for them. The reference is the
same tiny transformer dense (the fixture returns what every packed tensor dequantizes to), patched directly. A
LoRA layer that resolved to no module, a wrong scale, or a sidecar residual applied in the wrong space would all
move the result by far more than the GGUF kernels' fp16 rounding.
"""

import gguf
import pytest
import torch

from invokeai.backend.model_manager.configs.main import Main_GGUF_ErnieImage_Config
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders import ernie_image
from invokeai.backend.model_manager.load.model_loaders.ernie_image import ErnieImageGGUFModel
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.ernie_image_lora_constants import ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.lora_conversions.ernie_image_lora_conversion_utils import (
    lora_model_from_ernie_image_state_dict,
)
from tests.backend.model_manager.load.ernie_image_gguf_fixture import TINY_CONFIG, write_ernie_image_gguf
from tests.fixtures.loader_seams import Seam, prepare

SEAM = Seam(
    loader=ErnieImageGGUFModel, module=ernie_image, entry="_load_model", patches_device=True, casts_fp8_storage=False
)

# Every module a published ERNIE-Image LoRA reaches: the seven per block (ai-toolkit) and the top-level ones a PEFT
# extraction adds. Packed and BF16 stem Linears alike.
_TARGETS = (
    "layers.0.self_attention.to_q",
    "layers.0.self_attention.to_k",
    "layers.0.self_attention.to_v",
    "layers.0.self_attention.to_out.0",
    "layers.0.mlp.gate_proj",
    "layers.0.mlp.up_proj",
    "layers.0.mlp.linear_fc2",
    "adaLN_modulation.1",
    "final_norm.linear",
    "final_linear",
    "text_proj",
    "time_embedding.linear_1",
    "time_embedding.linear_2",
)


def _forward(model: torch.nn.Module) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        return model(
            hidden_states=torch.randn(1, TINY_CONFIG["in_channels"], 2, 2, generator=generator),
            timestep=torch.tensor([500.0]),
            text_bth=torch.randn(1, 3, TINY_CONFIG["text_in_dim"], generator=generator),
            text_lens=torch.tensor([3]),
            return_dict=False,
        )[0]


def _random_lora(reference: torch.nn.Module, rank: int = 4) -> dict[str, torch.Tensor]:
    """An ai-toolkit-style state dict (`diffusion_model.` prefix, PEFT pairs) over every target module."""
    generator = torch.Generator().manual_seed(2)
    state_dict: dict[str, torch.Tensor] = {}
    for path in _TARGETS:
        module = reference.get_submodule(path)
        assert isinstance(module, torch.nn.Linear)
        state_dict[f"diffusion_model.{path}.lora_A.weight"] = torch.randn(rank, module.in_features, generator=generator)
        state_dict[f"diffusion_model.{path}.lora_B.weight"] = (
            torch.randn(module.out_features, rank, generator=generator) * 0.02
        )
    return state_dict


@pytest.fixture
def gguf_and_dense(monkeypatch, tmp_path) -> tuple[torch.nn.Module, torch.nn.Module]:
    from diffusers import ErnieImageTransformer2DModel

    path = tmp_path / "ernie-image-turbo-Q8_0.gguf"
    meant = write_ernie_image_gguf(path, qtype=gguf.GGMLQuantizationType.Q8_0)
    config = Main_GGUF_ErnieImage_Config.model_construct(path=str(path), name="ernie-image-turbo")
    run = prepare(
        SEAM,
        monkeypatch,
        geometry=lambda patch: patch.setattr(ernie_image, "ERNIE_IMAGE_TRANSFORMER_CONFIG", TINY_CONFIG),
    )
    packed = run.load(config, SubModelType.Transformer)
    apply_custom_layers_to_model(packed)
    dense = ErnieImageTransformer2DModel(**TINY_CONFIG)
    dense.load_state_dict(meant)
    return packed, dense


def test_the_gguf_sidecar_computes_what_merging_the_lora_into_dense_weights_does(gguf_and_dense) -> None:
    packed, dense = gguf_and_dense
    patch = lora_model_from_ernie_image_state_dict(_random_lora(dense))
    unpatched = _forward(dense)

    with LayerPatcher.apply_smart_model_patches(
        model=dense,
        patches=[(patch, 0.8)],
        prefix=ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX,
        dtype=torch.float32,
        force_direct_patching=True,
    ):
        merged = _forward(dense)
    with LayerPatcher.apply_smart_model_patches(
        model=packed,
        patches=[(patch, 0.8)],
        prefix=ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX,
        dtype=torch.float32,
        force_sidecar_patching=True,
    ):
        patched_modules = [m for m in packed.modules() if hasattr(m, "get_num_patches") and m.get_num_patches() > 0]
        sidecar = _forward(packed)

    # Every layer reached its module; none was skipped as unresolvable.
    assert len(patched_modules) == len(patch.layers) == len(_TARGETS)
    # The LoRA moves the output well beyond the comparison tolerance, so the comparison below is not vacuous.
    assert (merged - unpatched).abs().max() > 0.1
    # Measured 5e-6 apart, the GGUF kernels' own rounding (2e-6 without the LoRA); a mis-scaled or misplaced layer
    # is off by tenths.
    assert torch.allclose(sidecar, merged, atol=1e-4, rtol=0)
    # And the packed model is back to its own output once the patch is released.
    assert torch.allclose(_forward(packed), unpatched, atol=1e-4, rtol=0)
