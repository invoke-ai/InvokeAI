"""ERNIE-Image GGUF transformers: identified by their keys, loaded into the diffusers model.

Both are driven over real (tiny) GGUF files. Identification goes through the whole factory, because
the published files name another model's architecture in their header (`wan`, `flux`) and the
question is which of all the candidate configs claims them. The loader is checked by its forward,
against a dense model holding the values the file dequantizes to.
"""

from pathlib import Path

import gguf
import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_ErnieImage_Config, Main_GGUF_ErnieImage_Config
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders import ernie_image
from invokeai.backend.model_manager.load.model_loaders.ernie_image import ErnieImageGGUFModel
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from tests.backend.model_manager.load.ernie_image_gguf_fixture import (
    K_QUANT_CONFIG,
    TINY_CONFIG,
    write_ernie_image_gguf,
)
from tests.fixtures.loader_seams import Seam, prepare

SEAM = Seam(
    loader=ErnieImageGGUFModel, module=ernie_image, entry="_load_model", patches_device=True, casts_fp8_storage=False
)


def _identify(path: Path):
    return ModelConfigFactory.from_model_on_disk(ModelOnDisk(path), {}, allow_unknown=True)


@pytest.mark.parametrize(
    ("filename", "architecture"),
    [("ernie-image-turbo-Q8_0.gguf", "wan"), ("ernie-image-Q8_0.gguf", "flux")],
    ids=["unsloth", "vantagewithai"],
)
def test_a_published_gguf_installs_as_ernie_image_whatever_its_header_says(
    tmp_path: Path, filename: str, architecture: str
) -> None:
    path = tmp_path / filename
    write_ernie_image_gguf(path, architecture=architecture)

    result = _identify(path)

    assert result.match_count == 1, [type(match).__name__ for match in result.all_matches]
    assert isinstance(result.config, Main_GGUF_ErnieImage_Config)
    # Turbo and base share an architecture; only the name tells them apart, for GGUF as for safetensors.
    assert result.config.default_settings.steps == (8 if "turbo" in filename else 50)


def test_a_safetensors_single_file_is_not_claimed_as_gguf(tmp_path: Path) -> None:
    meant = write_ernie_image_gguf(tmp_path / "unused.gguf")
    path = tmp_path / "ernie-image-turbo.safetensors"
    save_file({key: value.to(torch.bfloat16) for key, value in meant.items()}, path)

    result = _identify(path)

    assert result.match_count == 1, [type(match).__name__ for match in result.all_matches]
    assert isinstance(result.config, Main_Checkpoint_ErnieImage_Config)


def _load(monkeypatch, tmp_path: Path, qtype: gguf.GGMLQuantizationType, geometry: dict = TINY_CONFIG):
    path = tmp_path / f"ernie-image-turbo-{qtype.name}.gguf"
    meant = write_ernie_image_gguf(path, qtype=qtype, config=geometry)
    config = Main_GGUF_ErnieImage_Config.model_construct(path=str(path), name="ernie-image-turbo")
    run = prepare(
        SEAM,
        monkeypatch,
        geometry=lambda patch: patch.setattr(ernie_image, "ERNIE_IMAGE_TRANSFORMER_CONFIG", geometry),
    )
    return run.load(config, SubModelType.Transformer), meant, run


def _forward(model: torch.nn.Module, geometry: dict = TINY_CONFIG) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        return model(
            hidden_states=torch.randn(1, geometry["in_channels"], 2, 2, generator=generator),
            timestep=torch.tensor([500.0]),
            text_bth=torch.randn(1, 3, geometry["text_in_dim"], generator=generator),
            text_lens=torch.tensor([3]),
            return_dict=False,
        )[0]


@pytest.mark.parametrize(
    ("qtype", "geometry"),
    [
        (gguf.GGMLQuantizationType.Q8_0, TINY_CONFIG),
        (gguf.GGMLQuantizationType.Q5_1, TINY_CONFIG),
        (gguf.GGMLQuantizationType.Q4_0, TINY_CONFIG),
        # What the Q4_K_M starters mostly hold; its super-blocks need the 256-wide geometry.
        (gguf.GGMLQuantizationType.Q4_K, K_QUANT_CONFIG),
    ],
    ids=["Q8_0", "Q5_1", "Q4_0", "Q4_K"],
)
def test_the_gguf_transformer_computes_what_its_file_means(monkeypatch, tmp_path, qtype, geometry) -> None:
    """Not bit-exact: the torch kernels multiply codes by their fp16 block scale in fp16, the reference
    reader in float32. A dropped norm, an unconverted convolution or a wrong key is off by far more."""
    from diffusers import ErnieImageTransformer2DModel

    model, meant, _ = _load(monkeypatch, tmp_path, qtype, geometry)
    apply_custom_layers_to_model(model)
    reference = ErnieImageTransformer2DModel(**geometry)
    reference.load_state_dict(meant)

    assert torch.allclose(_forward(model, geometry), _forward(reference, geometry), atol=1e-3, rtol=1e-3)


def test_only_the_quantized_linears_stay_packed(monkeypatch, tmp_path) -> None:
    model, _, _ = _load(monkeypatch, tmp_path, gguf.GGMLQuantizationType.Q8_0)

    for name in ("layers.0.mlp.gate_proj.weight", "text_proj.weight"):
        assert isinstance(model.get_parameter(name).data, GGMLTensor), name
    # The 4-D BF16 patch convolution, a BF16 stem Linear and an F32 norm: none can run packed, or need to.
    for name in ("x_embedder.proj.weight", "adaLN_modulation.1.weight", "layers.0.adaLN_sa_ln.weight"):
        assert type(model.get_parameter(name).data) is torch.Tensor, name


def test_the_load_reserves_the_model_as_it_will_be_held(monkeypatch, tmp_path) -> None:
    """One absolute reservation, since `make_room` does not add to the framework's file-size one."""
    model, _, run = _load(monkeypatch, tmp_path, gguf.GGMLQuantizationType.Q8_0)
    held = {
        name: param.data.quantized_data.nbytes
        if isinstance(param.data, GGMLTensor)
        else param.numel() * param.element_size()
        for name, param in model.named_parameters()
    }
    largest_unpacked = max(
        size for name, size in held.items() if not isinstance(model.get_parameter(name).data, GGMLTensor)
    )

    assert run.reserved == [sum(held.values()) + largest_unpacked]
