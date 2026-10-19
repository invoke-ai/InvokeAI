"""Loader-level tests for Ideogram 4 GGUF transformers.

These read a real (tiny) GGUF through the real `gguf_sd_loader`, so every tensor arrives as the
`GGMLTensor` the loader has to cope with. What is asserted is the forward: a `GGMLTensor` only works
where torch dispatches the op to it, and `Ideogram4Transformer` reads several weights outside a
Linear's matmul -- an embedding lookup, `F.rms_norm`, and the dtype it casts its inputs to.
"""

import gguf
import pytest
import torch

from invokeai.backend.ideogram4.constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR
from invokeai.backend.ideogram4.modeling_ideogram4 import Ideogram4Transformer
from invokeai.backend.model_manager.configs.main import Main_GGUF_Ideogram4_Config
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders import ideogram4
from invokeai.backend.model_manager.load.model_loaders.ideogram4 import Ideogram4GGUFModel
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from tests.backend.model_manager.load.ideogram4_gguf_fixture import (
    TINY_CONFIG,
    Packs,
    packs_block_linears,
    packs_every_weight,
    write_ideogram4_gguf,
)
from tests.fixtures.loader_seams import Seam, SeamRun, prepare

SEAM = Seam(
    loader=Ideogram4GGUFModel,
    module=ideogram4,
    entry="_load_model",
    patches_device=True,
    casts_fp8_storage=False,
)

LAYOUTS = pytest.mark.parametrize("packs", [packs_block_linears, packs_every_weight], ids=["molbal", "rectangleworm"])


def _load(
    monkeypatch, tmp_path, packs: Packs, qtype: gguf.GGMLQuantizationType = gguf.GGMLQuantizationType.Q8_0
) -> tuple[torch.nn.Module, dict[str, torch.Tensor], SeamRun]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "ideogram4-transformer-q8_0.gguf"
    meant = write_ideogram4_gguf(path, packs, qtype=qtype)
    config = Main_GGUF_Ideogram4_Config.model_construct(path=str(path), name="ideogram4", branch="conditional")

    run = prepare(
        SEAM,
        monkeypatch,
        geometry=lambda patch: patch.setattr(
            "invokeai.backend.ideogram4.modeling_ideogram4.Ideogram4Config", lambda: TINY_CONFIG
        ),
    )
    return run.load(config, SubModelType.Transformer), meant, run


def _forward(model: torch.nn.Module) -> torch.Tensor:
    """One step over two text tokens and four image tokens, the packed layout the denoiser builds."""
    generator = torch.Generator().manual_seed(1)
    indicator = torch.tensor([[LLM_TOKEN_INDICATOR] * 2 + [OUTPUT_IMAGE_INDICATOR] * 4])
    with torch.no_grad():
        return model(
            llm_features=torch.randn(1, 6, TINY_CONFIG.llm_features_dim, generator=generator),
            x=torch.randn(1, 6, TINY_CONFIG.in_channels, generator=generator),
            t=torch.tensor([0.3]),
            position_ids=torch.randint(0, 4, (1, 6, 3), generator=generator),
            segment_ids=torch.zeros(1, 6, dtype=torch.long),
            indicator=indicator,
        )


@LAYOUTS
@pytest.mark.parametrize(
    "qtype",
    [gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q5_1, gguf.GGMLQuantizationType.Q4_0],
    ids=lambda qtype: qtype.name,
)
def test_the_gguf_branch_computes_what_its_file_means(
    monkeypatch, tmp_path, packs: Packs, qtype: gguf.GGMLQuantizationType
) -> None:
    """Against a dense model holding the values the file dequantizes to, so the comparison is not
    blurred by quantization error. Q4_0 and Q5_1 are the starter builds' types. Not bit-exact all the
    same: the torch kernels multiply codes
    by their fp16 block scale in fp16, the reference reader in float32. A dropped norm weight, a
    zeroed embedding or an activation cast to the wrong dtype is off by orders of magnitude more.

    Wrapped in the custom layers first, as the model cache does on every load: that is the path the
    packed Linear weights take in production.
    """
    model, meant, _ = _load(monkeypatch, tmp_path, packs, qtype)
    apply_custom_layers_to_model(model)
    reference = Ideogram4Transformer(TINY_CONFIG)
    reference.load_state_dict(meant)

    assert torch.allclose(_forward(model), _forward(reference), atol=1e-3, rtol=1e-3)


@LAYOUTS
def test_what_the_model_reads_outside_a_matmul_is_dequantized_at_load(monkeypatch, tmp_path, packs: Packs) -> None:
    model, _, _ = _load(monkeypatch, tmp_path, packs)

    # The embedding lookup, the RMSNorms, and the two Linears the model reads its compute dtype from.
    for name in (
        "embed_image_indicator.weight",
        "llm_cond_norm.weight",
        "layers.0.attention.norm_q.weight",
        "input_proj.weight",
        "t_embedding.mlp_in.weight",
    ):
        weight = model.get_parameter(name)
        assert type(weight.data) is torch.Tensor, name
        assert weight.dtype is torch.float32, name
    # Everything else that is quantized in the file stays packed, or there was no point to the format.
    assert isinstance(model.get_parameter("layers.0.feed_forward.w1.weight").data, GGMLTensor)


def test_an_unquantized_linear_weight_is_unpacked_and_a_quantized_one_is_not(monkeypatch, tmp_path) -> None:
    """`llm_cond_proj` is 53248 wide in the released geometry and BF16 in most releases.

    Kept packed, BF16 is dequantized on every forward like Q4_0 is -- a measured 1.87 GB transient
    for that one layer -- for no saving at all, since it is the same size either way. Quantized, as
    rectangleworm ships it, it stays packed like any other quantized Linear.
    """
    unpacked, _, _ = _load(monkeypatch, tmp_path / "bf16", packs_block_linears)
    packed, _, _ = _load(monkeypatch, tmp_path / "q8", packs_every_weight)

    assert type(unpacked.get_parameter("llm_cond_proj.weight").data) is torch.Tensor
    assert isinstance(packed.get_parameter("llm_cond_proj.weight").data, GGMLTensor)


@LAYOUTS
def test_the_load_reserves_the_model_as_it_will_be_held(monkeypatch, tmp_path, packs: Packs) -> None:
    """One absolute reservation: `make_room` makes that much room rather than adding to the file-size
    reservation the framework made before the loader ran, so reserving only what dequantizing adds
    would evict nothing. It covers every weight as the model ends up holding it -- packed or
    unpacked -- plus the largest unpacked tensor, which is briefly held twice while it is replaced.
    """
    model, _, run = _load(monkeypatch, tmp_path, packs)
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
