"""LTX-2 GGUF transformers: loaded into the diffusers model with their quantized Linears kept packed.

Driven over real (tiny) GGUF files packed the way the published releases pack theirs (see
``ltx2_transformer_fixture``). The loader is checked by its forward, against a dense model holding the values
the file dequantizes to.
"""

import dataclasses
from pathlib import Path

import gguf
import pytest
import torch

from invokeai.backend.ltx2 import component_configs as cc
from invokeai.backend.model_manager.configs.main import Main_GGUF_LTX2_Config
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders import ltx2
from invokeai.backend.model_manager.load.model_loaders.ltx2 import LTX2GGUFModel
from invokeai.backend.model_manager.taxonomy import LTX2VariantType, SubModelType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from tests.backend.model_manager.load.ltx2_transformer_fixture import (
    K_QUANT_TRANSFORMER,
    PREFIX,
    TINY_TRANSFORMER,
    packs_block_linears,
    packs_every_linear,
    run_forward,
    write_ltx2_gguf,
)
from tests.fixtures.loader_seams import Seam, prepare

SEAM = Seam(loader=LTX2GGUFModel, module=ltx2, entry="_load_model", patches_device=True, casts_fp8_storage=False)


def _load(monkeypatch, tmp_path: Path, geometry: dict = TINY_TRANSFORMER, seam: Seam = SEAM, **write_kwargs):
    path = tmp_path / "ltx-2.5-22b-distilled-transformer.gguf"
    meant = write_ltx2_gguf(path, geometry=geometry, **write_kwargs)
    config = Main_GGUF_LTX2_Config.model_construct(
        path=str(path), name=path.stem, variant=LTX2VariantType.Distilled, generation="2.5"
    )
    run = prepare(seam, monkeypatch, geometry=lambda patch: patch.setattr(cc, "LTX2_5_TRANSFORMER_CONFIG", geometry))
    return run.load(config, SubModelType.Transformer), meant, run


def _dense_reference(meant: dict[str, torch.Tensor], geometry: dict) -> torch.nn.Module:
    from diffusers import LTX2VideoTransformer3DModel

    reference = LTX2VideoTransformer3DModel(**{**geometry, "use_keyframes_abs_pos_embedding": True})
    reference.load_state_dict(meant)
    return reference.eval()


def _assert_computes_what_the_file_means(model: torch.nn.Module, meant: dict, geometry: dict) -> None:
    """Not bit-exact: the torch kernels multiply codes by their fp16 block scale in fp16, the reference reader
    in float32. A dropped norm, a misrouted rename or an unconverted tensor is off by far more."""
    apply_custom_layers_to_model(model)
    for got, want in zip(
        run_forward(model, geometry), run_forward(_dense_reference(meant, geometry), geometry), strict=True
    ):
        assert torch.allclose(got, want, atol=1e-3, rtol=1e-3), (got - want).abs().max()


@pytest.mark.parametrize(
    ("qtype", "geometry"),
    [
        (gguf.GGMLQuantizationType.Q8_0, TINY_TRANSFORMER),
        (gguf.GGMLQuantizationType.Q4_0, TINY_TRANSFORMER),
        # What the Q4_K_M releases mostly hold; its super-blocks need the 256-wide geometry.
        (gguf.GGMLQuantizationType.Q4_K, K_QUANT_TRANSFORMER),
    ],
    ids=["Q8_0", "Q4_0", "Q4_K"],
)
def test_the_gguf_transformer_computes_what_its_file_means(monkeypatch, tmp_path, qtype, geometry) -> None:
    model, meant, _ = _load(monkeypatch, tmp_path, geometry, qtype=qtype)

    _assert_computes_what_the_file_means(model, meant, geometry)


def test_a_build_that_quantizes_the_stem_and_adaln_still_computes_what_it_means(monkeypatch, tmp_path) -> None:
    """Two departures in one file: ruygar's builds quantize the adaLN, timestep-embedder and projection Linears
    too, and molbal's hold unquantized tensors in BF16, which is unpacked by reinterpreting its bytes rather than
    through a GGML kernel."""
    model, meant, _ = _load(monkeypatch, tmp_path, packs=packs_every_linear, unquantized=gguf.GGMLQuantizationType.BF16)

    assert isinstance(model.get_parameter("time_embed.linear.weight").data, GGMLTensor)
    _assert_computes_what_the_file_means(model, meant, TINY_TRANSFORMER)


def test_only_the_quantized_linears_stay_packed(monkeypatch, tmp_path) -> None:
    model, _, _ = _load(monkeypatch, tmp_path)

    # Renamed on the way in: the official file spells these `attn1.to_q` and `ff.net.0.proj` the same way,
    # but the stem and modulation layers arrive as `patchify_proj` / `adaln_single`.
    for name in ("transformer_blocks.0.attn1.to_q.weight", "transformer_blocks.0.ff.net.0.proj.weight"):
        assert isinstance(model.get_parameter(name).data, GGMLTensor), name
    # Read outside a matmul or not quantized at all: a norm, a modulation table, the flattened keyframe
    # embedding restored to its shape, an F16 gate and an F32 stem Linear.
    for name in (
        "transformer_blocks.0.attn1.norm_q.weight",
        "transformer_blocks.0.scale_shift_table",
        "keyframes_abs_pos_embedding",
        "transformer_blocks.0.attn1.to_gate_logits.weight",
        "time_embed.linear.weight",
    ):
        assert type(model.get_parameter(name).data) is torch.Tensor, name
    assert model.get_parameter("keyframes_abs_pos_embedding").shape == (1, TINY_TRANSFORMER["num_attention_heads"] * 64)


def test_a_quantized_keyframe_embedding_is_unpacked_to_its_values(monkeypatch, tmp_path) -> None:
    """molbal's builds quantize the keyframe embedding, which the model adds outside any matmul. No forward here
    reads it (only keyframe conditioning does), so its values are compared directly."""
    model, meant, _ = _load(
        monkeypatch,
        tmp_path,
        packs=lambda name, shape: packs_block_linears(name, shape) or name == "keyframes_abs_pos_embedding",
    )

    embedding = model.get_parameter("keyframes_abs_pos_embedding").data
    assert type(embedding) is torch.Tensor
    torch.testing.assert_close(embedding, meant["keyframes_abs_pos_embedding"], rtol=1e-3, atol=1e-6)


def test_the_weights_follow_the_devices_compute_dtype(monkeypatch, tmp_path) -> None:
    """On a GPU that dtype is bf16, not the float32 the forwards above run at: the unpacked tensors are held in it
    and the packed weights dequantize to it. float32 would double both."""
    model, _, _ = _load(monkeypatch, tmp_path, seam=dataclasses.replace(SEAM, compute_dtype=torch.bfloat16))

    packed = [param.data for param in model.parameters() if isinstance(param.data, GGMLTensor)]
    assert packed and all(weight.compute_dtype == torch.bfloat16 for weight in packed)
    assert all(param.dtype == torch.bfloat16 for param in model.parameters() if not isinstance(param.data, GGMLTensor))


def test_a_prefixed_gguf_loads_like_a_bare_one(monkeypatch, tmp_path) -> None:
    """ruygar's builds keep the official `model.diffusion_model.` prefix on every key."""
    model, meant, _ = _load(monkeypatch, tmp_path, prefix=PREFIX)

    _assert_computes_what_the_file_means(model, meant, TINY_TRANSFORMER)


def test_the_load_reserves_the_model_as_it_will_be_held(monkeypatch, tmp_path) -> None:
    """One absolute reservation, since `make_room` does not add to the framework's file-size one. The connectors
    are never read, so they are not in it."""
    model, _, run = _load(monkeypatch, tmp_path)
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


def test_only_the_transformer_submodel_is_served(monkeypatch, tmp_path) -> None:
    path = tmp_path / "ltx-2.5-22b-dev-transformer-Q8_0.gguf"
    write_ltx2_gguf(path)
    config = Main_GGUF_LTX2_Config.model_construct(
        path=str(path), name=path.stem, variant=LTX2VariantType.Dev, generation="2.5"
    )
    run = prepare(SEAM, monkeypatch)

    with pytest.raises(ValueError, match="serve only the transformer"):
        run.load(config, SubModelType.VAE)
    assert run.reserved == []


def test_a_q8_cr_file_behind_an_installed_record_fails_at_load_by_name(monkeypatch, tmp_path) -> None:
    """A Q8_CR file never installs as an LTX-2 GGUF, but a record outlives its file: one replaced on disk after the
    install must fail here by name, not at the module build on a dtype error that names neither file nor format."""
    with pytest.raises(ValueError, match="Q8_CR"):
        _load(monkeypatch, tmp_path, q8_cr_layer="transformer_blocks.0.attn1.to_q.weight")
