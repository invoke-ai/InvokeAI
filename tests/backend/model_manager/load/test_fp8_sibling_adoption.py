"""Registered sibling loaders run their native bodies against real shared stores and small geometry."""

import json
from contextlib import ExitStack

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.load import load_default
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import FluxVariantType, Qwen35VariantType, SubModelType
from invokeai.backend.util.devices import TorchDevice
from tests.backend.model_manager.load.test_fp8_review_regressions import _device_loaders
from tests.backend.model_manager.load.test_fp8_shared_weight_variants import FP8, _TinyModel
from tests.backend.model_manager.load.test_layerwise_storage_policy import _public_load


def _sibling(family, monkeypatch, tmp_path):
    import diffusers

    from invokeai.backend.anima import anima_transformer
    from invokeai.backend.model_manager.configs import main
    from invokeai.backend.model_manager.configs.qwen3_5_encoder import Qwen35Encoder_Checkpoint_Config
    from invokeai.backend.model_manager.load.model_loaders import anima, flux

    # Only the large geometry is substituted on these five transformer paths. No loader class,
    # cache lookup, quantization operation, admission or adoption implementation is replaced.
    config_cls = {
        "anima": main.Main_Checkpoint_Anima_Config,
        "flux": main.Main_Checkpoint_FLUX_Config,
        "flux2": main.Main_Checkpoint_Flux2_Config,
        "z": main.Main_Checkpoint_ZImage_Config,
        "qwen21": main.Main_Checkpoint_QwenImage21_Config,
        "qwen35": Qwen35Encoder_Checkpoint_Config,
        "moss-text": main.Main_Diffusers_MossImage_Config,
    }[family]
    subtype = SubModelType.TextEncoder if family in ("qwen35", "moss-text") else SubModelType.Transformer
    source = {"proj.weight": torch.ones(4, 4).to(FP8)}
    weight = "proj.weight"
    path = tmp_path / "sibling.safetensors"
    extra = {}
    if family == "anima":
        monkeypatch.setattr(anima_transformer, "AnimaTransformer", _TinyModel)
        monkeypatch.setattr(anima, "anima_transformer_config", lambda _sd: {})
        monkeypatch.setattr(TorchDevice, "choose_anima_inference_dtype", staticmethod(lambda _d: torch.float32))
    elif family == "flux":
        monkeypatch.setattr(flux, "Flux", _TinyModel)
        monkeypatch.setattr(flux, "get_flux_transformers_params", lambda _variant: None)
        extra["variant"] = FluxVariantType.Dev
    elif family == "flux2":
        monkeypatch.setattr(diffusers, "Flux2Transformer2DModel", _TinyModel)
    elif family in ("z", "qwen21"):
        prefix = "layers" if family == "z" else "transformer_blocks"

        class TinyBlocks(torch.nn.Module):
            def __init__(self, **_kwargs):
                super().__init__()
                setattr(self, prefix, torch.nn.ModuleList([_TinyModel()]))

        weight = f"{prefix}.0.proj.weight"
        source = {weight: torch.ones(4, 4).to(FP8)}
        name = "ZImageTransformer2DModel" if family == "z" else "QwenImage21Transformer2DModel"
        monkeypatch.setattr(diffusers, name, TinyBlocks)
    else:
        from invokeai.backend.qwen3_5 import qwen3_5_encoder
        from tests.backend.model_manager.load.test_moss_image_loader import _TINY_TEXT_ENCODER_CONFIG
        from tests.backend.qwen3_5.test_qwen3_5_loading import _config, _reference

        reference = _reference(final_norm=True)
        source = {name: value.detach().clone().contiguous() for name, value in reference.state_dict().items()}
        weight = "layers.3.self_attn.q_proj.weight"
        source[weight] = torch.ones_like(source[weight]).to(FP8)
        if family == "qwen35":
            monkeypatch.setattr(qwen3_5_encoder, "qwen3_5_text_config", lambda _variant: _config())
            extra["variant"] = Qwen35VariantType.Qwen35_9B
        else:
            path = tmp_path / "moss"
            (path / "text_encoder").mkdir(parents=True)
            (path / "text_encoder" / "config.json").write_text(json.dumps(_TINY_TEXT_ENCODER_CONFIG))
    checkpoint = path / "text_encoder" / "model.safetensors" if family == "moss-text" else path
    save_file(source, str(checkpoint))
    config = config_cls.model_construct(key="sibling", name="sibling", path=str(path), **extra)
    cls, config, subtype = ModelLoaderRegistry.get_implementation(config, subtype)
    return cls, config, subtype, weight


@pytest.mark.parametrize("family", ["anima", "flux", "flux2", "z", "qwen21", "qwen35", "moss-text"])
@pytest.mark.parametrize("reverse", [False, True])
def test_registered_checkpoint_siblings_partition_and_adopt(family, reverse, monkeypatch, tmp_path):
    cls, config, subtype, weight = _sibling(family, monkeypatch, tmp_path)
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _d: torch.float32))
    monkeypatch.setattr(load_default, "should_keep_fp8_weights", lambda d: d.index == 0)
    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", lambda *_args: False)
    capable, incapable, peer = _device_loaders(cls, tmp_path, ["cuda:0", "cuda:1", "cuda:0"])
    with ExitStack() as stack:
        for loader in (capable, incapable, peer):
            stack.callback(loader._ram_cache.shutdown)
        order = [incapable, capable] if reverse else [capable, incapable]
        models = {loader: _public_load(loader, config.model_copy(), subtype) for loader in order}
        packed, folded = (models[loader].get_parameter(weight) for loader in (capable, incapable))
        assert packed.dtype == FP8
        assert folded.dtype == (torch.bfloat16 if family in ("flux", "flux2") else torch.float32)
        torch.testing.assert_close(folded.float(), torch.ones_like(folded, dtype=torch.float32))
        assert packed.data_ptr() != folded.data_ptr()
        adopted = _public_load(peer, config.model_copy(), subtype)
        assert adopted.get_parameter(weight).data_ptr() == packed.data_ptr()
