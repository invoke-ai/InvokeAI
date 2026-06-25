"""Model loading for Ideogram 4 in InvokeAI.

The on-disk model is a diffusers pipeline folder bundling:
  - transformer/                 (Ideogram4Transformer, nf4 or fp8 quantized)
  - unconditional_transformer/   (Ideogram4Transformer, nf4 or fp8 quantized)
  - text_encoder/ + tokenizer/   (Qwen3-VL, nf4 or fp8)
  - vae/                         (FLUX.2-style AutoencoderKL; loaded via the vendored AutoEncoder)

The transformer is our vendored ``Ideogram4Transformer`` (not a diffusers class), so we
build it explicitly and load the prequantized state dict — mirroring how InvokeAI loads
FLUX nf4. Both transformer branches are returned as a single ``Ideogram4TransformerPair``.
"""

import json
from pathlib import Path
from typing import Any, Optional

import accelerate
import torch
from safetensors.torch import load_file

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import Main_Diffusers_Ideogram4_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.util.devices import TorchDevice


def _load_local_state_dict(folder: Path, basename: str) -> dict[str, torch.Tensor]:
    """Load a (possibly sharded) safetensors checkpoint from a local diffusers component folder."""
    index_path = folder / f"{basename}.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
        sd: dict[str, torch.Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            sd.update(load_file(folder / shard))
        return sd
    return load_file(folder / f"{basename}.safetensors")


@ModelLoaderRegistry.register(base=BaseModelType.Ideogram4, type=ModelType.Main, format=ModelFormat.Diffusers)
class Ideogram4DiffusersModel(ModelLoader):
    """Loads Ideogram 4 main models (nf4 / fp8) bundled in diffusers layout."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Main_Diffusers_Ideogram4_Config):
            raise ValueError(f"Expected Main_Diffusers_Ideogram4_Config, got {type(config).__name__}.")
        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading Ideogram 4 main pipelines.")

        model_path = Path(config.path)

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_transformer_pair(model_path)
            case SubModelType.TextEncoder:
                return self._load_text_encoder(model_path)
            case SubModelType.Tokenizer:
                from transformers import AutoTokenizer

                return AutoTokenizer.from_pretrained(model_path / "tokenizer", local_files_only=True)
            case SubModelType.VAE:
                return self._load_vae(model_path)

        raise ValueError(
            f"Unsupported submodel for Ideogram 4: {submodel_type.value if submodel_type else 'None'}. "
            "Supported: Transformer, TextEncoder, Tokenizer, VAE."
        )

    def _load_transformer_pair(self, model_path: Path) -> AnyModel:
        from invokeai.backend.ideogram4.transformer_pair import Ideogram4TransformerPair

        conditional = self._load_one_transformer(model_path / "transformer")
        unconditional = self._load_one_transformer(model_path / "unconditional_transformer")
        return Ideogram4TransformerPair(conditional=conditional, unconditional=unconditional)

    def _load_one_transformer(self, folder: Path) -> torch.nn.Module:
        from invokeai.backend.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
        from invokeai.backend.ideogram4.quantized_loading import (
            is_bnb4bit_state_dict,
            is_fp8_state_dict,
            load_fp8_state_dict,
            swap_linears_to_fp8,
        )
        from invokeai.backend.quantization.bnb_nf4 import quantize_model_nf4

        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = _load_local_state_dict(folder, "diffusion_pytorch_model")
        self._ram_cache.make_room(sum(t.nelement() * t.element_size() for t in sd.values()))

        if is_bnb4bit_state_dict(sd):
            # nf4: build the model with InvokeLinearNF4 layers (compress_statistics=False, matching
            # the on-disk single-quant format), then load the prequantized state dict. The model
            # stays on CPU/meta until the cache moves it to the GPU.
            with accelerate.init_empty_weights():
                model: torch.nn.Module = Ideogram4Transformer(Ideogram4Config())
                model = quantize_model_nf4(model, modules_to_not_convert=set(), compute_dtype=compute_dtype)
            model.load_state_dict(sd, strict=True, assign=True)
            return model

        if is_fp8_state_dict(sd):
            # Weight-only fp8 (e4m3): dequantizes to compute dtype at forward time; runs on any device.
            model = Ideogram4Transformer(Ideogram4Config())
            model.to(compute_dtype)
            swap_linears_to_fp8(model, sd, compute_dtype=compute_dtype)
            load_fp8_state_dict(model, sd, device=torch.device("cpu"), dtype=compute_dtype)
            model.eval()
            return model

        # Unquantized fallback.
        with accelerate.init_empty_weights():
            model = Ideogram4Transformer(Ideogram4Config())
        model.load_state_dict(sd, strict=True, assign=True)
        return model.to(compute_dtype)

    def _load_text_encoder(self, model_path: Path) -> AnyModel:
        import accelerate
        from transformers import AutoConfig, AutoModel

        from invokeai.backend.quantization.bnb_nf4 import quantize_model_nf4

        encoder_path = model_path / "text_encoder"
        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        raw_cfg = json.loads((encoder_path / "config.json").read_text(encoding="utf-8"))
        if raw_cfg.get("ideogram_fp8_weight_only", False):
            # The fp8 text encoder uses Ideogram's custom weight-only fp8 layout; supporting it
            # requires the vendored _load_fp8_text_encoder path. Deferred (nf4 is the 24 GB path).
            raise NotImplementedError(
                "Ideogram 4 fp8 text encoder loading is not yet implemented; use the nf4 build."
            )

        # Build the bare architecture from config, then quantize with InvokeAI's InvokeLinearNF4 and
        # load the prequantized weights. We must NOT use transformers' native bitsandbytes loading
        # (from_pretrained with a quantization_config) because the resulting bnb Linear4bit layers are
        # not compatible with InvokeAI's partial-loading model cache. This mirrors how the FLUX T5 bnb
        # encoder is loaded.
        cfg = AutoConfig.from_pretrained(encoder_path, local_files_only=True)
        # Drop the quantization_config so from_config builds a plain (unquantized) architecture.
        if hasattr(cfg, "quantization_config"):
            cfg.quantization_config = None

        sd = _load_local_state_dict(encoder_path, "model")
        self._ram_cache.make_room(sum(t.nelement() * t.element_size() for t in sd.values()))

        is_bnb_nf4 = "quantization_config" in raw_cfg and bool(
            raw_cfg["quantization_config"].get("load_in_4bit")
        )

        with accelerate.init_empty_weights():
            model: torch.nn.Module = AutoModel.from_config(cfg)
            if is_bnb_nf4:
                model = quantize_model_nf4(model, modules_to_not_convert=set(), compute_dtype=compute_dtype)

        model.load_state_dict(sd, strict=False, assign=True)
        if not is_bnb_nf4:
            model = model.to(compute_dtype)
        model.eval()
        return model

    def _load_vae(self, model_path: Path) -> AnyModel:
        from invokeai.backend.ideogram4.autoencoder import (
            AutoEncoder,
            AutoEncoderParams,
            convert_diffusers_state_dict,
        )

        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = load_file(model_path / "vae" / "diffusion_pytorch_model.safetensors")
        sd = convert_diffusers_state_dict(sd)
        ae = AutoEncoder(AutoEncoderParams())
        ae.load_state_dict(sd)
        ae.eval()
        return ae.to(model_dtype)
