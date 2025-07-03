# Copyright (c) 2024, Brandon W. Rising and the InvokeAI Development Team
"""Class for Flux model loading in InvokeAI."""

from pathlib import Path
from typing import Optional

import accelerate
import torch
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForTextEncoding,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from invokeai.app.services.config.config_default import get_config
from invokeai.backend.flux.controlnet.instantx_controlnet_flux import InstantXControlNetFlux
from invokeai.backend.flux.controlnet.state_dict_utils import (
    convert_diffusers_instantx_state_dict_to_bfl_format,
    infer_flux_params_from_state_dict,
    infer_instantx_num_control_modes_from_state_dict,
    is_state_dict_instantx_controlnet,
    is_state_dict_xlabs_controlnet,
)
from invokeai.backend.flux.controlnet.xlabs_controlnet_flux import XLabsControlNetFlux
from invokeai.backend.flux.ip_adapter.state_dict_utils import infer_xlabs_ip_adapter_params_from_state_dict
from invokeai.backend.flux.ip_adapter.xlabs_ip_adapter_flux import (
    XlabsIpAdapterFlux,
)
from invokeai.backend.flux.model import Flux
from invokeai.backend.flux.modules.autoencoder import AutoEncoder
from invokeai.backend.flux.redux.flux_redux_model import FluxReduxModel
from invokeai.backend.flux.util import ae_params, params
from invokeai.backend.model_manager.config import (
    AnyModelConfig,
    CheckpointConfigBase,
    CLIPEmbedDiffusersConfig,
    ControlNetCheckpointConfig,
    ControlNetDiffusersConfig,
    FluxReduxConfig,
    IPAdapterCheckpointConfig,
    MainBnbQuantized4bCheckpointConfig,
    MainCheckpointConfig,
    MainGGUFCheckpointConfig,
    T5EncoderBnbQuantizedLlmInt8bConfig,
    T5EncoderConfig,
    VAECheckpointConfig,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.model_manager.util.model_util import (
    convert_bundle_to_flux_transformer_checkpoint,
)
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader
from invokeai.backend.quantization.gguf.utils import TORCH_COMPATIBLE_QTYPES
from invokeai.backend.util.silence_warnings import SilenceWarnings

try:
    from invokeai.backend.quantization.bnb_llm_int8 import quantize_model_llm_int8
    from invokeai.backend.quantization.bnb_nf4 import quantize_model_nf4

    bnb_available = True
except ImportError:
    bnb_available = False

app_config = get_config()


@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.VAE, format=ModelFormat.Checkpoint)
class FluxVAELoader(ModelLoader):
    """Class to load VAE models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, VAECheckpointConfig):
            raise ValueError("Only VAECheckpointConfig models are currently supported here.")
        model_path = Path(config.path)

        with accelerate.init_empty_weights():
            model = AutoEncoder(ae_params[config.config_path])
        sd = load_file(model_path)
        model.load_state_dict(sd, assign=True)
        # VAE is broken in float16, which mps defaults to
        if self._torch_dtype == torch.float16:
            try:
                vae_dtype = torch.tensor([1.0], dtype=torch.bfloat16, device=self._torch_device).dtype
            except TypeError:
                vae_dtype = torch.float32
        else:
            vae_dtype = self._torch_dtype
        model.to(vae_dtype)

        return model


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.CLIPEmbed, format=ModelFormat.Diffusers)
class ClipCheckpointModel(ModelLoader):
    """Class to load main models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, CLIPEmbedDiffusersConfig):
            raise ValueError("Only CLIPEmbedDiffusersConfig models are currently supported here.")

        match submodel_type:
            case SubModelType.Tokenizer:
                return CLIPTokenizer.from_pretrained(Path(config.path) / "tokenizer")
            case SubModelType.TextEncoder:
                return CLIPTextModel.from_pretrained(Path(config.path) / "text_encoder")

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.T5Encoder, format=ModelFormat.BnbQuantizedLlmInt8b)
class BnbQuantizedLlmInt8bCheckpointModel(ModelLoader):
    """Class to load main models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, T5EncoderBnbQuantizedLlmInt8bConfig):
            raise ValueError("Only T5EncoderBnbQuantizedLlmInt8bConfig models are currently supported here.")
        if not bnb_available:
            raise ImportError(
                "The bnb modules are not available. Please install bitsandbytes if available on your platform."
            )
        match submodel_type:
            case SubModelType.Tokenizer2 | SubModelType.Tokenizer3:
                return T5TokenizerFast.from_pretrained(Path(config.path) / "tokenizer_2", max_length=512)
            case SubModelType.TextEncoder2 | SubModelType.TextEncoder3:
                te2_model_path = Path(config.path) / "text_encoder_2"
                model_config = AutoConfig.from_pretrained(te2_model_path)
                with accelerate.init_empty_weights():
                    model = AutoModelForTextEncoding.from_config(model_config)
                    model = quantize_model_llm_int8(model, modules_to_not_convert=set())

                state_dict_path = te2_model_path / "bnb_llm_int8_model.safetensors"
                state_dict = load_file(state_dict_path)
                self._load_state_dict_into_t5(model, state_dict)

                return model

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    @classmethod
    def _load_state_dict_into_t5(cls, model: T5EncoderModel, state_dict: dict[str, torch.Tensor]):
        # There is a shared reference to a single weight tensor in the model.
        # Both "encoder.embed_tokens.weight" and "shared.weight" refer to the same tensor, so only the latter should
        # be present in the state_dict.
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
        assert len(unexpected_keys) == 0
        assert set(missing_keys) == {"encoder.embed_tokens.weight"}
        # Assert that the layers we expect to be shared are actually shared.
        assert model.encoder.embed_tokens.weight is model.shared.weight


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.T5Encoder, format=ModelFormat.T5Encoder)
class T5EncoderCheckpointModel(ModelLoader):
    """Class to load main models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, T5EncoderConfig):
            raise ValueError("Only T5EncoderConfig models are currently supported here.")

        match submodel_type:
            case SubModelType.Tokenizer2 | SubModelType.Tokenizer3:
                return T5TokenizerFast.from_pretrained(Path(config.path) / "tokenizer_2", max_length=512)
            case SubModelType.TextEncoder2 | SubModelType.TextEncoder3:
                return T5EncoderModel.from_pretrained(
                    Path(config.path) / "text_encoder_2", torch_dtype="auto", low_cpu_mem_usage=True
                )

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )


@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.Main, format=ModelFormat.Checkpoint)
class FluxCheckpointModel(ModelLoader):
    """Class to load main models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, CheckpointConfigBase):
            raise ValueError("Only CheckpointConfigBase models are currently supported here.")

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_from_singlefile(config)

        raise ValueError(
            f"Only Transformer submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_singlefile(
        self,
        config: AnyModelConfig,
    ) -> AnyModel:
        assert isinstance(config, MainCheckpointConfig)
        model_path = Path(config.path)

        with accelerate.init_empty_weights():
            model = Flux(params[config.config_path])

        sd = load_file(model_path)
        if "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale" in sd:
            sd = convert_bundle_to_flux_transformer_checkpoint(sd)
        new_sd_size = sum([ten.nelement() * torch.bfloat16.itemsize for ten in sd.values()])
        self._ram_cache.make_room(new_sd_size)
        for k in sd.keys():
            # We need to cast to bfloat16 due to it being the only currently supported dtype for inference
            sd[k] = sd[k].to(torch.bfloat16)
        model.load_state_dict(sd, assign=True)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.Main, format=ModelFormat.GGUFQuantized)
class FluxGGUFCheckpointModel(ModelLoader):
    """Class to load GGUF main models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, CheckpointConfigBase):
            raise ValueError("Only CheckpointConfigBase models are currently supported here.")

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_from_singlefile(config)

        raise ValueError(
            f"Only Transformer submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_singlefile(
        self,
        config: AnyModelConfig,
    ) -> AnyModel:
        assert isinstance(config, MainGGUFCheckpointConfig)
        model_path = Path(config.path)

        with accelerate.init_empty_weights():
            model = Flux(params[config.config_path])

        # HACK(ryand): We shouldn't be hard-coding the compute_dtype here.
        sd = gguf_sd_loader(model_path, compute_dtype=torch.bfloat16)

        # HACK(ryand): There are some broken GGUF models in circulation that have the wrong shape for img_in.weight.
        # We override the shape here to fix the issue.
        # Example model with this issue (Q4_K_M): https://civitai.com/models/705823/ggufk-flux-unchained-km-quants
        img_in_weight = sd.get("img_in.weight", None)
        if img_in_weight is not None and img_in_weight._ggml_quantization_type in TORCH_COMPATIBLE_QTYPES:
            expected_img_in_weight_shape = model.img_in.weight.shape
            img_in_weight.quantized_data = img_in_weight.quantized_data.view(expected_img_in_weight_shape)
            img_in_weight.tensor_shape = expected_img_in_weight_shape

        model.load_state_dict(sd, assign=True)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.Main, format=ModelFormat.BnbQuantizednf4b)
class FluxBnbQuantizednf4bCheckpointModel(ModelLoader):
    """Class to load main models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, CheckpointConfigBase):
            raise ValueError("Only CheckpointConfigBase models are currently supported here.")

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_from_singlefile(config)

        raise ValueError(
            f"Only Transformer submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_singlefile(
        self,
        config: AnyModelConfig,
    ) -> AnyModel:
        assert isinstance(config, MainBnbQuantized4bCheckpointConfig)
        if not bnb_available:
            raise ImportError(
                "The bnb modules are not available. Please install bitsandbytes if available on your platform."
            )
        model_path = Path(config.path)

        with SilenceWarnings():
            with accelerate.init_empty_weights():
                model = Flux(params[config.config_path])
                model = quantize_model_nf4(model, modules_to_not_convert=set(), compute_dtype=torch.bfloat16)
            sd = load_file(model_path)
            if "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale" in sd:
                sd = convert_bundle_to_flux_transformer_checkpoint(sd)
            model.load_state_dict(sd, assign=True)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.ControlNet, format=ModelFormat.Checkpoint)
@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.ControlNet, format=ModelFormat.Diffusers)
class FluxControlnetModel(ModelLoader):
    """Class to load FLUX ControlNet models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, ControlNetCheckpointConfig):
            model_path = Path(config.path)
        elif isinstance(config, ControlNetDiffusersConfig):
            # If this is a diffusers directory, we simply ignore the config file and load from the weight file.
            model_path = Path(config.path) / "diffusion_pytorch_model.safetensors"
        else:
            raise ValueError(f"Unexpected ControlNet model config type: {type(config)}")

        sd = load_file(model_path)

        # Detect the FLUX ControlNet model type from the state dict.
        if is_state_dict_xlabs_controlnet(sd):
            return self._load_xlabs_controlnet(sd)
        elif is_state_dict_instantx_controlnet(sd):
            return self._load_instantx_controlnet(sd)
        else:
            raise ValueError("Do not recognize the state dict as an XLabs or InstantX ControlNet model.")

    def _load_xlabs_controlnet(self, sd: dict[str, torch.Tensor]) -> AnyModel:
        with accelerate.init_empty_weights():
            # HACK(ryand): Is it safe to assume dev here?
            model = XLabsControlNetFlux(params["flux-dev"])

        model.load_state_dict(sd, assign=True)
        return model

    def _load_instantx_controlnet(self, sd: dict[str, torch.Tensor]) -> AnyModel:
        sd = convert_diffusers_instantx_state_dict_to_bfl_format(sd)
        flux_params = infer_flux_params_from_state_dict(sd)
        num_control_modes = infer_instantx_num_control_modes_from_state_dict(sd)

        with accelerate.init_empty_weights():
            model = InstantXControlNetFlux(flux_params, num_control_modes)

        model.load_state_dict(sd, assign=True)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.IPAdapter, format=ModelFormat.Checkpoint)
class FluxIpAdapterModel(ModelLoader):
    """Class to load FLUX IP-Adapter models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, IPAdapterCheckpointConfig):
            raise ValueError(f"Unexpected model config type: {type(config)}.")

        sd = load_file(Path(config.path))

        params = infer_xlabs_ip_adapter_params_from_state_dict(sd)

        with accelerate.init_empty_weights():
            model = XlabsIpAdapterFlux(params=params)

        model.load_xlabs_state_dict(sd, assign=True)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.FluxRedux, format=ModelFormat.Checkpoint)
class FluxReduxModelLoader(ModelLoader):
    """Class to load FLUX Redux models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, FluxReduxConfig):
            raise ValueError(f"Unexpected model config type: {type(config)}.")

        sd = load_file(Path(config.path))

        with accelerate.init_empty_weights():
            model = FluxReduxModel()

        model.load_state_dict(sd, assign=True)
        model.to(dtype=torch.bfloat16)
        return model
