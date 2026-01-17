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
from invokeai.backend.flux.util import get_flux_ae_params, get_flux_transformers_params
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.clip_embed import CLIPEmbed_Diffusers_Config_Base
from invokeai.backend.model_manager.configs.controlnet import (
    ControlNet_Checkpoint_Config_Base,
    ControlNet_Diffusers_Config_Base,
)
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.flux_redux import FLUXRedux_Checkpoint_Config
from invokeai.backend.model_manager.configs.ip_adapter import IPAdapter_Checkpoint_Config_Base
from invokeai.backend.model_manager.configs.main import (
    Main_BnBNF4_FLUX_Config,
    Main_Checkpoint_FLUX_Config,
    Main_Checkpoint_Flux2_Config,
    Main_GGUF_FLUX_Config,
)
from invokeai.backend.model_manager.configs.t5_encoder import T5Encoder_BnBLLMint8_Config, T5Encoder_T5Encoder_Config
from invokeai.backend.model_manager.configs.vae import VAE_Checkpoint_Config_Base, VAE_Checkpoint_Flux2_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    FluxVariantType,
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
        if not isinstance(config, VAE_Checkpoint_Config_Base):
            raise ValueError("Only VAECheckpointConfig models are currently supported here.")
        model_path = Path(config.path)

        with accelerate.init_empty_weights():
            model = AutoEncoder(get_flux_ae_params())
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


@ModelLoaderRegistry.register(base=BaseModelType.Flux2, type=ModelType.VAE, format=ModelFormat.Checkpoint)
class Flux2VAELoader(ModelLoader):
    """Class to load FLUX.2 VAE models (AutoencoderKLFlux2 with 32 latent channels)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, VAE_Checkpoint_Flux2_Config):
            raise ValueError("Only VAE_Checkpoint_Flux2_Config models are currently supported here.")

        from diffusers import AutoencoderKLFlux2

        model_path = Path(config.path)

        # Load state dict manually since from_single_file may not support AutoencoderKLFlux2 yet
        sd = load_file(model_path)

        # Convert BFL format to diffusers format if needed
        # BFL format uses: encoder.down., decoder.up., decoder.mid.block_1, decoder.mid.attn_1, decoder.norm_out
        # Diffusers uses: encoder.down_blocks., decoder.up_blocks., decoder.mid_block.resnets., decoder.conv_norm_out
        is_bfl_format = any(
            k.startswith("encoder.down.") or
            k.startswith("decoder.up.") or
            k.startswith("decoder.mid.block_") or
            k.startswith("decoder.mid.attn_") or
            k.startswith("decoder.norm_out") or
            k.startswith("encoder.mid.block_") or
            k.startswith("encoder.mid.attn_") or
            k.startswith("encoder.norm_out")
            for k in sd.keys()
        )
        if is_bfl_format:
            sd = self._convert_flux2_vae_bfl_to_diffusers(sd)

        # FLUX.2 VAE configuration (32 latent channels)
        # Based on the official FLUX.2 VAE architecture
        # Use default config - AutoencoderKLFlux2 has built-in defaults
        with SilenceWarnings():
            with accelerate.init_empty_weights():
                model = AutoencoderKLFlux2()

        # Convert to bfloat16 and load
        for k in sd.keys():
            sd[k] = sd[k].to(torch.bfloat16)

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

    def _convert_flux2_vae_bfl_to_diffusers(self, sd: dict) -> dict:
        """Convert FLUX.2 VAE BFL format state dict to diffusers format.

        Key differences:
        - encoder.down.X.block.Y -> encoder.down_blocks.X.resnets.Y
        - encoder.down.X.downsample.conv -> encoder.down_blocks.X.downsamplers.0.conv
        - encoder.mid.block_1/2 -> encoder.mid_block.resnets.0/1
        - encoder.mid.attn_1.q/k/v -> encoder.mid_block.attentions.0.to_q/k/v
        - encoder.norm_out -> encoder.conv_norm_out
        - encoder.quant_conv -> quant_conv (top-level)
        - decoder.up.X -> decoder.up_blocks.(num_blocks-1-X) (reversed order!)
        - decoder.post_quant_conv -> post_quant_conv (top-level)
        - *.nin_shortcut -> *.conv_shortcut
        """
        import re

        converted = {}
        num_up_blocks = 4  # Standard VAE has 4 up blocks

        for old_key, tensor in sd.items():
            new_key = old_key

            # Encoder down blocks: encoder.down.X.block.Y -> encoder.down_blocks.X.resnets.Y
            match = re.match(r"encoder\.down\.(\d+)\.block\.(\d+)\.(.*)", old_key)
            if match:
                block_idx, resnet_idx, rest = match.groups()
                rest = rest.replace("nin_shortcut", "conv_shortcut")
                new_key = f"encoder.down_blocks.{block_idx}.resnets.{resnet_idx}.{rest}"
                converted[new_key] = tensor
                continue

            # Encoder downsamplers: encoder.down.X.downsample.conv -> encoder.down_blocks.X.downsamplers.0.conv
            match = re.match(r"encoder\.down\.(\d+)\.downsample\.conv\.(.*)", old_key)
            if match:
                block_idx, rest = match.groups()
                new_key = f"encoder.down_blocks.{block_idx}.downsamplers.0.conv.{rest}"
                converted[new_key] = tensor
                continue

            # Encoder mid block resnets: encoder.mid.block_1/2 -> encoder.mid_block.resnets.0/1
            match = re.match(r"encoder\.mid\.block_(\d+)\.(.*)", old_key)
            if match:
                block_num, rest = match.groups()
                resnet_idx = int(block_num) - 1  # block_1 -> resnets.0, block_2 -> resnets.1
                new_key = f"encoder.mid_block.resnets.{resnet_idx}.{rest}"
                converted[new_key] = tensor
                continue

            # Encoder mid block attention: encoder.mid.attn_1.* -> encoder.mid_block.attentions.0.*
            match = re.match(r"encoder\.mid\.attn_1\.(.*)", old_key)
            if match:
                rest = match.group(1)
                # Map attention keys
                # BFL uses Conv2d (shape [out, in, 1, 1]), diffusers uses Linear (shape [out, in])
                # Squeeze the extra dimensions for weight tensors
                if rest.startswith("q."):
                    new_key = f"encoder.mid_block.attentions.0.to_q.{rest[2:]}"
                    if rest.endswith(".weight") and tensor.dim() == 4:
                        tensor = tensor.squeeze(-1).squeeze(-1)
                elif rest.startswith("k."):
                    new_key = f"encoder.mid_block.attentions.0.to_k.{rest[2:]}"
                    if rest.endswith(".weight") and tensor.dim() == 4:
                        tensor = tensor.squeeze(-1).squeeze(-1)
                elif rest.startswith("v."):
                    new_key = f"encoder.mid_block.attentions.0.to_v.{rest[2:]}"
                    if rest.endswith(".weight") and tensor.dim() == 4:
                        tensor = tensor.squeeze(-1).squeeze(-1)
                elif rest.startswith("proj_out."):
                    new_key = f"encoder.mid_block.attentions.0.to_out.0.{rest[9:]}"
                    if rest.endswith(".weight") and tensor.dim() == 4:
                        tensor = tensor.squeeze(-1).squeeze(-1)
                elif rest.startswith("norm."):
                    new_key = f"encoder.mid_block.attentions.0.group_norm.{rest[5:]}"
                else:
                    new_key = f"encoder.mid_block.attentions.0.{rest}"
                converted[new_key] = tensor
                continue

            # Encoder norm_out -> conv_norm_out
            if old_key.startswith("encoder.norm_out."):
                new_key = old_key.replace("encoder.norm_out.", "encoder.conv_norm_out.")
                converted[new_key] = tensor
                continue

            # Encoder quant_conv -> quant_conv (move to top level)
            if old_key.startswith("encoder.quant_conv."):
                new_key = old_key.replace("encoder.quant_conv.", "quant_conv.")
                converted[new_key] = tensor
                continue

            # Decoder up blocks (reversed order!): decoder.up.X -> decoder.up_blocks.(num_blocks-1-X)
            match = re.match(r"decoder\.up\.(\d+)\.block\.(\d+)\.(.*)", old_key)
            if match:
                block_idx, resnet_idx, rest = match.groups()
                # Reverse the block index
                new_block_idx = num_up_blocks - 1 - int(block_idx)
                rest = rest.replace("nin_shortcut", "conv_shortcut")
                new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.{rest}"
                converted[new_key] = tensor
                continue

            # Decoder upsamplers (reversed order!)
            match = re.match(r"decoder\.up\.(\d+)\.upsample\.conv\.(.*)", old_key)
            if match:
                block_idx, rest = match.groups()
                new_block_idx = num_up_blocks - 1 - int(block_idx)
                new_key = f"decoder.up_blocks.{new_block_idx}.upsamplers.0.conv.{rest}"
                converted[new_key] = tensor
                continue

            # Decoder mid block resnets: decoder.mid.block_1/2 -> decoder.mid_block.resnets.0/1
            match = re.match(r"decoder\.mid\.block_(\d+)\.(.*)", old_key)
            if match:
                block_num, rest = match.groups()
                resnet_idx = int(block_num) - 1
                new_key = f"decoder.mid_block.resnets.{resnet_idx}.{rest}"
                converted[new_key] = tensor
                continue

            # Decoder mid block attention: decoder.mid.attn_1.* -> decoder.mid_block.attentions.0.*
            match = re.match(r"decoder\.mid\.attn_1\.(.*)", old_key)
            if match:
                rest = match.group(1)
                # BFL uses Conv2d (shape [out, in, 1, 1]), diffusers uses Linear (shape [out, in])
                # Squeeze the extra dimensions for weight tensors
                if rest.startswith("q."):
                    new_key = f"decoder.mid_block.attentions.0.to_q.{rest[2:]}"
                    if rest.endswith(".weight") and tensor.dim() == 4:
                        tensor = tensor.squeeze(-1).squeeze(-1)
                elif rest.startswith("k."):
                    new_key = f"decoder.mid_block.attentions.0.to_k.{rest[2:]}"
                    if rest.endswith(".weight") and tensor.dim() == 4:
                        tensor = tensor.squeeze(-1).squeeze(-1)
                elif rest.startswith("v."):
                    new_key = f"decoder.mid_block.attentions.0.to_v.{rest[2:]}"
                    if rest.endswith(".weight") and tensor.dim() == 4:
                        tensor = tensor.squeeze(-1).squeeze(-1)
                elif rest.startswith("proj_out."):
                    new_key = f"decoder.mid_block.attentions.0.to_out.0.{rest[9:]}"
                    if rest.endswith(".weight") and tensor.dim() == 4:
                        tensor = tensor.squeeze(-1).squeeze(-1)
                elif rest.startswith("norm."):
                    new_key = f"decoder.mid_block.attentions.0.group_norm.{rest[5:]}"
                else:
                    new_key = f"decoder.mid_block.attentions.0.{rest}"
                converted[new_key] = tensor
                continue

            # Decoder norm_out -> conv_norm_out
            if old_key.startswith("decoder.norm_out."):
                new_key = old_key.replace("decoder.norm_out.", "decoder.conv_norm_out.")
                converted[new_key] = tensor
                continue

            # Decoder post_quant_conv -> post_quant_conv (move to top level)
            if old_key.startswith("decoder.post_quant_conv."):
                new_key = old_key.replace("decoder.post_quant_conv.", "post_quant_conv.")
                converted[new_key] = tensor
                continue

            # Keep other keys as-is (like encoder.conv_in, decoder.conv_in, decoder.conv_out, bn.*)
            converted[new_key] = tensor

        return converted


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.CLIPEmbed, format=ModelFormat.Diffusers)
class CLIPDiffusersLoader(ModelLoader):
    """Class to load main models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, CLIPEmbed_Diffusers_Config_Base):
            raise ValueError("Only CLIPEmbedDiffusersConfig models are currently supported here.")

        match submodel_type:
            case SubModelType.Tokenizer:
                return CLIPTokenizer.from_pretrained(Path(config.path) / "tokenizer", local_files_only=True)
            case SubModelType.TextEncoder:
                return CLIPTextModel.from_pretrained(Path(config.path) / "text_encoder", local_files_only=True)

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
        if not isinstance(config, T5Encoder_BnBLLMint8_Config):
            raise ValueError("Only T5EncoderBnbQuantizedLlmInt8bConfig models are currently supported here.")
        if not bnb_available:
            raise ImportError(
                "The bnb modules are not available. Please install bitsandbytes if available on your platform."
            )
        match submodel_type:
            case SubModelType.Tokenizer2 | SubModelType.Tokenizer3:
                return T5TokenizerFast.from_pretrained(
                    Path(config.path) / "tokenizer_2", max_length=512, local_files_only=True
                )
            case SubModelType.TextEncoder2 | SubModelType.TextEncoder3:
                te2_model_path = Path(config.path) / "text_encoder_2"
                model_config = AutoConfig.from_pretrained(te2_model_path, local_files_only=True)
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
        if not isinstance(config, T5Encoder_T5Encoder_Config):
            raise ValueError("Only T5EncoderConfig models are currently supported here.")

        match submodel_type:
            case SubModelType.Tokenizer2 | SubModelType.Tokenizer3:
                return T5TokenizerFast.from_pretrained(
                    Path(config.path) / "tokenizer_2", max_length=512, local_files_only=True
                )
            case SubModelType.TextEncoder2 | SubModelType.TextEncoder3:
                return T5EncoderModel.from_pretrained(
                    Path(config.path) / "text_encoder_2",
                    torch_dtype="auto",
                    low_cpu_mem_usage=True,
                    local_files_only=True,
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
        if not isinstance(config, Checkpoint_Config_Base):
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
        assert isinstance(config, Main_Checkpoint_FLUX_Config)
        model_path = Path(config.path)

        with accelerate.init_empty_weights():
            model = Flux(get_flux_transformers_params(config.variant))

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
        if not isinstance(config, Checkpoint_Config_Base):
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
        assert isinstance(config, Main_GGUF_FLUX_Config)
        model_path = Path(config.path)

        with accelerate.init_empty_weights():
            model = Flux(get_flux_transformers_params(config.variant))

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
        if not isinstance(config, Checkpoint_Config_Base):
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
        assert isinstance(config, Main_BnBNF4_FLUX_Config)
        if not bnb_available:
            raise ImportError(
                "The bnb modules are not available. Please install bitsandbytes if available on your platform."
            )
        model_path = Path(config.path)

        with SilenceWarnings():
            with accelerate.init_empty_weights():
                model = Flux(get_flux_transformers_params(config.variant))
                model = quantize_model_nf4(model, modules_to_not_convert=set(), compute_dtype=torch.bfloat16)
            sd = load_file(model_path)
            if "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale" in sd:
                sd = convert_bundle_to_flux_transformer_checkpoint(sd)
            model.load_state_dict(sd, assign=True)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.Main, format=ModelFormat.Diffusers)
class FluxDiffusersModel(GenericDiffusersLoader):
    """Class to load FLUX.1 main models in diffusers format."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, Checkpoint_Config_Base):
            raise NotImplementedError("CheckpointConfigBase is not implemented for FLUX diffusers models.")

        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading main pipelines.")

        model_path = Path(config.path)
        load_class = self.get_hf_load_class(model_path, submodel_type)
        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
        variant = repo_variant.value if repo_variant else None
        model_path = model_path / submodel_type.value

        # We force bfloat16 for FLUX models. This is required for correct inference.
        dtype = torch.bfloat16
        try:
            result: AnyModel = load_class.from_pretrained(
                model_path,
                torch_dtype=dtype,
                variant=variant,
                local_files_only=True,
            )
        except OSError as e:
            if variant and "no file named" in str(
                e
            ):  # try without the variant, just in case user's preferences changed
                result = load_class.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True)
            else:
                raise e

        return result


@ModelLoaderRegistry.register(base=BaseModelType.Flux2, type=ModelType.Main, format=ModelFormat.Diffusers)
class Flux2DiffusersModel(GenericDiffusersLoader):
    """Class to load FLUX.2 main models in diffusers format (e.g. FLUX.2 Klein)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, Checkpoint_Config_Base):
            raise NotImplementedError("CheckpointConfigBase is not implemented for FLUX.2 diffusers models.")

        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading main pipelines.")

        model_path = Path(config.path)
        load_class = self.get_hf_load_class(model_path, submodel_type)
        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
        variant = repo_variant.value if repo_variant else None
        model_path = model_path / submodel_type.value

        # We force bfloat16 for FLUX.2 models. This is required for correct inference.
        # We use low_cpu_mem_usage=False to avoid meta tensors for weights not in checkpoint.
        # FLUX.2 Klein models may have guidance_embeds=False, so the guidance_embed layers
        # won't be in the checkpoint but the model class still creates them.
        # We use SilenceWarnings to suppress the "guidance_embeds is not expected" warning
        # from diffusers Flux2Transformer2DModel.
        dtype = torch.bfloat16
        with SilenceWarnings():
            try:
                result: AnyModel = load_class.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    variant=variant,
                    local_files_only=True,
                    low_cpu_mem_usage=False,
                )
            except OSError as e:
                if variant and "no file named" in str(
                    e
                ):  # try without the variant, just in case user's preferences changed
                    result = load_class.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                        local_files_only=True,
                        low_cpu_mem_usage=False,
                    )
                else:
                    raise e

        return result


@ModelLoaderRegistry.register(base=BaseModelType.Flux2, type=ModelType.Main, format=ModelFormat.Checkpoint)
class Flux2CheckpointModel(ModelLoader):
    """Class to load FLUX.2 transformer models from single-file checkpoints (safetensors)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Checkpoint_Config_Base):
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
        from diffusers import Flux2Transformer2DModel

        if not isinstance(config, Main_Checkpoint_Flux2_Config):
            raise TypeError(
                f"Expected Main_Checkpoint_Flux2_Config, got {type(config).__name__}. "
                "Model configuration type mismatch."
            )
        model_path = Path(config.path)

        # Load state dict
        sd = load_file(model_path)

        # Check if keys have ComfyUI-style prefix and strip if needed
        prefix_to_strip = None
        for prefix in ["model.diffusion_model.", "diffusion_model."]:
            if any(k.startswith(prefix) for k in sd.keys() if isinstance(k, str)):
                prefix_to_strip = prefix
                break

        if prefix_to_strip:
            sd = {
                (k[len(prefix_to_strip) :] if isinstance(k, str) and k.startswith(prefix_to_strip) else k): v
                for k, v in sd.items()
            }

        # Convert BFL format state dict to diffusers format
        converted_sd = self._convert_flux2_bfl_to_diffusers(sd)

        # Detect architecture from checkpoint keys
        double_block_indices = [
            int(k.split(".")[1]) for k in converted_sd.keys() if isinstance(k, str) and k.startswith("transformer_blocks.")
        ]
        single_block_indices = [
            int(k.split(".")[1]) for k in converted_sd.keys() if isinstance(k, str) and k.startswith("single_transformer_blocks.")
        ]

        num_layers = max(double_block_indices) + 1 if double_block_indices else 5
        num_single_layers = max(single_block_indices) + 1 if single_block_indices else 20

        # Get dimensions from weights
        context_embedder_weight = converted_sd.get("context_embedder.weight")
        if context_embedder_weight is not None:
            joint_attention_dim = context_embedder_weight.shape[1]
        else:
            joint_attention_dim = 7680

        x_embedder_weight = converted_sd.get("x_embedder.weight")
        if x_embedder_weight is not None:
            in_channels = x_embedder_weight.shape[1]
        else:
            in_channels = 128

        # Klein models don't have guidance embeddings - check if they're in the checkpoint
        has_guidance = "time_guidance_embed.guidance_embedder.linear_1.weight" in converted_sd

        # Create model with detected configuration
        with SilenceWarnings():
            with accelerate.init_empty_weights():
                model = Flux2Transformer2DModel(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    num_layers=num_layers,
                    num_single_layers=num_single_layers,
                    attention_head_dim=128,
                    num_attention_heads=24,
                    joint_attention_dim=joint_attention_dim,
                    patch_size=1,
                )

        # If Klein model without guidance, initialize guidance embedder with zeros
        if not has_guidance:
            # Get the expected dimensions from timestep embedder (they should match)
            timestep_linear1 = converted_sd.get("time_guidance_embed.timestep_embedder.linear_1.weight")
            if timestep_linear1 is not None:
                in_features = timestep_linear1.shape[1]
                out_features = timestep_linear1.shape[0]
                # Initialize guidance embedder with same shape as timestep embedder
                converted_sd["time_guidance_embed.guidance_embedder.linear_1.weight"] = torch.zeros(
                    out_features, in_features, dtype=torch.bfloat16
                )
                timestep_linear2 = converted_sd.get("time_guidance_embed.timestep_embedder.linear_2.weight")
                if timestep_linear2 is not None:
                    in_features2 = timestep_linear2.shape[1]
                    out_features2 = timestep_linear2.shape[0]
                    converted_sd["time_guidance_embed.guidance_embedder.linear_2.weight"] = torch.zeros(
                        out_features2, in_features2, dtype=torch.bfloat16
                    )

        # Convert to bfloat16 and load
        for k in converted_sd.keys():
            converted_sd[k] = converted_sd[k].to(torch.bfloat16)

        # Load the state dict - guidance weights were already initialized above if missing
        model.load_state_dict(converted_sd, assign=True)

        return model


    def _convert_flux2_bfl_to_diffusers(self, sd: dict) -> dict:
        """Convert FLUX.2 BFL format state dict to diffusers format.

        Based on diffusers convert_flux2_to_diffusers.py key mappings.
        """
        converted = {}

        # Basic key renames
        key_renames = {
            "img_in.weight": "x_embedder.weight",
            "txt_in.weight": "context_embedder.weight",
            "time_in.in_layer.weight": "time_guidance_embed.timestep_embedder.linear_1.weight",
            "time_in.out_layer.weight": "time_guidance_embed.timestep_embedder.linear_2.weight",
            "guidance_in.in_layer.weight": "time_guidance_embed.guidance_embedder.linear_1.weight",
            "guidance_in.out_layer.weight": "time_guidance_embed.guidance_embedder.linear_2.weight",
            "double_stream_modulation_img.lin.weight": "double_stream_modulation_img.linear.weight",
            "double_stream_modulation_txt.lin.weight": "double_stream_modulation_txt.linear.weight",
            "single_stream_modulation.lin.weight": "single_stream_modulation.linear.weight",
            "final_layer.linear.weight": "proj_out.weight",
            "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
        }

        for old_key, tensor in sd.items():
            new_key = old_key

            # Apply basic renames
            if old_key in key_renames:
                new_key = key_renames[old_key]
                # Apply scale-shift swap for adaLN modulation weights
                # BFL and diffusers use different parameter ordering for AdaLayerNorm
                if old_key == "final_layer.adaLN_modulation.1.weight":
                    tensor = self._swap_scale_shift(tensor)
                converted[new_key] = tensor
                continue

            # Convert double_blocks.X.* to transformer_blocks.X.*
            if old_key.startswith("double_blocks."):
                new_key = self._convert_double_block_key(old_key, tensor, converted)
                if new_key is None:
                    continue  # Key was handled specially
            # Convert single_blocks.X.* to single_transformer_blocks.X.*
            elif old_key.startswith("single_blocks."):
                new_key = self._convert_single_block_key(old_key, tensor, converted)
                if new_key is None:
                    continue  # Key was handled specially

            if new_key != old_key or new_key not in converted:
                converted[new_key] = tensor

        return converted

    def _convert_double_block_key(self, key: str, tensor: torch.Tensor, converted: dict) -> str | None:
        """Convert double_blocks key to transformer_blocks format."""
        parts = key.split(".")
        block_idx = parts[1]
        rest = ".".join(parts[2:])

        prefix = f"transformer_blocks.{block_idx}"

        # Attention QKV conversion - BFL uses fused qkv, diffusers uses separate
        if "img_attn.qkv.weight" in rest:
            # Split fused QKV into separate Q, K, V
            q, k, v = tensor.chunk(3, dim=0)
            converted[f"{prefix}.attn.to_q.weight"] = q
            converted[f"{prefix}.attn.to_k.weight"] = k
            converted[f"{prefix}.attn.to_v.weight"] = v
            return None
        elif "txt_attn.qkv.weight" in rest:
            q, k, v = tensor.chunk(3, dim=0)
            converted[f"{prefix}.attn.add_q_proj.weight"] = q
            converted[f"{prefix}.attn.add_k_proj.weight"] = k
            converted[f"{prefix}.attn.add_v_proj.weight"] = v
            return None

        # Attention output projection
        if "img_attn.proj.weight" in rest:
            return f"{prefix}.attn.to_out.0.weight"
        elif "txt_attn.proj.weight" in rest:
            return f"{prefix}.attn.to_add_out.weight"

        # Attention norms
        if "img_attn.norm.query_norm.scale" in rest:
            return f"{prefix}.attn.norm_q.weight"
        elif "img_attn.norm.key_norm.scale" in rest:
            return f"{prefix}.attn.norm_k.weight"
        elif "txt_attn.norm.query_norm.scale" in rest:
            return f"{prefix}.attn.norm_added_q.weight"
        elif "txt_attn.norm.key_norm.scale" in rest:
            return f"{prefix}.attn.norm_added_k.weight"

        # MLP layers
        if "img_mlp.0.weight" in rest:
            return f"{prefix}.ff.linear_in.weight"
        elif "img_mlp.2.weight" in rest:
            return f"{prefix}.ff.linear_out.weight"
        elif "txt_mlp.0.weight" in rest:
            return f"{prefix}.ff_context.linear_in.weight"
        elif "txt_mlp.2.weight" in rest:
            return f"{prefix}.ff_context.linear_out.weight"

        return key

    def _convert_single_block_key(self, key: str, tensor: torch.Tensor, converted: dict) -> str | None:
        """Convert single_blocks key to single_transformer_blocks format."""
        parts = key.split(".")
        block_idx = parts[1]
        rest = ".".join(parts[2:])

        prefix = f"single_transformer_blocks.{block_idx}"

        # linear1 is the fused QKV+MLP projection
        if "linear1.weight" in rest:
            return f"{prefix}.attn.to_qkv_mlp_proj.weight"
        elif "linear2.weight" in rest:
            return f"{prefix}.attn.to_out.weight"

        # Norms
        if "norm.query_norm.scale" in rest:
            return f"{prefix}.attn.norm_q.weight"
        elif "norm.key_norm.scale" in rest:
            return f"{prefix}.attn.norm_k.weight"

        return key

    def _swap_scale_shift(self, weight: torch.Tensor) -> torch.Tensor:
        """Swap scale and shift in AdaLayerNorm weights.

        BFL and diffusers use different parameter ordering for AdaLayerNorm.
        This function swaps the two halves of the weight tensor.

        Args:
            weight: Weight tensor of shape (out_features,) or (out_features, in_features)

        Returns:
            Weight tensor with scale and shift swapped.
        """
        # Split in half along the first dimension and swap
        shift, scale = weight.chunk(2, dim=0)
        return torch.cat([scale, shift], dim=0)


@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.ControlNet, format=ModelFormat.Checkpoint)
@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.ControlNet, format=ModelFormat.Diffusers)
class FluxControlnetModel(ModelLoader):
    """Class to load FLUX ControlNet models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, ControlNet_Checkpoint_Config_Base):
            model_path = Path(config.path)
        elif isinstance(config, ControlNet_Diffusers_Config_Base):
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
            model = XLabsControlNetFlux(get_flux_transformers_params(FluxVariantType.Dev))

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
        if not isinstance(config, IPAdapter_Checkpoint_Config_Base):
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
        if not isinstance(config, FLUXRedux_Checkpoint_Config):
            raise ValueError(f"Unexpected model config type: {type(config)}.")

        sd = load_file(Path(config.path))

        with accelerate.init_empty_weights():
            model = FluxReduxModel()

        model.load_state_dict(sd, assign=True)
        model.to(dtype=torch.bfloat16)
        return model
