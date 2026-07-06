# Copyright (c) 2024, Brandon W. Rising and the InvokeAI Development Team
"""Class for Flux model loading in InvokeAI."""

from pathlib import Path
from typing import Optional

import accelerate
import torch
from diffusers import AutoencoderKL
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForTextEncoding,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
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
    Main_Checkpoint_Flux2_Config,
    Main_Checkpoint_FLUX_Config,
    Main_GGUF_Flux2_Config,
    Main_GGUF_FLUX_Config,
    Main_SDNQ_Diffusers_Flux2_Config,
    Main_SDNQ_Diffusers_FLUX_Config,
    Main_SDNQ_Flux2_Config,
    Main_SDNQ_FLUX_Config,
)
from invokeai.backend.model_manager.configs.t5_encoder import (
    T5Encoder_BnBLLMint8_Config,
    T5Encoder_SDNQ_Config,
    T5Encoder_T5Encoder_Config,
)
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
from invokeai.backend.quantization.sdnq.loaders import sdnq_sd_loader
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


@ModelLoaderRegistry.register(base=BaseModelType.Flux2, type=ModelType.VAE, format=ModelFormat.Diffusers)
class Flux2VAEDiffusersLoader(ModelLoader):
    """Class to load FLUX.2 VAE models in diffusers format (AutoencoderKLFlux2 with 32 latent channels)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        from diffusers import AutoencoderKLFlux2

        model_path = Path(config.path)

        # VAE is broken in float16, which mps defaults to
        if self._torch_dtype == torch.float16:
            try:
                vae_dtype = torch.tensor([1.0], dtype=torch.bfloat16, device=self._torch_device).dtype
            except TypeError:
                vae_dtype = torch.float32
        else:
            vae_dtype = self._torch_dtype

        model = AutoencoderKLFlux2.from_pretrained(
            model_path,
            torch_dtype=vae_dtype,
            local_files_only=True,
        )

        model = self._apply_fp8_layerwise_casting(model, config, submodel_type)
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
            k.startswith("encoder.down.")
            or k.startswith("decoder.up.")
            or k.startswith("decoder.mid.block_")
            or k.startswith("decoder.mid.attn_")
            or k.startswith("decoder.norm_out")
            or k.startswith("encoder.mid.block_")
            or k.startswith("encoder.mid.attn_")
            or k.startswith("encoder.norm_out")
            for k in sd.keys()
        )
        if is_bfl_format:
            sd = self._convert_flux2_vae_bfl_to_diffusers(sd)

        # FLUX.2 VAE configuration (32 latent channels).
        # The standard FLUX.2 VAE uses block_out_channels=(128,256,512,512) for both
        # encoder and decoder. The "small decoder" variant from
        # black-forest-labs/FLUX.2-small-decoder keeps the full encoder but uses a
        # narrower decoder with channels (96,192,384,384). AutoencoderKLFlux2 only
        # exposes a single block_out_channels, so we build the model with the
        # encoder's channels and, if the decoder differs, replace just the decoder
        # submodule with a matching one before loading the state dict.
        encoder_block_out_channels = (128, 256, 512, 512)
        decoder_block_out_channels = encoder_block_out_channels
        if "encoder.conv_in.weight" in sd and "encoder.conv_norm_out.weight" in sd:
            enc_last = int(sd["encoder.conv_norm_out.weight"].shape[0])
            enc_first = int(sd["encoder.conv_in.weight"].shape[0])
            encoder_block_out_channels = (enc_first, enc_first * 2, enc_last, enc_last)
        if "decoder.conv_in.weight" in sd and "decoder.conv_norm_out.weight" in sd:
            dec_last = int(sd["decoder.conv_in.weight"].shape[0])
            dec_first = int(sd["decoder.conv_norm_out.weight"].shape[0])
            decoder_block_out_channels = (dec_first, dec_first * 2, dec_last, dec_last)

        with SilenceWarnings():
            with accelerate.init_empty_weights():
                model = AutoencoderKLFlux2(block_out_channels=encoder_block_out_channels)
                if decoder_block_out_channels != encoder_block_out_channels:
                    # Rebuild the decoder with the smaller channel widths.
                    from diffusers.models.autoencoders.vae import Decoder

                    cfg = model.config
                    model.decoder = Decoder(
                        in_channels=cfg.latent_channels,
                        out_channels=cfg.out_channels,
                        up_block_types=cfg.up_block_types,
                        block_out_channels=decoder_block_out_channels,
                        layers_per_block=cfg.layers_per_block,
                        norm_num_groups=cfg.norm_num_groups,
                        act_fn=cfg.act_fn,
                        mid_block_add_attention=cfg.mid_block_add_attention,
                    )

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

        model = self._apply_fp8_layerwise_casting(model, config, submodel_type)
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
                return T5Tokenizer.from_pretrained(
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
        # Re-tie shared weights. In transformers 5.x, weight tying is implemented at the
        # parameter level (via _tie_weights / tie_weights) rather than as a Python object
        # alias.  load_state_dict(assign=True) replaces parameters in-place, which severs
        # the parameter-level tie.  Calling tie_weights() re-establishes it.
        model.tie_weights()


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
                return T5Tokenizer.from_pretrained(
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


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.T5Encoder, format=ModelFormat.SDNQQuantized)
class T5EncoderSDNQLoader(ModelLoader):
    """Class to load SDNQ-quantized T5 Encoder models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, T5Encoder_SDNQ_Config):
            raise ValueError("Only T5Encoder_SDNQ_Config models are supported here.")

        match submodel_type:
            case SubModelType.Tokenizer2 | SubModelType.Tokenizer3:
                return T5TokenizerFast.from_pretrained(
                    Path(config.path) / "tokenizer_2", max_length=512, local_files_only=True
                )
            case SubModelType.TextEncoder2 | SubModelType.TextEncoder3:
                return self._load_text_encoder(config)

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_text_encoder(self, config: T5Encoder_SDNQ_Config) -> AnyModel:
        # Two layouts: either config.path is the pipeline root (T5 lives under text_encoder_2/),
        # or config.path is the text_encoder_2 folder itself (FluxPipeline submodel case).
        base = Path(config.path)
        nested = base / "text_encoder_2"
        te_dir = nested if (nested / "config.json").exists() else base

        model_config = AutoConfig.from_pretrained(te_dir, local_files_only=True)
        with accelerate.init_empty_weights():
            model = AutoModelForTextEncoding.from_config(model_config)

        sd = sdnq_sd_loader(te_dir, compute_dtype=torch.bfloat16)

        # T5's embed_tokens and shared point to the same parameter; the SDNQ state dict only carries one of them.
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False, assign=True)
        assert len(unexpected_keys) == 0, f"Unexpected keys loading SDNQ T5: {unexpected_keys}"
        assert set(missing_keys) <= {"encoder.embed_tokens.weight"}, (
            f"Unexpected missing keys loading SDNQ T5: {missing_keys}"
        )
        if "encoder.embed_tokens.weight" in missing_keys:
            model.encoder.embed_tokens.weight = model.shared.weight
        return model


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
                model = self._load_from_singlefile(config)
                model = self._apply_fp8_layerwise_casting(model, config, submodel_type)
                return model

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

        result = self._apply_fp8_layerwise_casting(result, config, submodel_type)
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

        # For Klein models without guidance_embeds, zero out the guidance_embedder weights
        # that were randomly initialized by diffusers. This prevents noise from affecting
        # the time embeddings.
        if submodel_type == SubModelType.Transformer and hasattr(result, "time_guidance_embed"):
            # Check if this is a Klein model without guidance (guidance_embeds=False in config)
            transformer_config_path = model_path / "config.json"
            if transformer_config_path.exists():
                import json

                with open(transformer_config_path, "r") as f:
                    transformer_config = json.load(f)
                if not transformer_config.get("guidance_embeds", True):
                    # Zero out the guidance embedder weights
                    guidance_emb = result.time_guidance_embed.guidance_embedder
                    if hasattr(guidance_emb, "linear_1"):
                        guidance_emb.linear_1.weight.data.zero_()
                        if guidance_emb.linear_1.bias is not None:
                            guidance_emb.linear_1.bias.data.zero_()
                    if hasattr(guidance_emb, "linear_2"):
                        guidance_emb.linear_2.weight.data.zero_()
                        if guidance_emb.linear_2.bias is not None:
                            guidance_emb.linear_2.bias.data.zero_()

        result = self._apply_fp8_layerwise_casting(result, config, submodel_type)
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
                model = self._load_from_singlefile(config)
                model = self._apply_fp8_layerwise_casting(model, config, submodel_type)
                return model

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

        # Handle FP8 quantized weights (ComfyUI-style or scaled FP8)
        # These store weights as: layer.weight (FP8) + layer.weight_scale (FP32 scalar)
        sd = self._dequantize_fp8_weights(sd)

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
            int(k.split(".")[1])
            for k in converted_sd.keys()
            if isinstance(k, str) and k.startswith("transformer_blocks.")
        ]
        single_block_indices = [
            int(k.split(".")[1])
            for k in converted_sd.keys()
            if isinstance(k, str) and k.startswith("single_transformer_blocks.")
        ]

        num_layers = max(double_block_indices) + 1 if double_block_indices else 5
        num_single_layers = max(single_block_indices) + 1 if single_block_indices else 20

        # Get dimensions from weights
        # context_embedder.weight shape: [hidden_size, joint_attention_dim]
        context_embedder_weight = converted_sd.get("context_embedder.weight")
        if context_embedder_weight is not None:
            hidden_size = context_embedder_weight.shape[0]
            joint_attention_dim = context_embedder_weight.shape[1]
        else:
            # Default to Klein 4B dimensions
            hidden_size = 3072
            joint_attention_dim = 7680

        x_embedder_weight = converted_sd.get("x_embedder.weight")
        if x_embedder_weight is not None:
            in_channels = x_embedder_weight.shape[1]
        else:
            in_channels = 128

        # Calculate num_attention_heads from hidden_size
        # Klein 4B: hidden_size=3072, num_attention_heads=24 (3072/128=24)
        # Klein 9B: hidden_size=4096, num_attention_heads=32 (4096/128=32)
        attention_head_dim = 128
        num_attention_heads = hidden_size // attention_head_dim

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
                    attention_head_dim=attention_head_dim,
                    num_attention_heads=num_attention_heads,
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
            # Defensive check: ensure tensor has at least 1 dimension and can be split into 3
            if tensor.dim() < 1 or tensor.shape[0] % 3 != 0:
                # Skip malformed tensors (might be metadata or corrupted)
                return key
            q, k, v = tensor.chunk(3, dim=0)
            converted[f"{prefix}.attn.to_q.weight"] = q
            converted[f"{prefix}.attn.to_k.weight"] = k
            converted[f"{prefix}.attn.to_v.weight"] = v
            return None
        elif "txt_attn.qkv.weight" in rest:
            # Defensive check
            if tensor.dim() < 1 or tensor.shape[0] % 3 != 0:
                return key
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
        if "img_attn.norm.query_norm.scale" in rest or "img_attn.norm.query_norm.weight" in rest:
            return f"{prefix}.attn.norm_q.weight"
        elif "img_attn.norm.key_norm.scale" in rest or "img_attn.norm.key_norm.weight" in rest:
            return f"{prefix}.attn.norm_k.weight"
        elif "txt_attn.norm.query_norm.scale" in rest or "txt_attn.norm.query_norm.weight" in rest:
            return f"{prefix}.attn.norm_added_q.weight"
        elif "txt_attn.norm.key_norm.scale" in rest or "txt_attn.norm.key_norm.weight" in rest:
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
        if "norm.query_norm.scale" in rest or "norm.query_norm.weight" in rest:
            return f"{prefix}.attn.norm_q.weight"
        elif "norm.key_norm.scale" in rest or "norm.key_norm.weight" in rest:
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
        # Defensive check: ensure tensor can be split
        if weight.dim() < 1 or weight.shape[0] % 2 != 0:
            return weight
        # Split in half along the first dimension and swap
        shift, scale = weight.chunk(2, dim=0)
        return torch.cat([scale, shift], dim=0)

    def _dequantize_fp8_weights(self, sd: dict) -> dict:
        """Dequantize FP8 quantized weights in the state dict.

        ComfyUI and some FLUX.2 models store quantized weights as:
        - layer.weight: quantized FP8 data
        - layer.weight_scale: scale factor (FP32 scalar or per-channel)

        Dequantization formula: dequantized = weight.to(float) * weight_scale

        Also handles FP8 tensors stored with float8_e4m3fn dtype by converting to float.
        """
        # Check for ComfyUI-style scale factors
        weight_scale_keys = [k for k in sd.keys() if isinstance(k, str) and k.endswith(".weight_scale")]

        for scale_key in weight_scale_keys:
            # Get the corresponding weight key
            weight_key = scale_key.replace(".weight_scale", ".weight")
            if weight_key in sd:
                weight = sd[weight_key]
                scale = sd[scale_key]

                # Dequantize: convert FP8 to float and multiply by scale
                # Note: Float8 types require .float() instead of .to(torch.float32)
                weight_float = weight.float()
                scale = scale.float()

                # Handle block-wise quantization where scale may have different shape
                if scale.dim() > 0 and scale.shape != weight_float.shape and scale.numel() > 1:
                    for dim in range(len(weight_float.shape)):
                        if dim < len(scale.shape) and scale.shape[dim] != weight_float.shape[dim]:
                            block_size = weight_float.shape[dim] // scale.shape[dim]
                            if block_size > 1:
                                scale = scale.repeat_interleave(block_size, dim=dim)

                sd[weight_key] = weight_float * scale

        # Filter out scale metadata keys and other FP8 metadata
        keys_to_remove = [
            k
            for k in sd.keys()
            if isinstance(k, str)
            and (k.endswith(".weight_scale") or k.endswith(".scale_weight") or "comfy_quant" in k or k == "scaled_fp8")
        ]
        for k in keys_to_remove:
            del sd[k]

        # Handle native FP8 tensors (float8_e4m3fn dtype) that aren't already dequantized
        # Also filter out 0-dimensional tensors (scalars) which are typically metadata
        keys_to_convert = []
        keys_to_remove_scalars = []
        for key in list(sd.keys()):
            tensor = sd[key]
            if hasattr(tensor, "dim"):
                if tensor.dim() == 0:
                    # 0-dimensional tensor (scalar) - likely metadata, remove it
                    keys_to_remove_scalars.append(key)
                elif hasattr(tensor, "dtype") and "float8" in str(tensor.dtype):
                    # Native FP8 tensor - mark for conversion
                    keys_to_convert.append(key)

        for k in keys_to_remove_scalars:
            del sd[k]

        for key in keys_to_convert:
            # Convert FP8 tensor to float32
            sd[key] = sd[key].float()

        return sd


@ModelLoaderRegistry.register(base=BaseModelType.Flux2, type=ModelType.Main, format=ModelFormat.SDNQQuantized)
class Flux2SDNQCheckpointModel(ModelLoader):
    """Class to load SDNQ-quantized FLUX.2 transformer models (e.g. Klein 4B / 9B).

    The checkpoint is expected to be in diffusers layout (i.e. the same key naming as
    Flux2Transformer2DModel.state_dict()), since SDNQ tooling typically operates on
    diffusers state dicts. BFL-layout SDNQ FLUX.2 checkpoints are not supported here.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, (Main_SDNQ_Flux2_Config, Main_SDNQ_Diffusers_Flux2_Config)):
            raise ValueError(
                "Only Main_SDNQ_Flux2_Config or Main_SDNQ_Diffusers_Flux2_Config models are supported here."
            )

        # Single-file SDNQ FLUX.2 checkpoints only ship the transformer.
        if isinstance(config, Main_SDNQ_Flux2_Config):
            if submodel_type == SubModelType.Transformer:
                return self._load_from_singlefile(config)
            raise ValueError(
                f"Single-file SDNQ FLUX.2 checkpoints only provide the Transformer submodel. "
                f"Received: {submodel_type.value if submodel_type else 'None'}"
            )

        # Full Flux2 pipeline folder — dispatch each submodel from its own subfolder.
        match submodel_type:
            case SubModelType.Transformer:
                return self._load_transformer_from_folder(config)
            case SubModelType.TextEncoder:
                return self._load_text_encoder(config)
            case SubModelType.Tokenizer:
                return self._load_tokenizer(config)
            case SubModelType.VAE:
                return self._load_vae(config)

        raise ValueError(
            f"Unsupported submodel type for SDNQ FLUX.2 pipeline: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_transformer_from_folder(self, config: Main_SDNQ_Diffusers_Flux2_Config) -> AnyModel:
        from diffusers import Flux2Transformer2DModel

        model_path = Path(config.path)
        transformer_path = model_path / "transformer" if (model_path / "transformer").is_dir() else model_path

        with accelerate.init_empty_weights():
            model = Flux2Transformer2DModel.from_config(
                Flux2Transformer2DModel.load_config(transformer_path, local_files_only=True)
            )

        sd = sdnq_sd_loader(transformer_path, compute_dtype=torch.bfloat16)
        model.load_state_dict(sd, assign=True, strict=False)
        return model

    def _load_text_encoder(self, config: Main_SDNQ_Diffusers_Flux2_Config) -> AnyModel:
        from transformers import AutoConfig, Qwen3ForCausalLM

        te_dir = Path(config.path) / "text_encoder"
        te_config = AutoConfig.from_pretrained(te_dir, local_files_only=True)
        with accelerate.init_empty_weights():
            model = Qwen3ForCausalLM(te_config)

        sd = sdnq_sd_loader(te_dir, compute_dtype=torch.bfloat16)
        missing, unexpected = model.load_state_dict(sd, assign=True, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected keys loading SDNQ Qwen3 text encoder: {unexpected}")
        if missing and missing != ["lm_head.weight"]:
            raise ValueError(f"Unexpected missing keys loading SDNQ Qwen3 text encoder: {missing}")
        if missing == ["lm_head.weight"]:
            model.lm_head.weight = model.model.embed_tokens.weight
        return model

    def _load_tokenizer(self, config: Main_SDNQ_Diffusers_Flux2_Config) -> AnyModel:
        from transformers import AutoTokenizer

        tok_dir = Path(config.path) / "tokenizer"
        return AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)

    def _load_vae(self, config: Main_SDNQ_Diffusers_Flux2_Config) -> AnyModel:
        # FLUX.2 Klein uses AutoencoderKLFlux2 (not the generic AutoencoderKL). Both ship as
        # plain bf16 in this pipeline (the VAE itself isn't SDNQ-quantized).
        from diffusers import AutoencoderKL, AutoencoderKLFlux2

        vae_dir = Path(config.path) / "vae"
        # Pick the right class based on what the on-disk config.json declares.
        try:
            cls_name = AutoencoderKL.load_config(vae_dir, local_files_only=True).get("_class_name", "")
        except Exception:
            cls_name = ""
        if cls_name == "AutoencoderKLFlux2":
            return AutoencoderKLFlux2.from_pretrained(vae_dir, local_files_only=True)
        return AutoencoderKL.from_pretrained(vae_dir, local_files_only=True)

    def _load_from_singlefile(self, config: Main_SDNQ_Flux2_Config) -> AnyModel:
        from diffusers import Flux2Transformer2DModel

        model_path = Path(config.path)

        sd = sdnq_sd_loader(model_path, compute_dtype=torch.bfloat16)

        # Detect architecture from state dict shapes. SDNQTensor.shape returns the
        # *dequantized* shape, so this works identically to the fp16 path.
        double_block_indices = [
            int(k.split(".")[1]) for k in sd.keys() if isinstance(k, str) and k.startswith("transformer_blocks.")
        ]
        single_block_indices = [
            int(k.split(".")[1]) for k in sd.keys() if isinstance(k, str) and k.startswith("single_transformer_blocks.")
        ]
        num_layers = max(double_block_indices) + 1 if double_block_indices else 5
        num_single_layers = max(single_block_indices) + 1 if single_block_indices else 20

        context_embedder_weight = sd.get("context_embedder.weight")
        if context_embedder_weight is not None:
            hidden_size = context_embedder_weight.shape[0]
            joint_attention_dim = context_embedder_weight.shape[1]
        else:
            hidden_size = 3072
            joint_attention_dim = 7680

        x_embedder_weight = sd.get("x_embedder.weight")
        in_channels = x_embedder_weight.shape[1] if x_embedder_weight is not None else 128

        attention_head_dim = 128
        num_attention_heads = hidden_size // attention_head_dim

        has_guidance = "time_guidance_embed.guidance_embedder.linear_1.weight" in sd

        with SilenceWarnings():
            with accelerate.init_empty_weights():
                model = Flux2Transformer2DModel(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    num_layers=num_layers,
                    num_single_layers=num_single_layers,
                    attention_head_dim=attention_head_dim,
                    num_attention_heads=num_attention_heads,
                    joint_attention_dim=joint_attention_dim,
                    patch_size=1,
                )

        # Klein variants ship without guidance embeddings — zero-fill from the timestep
        # embedder dimensions so load_state_dict has a tensor for those slots.
        if not has_guidance:
            timestep_linear1 = sd.get("time_guidance_embed.timestep_embedder.linear_1.weight")
            if timestep_linear1 is not None:
                out_features, in_features = timestep_linear1.shape[0], timestep_linear1.shape[1]
                sd["time_guidance_embed.guidance_embedder.linear_1.weight"] = torch.zeros(
                    out_features, in_features, dtype=torch.bfloat16
                )
                timestep_linear2 = sd.get("time_guidance_embed.timestep_embedder.linear_2.weight")
                if timestep_linear2 is not None:
                    out2, in2 = timestep_linear2.shape[0], timestep_linear2.shape[1]
                    sd["time_guidance_embed.guidance_embedder.linear_2.weight"] = torch.zeros(
                        out2, in2, dtype=torch.bfloat16
                    )

        model.load_state_dict(sd, assign=True)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Flux2, type=ModelType.Main, format=ModelFormat.GGUFQuantized)
class Flux2GGUFCheckpointModel(ModelLoader):
    """Class to load GGUF-quantized FLUX.2 transformer models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Main_GGUF_Flux2_Config):
            raise ValueError("Only Main_GGUF_Flux2_Config models are currently supported here.")

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_from_singlefile(config)

        raise ValueError(
            f"Only Transformer submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_singlefile(
        self,
        config: Main_GGUF_Flux2_Config,
    ) -> AnyModel:
        from diffusers import Flux2Transformer2DModel

        model_path = Path(config.path)

        # Load GGUF state dict
        sd = gguf_sd_loader(model_path, compute_dtype=torch.bfloat16)

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
            int(k.split(".")[1])
            for k in converted_sd.keys()
            if isinstance(k, str) and k.startswith("transformer_blocks.")
        ]
        single_block_indices = [
            int(k.split(".")[1])
            for k in converted_sd.keys()
            if isinstance(k, str) and k.startswith("single_transformer_blocks.")
        ]

        num_layers = max(double_block_indices) + 1 if double_block_indices else 5
        num_single_layers = max(single_block_indices) + 1 if single_block_indices else 20

        # Get dimensions from weights
        # context_embedder.weight shape: [hidden_size, joint_attention_dim]
        context_embedder_weight = converted_sd.get("context_embedder.weight")
        if context_embedder_weight is not None:
            if hasattr(context_embedder_weight, "tensor_shape"):
                hidden_size = context_embedder_weight.tensor_shape[0]
                joint_attention_dim = context_embedder_weight.tensor_shape[1]
            else:
                hidden_size = context_embedder_weight.shape[0]
                joint_attention_dim = context_embedder_weight.shape[1]
        else:
            # Default to Klein 4B dimensions
            hidden_size = 3072
            joint_attention_dim = 7680

        x_embedder_weight = converted_sd.get("x_embedder.weight")
        if x_embedder_weight is not None:
            in_channels = (
                x_embedder_weight.tensor_shape[1]
                if hasattr(x_embedder_weight, "tensor_shape")
                else x_embedder_weight.shape[1]
            )
        else:
            in_channels = 128

        # Calculate num_attention_heads from hidden_size
        # Klein 4B: hidden_size=3072, num_attention_heads=24 (3072/128=24)
        # Klein 9B: hidden_size=4096, num_attention_heads=32 (4096/128=32)
        attention_head_dim = 128
        num_attention_heads = hidden_size // attention_head_dim

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
                    attention_head_dim=attention_head_dim,
                    num_attention_heads=num_attention_heads,
                    joint_attention_dim=joint_attention_dim,
                    patch_size=1,
                )

        # If Klein model without guidance, initialize guidance embedder with zeros
        if not has_guidance:
            timestep_linear1 = converted_sd.get("time_guidance_embed.timestep_embedder.linear_1.weight")
            if timestep_linear1 is not None:
                in_features = (
                    timestep_linear1.tensor_shape[1]
                    if hasattr(timestep_linear1, "tensor_shape")
                    else timestep_linear1.shape[1]
                )
                out_features = (
                    timestep_linear1.tensor_shape[0]
                    if hasattr(timestep_linear1, "tensor_shape")
                    else timestep_linear1.shape[0]
                )
                converted_sd["time_guidance_embed.guidance_embedder.linear_1.weight"] = torch.zeros(
                    out_features, in_features, dtype=torch.bfloat16
                )
                timestep_linear2 = converted_sd.get("time_guidance_embed.timestep_embedder.linear_2.weight")
                if timestep_linear2 is not None:
                    in_features2 = (
                        timestep_linear2.tensor_shape[1]
                        if hasattr(timestep_linear2, "tensor_shape")
                        else timestep_linear2.shape[1]
                    )
                    out_features2 = (
                        timestep_linear2.tensor_shape[0]
                        if hasattr(timestep_linear2, "tensor_shape")
                        else timestep_linear2.shape[0]
                    )
                    converted_sd["time_guidance_embed.guidance_embedder.linear_2.weight"] = torch.zeros(
                        out_features2, in_features2, dtype=torch.bfloat16
                    )

        model.load_state_dict(converted_sd, assign=True)
        return model

    def _convert_flux2_bfl_to_diffusers(self, sd: dict) -> dict:
        """Convert FLUX.2 BFL format state dict to diffusers format."""
        converted = {}

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

            if old_key in key_renames:
                new_key = key_renames[old_key]
                if old_key == "final_layer.adaLN_modulation.1.weight":
                    tensor = self._swap_scale_shift(tensor)
                converted[new_key] = tensor
                continue

            if old_key.startswith("double_blocks."):
                new_key = self._convert_double_block_key(old_key, tensor, converted)
                if new_key is None:
                    continue
            elif old_key.startswith("single_blocks."):
                new_key = self._convert_single_block_key(old_key, tensor, converted)
                if new_key is None:
                    continue

            if new_key != old_key or new_key not in converted:
                converted[new_key] = tensor

        return converted

    def _convert_double_block_key(self, key: str, tensor, converted: dict) -> str | None:
        parts = key.split(".")
        block_idx = parts[1]
        rest = ".".join(parts[2:])
        prefix = f"transformer_blocks.{block_idx}"

        if "img_attn.qkv.weight" in rest:
            q, k, v = self._chunk_tensor(tensor, 3)
            converted[f"{prefix}.attn.to_q.weight"] = q
            converted[f"{prefix}.attn.to_k.weight"] = k
            converted[f"{prefix}.attn.to_v.weight"] = v
            return None
        elif "txt_attn.qkv.weight" in rest:
            q, k, v = self._chunk_tensor(tensor, 3)
            converted[f"{prefix}.attn.add_q_proj.weight"] = q
            converted[f"{prefix}.attn.add_k_proj.weight"] = k
            converted[f"{prefix}.attn.add_v_proj.weight"] = v
            return None

        if "img_attn.proj.weight" in rest:
            return f"{prefix}.attn.to_out.0.weight"
        elif "txt_attn.proj.weight" in rest:
            return f"{prefix}.attn.to_add_out.weight"

        if "img_attn.norm.query_norm.scale" in rest or "img_attn.norm.query_norm.weight" in rest:
            return f"{prefix}.attn.norm_q.weight"
        elif "img_attn.norm.key_norm.scale" in rest or "img_attn.norm.key_norm.weight" in rest:
            return f"{prefix}.attn.norm_k.weight"
        elif "txt_attn.norm.query_norm.scale" in rest or "txt_attn.norm.query_norm.weight" in rest:
            return f"{prefix}.attn.norm_added_q.weight"
        elif "txt_attn.norm.key_norm.scale" in rest or "txt_attn.norm.key_norm.weight" in rest:
            return f"{prefix}.attn.norm_added_k.weight"

        if "img_mlp.0.weight" in rest:
            return f"{prefix}.ff.linear_in.weight"
        elif "img_mlp.2.weight" in rest:
            return f"{prefix}.ff.linear_out.weight"
        elif "txt_mlp.0.weight" in rest:
            return f"{prefix}.ff_context.linear_in.weight"
        elif "txt_mlp.2.weight" in rest:
            return f"{prefix}.ff_context.linear_out.weight"

        return key

    def _convert_single_block_key(self, key: str, tensor, converted: dict) -> str | None:
        parts = key.split(".")
        block_idx = parts[1]
        rest = ".".join(parts[2:])
        prefix = f"single_transformer_blocks.{block_idx}"

        if "linear1.weight" in rest:
            return f"{prefix}.attn.to_qkv_mlp_proj.weight"
        elif "linear2.weight" in rest:
            return f"{prefix}.attn.to_out.weight"

        if "norm.query_norm.scale" in rest or "norm.query_norm.weight" in rest:
            return f"{prefix}.attn.norm_q.weight"
        elif "norm.key_norm.scale" in rest or "norm.key_norm.weight" in rest:
            return f"{prefix}.attn.norm_k.weight"

        return key

    def _chunk_tensor(self, tensor, chunks: int):
        """Chunk a tensor, handling both regular tensors and GGUF quantized tensors."""
        if hasattr(tensor, "get_dequantized_tensor"):
            # GGUF quantized tensor - dequantize first, then chunk
            # This loses quantization for the split weights, but is necessary
            # because diffusers uses separate Q/K/V projections
            tensor = tensor.get_dequantized_tensor()
        return tensor.chunk(chunks, dim=0)

    def _swap_scale_shift(self, weight) -> torch.Tensor:
        """Swap scale and shift in AdaLayerNorm weights."""
        if hasattr(weight, "get_dequantized_tensor"):
            # For GGUF, dequantize first
            weight = weight.get_dequantized_tensor()
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


def _is_sdnq_folder(folder_path: Path) -> bool:
    """Check if a folder contains SDNQ-quantized model weights."""
    import json

    quant_config_path = folder_path / "quantization_config.json"
    if quant_config_path.exists():
        try:
            with open(quant_config_path, "r", encoding="utf-8") as f:
                quant_config = json.load(f)
            if quant_config.get("quant_method") == "sdnq":
                return True
        except (json.JSONDecodeError, OSError):
            pass
    return False


@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.Main, format=ModelFormat.SDNQQuantized)
class FluxSDNQDiffusersModel(ModelLoader):
    """Class to load SDNQ-quantized Flux models in diffusers format."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        print(
            f"[SDNQ] FluxSDNQDiffusersModel._load_model called with config={type(config).__name__}, submodel={submodel_type}"
        )
        # Handle single-file SDNQ checkpoint (Main_SDNQ_FLUX_Config)
        if isinstance(config, Main_SDNQ_FLUX_Config):
            if submodel_type == SubModelType.Transformer:
                return self._load_sdnq_transformer_checkpoint(config)
            raise ValueError(
                f"Only Transformer submodels are supported for checkpoint format. Received: {submodel_type}"
            )

        # Handle diffusers-format SDNQ model (Main_SDNQ_Diffusers_FLUX_Config)
        if not isinstance(config, Main_SDNQ_Diffusers_FLUX_Config):
            raise ValueError(f"Expected Main_SDNQ_Diffusers_FLUX_Config, got {type(config).__name__}")

        if submodel_type is None:
            raise ValueError("A submodel type must be provided when loading main pipelines.")

        model_path = Path(config.path)
        submodel_path = model_path / submodel_type.value

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_sdnq_transformer(submodel_path, config)
            case SubModelType.TextEncoder:
                return self._load_text_encoder(submodel_path)
            case SubModelType.TextEncoder2:
                return self._load_text_encoder_2(submodel_path)
            case SubModelType.Tokenizer:
                return CLIPTokenizer.from_pretrained(submodel_path, local_files_only=True)
            case SubModelType.Tokenizer2:
                return T5TokenizerFast.from_pretrained(submodel_path, max_length=512, local_files_only=True)
            case SubModelType.VAE:
                return self._load_vae(submodel_path)
            case _:
                raise ValueError(f"Unsupported submodel type: {submodel_type}")

    def _load_sdnq_transformer_checkpoint(self, config: Main_SDNQ_FLUX_Config) -> AnyModel:
        """Load SDNQ transformer from single-file checkpoint."""
        model_path = Path(config.path)

        with accelerate.init_empty_weights():
            model = Flux(get_flux_transformers_params(config.variant))

        sd = sdnq_sd_loader(model_path, compute_dtype=torch.bfloat16)

        # Handle ComfyUI bundle format
        if "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale" in sd:
            sd = convert_bundle_to_flux_transformer_checkpoint(sd)

        model.load_state_dict(sd, assign=True)
        return model

    def _load_sdnq_transformer(self, transformer_path: Path, config: Main_SDNQ_Diffusers_FLUX_Config) -> AnyModel:
        """Load SDNQ-quantized transformer from diffusers folder."""
        print(f"[SDNQ] _load_sdnq_transformer called for {transformer_path}")
        with accelerate.init_empty_weights():
            model = Flux(get_flux_transformers_params(config.variant))

        sd = sdnq_sd_loader(transformer_path, compute_dtype=torch.bfloat16)

        # Convert from diffusers format to BFL format
        sd = self._convert_diffusers_sd_to_bfl(sd)

        model.load_state_dict(sd, assign=True)
        return model

    def _convert_diffusers_sd_to_bfl(self, sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert a Flux transformer state dict from diffusers format to BFL format.

        Note: For SDNQTensor objects, Q/K/V tensors are dequantized before fusion since
        torch.cat doesn't work with quantized tensors. Other layers retain quantization.
        """
        from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor

        # Helper to dequantize SDNQTensor or return as-is
        def maybe_dequantize(t: torch.Tensor) -> torch.Tensor:
            if isinstance(t, SDNQTensor):
                return t.get_dequantized_tensor()
            return t

        # Helper to fuse weights, handling SDNQTensor
        def fuse_weights(*tensors: torch.Tensor) -> torch.Tensor:
            dequantized = [maybe_dequantize(t) for t in tensors]
            return torch.cat(dequantized, dim=0)

        def _swap_scale_shift_halves(t: torch.Tensor) -> torch.Tensor:
            """Swap the (scale, shift) halves along dim 0 to (shift, scale).

            diffusers' AdaLayerNormContinuous packs (scale, shift); BFL's LastLayer expects
            (shift, scale). Same memory, different interpretation — without this swap the final
            normalisation modulation is permuted and the output is high-frequency noise.
            """
            t = maybe_dequantize(t)
            if t.dim() < 1 or t.shape[0] % 2 != 0:
                return t
            scale, shift = t.chunk(2, dim=0)
            return torch.cat([shift, scale], dim=0)

        # Make a shallow copy so we can pop keys
        sd = sd.copy()
        new_sd: dict[str, torch.Tensor] = {}

        # Basic 1-to-1 key conversions
        basic_key_map = {
            # txt_in keys
            "context_embedder.bias": "txt_in.bias",
            "context_embedder.weight": "txt_in.weight",
            # guidance_in MLPEmbedder keys
            "time_text_embed.guidance_embedder.linear_1.bias": "guidance_in.in_layer.bias",
            "time_text_embed.guidance_embedder.linear_1.weight": "guidance_in.in_layer.weight",
            "time_text_embed.guidance_embedder.linear_2.bias": "guidance_in.out_layer.bias",
            "time_text_embed.guidance_embedder.linear_2.weight": "guidance_in.out_layer.weight",
            # vector_in MLPEmbedder keys
            "time_text_embed.text_embedder.linear_1.bias": "vector_in.in_layer.bias",
            "time_text_embed.text_embedder.linear_1.weight": "vector_in.in_layer.weight",
            "time_text_embed.text_embedder.linear_2.bias": "vector_in.out_layer.bias",
            "time_text_embed.text_embedder.linear_2.weight": "vector_in.out_layer.weight",
            # time_in MLPEmbedder keys
            "time_text_embed.timestep_embedder.linear_1.bias": "time_in.in_layer.bias",
            "time_text_embed.timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
            "time_text_embed.timestep_embedder.linear_2.bias": "time_in.out_layer.bias",
            "time_text_embed.timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
            # img_in keys
            "x_embedder.bias": "img_in.bias",
            "x_embedder.weight": "img_in.weight",
            # final_layer keys
            "proj_out.bias": "final_layer.linear.bias",
            "proj_out.weight": "final_layer.linear.weight",
            # norm_out.linear is the final AdaLayerNormContinuous. diffusers packs the linear
            # output as (scale, shift); BFL's LastLayer packs as (shift, scale). Swap the
            # halves of the weight and bias to keep the math correct.
            "norm_out.linear.bias": "final_layer.adaLN_modulation.1.bias",
            "norm_out.linear.weight": "final_layer.adaLN_modulation.1.weight",
        }
        # Keys whose first-axis halves (scale, shift) must be swapped to (shift, scale) for BFL.
        SWAP_SCALE_SHIFT_KEYS = {
            "norm_out.linear.bias",
            "norm_out.linear.weight",
        }
        for old_key, new_key in basic_key_map.items():
            v = sd.pop(old_key, None)
            if v is not None:
                if old_key in SWAP_SCALE_SHIFT_KEYS:
                    v = _swap_scale_shift_halves(v)
                new_sd[new_key] = v

        # Handle the double_blocks (19 blocks for FLUX)
        block_index = 0
        while f"transformer_blocks.{block_index}.attn.add_q_proj.bias" in sd:
            from_prefix = f"transformer_blocks.{block_index}"
            to_prefix = f"double_blocks.{block_index}"

            # txt_attn.qkv (fuse add_q, add_k, add_v)
            new_sd[f"{to_prefix}.txt_attn.qkv.bias"] = fuse_weights(
                sd.pop(f"{from_prefix}.attn.add_q_proj.bias"),
                sd.pop(f"{from_prefix}.attn.add_k_proj.bias"),
                sd.pop(f"{from_prefix}.attn.add_v_proj.bias"),
            )
            new_sd[f"{to_prefix}.txt_attn.qkv.weight"] = fuse_weights(
                sd.pop(f"{from_prefix}.attn.add_q_proj.weight"),
                sd.pop(f"{from_prefix}.attn.add_k_proj.weight"),
                sd.pop(f"{from_prefix}.attn.add_v_proj.weight"),
            )

            # img_attn.qkv (fuse to_q, to_k, to_v)
            new_sd[f"{to_prefix}.img_attn.qkv.bias"] = fuse_weights(
                sd.pop(f"{from_prefix}.attn.to_q.bias"),
                sd.pop(f"{from_prefix}.attn.to_k.bias"),
                sd.pop(f"{from_prefix}.attn.to_v.bias"),
            )
            new_sd[f"{to_prefix}.img_attn.qkv.weight"] = fuse_weights(
                sd.pop(f"{from_prefix}.attn.to_q.weight"),
                sd.pop(f"{from_prefix}.attn.to_k.weight"),
                sd.pop(f"{from_prefix}.attn.to_v.weight"),
            )

            # 1-to-1 key mappings for double block
            double_block_key_map = {
                # img_attn
                "attn.norm_k.weight": "img_attn.norm.key_norm.scale",
                "attn.norm_q.weight": "img_attn.norm.query_norm.scale",
                "attn.to_out.0.weight": "img_attn.proj.weight",
                "attn.to_out.0.bias": "img_attn.proj.bias",
                # img_mlp
                "ff.net.0.proj.weight": "img_mlp.0.weight",
                "ff.net.0.proj.bias": "img_mlp.0.bias",
                "ff.net.2.weight": "img_mlp.2.weight",
                "ff.net.2.bias": "img_mlp.2.bias",
                # img_mod
                "norm1.linear.weight": "img_mod.lin.weight",
                "norm1.linear.bias": "img_mod.lin.bias",
                # txt_attn
                "attn.norm_added_q.weight": "txt_attn.norm.query_norm.scale",
                "attn.norm_added_k.weight": "txt_attn.norm.key_norm.scale",
                "attn.to_add_out.weight": "txt_attn.proj.weight",
                "attn.to_add_out.bias": "txt_attn.proj.bias",
                # txt_mlp
                "ff_context.net.0.proj.weight": "txt_mlp.0.weight",
                "ff_context.net.0.proj.bias": "txt_mlp.0.bias",
                "ff_context.net.2.weight": "txt_mlp.2.weight",
                "ff_context.net.2.bias": "txt_mlp.2.bias",
                # txt_mod
                "norm1_context.linear.weight": "txt_mod.lin.weight",
                "norm1_context.linear.bias": "txt_mod.lin.bias",
            }
            for from_key, to_key in double_block_key_map.items():
                v = sd.pop(f"{from_prefix}.{from_key}", None)
                if v is not None:
                    new_sd[f"{to_prefix}.{to_key}"] = v

            block_index += 1

        # Handle the single_blocks (38 blocks for FLUX)
        block_index = 0
        while f"single_transformer_blocks.{block_index}.attn.to_q.bias" in sd:
            from_prefix = f"single_transformer_blocks.{block_index}"
            to_prefix = f"single_blocks.{block_index}"

            # linear1 (fuse to_q, to_k, to_v, proj_mlp)
            new_sd[f"{to_prefix}.linear1.bias"] = fuse_weights(
                sd.pop(f"{from_prefix}.attn.to_q.bias"),
                sd.pop(f"{from_prefix}.attn.to_k.bias"),
                sd.pop(f"{from_prefix}.attn.to_v.bias"),
                sd.pop(f"{from_prefix}.proj_mlp.bias"),
            )
            new_sd[f"{to_prefix}.linear1.weight"] = fuse_weights(
                sd.pop(f"{from_prefix}.attn.to_q.weight"),
                sd.pop(f"{from_prefix}.attn.to_k.weight"),
                sd.pop(f"{from_prefix}.attn.to_v.weight"),
                sd.pop(f"{from_prefix}.proj_mlp.weight"),
            )

            # 1-to-1 key mappings for single block
            single_block_key_map = {
                # linear2
                "proj_out.weight": "linear2.weight",
                "proj_out.bias": "linear2.bias",
                # modulation
                "norm.linear.weight": "modulation.lin.weight",
                "norm.linear.bias": "modulation.lin.bias",
                # norm
                "attn.norm_k.weight": "norm.key_norm.scale",
                "attn.norm_q.weight": "norm.query_norm.scale",
            }
            for from_key, to_key in single_block_key_map.items():
                v = sd.pop(f"{from_prefix}.{from_key}", None)
                if v is not None:
                    new_sd[f"{to_prefix}.{to_key}"] = v

            block_index += 1

        # Any remaining keys that weren't converted - just pass through
        for k, v in sd.items():
            if k not in new_sd:
                new_sd[k] = v

        return new_sd

    def _load_text_encoder(self, text_encoder_path: Path) -> AnyModel:
        """Load text encoder (CLIP) - SDNQ or normal."""
        if _is_sdnq_folder(text_encoder_path):
            # SDNQ CLIP - need custom loading
            return self._load_sdnq_clip(text_encoder_path)
        # Normal CLIP
        return CLIPTextModel.from_pretrained(text_encoder_path, local_files_only=True)

    def _load_text_encoder_2(self, text_encoder_path: Path) -> AnyModel:
        """Load text encoder 2 (T5) - SDNQ or normal."""
        if _is_sdnq_folder(text_encoder_path):
            # SDNQ T5 - need custom loading
            return self._load_sdnq_t5(text_encoder_path)
        # Normal T5
        return T5EncoderModel.from_pretrained(
            text_encoder_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )

    def _load_sdnq_clip(self, clip_path: Path) -> AnyModel:
        """Load SDNQ-quantized CLIP text encoder."""
        from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor

        # Load SDNQ state dict
        sd = sdnq_sd_loader(clip_path, compute_dtype=torch.bfloat16)

        # Load config and create model
        model_config = AutoConfig.from_pretrained(clip_path, local_files_only=True)
        with accelerate.init_empty_weights():
            model = CLIPTextModel(model_config)

        model.load_state_dict(sd, strict=False, assign=True)

        # Dequantize embedding layer
        if hasattr(model, "text_model") and hasattr(model.text_model, "embeddings"):
            embed_weight = model.text_model.embeddings.token_embedding.weight
            if isinstance(embed_weight, SDNQTensor):
                dequantized = embed_weight.get_dequantized_tensor()
                model.text_model.embeddings.token_embedding.weight = torch.nn.Parameter(
                    dequantized, requires_grad=False
                )

        return model

    def _load_sdnq_t5(self, t5_path: Path) -> AnyModel:
        """Load SDNQ-quantized T5 text encoder."""
        from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor

        # Load SDNQ state dict
        sd = sdnq_sd_loader(t5_path, compute_dtype=torch.bfloat16)

        # Load config and create model
        model_config = AutoConfig.from_pretrained(t5_path, local_files_only=True)
        with accelerate.init_empty_weights():
            model = AutoModelForTextEncoding.from_config(model_config)

        model.load_state_dict(sd, strict=False, assign=True)

        # Dequantize shared embedding
        if hasattr(model, "shared") and isinstance(model.shared.weight, SDNQTensor):
            dequantized = model.shared.weight.get_dequantized_tensor()
            model.shared.weight = torch.nn.Parameter(dequantized, requires_grad=False)

        # Re-tie weights after dequantization
        if hasattr(model, "encoder") and hasattr(model.encoder, "embed_tokens"):
            if model.encoder.embed_tokens.weight is not model.shared.weight:
                model.encoder.embed_tokens.weight = model.shared.weight

        return model

    def _load_vae(self, vae_path: Path) -> AnyModel:
        """Load VAE - SDNQ or normal."""
        if _is_sdnq_folder(vae_path):
            return self._load_sdnq_vae(vae_path)
        # Normal VAE
        return AutoencoderKL.from_pretrained(vae_path, local_files_only=True)

    def _load_sdnq_vae(self, vae_path: Path) -> AnyModel:
        """Load SDNQ-quantized VAE."""
        # Load SDNQ state dict
        sd = sdnq_sd_loader(vae_path, compute_dtype=torch.bfloat16)

        # Load config and create model
        with accelerate.init_empty_weights():
            model = AutoencoderKL.from_config(AutoencoderKL.load_config(vae_path, local_files_only=True))

        model.load_state_dict(sd, strict=False, assign=True)
        return model
