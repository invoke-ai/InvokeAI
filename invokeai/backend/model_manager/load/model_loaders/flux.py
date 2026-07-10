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
)
from invokeai.backend.model_manager.configs.t5_encoder import T5Encoder_BnBLLMint8_Config, T5Encoder_T5Encoder_Config
from invokeai.backend.model_manager.configs.vae import VAE_Checkpoint_Config_Base, VAE_Checkpoint_Flux2_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.flux2_state_dict_utils import (
    convert_flux2_bfl_to_diffusers,
    convert_flux2_vae_bfl_to_diffusers,
)
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
            sd = convert_flux2_vae_bfl_to_diffusers(sd)

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
        converted_sd = convert_flux2_bfl_to_diffusers(sd)

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

                # Do the multiply in float32 for precision, but store bf16 (FLUX.2's compute dtype)
                # immediately so the *whole* model is never materialized in float32. Holding every
                # dequantized weight as float32 here doubled RAM transiently (~36GB vs ~17GB for a 9B
                # model) and was the dominant cold-load spike, especially with two GPUs. The result is
                # identical to the previous code, which cast the same values to bf16 a few steps later.
                sd[weight_key] = (weight_float * scale).to(torch.bfloat16)
                del weight_float

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
            # Convert native FP8 tensors straight to bf16 (FLUX.2's compute dtype) rather than float32,
            # so a cold load never transiently holds the whole model in float32 (see the scaled path).
            sd[key] = sd[key].to(torch.bfloat16)

        return sd


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
        converted_sd = convert_flux2_bfl_to_diffusers(sd)

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
