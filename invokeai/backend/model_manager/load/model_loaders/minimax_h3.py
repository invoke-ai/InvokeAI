"""Loader registrations for MiniMax H3 (Hailuo 3.0) audio-video generation models.

Currently covers the diffusers-format Modular Diffusers layout only (the layout at the root of
the ``MiniMaxAI/MiniMax-H3`` HF repo). The model classes are vendored under
``invokeai.backend.minimax_h3`` because they only exist in an unreleased diffusers branch, so this
loader dispatches submodels explicitly instead of subclassing ``GenericDiffusersLoader`` (whose
``get_hf_load_class`` resolves class names against the installed diffusers and would fail).

Submodel map (subfolder == ``SubModelType`` value):
- ``transformer``  -> vendored ``MiniMaxH3Transformer3DModel`` (33B DiT, bf16)
- ``text_encoder`` -> ``transformers.Qwen3VLForConditionalGeneration`` (Qwen3-VL-32B, bf16; H3
  conditions on layer-50 hidden states and never uses the LM head, but the checkpoint is the full
  model per the repo's modular_model_index.json)
- ``tokenizer``    -> ``AutoTokenizer`` (Qwen2TokenizerFast)
- ``processor``    -> ``AutoProcessor`` (Qwen3VLProcessor; needed even for text-only encoding,
  which uses its multimodal token-type ids)
- ``vae``          -> vendored ``AutoencoderKLMiniMaxH3`` (video VAE, bf16)
- ``audio_vae``    -> vendored ``AutoencoderKLMiniMaxH3Audio`` (kept fp32: it is ~0.6 GB and
  half-precision artifacts in decoded audio are audible)

The two ``MiniMaxH3Scheduler`` instances (video shift 12.0, audio shift 3.0) are constructed
directly by the denoise invocation - they are stateless configs, not loaded weights.
"""

from pathlib import Path
from typing import Optional

import torch

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_MiniMaxH3_Config,
    Main_Diffusers_MiniMaxH3_Config,
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
from invokeai.backend.model_manager.util.qwen3_vl import normalize_qwen3vl_rope_config
from invokeai.backend.util.devices import TorchDevice


def _raise_if_no_weight_shards(submodel_path: Path, submodel_label: str) -> None:
    """Fail with an actionable message when a components-only (slim) install is asked for a
    submodel whose weights it does not carry.

    A slim MiniMax H3 install ships the shared components (tokenizer, processor, VAEs) plus bare
    config JSONs, so a generic from_pretrained() here would otherwise die on a cryptic
    missing-shard error deep inside diffusers/transformers.
    """
    if not any(any(submodel_path.glob(pattern)) for pattern in ("*.safetensors", "*.bin", "*.pth", "*.pt", "*.ckpt")):
        raise ValueError(
            f"This MiniMax H3 model folder has no {submodel_label} weights - it is a components-only "
            f"(slim) install. In the MiniMax H3 Model Loader, select a single-file {submodel_label} "
            f"(for example the Comfy-Org int8 release) to use with it."
        )


@ModelLoaderRegistry.register(base=BaseModelType.MiniMaxH3, type=ModelType.Main, format=ModelFormat.Diffusers)
class MiniMaxH3DiffusersModel(ModelLoader):
    """Loader for MiniMax H3 diffusers-format models (FL2VA)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Main_Diffusers_MiniMaxH3_Config):
            raise ValueError(f"Unexpected config type {type(config).__name__} for a MiniMax H3 loader.")
        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading MiniMax H3 main models.")

        model_path = Path(config.path)
        submodel_path = model_path / submodel_type.value

        target_device = TorchDevice.choose_torch_device()
        dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        match submodel_type:
            case SubModelType.Transformer:
                _raise_if_no_weight_shards(submodel_path, "transformer")

                from invokeai.backend.minimax_h3 import MiniMaxH3Transformer3DModel

                return MiniMaxH3Transformer3DModel.from_pretrained(
                    submodel_path, torch_dtype=dtype, local_files_only=True
                )
            case SubModelType.TextEncoder:
                _raise_if_no_weight_shards(submodel_path, "text encoder")

                from transformers import AutoConfig, Qwen3VLForConditionalGeneration

                te_config = normalize_qwen3vl_rope_config(
                    AutoConfig.from_pretrained(submodel_path, local_files_only=True)
                )
                return Qwen3VLForConditionalGeneration.from_pretrained(
                    submodel_path,
                    config=te_config,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
            case SubModelType.Tokenizer:
                from transformers import AutoTokenizer

                return AutoTokenizer.from_pretrained(submodel_path, local_files_only=True)
            case SubModelType.Processor:
                from transformers import AutoProcessor

                return AutoProcessor.from_pretrained(submodel_path, local_files_only=True)
            case SubModelType.VAE:
                from invokeai.backend.minimax_h3 import AutoencoderKLMiniMaxH3
                from invokeai.backend.minimax_h3.rocm_causal_conv3d import patch_minimax_h3_causal_conv3d_for_rocm

                # The convolutional encoder hits MIOpen's Im3d2Col conv3d fallback on ROCm
                # (~48x, same failure as the Wan VAE); decompose to per-temporal-tap conv2d.
                # The ViT decoder is unaffected either way.
                patch_minimax_h3_causal_conv3d_for_rocm()
                return AutoencoderKLMiniMaxH3.from_pretrained(submodel_path, torch_dtype=dtype, local_files_only=True)
            case SubModelType.AudioVAE:
                from invokeai.backend.minimax_h3 import AutoencoderKLMiniMaxH3Audio

                return AutoencoderKLMiniMaxH3Audio.from_pretrained(
                    submodel_path, torch_dtype=torch.float32, local_files_only=True
                )
            case _:
                raise ValueError(f"Unsupported submodel type {submodel_type} for MiniMax H3 models.")


@ModelLoaderRegistry.register(base=BaseModelType.MiniMaxH3, type=ModelType.Main, format=ModelFormat.Checkpoint)
class MiniMaxH3CheckpointModel(ModelLoader):
    """Loader for MiniMax H3 single-file transformer checkpoints (bf16 or Comfy int8-convrot,
    full or AdaLN-pruned).

    The file holds ONLY the transformer. Quantized linears keep their int8 weights + per-channel
    scales resident (`Int8ConvrotLinear` dequantizes and derotates per forward call); everything
    else loads at the dtype stored in the file (the fp32 patch projections / output heads and the
    bf16 block stack mirror the diffusers folder's mixed-precision islands).
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Main_Checkpoint_MiniMaxH3_Config):
            raise ValueError(f"Unexpected config type {type(config).__name__} for a MiniMax H3 checkpoint loader.")
        if submodel_type is not SubModelType.Transformer:
            raise ValueError(
                "MiniMax H3 single-file checkpoints contain only the transformer. Install the MiniMax H3 "
                "diffusers folder and select it as the main model; the single file can then be used as the "
                f"transformer override. Received: {submodel_type.value if submodel_type else 'None'}"
            )
        return self._load_transformer_from_singlefile(config)

    def _load_transformer_from_singlefile(self, config: Main_Checkpoint_MiniMaxH3_Config) -> AnyModel:
        import accelerate
        from safetensors.torch import load_file

        from invokeai.backend.minimax_h3 import MiniMaxH3Transformer3DModel
        from invokeai.backend.minimax_h3.int8_convrot import Int8ConvrotLinear
        from invokeai.backend.minimax_h3.transformer_minimax_h3_pruned import (
            MiniMaxH3PrunedTransformer3DModel,
            set_curve_modulation_dtype,
        )
        from invokeai.backend.model_manager.load.model_loaders.minimax_h3_state_dict_utils import (
            convert_minimax_h3_checkpoint_to_diffusers,
            read_comfy_quant_markers,
        )

        model_path = Path(config.path)

        # Reject unsupported quantization formats from the header alone, before committing to
        # the ~20 GiB tensor read (the fp8_scaled repacks share this key layout).
        unsupported = sorted(
            {
                str(marker.get("format"))
                for marker in read_comfy_quant_markers(model_path).values()
                if marker.get("format") != "int8_tensorwise"
            }
        )
        if unsupported:
            raise ValueError(
                f"Unsupported quantization format(s) {unsupported} in MiniMax H3 checkpoint {model_path.name}. "
                "Only unquantized and Comfy 'int8_tensorwise' (int8/int8-convrot) single files are supported."
            )

        sd = load_file(model_path)
        rope_freq_dim = sd["rope.inv_freq"].shape[0] if "rope.inv_freq" in sd else 16
        sd, quant_markers = convert_minimax_h3_checkpoint_to_diffusers(sd)

        self._ram_cache.make_room(sum(t.nelement() * t.element_size() for t in sd.values()))

        # The AdaLN-curve projections are stored f16 but run float32 (see the pruned class).
        if config.pruned:
            for key in list(sd.keys()):
                if ".adaln_proj.linear." in key or key.startswith("norm_out.linear."):
                    sd[key] = sd[key].to(torch.float32)

        # Architecture from tensor shapes; patch_size is not recoverable from shapes and is
        # constant across every released H3 checkpoint.
        model_params = {
            "hidden_size": sd["transformer_blocks.0.norm1.weight"].shape[0],
            "attention_head_dim": sd["transformer_blocks.0.attn.norm_q.weight"].shape[0],
            "num_attention_heads": sd["transformer_blocks.0.attn.to_q.weight"].shape[0]
            // sd["transformer_blocks.0.attn.norm_q.weight"].shape[0],
            "ffn_dim": sd["transformer_blocks.0.ff.net.0.proj.weight"].shape[0] // 2,
            "num_layers": 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("transformer_blocks.")),
            "num_refiner_layers": 1
            + max(int(k.split(".")[2]) for k in sd if k.startswith("token_refiner.refiner_blocks.")),
            "text_dim": sd["context_embedder.weight"].shape[1],
            "audio_in_channels": sd["audio_proj_in.weight"].shape[1],
            "in_channels": sd["proj_in.weight"].shape[1] // 4,
            "patch_size": (1, 2, 2),
            "rope_freq_dim": rope_freq_dim,
        }

        with accelerate.init_empty_weights():
            if config.pruned:
                grid, curve_dim = sd["adaln_t_table"].shape
                model = MiniMaxH3PrunedTransformer3DModel(
                    **model_params, adaln_curve_grid=grid, adaln_curve_dim=curve_dim
                )
            else:
                model = MiniMaxH3Transformer3DModel(**model_params)

        # Swap each quantized nn.Linear for an Int8ConvrotLinear sized from the state dict. Its
        # persistent buffers are named `weight`/`weight_scale`, so the strict load below consumes
        # the quantized tensors directly.
        for module_name, marker in quant_markers.items():
            if marker.get("format") != "int8_tensorwise":
                raise ValueError(f"Unsupported comfy_quant format {marker!r} on {module_name}")
            parent = model.get_submodule(module_name.rsplit(".", 1)[0])
            setattr(
                parent,
                module_name.rsplit(".", 1)[1],
                Int8ConvrotLinear(
                    weight=sd[module_name + ".weight"],
                    weight_scale=sd[module_name + ".weight_scale"],
                    convrot=bool(marker.get("convrot", False)),
                ),
            )

        model.load_state_dict(sd, strict=True, assign=True)

        if isinstance(model, MiniMaxH3PrunedTransformer3DModel):
            # Match the modulation dtype to the LOADED block stack (bf16 from the file), not the
            # device's preferred compute dtype: on fp16-fallback devices the latter would emit
            # fp16 modulation into a bf16 stream and promote every block's activations to fp32.
            set_curve_modulation_dtype(model, model.context_embedder.weight.dtype)

        return model


@ModelLoaderRegistry.register(
    base=BaseModelType.MiniMaxH3, type=ModelType.Qwen3VLEncoder, format=ModelFormat.Checkpoint
)
class MiniMaxH3TextEncoderCheckpointModel(ModelLoader):
    """Loader for MiniMax H3's truncated Qwen3-VL-32B conditioning-encoder single files
    (Comfy-Org ``qwen3vl_32b_minimax_h3_bf16/int8_convrot`` and mirrors).

    The file holds a 50-layer truncation of the language stack (H3 conditions on the
    UNNORMALIZED hidden state after layer 50 and never runs the final norm or LM head, so
    neither is stored) plus the full bf16 vision tower. Quantized files keep their int8
    weights + per-channel scales resident (``Int8ConvrotLinear``); the tokenizer and processor
    still come from the H3 diffusers folder selected as the main model.

    Like the H3 transformer checkpoints, quantized layers are not autocast-wrapped, so
    residency is all-or-nothing: the int8 file needs ~26 GiB free VRAM while encoding (it
    idle-offloads afterwards); graceful partial-load degradation is the folder encoder's job.
    If partial load does engage, the tied lm_head/embed_tokens alias additionally splits into
    two device tensors (~1.56 GB overhead) - see the tie_weights() note in the load path.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        from invokeai.backend.model_manager.configs.qwen3_vl_encoder import (
            Qwen3VLEncoder_Checkpoint_MiniMaxH3_Config,
        )

        if not isinstance(config, Qwen3VLEncoder_Checkpoint_MiniMaxH3_Config):
            raise ValueError(f"Unexpected config type {type(config).__name__} for a MiniMax H3 encoder loader.")
        if submodel_type is not SubModelType.TextEncoder:
            raise ValueError(
                "MiniMax H3 single-file encoders contain only the text-encoder weights. The tokenizer and "
                "processor come from the MiniMax H3 diffusers folder selected as the main model. "
                f"Received: {submodel_type.value if submodel_type else 'None'}"
            )
        return self._load_text_encoder_from_singlefile(config)

    def _load_text_encoder_from_singlefile(self, config: AnyModelConfig) -> AnyModel:
        import json

        import accelerate
        from safetensors.torch import load_file
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

        from invokeai.backend.minimax_h3.int8_convrot import Int8ConvrotLinear
        from invokeai.backend.minimax_h3.text_conditioning import MINIMAX_H3_TEXT_ENCODER_LAYER
        from invokeai.backend.model_manager.load.model_loaders.minimax_h3_state_dict_utils import (
            convert_minimax_h3_text_encoder_checkpoint,
            read_comfy_quant_markers,
        )

        model_path = Path(config.path)

        # Reject unsupported quantization formats from the header alone, before the ~25 GiB
        # tensor read (the nvfp4_awq repacks share this key layout).
        unsupported = sorted(
            {
                str(marker.get("format"))
                for marker in read_comfy_quant_markers(model_path).values()
                if marker.get("format") != "int8_tensorwise"
            }
        )
        if unsupported:
            raise ValueError(
                f"Unsupported quantization format(s) {unsupported} in MiniMax H3 text encoder "
                f"{model_path.name}. Only unquantized and Comfy 'int8_tensorwise' (int8/int8-convrot) "
                "single files are supported."
            )

        sd = load_file(model_path)
        sd, quant_markers = convert_minimax_h3_text_encoder_checkpoint(sd)

        self._ram_cache.make_room(sum(t.nelement() * t.element_size() for t in sd.values()))

        layer_prefix = "model.language_model.layers."
        num_layers = 1 + max(int(k.removeprefix(layer_prefix).split(".")[0]) for k in sd if k.startswith(layer_prefix))
        if num_layers != MINIMAX_H3_TEXT_ENCODER_LAYER:
            raise ValueError(
                f"MiniMax H3 conditions on the unnormalized hidden state after layer "
                f"{MINIMAX_H3_TEXT_ENCODER_LAYER}; this file has {num_layers} layers. Only the H3-truncated "
                f"{MINIMAX_H3_TEXT_ENCODER_LAYER}-layer conditioning encoders are supported."
            )

        # The bundled config is the Apache-2.0 Qwen3-VL-32B config from the H3 repo, truncated to
        # the stored layer count. tie_word_embeddings avoids materializing a dead 1.5 GB LM head
        # (H3 never runs it - conditioning is hidden states only).
        bundled_config = (
            Path(__file__).parents[4] / "backend" / "minimax_h3" / "qwen3vl_32b_h3_text_encoder_config.json"
        )
        config_dict = json.loads(bundled_config.read_text(encoding="utf-8"))
        config_dict["text_config"]["num_hidden_layers"] = num_layers
        config_dict["tie_word_embeddings"] = True
        te_config = normalize_qwen3vl_rope_config(Qwen3VLConfig.from_dict(config_dict))

        with accelerate.init_empty_weights():
            model = Qwen3VLForConditionalGeneration._from_config(te_config)

        # The file stores no final norm: the conditioning contract is the UNNORMALIZED hidden
        # state after layer 50. With the truncated stack, transformers appends the post-"norm"
        # output as hidden_states[num_layers] - an Identity norm makes that entry exactly the
        # unnormalized tensor the folder model exposes at the same index. encode_prompt() checks
        # for this Identity before accepting a truncated encoder.
        model.model.language_model.norm = torch.nn.Identity()

        for module_name, marker in quant_markers.items():
            if marker.get("format") != "int8_tensorwise":
                raise ValueError(f"Unsupported comfy_quant format {marker!r} on {module_name}")
            parent = model.get_submodule(module_name.rsplit(".", 1)[0])
            setattr(
                parent,
                module_name.rsplit(".", 1)[1],
                Int8ConvrotLinear(
                    weight=sd[module_name + ".weight"],
                    weight_scale=sd[module_name + ".weight_scale"],
                    convrot=bool(marker.get("convrot", False)),
                ),
            )

        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        if unexpected:
            raise RuntimeError(f"Unexpected keys loading MiniMax H3 text encoder: {sorted(unexpected)[:5]}...")
        if set(missing) != {"lm_head.weight"}:
            raise RuntimeError(
                f"Missing keys loading MiniMax H3 text encoder (expected only the tied lm_head): {sorted(missing)[:5]}"
            )
        # Re-tie now that embed_tokens holds the loaded tensor (assign=True replaced the meta
        # parameter the original tie pointed at). The head is never run; tying keeps the module
        # free of meta tensors and adds no RAM (aliased storage). Caveat: the model cache's
        # partial-load path moves state-dict keys independently (no data_ptr dedupe), so on that
        # path the tied pair splits into two device tensors (~1.56 GB extra VRAM, double-counted
        # in the cache's accounting). Fully-resident loads - the intended regime - keep the alias.
        model.tie_weights()

        return model
