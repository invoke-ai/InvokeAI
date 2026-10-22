"""Loaders for Qwen-Image-2.1 main models: diffusers pipelines, single-file and GGUF transformers."""

import json
import math
from pathlib import Path
from typing import Any, Optional

import accelerate
import torch
from transformers import AutoTokenizer

from invokeai.backend.model_manager.checkpoint_prefix import CheckpointPrefix
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_QwenImage21_Config,
    Main_GGUF_QwenImage21_Config,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader, _model_declared_skip_patterns
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.model_manager.util.qwen3_vl import drop_qwen3vl_visual_tower
from invokeai.backend.quantization.fp8_scaled import (
    attach_fp8_scales,
    cast_state_dict,
    dequantize_fp8_scaled,
    extract_comfy_quant_hints,
    extract_fp8_scaled_layers,
    parse_quantization_metadata,
    read_safetensors_metadata,
    split_fp8_scaled_layers,
    strip_layer_path_prefix,
    warn_on_unattached_scales,
)
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader, unpack_ggml_at_load
from invokeai.backend.quantization.int8_convrot import (
    drop_unconsumed_quantization_sidecars,
    extract_int8_convrot_markers,
    install_int8_convrot_layers,
    reject_unmarked_int8_weights,
    resolve_quantized_module_paths,
)
from invokeai.backend.quantization.load_plan import reserve_for_load
from invokeai.backend.quantization.nvfp4 import pop_nvfp4_layers
from invokeai.backend.qwen_image_2_1.checkpoint_layout import (
    count_transformer_blocks,
    split_fused_gate_up,
    split_fused_layer_names,
)
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.state_dict_loading import load_state_dict_ignoring_extras, reject_incomplete_load


def _build_transformer(sd: dict[str, Any]) -> torch.nn.Module:
    """An empty transformer as deep as the checkpoint; every other hyperparameter is the released model's."""
    from diffusers import QwenImage21Transformer2DModel

    with accelerate.init_empty_weights():
        return QwenImage21Transformer2DModel(num_layers=count_transformer_blocks(sd))


@ModelLoaderRegistry.register(base=BaseModelType.QwenImage21, type=ModelType.Main, format=ModelFormat.Diffusers)
class QwenImage21DiffusersModel(GenericDiffusersLoader):
    """Loads the submodels of a Qwen-Image-2.1 diffusers pipeline.

    The tokenizer lives in `processor/` (the pipeline declares a `Qwen3VLProcessor`); the processor itself, which
    also cuts reference images into patches, loads only for an edit. The text encoder loads as `Qwen3VLModel`
    rather than the declared `Qwen3VLForConditionalGeneration`: the LM head is ~1.2 GiB the encoder never runs,
    and `Qwen3VLModel` is what the standalone Qwen3-VL loaders return, so the text encoder node sees one class.

    `TextEncoder` drops the vision tower, as the standalone loaders do: text prompts never reach it.
    `VisionTower` is that tower alone (~1.1 GiB), read from the same folder for edits, which attach it to the
    cached encoder while they encode. The language model's weights are cached once for both.
    """

    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        if submodel_type is SubModelType.VisionTower:
            # A few hundred tensors of the encoder's folder, not the folder: reserving its ~17 GB for the tower
            # would evict models for nothing. Sized at the dtype they load in, whatever the file stores.
            element_size = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device()).itemsize
            return sum(
                math.prod(shape) * element_size
                for _shard, _key, shape in _vision_tower_tensors(model_path / SubModelType.TextEncoder.value)
            )
        return super().get_size_fs(config, model_path, submodel_type)

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        if isinstance(config, Checkpoint_Config_Base):
            raise NotImplementedError(
                "CheckpointConfigBase is not implemented for the Qwen-Image-2.1 diffusers loader."
            )
        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading main pipelines.")

        model_path = Path(config.path)
        if submodel_type is SubModelType.Tokenizer:
            return AutoTokenizer.from_pretrained(model_path / "processor", local_files_only=True)
        if submodel_type is SubModelType.Processor:
            from transformers import AutoProcessor

            return AutoProcessor.from_pretrained(model_path / "processor", local_files_only=True)

        if submodel_type is SubModelType.VAE:
            # The configured precision, except float16, which is unstable on this autoencoder: bfloat16 where the
            # device has it, float32 otherwise.
            dtype = self._torch_dtype_avoiding_float16()
        else:
            dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())

        if submodel_type is SubModelType.VisionTower:
            tower = _load_vision_tower(model_path / SubModelType.TextEncoder.value, dtype)
            return self._apply_fp8_layerwise_casting(tower, config, submodel_type)

        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
        variant = repo_variant.value if repo_variant else None
        if submodel_type is SubModelType.TextEncoder:
            from transformers import Qwen3VLModel

            load_class: Any = Qwen3VLModel
        else:
            load_class = self.get_hf_load_class(model_path, submodel_type)

        try:
            result: AnyModel = load_class.from_pretrained(
                model_path / submodel_type.value, torch_dtype=dtype, variant=variant, local_files_only=True
            )
        except OSError as e:
            if variant and "no file named" in str(e):
                # The user's repo-variant preference may have changed since install.
                result = load_class.from_pretrained(
                    model_path / submodel_type.value, torch_dtype=dtype, local_files_only=True
                )
            else:
                raise

        if submodel_type is SubModelType.TextEncoder:
            # Text prompts never reach the vision tower; edits load it as `VisionTower` and attach it while they
            # encode.
            drop_qwen3vl_visual_tower(result)

        return self._apply_fp8_layerwise_casting(result, config, submodel_type)


# The tower's tensors in an encoder folder: `model.visual.*` as `Qwen3VLForConditionalGeneration` saves them (the
# released pipeline), `visual.*` as `Qwen3VLModel` does.
_VISION_TOWER_PREFIXES = ("model.visual.", "visual.")


def _vision_tower_tensors(folder: Path) -> list[tuple[Path, str, list[int]]]:
    """(shard, key, shape) of every vision-tower tensor in the folder, read from the headers.

    The shard index names the shards that hold them; without an index, every safetensors file in the folder.
    """
    from safetensors import safe_open

    index = folder / "model.safetensors.index.json"
    if index.exists():
        weight_map: dict[str, str] = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted({folder / shard for key, shard in weight_map.items() if key.startswith(_VISION_TOWER_PREFIXES)})
    else:
        shards = sorted(folder.glob("*.safetensors"))
    tensors = []
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(_VISION_TOWER_PREFIXES):
                    tensors.append((shard, key, f.get_slice(key).get_shape()))
    if not tensors:
        raise ValueError(f"{folder} holds no Qwen3-VL vision tower ({' or '.join(_VISION_TOWER_PREFIXES)} tensors).")
    return tensors


def _load_vision_tower(folder: Path, dtype: torch.dtype) -> torch.nn.Module:
    """The encoder's vision tower alone, as `Qwen3VLModel.visual` would hold it, without reading the language model."""
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    vision_config = AutoConfig.from_pretrained(folder, local_files_only=True).vision_config
    with accelerate.init_empty_weights():
        tower = Qwen3VLVisionModel._from_config(vision_config, torch_dtype=dtype)

    state_dict: dict[str, torch.Tensor] = {}
    tensors = _vision_tower_tensors(folder)
    for shard in sorted({shard for shard, *_ in tensors}):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for _shard, key, _shape in (t for t in tensors if t[0] == shard):
                prefix = next(p for p in _VISION_TOWER_PREFIXES if key.startswith(p))
                state_dict[key[len(prefix) :]] = f.get_tensor(key).to(dtype)
    load_state_dict_ignoring_extras(tower, state_dict, source=f"Qwen3-VL vision tower in {folder}", assign=True)
    reject_incomplete_load(tower, what="Qwen3-VL vision tower")
    return tower.eval()


@ModelLoaderRegistry.register(base=BaseModelType.QwenImage21, type=ModelType.Main, format=ModelFormat.Checkpoint)
class QwenImage21CheckpointModel(ModelLoader):
    """Loads a Qwen-Image-2.1 transformer from a single safetensors file.

    Plain bf16/fp16 files, ComfyUI 'scaled fp8' (fp8 weights + `weight_scale`) and ComfyUI `int8_tensorwise`
    (+convrot) files, with or without ComfyUI's prefix. The fused `img_mlp.gate_up` is split before any
    quantization side channel is read, so every quantized layer names a diffusers Linear.
    """

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        if not isinstance(config, Main_Checkpoint_QwenImage21_Config):
            raise TypeError(f"Expected Main_Checkpoint_QwenImage21_Config, got {type(config).__name__}.")
        if submodel_type is not SubModelType.Transformer:
            raise ValueError(f"Only the transformer is in a single file. Received: {submodel_type}")

        from safetensors.torch import load_file

        model_path = Path(config.path)
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())

        sd = load_file(model_path)
        sd = CheckpointPrefix.detect(sd).strip(sd)
        header_layers = split_fused_layer_names(
            strip_layer_path_prefix(parse_quantization_metadata(read_safetensors_metadata(model_path, self._logger)))
        )
        if pop_nvfp4_layers(dict(sd), header_layers=header_layers):
            raise ValueError(
                f"{model_path.name} is an nvfp4 checkpoint, which Qwen-Image-2.1 does not load yet. "
                "Use the bf16, fp8 or int8 build, or a GGUF."
            )
        sd = split_fused_gate_up(sd)

        int8_markers = extract_int8_convrot_markers(sd)
        reject_unmarked_int8_weights(sd, int8_markers, "Qwen-Image-2.1")
        fp8_layers: dict[str, Any] = {}
        keep_fp8 = False
        if int8_markers:
            sd = drop_unconsumed_quantization_sidecars(sd)
        else:
            layer_hints = {**extract_comfy_quant_hints(sd), **header_layers}
            fp8_layers = extract_fp8_scaled_layers(sd, layer_hints=layer_hints)
            keep_fp8 = self._keep_fp8_weights(config, SubModelType.Transformer)

        model = _build_transformer(sd)
        skip_patterns = _model_declared_skip_patterns(model)

        kept = 0
        if int8_markers:
            quantized = install_int8_convrot_layers(
                model,
                sd,
                resolve_quantized_module_paths(int8_markers, {}),
                model_dtype,
                architecture="Qwen-Image-2.1",
                reserve=self._ram_cache.make_room,
                skip_patterns=skip_patterns,
            )
            self._logger.info(
                f"Qwen-Image-2.1: kept {len(quantized)} of {len(int8_markers)} layer(s) in int8 "
                "(int8_tensorwise checkpoint, dequantized per forward)"
            )
        else:
            # Reserve before the fold or split below widens anything.
            reserve_for_load(
                self._ram_cache.make_room,
                sd,
                model_dtype,
                keep_fp8=keep_fp8,
                model=model,
                skip_patterns=skip_patterns,
                fp8_layers=fp8_layers,
                nvfp4_payloads={},
            )
            if fp8_layers and not keep_fp8:
                dequantize_fp8_scaled(sd, fp8_layers, model_dtype)
                fp8_layers = {}
            fp8_layers = split_fp8_scaled_layers(sd, fp8_layers, model_dtype, model=model, skip_patterns=skip_patterns)
            kept = cast_state_dict(sd, model_dtype, keep_fp8=keep_fp8, model=model, skip_patterns=skip_patterns)

        load_state_dict_ignoring_extras(
            model, sd, source="Qwen-Image-2.1 single-file checkpoint", assign=True, allow_missing=True
        )
        reject_incomplete_load(model, what="Qwen-Image-2.1 single-file checkpoint")
        # `assign=True` aliased every parameter to its `sd` tensor; drop the dict before any fp8 cast allocates.
        sd.clear()

        if fp8_layers:
            attached = attach_fp8_scales(model, fp8_layers)
            self._logger.info(
                f"Qwen-Image-2.1: kept {attached} layer(s) in fp8 (scaled fp8 checkpoint, kept for {self._fp8_kept_reason()})"
            )
            warn_on_unattached_scales(self._logger, "Qwen-Image-2.1", attached, fp8_layers)
            # Already fp8: the layerwise cast would undo the scales, so it must not run.
            return model
        if kept:
            self._logger.info(
                f"Qwen-Image-2.1: kept {kept} raw fp8 weight(s) quantized (no weight_scale in the checkpoint)."
            )
        if int8_markers:
            return model
        return self._apply_fp8_layerwise_casting(model, config, SubModelType.Transformer)


@ModelLoaderRegistry.register(base=BaseModelType.QwenImage21, type=ModelType.Main, format=ModelFormat.GGUFQuantized)
class QwenImage21GGUFCheckpointModel(ModelLoader):
    """Loads a GGUF-quantized Qwen-Image-2.1 transformer; its tensors stay GGML and dequantize per forward.

    The fused `img_mlp.gate_up` is split by its quantized rows, which GGML blocks never cross, so the split
    is exact without dequantizing.
    """

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        if not isinstance(config, Main_GGUF_QwenImage21_Config):
            raise TypeError(f"Expected Main_GGUF_QwenImage21_Config, got {type(config).__name__}.")
        if submodel_type is not SubModelType.Transformer:
            raise ValueError(f"Only the transformer is in a GGUF. Received: {submodel_type}")

        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())
        sd = gguf_sd_loader(Path(config.path), compute_dtype=compute_dtype)
        sd = split_fused_gate_up(CheckpointPrefix.detect(sd).strip(sd))

        model = _build_transformer(sd)
        # What the model reads outside a matmul (the norms, `txt_in.text_norm` among them) and what the GGUF stores
        # unquantized cannot, or need not, stay packed.
        unpack_ggml_at_load(sd, model, _model_declared_skip_patterns(model), self._ram_cache.make_room)

        load_state_dict_ignoring_extras(
            model, sd, source="Qwen-Image-2.1 GGUF checkpoint", assign=True, allow_missing=True
        )
        reject_incomplete_load(model, what="Qwen-Image-2.1 GGUF checkpoint")
        return model
