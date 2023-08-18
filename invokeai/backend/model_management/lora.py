from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Optional, Dict, Tuple, Any, Union, List
from pathlib import Path

import numpy as np
import torch
from compel.embeddings_provider import BaseTextualInversionManager
from diffusers.models import UNet2DConditionModel
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer

from .models.lora import LoRAModel


"""
loras = [
    (lora_model1, 0.7),
    (lora_model2, 0.4),
]
with LoRAHelper.apply_lora_unet(unet, loras):
    # unet with applied loras
# unmodified unet

"""


# TODO: rename smth like ModelPatcher and add TI method?
class ModelPatcher:
    @staticmethod
    def _resolve_lora_key(model: torch.nn.Module, lora_key: str, prefix: str) -> Tuple[str, torch.nn.Module]:
        assert "." not in lora_key

        if not lora_key.startswith(prefix):
            raise Exception(f"lora_key with invalid prefix: {lora_key}, {prefix}")

        module = model
        module_key = ""
        key_parts = lora_key[len(prefix) :].split("_")

        submodule_name = key_parts.pop(0)

        while len(key_parts) > 0:
            try:
                module = module.get_submodule(submodule_name)
                module_key += "." + submodule_name
                submodule_name = key_parts.pop(0)
            except Exception:
                submodule_name += "_" + key_parts.pop(0)

        module = module.get_submodule(submodule_name)
        module_key = (module_key + "." + submodule_name).lstrip(".")

        return (module_key, module)

    @staticmethod
    def _lora_forward_hook(
        applied_loras: List[Tuple[LoRAModel, float]],
        layer_name: str,
    ):
        def lora_forward(module, input_h, output):
            if len(applied_loras) == 0:
                return output

            for lora, weight in applied_loras:
                layer = lora.layers.get(layer_name, None)
                if layer is None:
                    continue
                output += layer.forward(module, input_h, weight)
            return output

        return lora_forward

    @classmethod
    @contextmanager
    def apply_lora_unet(
        cls,
        unet: UNet2DConditionModel,
        loras: List[Tuple[LoRAModel, float]],
    ):
        with cls.apply_lora(unet, loras, "lora_unet_"):
            yield

    @classmethod
    @contextmanager
    def apply_lora_text_encoder(
        cls,
        text_encoder: CLIPTextModel,
        loras: List[Tuple[LoRAModel, float]],
    ):
        with cls.apply_lora(text_encoder, loras, "lora_te_"):
            yield

    @classmethod
    @contextmanager
    def apply_sdxl_lora_text_encoder(
        cls,
        text_encoder: CLIPTextModel,
        loras: List[Tuple[LoRAModel, float]],
    ):
        with cls.apply_lora(text_encoder, loras, "lora_te1_"):
            yield

    @classmethod
    @contextmanager
    def apply_sdxl_lora_text_encoder2(
        cls,
        text_encoder: CLIPTextModel,
        loras: List[Tuple[LoRAModel, float]],
    ):
        with cls.apply_lora(text_encoder, loras, "lora_te2_"):
            yield

    @classmethod
    @contextmanager
    def apply_lora(
        cls,
        model: torch.nn.Module,
        loras: List[Tuple[LoRAModel, float]],
        prefix: str,
    ):
        original_weights = dict()
        try:
            with torch.no_grad():
                for lora, lora_weight in loras:
                    # assert lora.device.type == "cpu"
                    for layer_key, layer in lora.layers.items():
                        if not layer_key.startswith(prefix):
                            continue

                        module_key, module = cls._resolve_lora_key(model, layer_key, prefix)
                        if module_key not in original_weights:
                            original_weights[module_key] = module.weight.detach().to(device="cpu", copy=True)

                        # enable autocast to calc fp16 loras on cpu
                        # with torch.autocast(device_type="cpu"):
                        layer.to(dtype=torch.float32)
                        layer_scale = layer.alpha / layer.rank if (layer.alpha and layer.rank) else 1.0
                        layer_weight = layer.get_weight(original_weights[module_key]) * lora_weight * layer_scale

                        if module.weight.shape != layer_weight.shape:
                            # TODO: debug on lycoris
                            layer_weight = layer_weight.reshape(module.weight.shape)

                        module.weight += layer_weight.to(device=module.weight.device, dtype=module.weight.dtype)

            yield  # wait for context manager exit

        finally:
            with torch.no_grad():
                for module_key, weight in original_weights.items():
                    model.get_submodule(module_key).weight.copy_(weight)

    @classmethod
    @contextmanager
    def apply_ti(
        cls,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        ti_list: List[Tuple[str, Any]],
    ) -> Tuple[CLIPTokenizer, TextualInversionManager]:
        init_tokens_count = None
        new_tokens_added = None

        try:
            ti_tokenizer = copy.deepcopy(tokenizer)
            ti_manager = TextualInversionManager(ti_tokenizer)
            init_tokens_count = text_encoder.resize_token_embeddings(None).num_embeddings

            def _get_trigger(ti_name, index):
                trigger = ti_name
                if index > 0:
                    trigger += f"-!pad-{i}"
                return f"<{trigger}>"

            # modify tokenizer
            new_tokens_added = 0
            for ti_name, ti in ti_list:
                for i in range(ti.embedding.shape[0]):
                    new_tokens_added += ti_tokenizer.add_tokens(_get_trigger(ti_name, i))

            # modify text_encoder
            text_encoder.resize_token_embeddings(init_tokens_count + new_tokens_added)
            model_embeddings = text_encoder.get_input_embeddings()

            for ti_name, ti in ti_list:
                ti_tokens = []
                for i in range(ti.embedding.shape[0]):
                    embedding = ti.embedding[i]
                    trigger = _get_trigger(ti_name, i)

                    token_id = ti_tokenizer.convert_tokens_to_ids(trigger)
                    if token_id == ti_tokenizer.unk_token_id:
                        raise RuntimeError(f"Unable to find token id for token '{trigger}'")

                    if model_embeddings.weight.data[token_id].shape != embedding.shape:
                        raise ValueError(
                            f"Cannot load embedding for {trigger}. It was trained on a model with token dimension {embedding.shape[0]}, but the current model has token dimension {model_embeddings.weight.data[token_id].shape[0]}."
                        )

                    model_embeddings.weight.data[token_id] = embedding.to(
                        device=text_encoder.device, dtype=text_encoder.dtype
                    )
                    ti_tokens.append(token_id)

                if len(ti_tokens) > 1:
                    ti_manager.pad_tokens[ti_tokens[0]] = ti_tokens[1:]

            yield ti_tokenizer, ti_manager

        finally:
            if init_tokens_count and new_tokens_added:
                text_encoder.resize_token_embeddings(init_tokens_count)

    @classmethod
    @contextmanager
    def apply_clip_skip(
        cls,
        text_encoder: CLIPTextModel,
        clip_skip: int,
    ):
        skipped_layers = []
        try:
            for i in range(clip_skip):
                skipped_layers.append(text_encoder.text_model.encoder.layers.pop(-1))

            yield

        finally:
            while len(skipped_layers) > 0:
                text_encoder.text_model.encoder.layers.append(skipped_layers.pop())


class TextualInversionModel:
    embedding: torch.Tensor  # [n, 768]|[n, 1280]

    @classmethod
    def from_checkpoint(
        cls,
        file_path: Union[str, Path],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        result = cls()  # TODO:

        if file_path.suffix == ".safetensors":
            state_dict = load_file(file_path.absolute().as_posix(), device="cpu")
        else:
            state_dict = torch.load(file_path, map_location="cpu")

        # both v1 and v2 format embeddings
        # difference mostly in metadata
        if "string_to_param" in state_dict:
            if len(state_dict["string_to_param"]) > 1:
                print(
                    f'Warn: Embedding "{file_path.name}" contains multiple tokens, which is not supported. The first token will be used.'
                )

            result.embedding = next(iter(state_dict["string_to_param"].values()))

        # v3 (easynegative)
        elif "emb_params" in state_dict:
            result.embedding = state_dict["emb_params"]

        # v4(diffusers bin files)
        else:
            result.embedding = next(iter(state_dict.values()))

            if len(result.embedding.shape) == 1:
                result.embedding = result.embedding.unsqueeze(0)

            if not isinstance(result.embedding, torch.Tensor):
                raise ValueError(f"Invalid embeddings file: {file_path.name}")

        return result


class TextualInversionManager(BaseTextualInversionManager):
    pad_tokens: Dict[int, List[int]]
    tokenizer: CLIPTokenizer

    def __init__(self, tokenizer: CLIPTokenizer):
        self.pad_tokens = dict()
        self.tokenizer = tokenizer

    def expand_textual_inversion_token_ids_if_necessary(self, token_ids: list[int]) -> list[int]:
        if len(self.pad_tokens) == 0:
            return token_ids

        if token_ids[0] == self.tokenizer.bos_token_id:
            raise ValueError("token_ids must not start with bos_token_id")
        if token_ids[-1] == self.tokenizer.eos_token_id:
            raise ValueError("token_ids must not end with eos_token_id")

        new_token_ids = []
        for token_id in token_ids:
            new_token_ids.append(token_id)
            if token_id in self.pad_tokens:
                new_token_ids.extend(self.pad_tokens[token_id])

        return new_token_ids


class ONNXModelPatcher:
    from .models.base import IAIOnnxRuntimeModel
    from diffusers import OnnxRuntimeModel

    @classmethod
    @contextmanager
    def apply_lora_unet(
        cls,
        unet: OnnxRuntimeModel,
        loras: List[Tuple[LoRAModel, float]],
    ):
        with cls.apply_lora(unet, loras, "lora_unet_"):
            yield

    @classmethod
    @contextmanager
    def apply_lora_text_encoder(
        cls,
        text_encoder: OnnxRuntimeModel,
        loras: List[Tuple[LoRAModel, float]],
    ):
        with cls.apply_lora(text_encoder, loras, "lora_te_"):
            yield

    # based on
    # https://github.com/ssube/onnx-web/blob/ca2e436f0623e18b4cfe8a0363fcfcf10508acf7/api/onnx_web/convert/diffusion/lora.py#L323
    @classmethod
    @contextmanager
    def apply_lora(
        cls,
        model: IAIOnnxRuntimeModel,
        loras: List[Tuple[LoRAModel, float]],
        prefix: str,
    ):
        from .models.base import IAIOnnxRuntimeModel

        if not isinstance(model, IAIOnnxRuntimeModel):
            raise Exception("Only IAIOnnxRuntimeModel models supported")

        orig_weights = dict()

        try:
            blended_loras = dict()

            for lora, lora_weight in loras:
                for layer_key, layer in lora.layers.items():
                    if not layer_key.startswith(prefix):
                        continue

                    layer.to(dtype=torch.float32)
                    layer_key = layer_key.replace(prefix, "")
                    # TODO: rewrite to pass original tensor weight(required by ia3)
                    layer_weight = layer.get_weight(None).detach().cpu().numpy() * lora_weight
                    if layer_key is blended_loras:
                        blended_loras[layer_key] += layer_weight
                    else:
                        blended_loras[layer_key] = layer_weight

            node_names = dict()
            for node in model.nodes.values():
                node_names[node.name.replace("/", "_").replace(".", "_").lstrip("_")] = node.name

            for layer_key, lora_weight in blended_loras.items():
                conv_key = layer_key + "_Conv"
                gemm_key = layer_key + "_Gemm"
                matmul_key = layer_key + "_MatMul"

                if conv_key in node_names or gemm_key in node_names:
                    if conv_key in node_names:
                        conv_node = model.nodes[node_names[conv_key]]
                    else:
                        conv_node = model.nodes[node_names[gemm_key]]

                    weight_name = [n for n in conv_node.input if ".weight" in n][0]
                    orig_weight = model.tensors[weight_name]

                    if orig_weight.shape[-2:] == (1, 1):
                        if lora_weight.shape[-2:] == (1, 1):
                            new_weight = orig_weight.squeeze((3, 2)) + lora_weight.squeeze((3, 2))
                        else:
                            new_weight = orig_weight.squeeze((3, 2)) + lora_weight

                        new_weight = np.expand_dims(new_weight, (2, 3))
                    else:
                        if orig_weight.shape != lora_weight.shape:
                            new_weight = orig_weight + lora_weight.reshape(orig_weight.shape)
                        else:
                            new_weight = orig_weight + lora_weight

                    orig_weights[weight_name] = orig_weight
                    model.tensors[weight_name] = new_weight.astype(orig_weight.dtype)

                elif matmul_key in node_names:
                    weight_node = model.nodes[node_names[matmul_key]]
                    matmul_name = [n for n in weight_node.input if "MatMul" in n][0]

                    orig_weight = model.tensors[matmul_name]
                    new_weight = orig_weight + lora_weight.transpose()

                    orig_weights[matmul_name] = orig_weight
                    model.tensors[matmul_name] = new_weight.astype(orig_weight.dtype)

                else:
                    # warn? err?
                    pass

            yield

        finally:
            # restore original weights
            for name, orig_weight in orig_weights.items():
                model.tensors[name] = orig_weight

    @classmethod
    @contextmanager
    def apply_ti(
        cls,
        tokenizer: CLIPTokenizer,
        text_encoder: IAIOnnxRuntimeModel,
        ti_list: List[Tuple[str, Any]],
    ) -> Tuple[CLIPTokenizer, TextualInversionManager]:
        from .models.base import IAIOnnxRuntimeModel

        if not isinstance(text_encoder, IAIOnnxRuntimeModel):
            raise Exception("Only IAIOnnxRuntimeModel models supported")

        orig_embeddings = None

        try:
            ti_tokenizer = copy.deepcopy(tokenizer)
            ti_manager = TextualInversionManager(ti_tokenizer)

            def _get_trigger(ti_name, index):
                trigger = ti_name
                if index > 0:
                    trigger += f"-!pad-{i}"
                return f"<{trigger}>"

            # modify tokenizer
            new_tokens_added = 0
            for ti_name, ti in ti_list:
                for i in range(ti.embedding.shape[0]):
                    new_tokens_added += ti_tokenizer.add_tokens(_get_trigger(ti_name, i))

            # modify text_encoder
            orig_embeddings = text_encoder.tensors["text_model.embeddings.token_embedding.weight"]

            embeddings = np.concatenate(
                (np.copy(orig_embeddings), np.zeros((new_tokens_added, orig_embeddings.shape[1]))),
                axis=0,
            )

            for ti_name, ti in ti_list:
                ti_tokens = []
                for i in range(ti.embedding.shape[0]):
                    embedding = ti.embedding[i].detach().numpy()
                    trigger = _get_trigger(ti_name, i)

                    token_id = ti_tokenizer.convert_tokens_to_ids(trigger)
                    if token_id == ti_tokenizer.unk_token_id:
                        raise RuntimeError(f"Unable to find token id for token '{trigger}'")

                    if embeddings[token_id].shape != embedding.shape:
                        raise ValueError(
                            f"Cannot load embedding for {trigger}. It was trained on a model with token dimension {embedding.shape[0]}, but the current model has token dimension {embeddings[token_id].shape[0]}."
                        )

                    embeddings[token_id] = embedding
                    ti_tokens.append(token_id)

                if len(ti_tokens) > 1:
                    ti_manager.pad_tokens[ti_tokens[0]] = ti_tokens[1:]

            text_encoder.tensors["text_model.embeddings.token_embedding.weight"] = embeddings.astype(
                orig_embeddings.dtype
            )

            yield ti_tokenizer, ti_manager

        finally:
            # restore
            if orig_embeddings is not None:
                text_encoder.tensors["text_model.embeddings.token_embedding.weight"] = orig_embeddings
