import json
import os
from typing import Union

from diffusers.models.model_loading_utils import load_state_dict
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.utils import (
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    _get_checkpoint_shard_files,
    is_accelerate_available,
)
from optimum.quanto.models import QuantizedDiffusersModel
from optimum.quanto.models.shared_dict import ShardedStateDict

from invokeai.backend.quantization.requantize import requantize


class FastQuantizedDiffusersModel(QuantizedDiffusersModel):
    @classmethod
    def from_pretrained(cls, model_name_or_path: Union[str, os.PathLike], base_class=FluxTransformer2DModel, **kwargs):
        """We override the `from_pretrained()` method in order to use our custom `requantize()` implementation."""
        base_class = base_class or cls.base_class
        if base_class is None:
            raise ValueError("The `base_class` attribute needs to be configured.")

        if not is_accelerate_available():
            raise ValueError("Reloading a quantized diffusers model requires the accelerate library.")
        from accelerate import init_empty_weights

        if os.path.isdir(model_name_or_path):
            # Look for a quantization map
            qmap_path = os.path.join(model_name_or_path, cls._qmap_name())
            if not os.path.exists(qmap_path):
                raise ValueError(f"No quantization map found in {model_name_or_path}: is this a quantized model ?")

            # Look for original model config file.
            model_config_path = os.path.join(model_name_or_path, CONFIG_NAME)
            if not os.path.exists(model_config_path):
                raise ValueError(f"{CONFIG_NAME} not found in {model_name_or_path}.")

            with open(qmap_path, "r", encoding="utf-8") as f:
                qmap = json.load(f)

            with open(model_config_path, "r", encoding="utf-8") as f:
                original_model_cls_name = json.load(f)["_class_name"]
            configured_cls_name = base_class.__name__
            if configured_cls_name != original_model_cls_name:
                raise ValueError(
                    f"Configured base class ({configured_cls_name}) differs from what was derived from the provided configuration ({original_model_cls_name})."
                )

            # Create an empty model
            config = base_class.load_config(model_name_or_path)
            with init_empty_weights():
                model = base_class.from_config(config)

            # Look for the index of a sharded checkpoint
            checkpoint_file = os.path.join(model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
            if os.path.exists(checkpoint_file):
                # Convert the checkpoint path to a list of shards
                _, sharded_metadata = _get_checkpoint_shard_files(model_name_or_path, checkpoint_file)
                # Create a mapping for the sharded safetensor files
                state_dict = ShardedStateDict(model_name_or_path, sharded_metadata["weight_map"])
            else:
                # Look for a single checkpoint file
                checkpoint_file = os.path.join(model_name_or_path, SAFETENSORS_WEIGHTS_NAME)
                if not os.path.exists(checkpoint_file):
                    raise ValueError(f"No safetensor weights found in {model_name_or_path}.")
                # Get state_dict from model checkpoint
                state_dict = load_state_dict(checkpoint_file)

            # Requantize and load quantized weights from state_dict
            requantize(model, state_dict=state_dict, quantization_map=qmap)
            model.eval()
            return cls(model)._wrapped
        else:
            raise NotImplementedError("Reloading quantized models directly from the hub is not supported yet.")
