import json
import os
import time
from pathlib import Path
from typing import Union

import torch
from diffusers.models.model_loading_utils import load_state_dict
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.utils import (
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    _get_checkpoint_shard_files,
    is_accelerate_available,
)
from optimum.quanto import qfloat8
from optimum.quanto.models import QuantizedDiffusersModel
from optimum.quanto.models.shared_dict import ShardedStateDict

from invokeai.backend.requantize import requantize


class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel

    @classmethod
    def from_pretrained(cls, model_name_or_path: Union[str, os.PathLike]):
        if cls.base_class is None:
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
            configured_cls_name = cls.base_class.__name__
            if configured_cls_name != original_model_cls_name:
                raise ValueError(
                    f"Configured base class ({configured_cls_name}) differs from what was derived from the provided configuration ({original_model_cls_name})."
                )

            # Create an empty model
            config = cls.base_class.load_config(model_name_or_path)
            with init_empty_weights():
                model = cls.base_class.from_config(config)

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
            return cls(model)
        else:
            raise NotImplementedError("Reloading quantized models directly from the hub is not supported yet.")


def load_flux_transformer(path: Path) -> FluxTransformer2DModel:
    # model = FluxTransformer2DModel.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16)
    model_8bit_path = path / "quantized"
    if model_8bit_path.exists():
        # The quantized model exists, load it.
        # TODO(ryand): The requantize(...) operation in from_pretrained(...) is very slow. This seems like
        # something that we should be able to make much faster.
        q_model = QuantizedFluxTransformer2DModel.from_pretrained(model_8bit_path)

        # Access the underlying wrapped model.
        # We access the wrapped model, even though it is private, because it simplifies the type checking by
        # always returning a FluxTransformer2DModel from this function.
        model = q_model._wrapped
    else:
        # The quantized model does not exist yet, quantize and save it.
        # TODO(ryand): Loading in float16 and then quantizing seems to result in NaNs. In order to run this on
        # GPUs that don't support bfloat16, we would need to host the quantized model instead of generating it
        # here.
        model = FluxTransformer2DModel.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16)
        assert isinstance(model, FluxTransformer2DModel)

        q_model = QuantizedFluxTransformer2DModel.quantize(model, weights=qfloat8)

        model_8bit_path.mkdir(parents=True, exist_ok=True)
        q_model.save_pretrained(model_8bit_path)

        # (See earlier comment about accessing the wrapped model.)
        model = q_model._wrapped

    assert isinstance(model, FluxTransformer2DModel)
    return model


def main():
    start = time.time()
    model = load_flux_transformer(
        Path("/data/invokeai/models/.download_cache/black-forest-labs_flux.1-schnell/FLUX.1-schnell/transformer/")
    )
    print(f"Time to load: {time.time() - start}s")
    print("hi")


if __name__ == "__main__":
    main()
