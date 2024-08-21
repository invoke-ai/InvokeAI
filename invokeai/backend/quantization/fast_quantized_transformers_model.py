import json
import os
from typing import Union

from optimum.quanto.models import QuantizedTransformersModel
from optimum.quanto.models.shared_dict import ShardedStateDict
from transformers import AutoConfig
from transformers.modeling_utils import get_checkpoint_shard_files, load_state_dict
from transformers.models.auto import AutoModelForTextEncoding
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, is_accelerate_available

from invokeai.backend.quantization.requantize import requantize


class FastQuantizedTransformersModel(QuantizedTransformersModel):
    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike], auto_class=AutoModelForTextEncoding, **kwargs
    ):
        """We override the `from_pretrained()` method in order to use our custom `requantize()` implementation."""
        auto_class = auto_class or cls.auto_class
        if auto_class is None:
            raise ValueError(
                "Quantized models cannot be reloaded using {cls}: use a specialized quantized class such as QuantizedModelForCausalLM instead."
            )
        if not is_accelerate_available():
            raise ValueError("Reloading a quantized transformers model requires the accelerate library.")
        from accelerate import init_empty_weights

        if os.path.isdir(model_name_or_path):
            # Look for a quantization map
            qmap_path = os.path.join(model_name_or_path, cls._qmap_name())
            if not os.path.exists(qmap_path):
                raise ValueError(f"No quantization map found in {model_name_or_path}: is this a quantized model ?")
            with open(qmap_path, "r", encoding="utf-8") as f:
                qmap = json.load(f)
            # Create an empty model
            config = AutoConfig.from_pretrained(model_name_or_path)
            with init_empty_weights():
                model = auto_class.from_config(config)
            # Look for the index of a sharded checkpoint
            checkpoint_file = os.path.join(model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
            if os.path.exists(checkpoint_file):
                # Convert the checkpoint path to a list of shards
                checkpoint_file, sharded_metadata = get_checkpoint_shard_files(model_name_or_path, checkpoint_file)
                # Create a mapping for the sharded safetensor files
                state_dict = ShardedStateDict(model_name_or_path, sharded_metadata["weight_map"])
            else:
                # Look for a single checkpoint file
                checkpoint_file = os.path.join(model_name_or_path, SAFE_WEIGHTS_NAME)
                if not os.path.exists(checkpoint_file):
                    raise ValueError(f"No safetensor weights found in {model_name_or_path}.")
                # Get state_dict from model checkpoint
                state_dict = load_state_dict(checkpoint_file)
            # Requantize and load quantized weights from state_dict
            requantize(model, state_dict=state_dict, quantization_map=qmap)
            if getattr(model.config, "tie_word_embeddings", True):
                # Tie output weight embeddings to input weight embeddings
                # Note that if they were quantized they would NOT be tied
                model.tie_weights()
            # Set model in evaluation mode as it is done in transformers
            model.eval()
            return cls(model)._wrapped
        else:
            raise NotImplementedError("Reloading quantized models directly from the hub is not supported yet.")
