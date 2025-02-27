from pathlib import Path

import accelerate
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForTextEncoding, T5EncoderModel

from invokeai.backend.quantization.bnb_llm_int8 import quantize_model_llm_int8
from invokeai.backend.quantization.scripts.load_flux_model_bnb_nf4 import log_time


def load_state_dict_into_t5(model: T5EncoderModel, state_dict: dict):
    # There is a shared reference to a single weight tensor in the model.
    # Both "encoder.embed_tokens.weight" and "shared.weight" refer to the same tensor, so only the latter should
    # be present in the state_dict.
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
    assert len(unexpected_keys) == 0
    assert set(missing_keys) == {"encoder.embed_tokens.weight"}
    # Assert that the layers we expect to be shared are actually shared.
    assert model.encoder.embed_tokens.weight is model.shared.weight


def main():
    """A script for quantizing a T5 text encoder model using the bitsandbytes LLM.int8() quantization method.

    This script is primarily intended for reference. The script params (e.g. the model_path, modules_to_not_convert,
    etc.) are hardcoded and would need to be modified for other use cases.
    """
    model_path = Path("/data/misc/text_encoder_2")

    with log_time("Intialize T5 on meta device"):
        model_config = AutoConfig.from_pretrained(model_path)
        with accelerate.init_empty_weights():
            model = AutoModelForTextEncoding.from_config(model_config)

    # TODO(ryand): We may want to add some modules to not quantize here (e.g. the proj_out layer). See the accelerate
    # `get_keys_to_not_convert(...)` function for a heuristic to determine which modules to not quantize.
    modules_to_not_convert: set[str] = set()

    model_int8_path = model_path / "bnb_llm_int8.safetensors"
    if model_int8_path.exists():
        # The quantized model already exists, load it and return it.
        print(f"A pre-quantized model already exists at '{model_int8_path}'. Attempting to load it...")

        # Replace the linear layers with LLM.int8() quantized linear layers (still on the meta device).
        with log_time("Replace linear layers with LLM.int8() layers"), accelerate.init_empty_weights():
            model = quantize_model_llm_int8(model, modules_to_not_convert=modules_to_not_convert)

        with log_time("Load state dict into model"):
            sd = load_file(model_int8_path)
            load_state_dict_into_t5(model, sd)

        with log_time("Move model to cuda"):
            model = model.to("cuda")

        print(f"Successfully loaded pre-quantized model from '{model_int8_path}'.")

    else:
        # The quantized model does not exist, quantize the model and save it.
        print(f"No pre-quantized model found at '{model_int8_path}'. Quantizing the model...")

        with log_time("Replace linear layers with LLM.int8() layers"), accelerate.init_empty_weights():
            model = quantize_model_llm_int8(model, modules_to_not_convert=modules_to_not_convert)

        with log_time("Load state dict into model"):
            # Load sharded state dict.
            files = list(model_path.glob("*.safetensors"))
            state_dict = {}
            for file in files:
                sd = load_file(file)
                state_dict.update(sd)
            load_state_dict_into_t5(model, state_dict)

        with log_time("Move model to cuda and quantize"):
            model = model.to("cuda")

        with log_time("Save quantized model"):
            model_int8_path.parent.mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict.pop("encoder.embed_tokens.weight")
            save_file(state_dict, model_int8_path)
            # This handling of shared weights could also be achieved with save_model(...), but then we'd lose control
            # over which keys are kept. And, the corresponding load_model(...) function does not support assign=True.
            # save_model(model, model_int8_path)

        print(f"Successfully quantized and saved model to '{model_int8_path}'.")

    assert isinstance(model, T5EncoderModel)
    return model


if __name__ == "__main__":
    main()
