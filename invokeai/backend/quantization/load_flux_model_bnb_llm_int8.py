import time
from contextlib import contextmanager
from pathlib import Path

import accelerate
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from safetensors.torch import load_file, save_file

from invokeai.backend.quantization.bnb_llm_int8 import quantize_model_llm_int8


@contextmanager
def log_time(name: str):
    """Helper context manager to log the time taken by a block of code."""
    start = time.time()
    try:
        yield None
    finally:
        end = time.time()
        print(f"'{name}' took {end - start:.4f} secs")


def main():
    # Load the FLUX transformer model onto the meta device.
    model_path = Path(
        "/data/invokeai/models/.download_cache/black-forest-labs_flux.1-schnell/FLUX.1-schnell/transformer/"
    )

    with log_time("Initialize FLUX transformer on meta device"):
        model_config = FluxTransformer2DModel.load_config(model_path, local_files_only=True)
        with accelerate.init_empty_weights():
            empty_model = FluxTransformer2DModel.from_config(model_config)
        assert isinstance(empty_model, FluxTransformer2DModel)

    # TODO(ryand): We may want to add some modules to not quantize here (e.g. the proj_out layer). See the accelerate
    # `get_keys_to_not_convert(...)` function for a heuristic to determine which modules to not quantize.
    modules_to_not_convert: set[str] = set()

    model_int8_path = model_path / "bnb_llm_int8"
    if model_int8_path.exists():
        # The quantized model already exists, load it and return it.
        print(f"A pre-quantized model already exists at '{model_int8_path}'. Attempting to load it...")

        # Replace the linear layers with LLM.int8() quantized linear layers (still on the meta device).
        with log_time("Replace linear layers with LLM.int8() layers"), accelerate.init_empty_weights():
            model = quantize_model_llm_int8(empty_model, modules_to_not_convert=modules_to_not_convert)

        with log_time("Load state dict into model"):
            sd = load_file(model_int8_path / "model.safetensors")
            model.load_state_dict(sd, strict=True, assign=True)

        with log_time("Move model to cuda"):
            model = model.to("cuda")

        print(f"Successfully loaded pre-quantized model from '{model_int8_path}'.")

    else:
        # The quantized model does not exist, quantize the model and save it.
        print(f"No pre-quantized model found at '{model_int8_path}'. Quantizing the model...")

        with log_time("Replace linear layers with LLM.int8() layers"), accelerate.init_empty_weights():
            model = quantize_model_llm_int8(empty_model, modules_to_not_convert=modules_to_not_convert)

        with log_time("Load state dict into model"):
            # Load sharded state dict.
            files = list(model_path.glob("*.safetensors"))
            state_dict = {}
            for file in files:
                sd = load_file(file)
                state_dict.update(sd)

            model.load_state_dict(state_dict, strict=True, assign=True)

        with log_time("Move model to cuda and quantize"):
            model = model.to("cuda")

        with log_time("Save quantized model"):
            model_int8_path.mkdir(parents=True, exist_ok=True)
            output_path = model_int8_path / "model.safetensors"
            save_file(model.state_dict(), output_path)

        print(f"Successfully quantized and saved model to '{output_path}'.")

    assert isinstance(model, FluxTransformer2DModel)
    return model


if __name__ == "__main__":
    main()
