from pathlib import Path

import accelerate
from safetensors.torch import load_file, save_file

from invokeai.backend.flux.model import Flux
from invokeai.backend.flux.util import params
from invokeai.backend.quantization.bnb_llm_int8 import quantize_model_llm_int8
from invokeai.backend.quantization.scripts.load_flux_model_bnb_nf4 import log_time


def main():
    """A script for quantizing a FLUX transformer model using the bitsandbytes LLM.int8() quantization method.

    This script is primarily intended for reference. The script params (e.g. the model_path, modules_to_not_convert,
    etc.) are hardcoded and would need to be modified for other use cases.
    """
    # Load the FLUX transformer model onto the meta device.
    model_path = Path(
        "/data/invokeai/models/.download_cache/https__huggingface.co_black-forest-labs_flux.1-schnell_resolve_main_flux1-schnell.safetensors/flux1-schnell.safetensors"
    )

    with log_time("Intialize FLUX transformer on meta device"):
        # TODO(ryand): Determine if this is a schnell model or a dev model and load the appropriate config.
        p = params["flux-schnell"]

        # Initialize the model on the "meta" device.
        with accelerate.init_empty_weights():
            model = Flux(p)

    # TODO(ryand): We may want to add some modules to not quantize here (e.g. the proj_out layer). See the accelerate
    # `get_keys_to_not_convert(...)` function for a heuristic to determine which modules to not quantize.
    modules_to_not_convert: set[str] = set()

    model_int8_path = model_path.parent / "bnb_llm_int8.safetensors"
    if model_int8_path.exists():
        # The quantized model already exists, load it and return it.
        print(f"A pre-quantized model already exists at '{model_int8_path}'. Attempting to load it...")

        # Replace the linear layers with LLM.int8() quantized linear layers (still on the meta device).
        with log_time("Replace linear layers with LLM.int8() layers"), accelerate.init_empty_weights():
            model = quantize_model_llm_int8(model, modules_to_not_convert=modules_to_not_convert)

        with log_time("Load state dict into model"):
            sd = load_file(model_int8_path)
            model.load_state_dict(sd, strict=True, assign=True)

        with log_time("Move model to cuda"):
            model = model.to("cuda")

        print(f"Successfully loaded pre-quantized model from '{model_int8_path}'.")

    else:
        # The quantized model does not exist, quantize the model and save it.
        print(f"No pre-quantized model found at '{model_int8_path}'. Quantizing the model...")

        with log_time("Replace linear layers with LLM.int8() layers"), accelerate.init_empty_weights():
            model = quantize_model_llm_int8(model, modules_to_not_convert=modules_to_not_convert)

        with log_time("Load state dict into model"):
            state_dict = load_file(model_path)
            # TODO(ryand): Cast the state_dict to the appropriate dtype?
            model.load_state_dict(state_dict, strict=True, assign=True)

        with log_time("Move model to cuda and quantize"):
            model = model.to("cuda")

        with log_time("Save quantized model"):
            model_int8_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(model.state_dict(), model_int8_path)

        print(f"Successfully quantized and saved model to '{model_int8_path}'.")

    assert isinstance(model, Flux)
    return model


if __name__ == "__main__":
    main()
