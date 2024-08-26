import time
from contextlib import contextmanager
from pathlib import Path

import accelerate
import torch
from safetensors.torch import load_file, save_file

from invokeai.backend.flux.model import Flux
from invokeai.backend.flux.util import params
from invokeai.backend.quantization.bnb_nf4 import quantize_model_nf4


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
    """A script for quantizing a FLUX transformer model using the bitsandbytes NF4 quantization method.

    This script is primarily intended for reference. The script params (e.g. the model_path, modules_to_not_convert,
    etc.) are hardcoded and would need to be modified for other use cases.
    """
    model_path = Path(
        "/data/invokeai/models/.download_cache/https__huggingface.co_black-forest-labs_flux.1-schnell_resolve_main_flux1-schnell.safetensors/flux1-schnell.safetensors"
    )

    # inference_dtype = torch.bfloat16
    with log_time("Intialize FLUX transformer on meta device"):
        # TODO(ryand): Determine if this is a schnell model or a dev model and load the appropriate config.
        p = params["flux-schnell"]

        # Initialize the model on the "meta" device.
        with accelerate.init_empty_weights():
            model = Flux(p)

    # TODO(ryand): We may want to add some modules to not quantize here (e.g. the proj_out layer). See the accelerate
    # `get_keys_to_not_convert(...)` function for a heuristic to determine which modules to not quantize.
    modules_to_not_convert: set[str] = set()

    model_nf4_path = model_path.parent / "bnb_nf4.safetensors"
    if model_nf4_path.exists():
        # The quantized model already exists, load it and return it.
        print(f"A pre-quantized model already exists at '{model_nf4_path}'. Attempting to load it...")

        # Replace the linear layers with NF4 quantized linear layers (still on the meta device).
        with log_time("Replace linear layers with NF4 layers"), accelerate.init_empty_weights():
            model = quantize_model_nf4(
                model, modules_to_not_convert=modules_to_not_convert, compute_dtype=torch.bfloat16
            )

        with log_time("Load state dict into model"):
            state_dict = load_file(model_nf4_path)
            model.load_state_dict(state_dict, strict=True, assign=True)

        with log_time("Move model to cuda"):
            model = model.to("cuda")

        print(f"Successfully loaded pre-quantized model from '{model_nf4_path}'.")

    else:
        # The quantized model does not exist, quantize the model and save it.
        print(f"No pre-quantized model found at '{model_nf4_path}'. Quantizing the model...")

        with log_time("Replace linear layers with NF4 layers"), accelerate.init_empty_weights():
            model = quantize_model_nf4(
                model, modules_to_not_convert=modules_to_not_convert, compute_dtype=torch.bfloat16
            )

        with log_time("Load state dict into model"):
            state_dict = load_file(model_path)
            # TODO(ryand): Cast the state_dict to the appropriate dtype?
            model.load_state_dict(state_dict, strict=True, assign=True)

        with log_time("Move model to cuda and quantize"):
            model = model.to("cuda")

        with log_time("Save quantized model"):
            model_nf4_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(model.state_dict(), model_nf4_path)

        print(f"Successfully quantized and saved model to '{model_nf4_path}'.")

    assert isinstance(model, Flux)
    return model


if __name__ == "__main__":
    main()
