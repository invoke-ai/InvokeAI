import time
from pathlib import Path

import accelerate
import torch
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from accelerate.utils.bnb import get_keys_to_not_convert
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from safetensors.torch import load_file

from invokeai.backend.bnb import quantize_model_llm_int8

# Docs:
# https://huggingface.co/docs/accelerate/usage_guides/quantization
# https://huggingface.co/docs/bitsandbytes/v0.43.3/en/integrations#accelerate


def get_parameter_device(parameter: torch.nn.Module):
    return next(parameter.parameters()).device


# def quantize_model_llm_int8(model: torch.nn.Module, modules_to_not_convert: set[str], llm_int8_threshold: int = 6):
#     """Apply bitsandbytes LLM.8bit() quantization to the model."""
#     model_device = get_parameter_device(model)
#     if model_device.type != "meta":
#         # Note: This is not strictly required, but I can't think of a good reason to quantize a model that's not on the
#         # meta device, so we enforce it for now.
#         raise RuntimeError("The model should be on the meta device to apply LLM.8bit() quantization.")

#     bnb_quantization_config = BnbQuantizationConfig(
#         load_in_8bit=True,
#         llm_int8_threshold=llm_int8_threshold,
#     )

#     with accelerate.init_empty_weights():
#         model = replace_with_bnb_layers(model, bnb_quantization_config, modules_to_not_convert=modules_to_not_convert)

#     return model


def load_flux_transformer(path: Path) -> FluxTransformer2DModel:
    model_config = FluxTransformer2DModel.load_config(path, local_files_only=True)
    with accelerate.init_empty_weights():
        empty_model = FluxTransformer2DModel.from_config(model_config)
    assert isinstance(empty_model, FluxTransformer2DModel)

    bnb_quantization_config = BnbQuantizationConfig(
        load_in_8bit=True,
        llm_int8_threshold=6,
    )

    model_8bit_path = path / "bnb_llm_int8"
    if model_8bit_path.exists():
        # The quantized model already exists, load it and return it.
        # Note that the model loading code is the same when loading from quantized vs original weights. The only
        # difference is the weights_location.
        # model = load_and_quantize_model(
        #     empty_model,
        #     weights_location=model_8bit_path,
        #     bnb_quantization_config=bnb_quantization_config,
        #     # device_map="auto",
        #     device_map={"": "cpu"},
        # )

        # TODO: Handle the keys that were not quantized (get_keys_to_not_convert).
        model = quantize_model_llm_int8(empty_model, modules_to_not_convert=set())

        # model = quantize_model_llm_int8(empty_model, set())

        # Load sharded state dict.
        files = list(path.glob("*.safetensors"))
        state_dict = dict()
        for file in files:
            sd = load_file(file)
            state_dict.update(sd)

    else:
        # The quantized model does not exist yet, quantize and save it.
        model = load_and_quantize_model(
            empty_model,
            weights_location=path,
            bnb_quantization_config=bnb_quantization_config,
            device_map="auto",
        )

        keys_to_not_convert = get_keys_to_not_convert(empty_model)  # TODO

        model_8bit_path.mkdir(parents=True, exist_ok=True)
        accl = accelerate.Accelerator()
        accl.save_model(model, model_8bit_path)

        # ---------------------

        # model = quantize_model_llm_int8(empty_model, set())

        # # Load sharded state dict.
        # files = list(path.glob("*.safetensors"))
        # state_dict = dict()
        # for file in files:
        #     sd = load_file(file)
        #     state_dict.update(sd)

        # # Load the state dict into the model. The bitsandbytes layers know how to load from both quantized and
        # # non-quantized state dicts.
        # result = model.load_state_dict(state_dict, strict=True)
        # model = model.to("cuda")

        # ---------------------

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
