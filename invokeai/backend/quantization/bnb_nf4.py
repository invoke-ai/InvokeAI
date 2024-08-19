import bitsandbytes as bnb
import torch

# This file contains utils for working with models that use bitsandbytes NF4 quantization.
# The utils in this file are partially inspired by:
# https://github.com/Lightning-AI/pytorch-lightning/blob/1551a16b94f5234a4a78801098f64d0732ef5cb5/src/lightning/fabric/plugins/precision/bitsandbytes.py

# NOTE(ryand): All of the custom state_dict manipulation logic in this file is pretty hacky. This could be made much
# cleaner by re-implementing bnb.nn.LinearNF4 with proper use of buffers and less magic. But, for now, we try to stick
# close to the bitsandbytes classes to make interoperability easier with other models that might use bitsandbytes.


class InvokeLinearNF4(bnb.nn.LinearNF4):
    """A class that extends `bnb.nn.LinearNF4` to add the following functionality:
    - Ability to load Linear NF4 layers from a pre-quantized state_dict.
    - Ability to load Linear NF4 layers from a state_dict when the model is on the "meta" device.
    """

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """This method is based on the logic in the bitsandbytes serialization unit tests for `Linear4bit`:
        https://github.com/bitsandbytes-foundation/bitsandbytes/blob/6d714a5cce3db5bd7f577bc447becc7a92d5ccc7/tests/test_linear4bit.py#L52-L71
        """
        weight = state_dict.pop(prefix + "weight")
        bias = state_dict.pop(prefix + "bias", None)
        # We expect the remaining keys to be quant_state keys.
        quant_state_sd = state_dict

        # During serialization, the quant_state is stored as subkeys of "weight." (See
        # `bnb.nn.LinearNF4._save_to_state_dict()`). We validate that they at least have the correct prefix.
        # TODO(ryand): Technically, we should be using `strict`, `missing_keys`, `unexpected_keys`, and `error_msgs`
        # rather than raising an exception to correctly implement this API.
        assert all(k.startswith(prefix + "weight.") for k in quant_state_sd.keys())

        if len(quant_state_sd) > 0:
            # We are loading a pre-quantized state dict.
            self.weight = bnb.nn.Params4bit.from_prequantized(
                data=weight, quantized_stats=quant_state_sd, device=weight.device
            )
            self.bias = bias if bias is None else torch.nn.Parameter(bias, requires_grad=False)
        else:
            # We are loading a non-quantized state dict.

            # We could simply call the `super()._load_from_state_dict()` method here, but then we wouldn't be able to
            # load from a state_dict into a model on the "meta" device. Attempting to load into a model on the "meta"
            # device requires setting `assign=True`, doing this with the default `super()._load_from_state_dict()`
            # implementation causes `Params4Bit` to be replaced by a `torch.nn.Parameter`. By initializing a new
            # `Params4bit` object, we work around this issue. It's a bit hacky, but it gets the job done.
            self.weight = bnb.nn.Params4bit(
                data=weight,
                requires_grad=self.weight.requires_grad,
                compress_statistics=self.weight.compress_statistics,
                quant_type=self.weight.quant_type,
                quant_storage=self.weight.quant_storage,
                module=self,
            )
            self.bias = bias if bias is None else torch.nn.Parameter(bias)


def _replace_param(
    param: torch.nn.Parameter | bnb.nn.Params4bit,
    data: torch.Tensor,
) -> torch.nn.Parameter:
    """A helper function to replace the data of a model parameter with new data in a way that allows replacing params on
    the "meta" device.

    Supports both `torch.nn.Parameter` and `bnb.nn.Params4bit` parameters.
    """
    if param.device.type == "meta":
        # Doing `param.data = data` raises a RuntimeError if param.data was on the "meta" device, so we need to
        # re-create the param instead of overwriting the data.
        if isinstance(param, bnb.nn.Params4bit):
            return bnb.nn.Params4bit(
                data,
                requires_grad=data.requires_grad,
                quant_state=param.quant_state,
                compress_statistics=param.compress_statistics,
                quant_type=param.quant_type,
            )
        return torch.nn.Parameter(data, requires_grad=data.requires_grad)

    param.data = data
    return param


def _convert_linear_layers_to_nf4(
    module: torch.nn.Module,
    ignore_modules: set[str],
    compute_dtype: torch.dtype,
    compress_statistics: bool = False,
    prefix: str = "",
) -> None:
    """Convert all linear layers in the model to NF4 quantized linear layers.

    Args:
        module: All linear layers in this module will be converted.
        ignore_modules: A set of module prefixes to ignore when converting linear layers.
        compute_dtype: The dtype to use for computation in the quantized linear layers.
        compress_statistics: Whether to enable nested quantization (aka double quantization) where the quantization
           constants from the first quantization are quantized again.
        prefix: The prefix of the current module in the model. Used to call this function recursively.
    """
    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear) and not any(fullname.startswith(s) for s in ignore_modules):
            has_bias = child.bias is not None
            replacement = InvokeLinearNF4(
                child.in_features,
                child.out_features,
                bias=has_bias,
                compute_dtype=compute_dtype,
                compress_statistics=compress_statistics,
            )
            if has_bias:
                replacement.bias = _replace_param(replacement.bias, child.bias.data)
            replacement.weight = _replace_param(replacement.weight, child.weight.data)
            replacement.requires_grad_(False)
            module.__setattr__(name, replacement)
        else:
            _convert_linear_layers_to_nf4(child, ignore_modules, compute_dtype=compute_dtype, prefix=fullname)


def quantize_model_nf4(model: torch.nn.Module, modules_to_not_convert: set[str], compute_dtype: torch.dtype):
    """Apply bitsandbytes nf4 quantization to the model.

    You likely want to call this function inside a `accelerate.init_empty_weights()` context.

    Example usage:
    ```
    # Initialize the model from a config on the meta device.
    with accelerate.init_empty_weights():
        model = ModelClass.from_config(...)

    # Add NF4 quantization linear layers to the model - still on the meta device.
    with accelerate.init_empty_weights():
        model = quantize_model_nf4(model, modules_to_not_convert=set(), compute_dtype=torch.float16)

    # Load a state_dict into the model. (Could be either a prequantized or non-quantized state_dict.)
    model.load_state_dict(state_dict, strict=True, assign=True)

    # Move the model to the "cuda" device. If the model was non-quantized, this is where the weight quantization takes
    # place.
    model.to("cuda")
    ```
    """
    _convert_linear_layers_to_nf4(module=model, ignore_modules=modules_to_not_convert, compute_dtype=compute_dtype)

    return model
