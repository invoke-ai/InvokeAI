from typing import Any, Optional, Set, Tuple, Type

import accelerate
import bitsandbytes as bnb
import torch

# The utils in this file take ideas from
# https://github.com/Lightning-AI/pytorch-lightning/blob/1551a16b94f5234a4a78801098f64d0732ef5cb5/src/lightning/fabric/plugins/precision/bitsandbytes.py


# Patterns:
# - Quantize:
#   - Initialize model on meta device
#   - Replace layers
#   - Load state_dict to cpu
#   - Load state_dict into model
#   - Quantize on GPU
#   - Extract state_dict
#   - Save

# - Load:
#   - Initialize model on meta device
#   - Replace layers
#   - Load state_dict to cpu
#   - Load state_dict into model on cpu
#   - Move to GPU


# class InvokeInt8Params(bnb.nn.Int8Params):
#     """Overrides `bnb.nn.Int8Params` to add the following functionality:
#     - Make it possible to load a quantized state dict without putting the weight on a "cuda" device.
#     """

#     def quantize(self, device: Optional[torch.device] = None):
#         device = device or torch.device("cuda")
#         if device.type != "cuda":
#             raise RuntimeError(f"Int8Params quantization is only supported on CUDA devices ({device=}).")

#         # https://github.com/TimDettmers/bitsandbytes/blob/0.41.0/bitsandbytes/nn/modules.py#L291-L302
#         B = self.data.contiguous().half().cuda(device)
#         if self.has_fp16_weights:
#             self.data = B
#         else:
#             # we store the 8-bit rows-major weight
#             # we convert this weight to the turning/ampere weight during the first inference pass
#             CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
#             del CBt
#             del SCBt
#             self.data = CB
#             self.CB = CB
#             self.SCB = SCB


class InvokeLinearNF4(bnb.nn.LinearNF4):
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """This method is based on the logic in the bitsandbytes serialization unit tests for `Linear4bit`:
        https://github.com/bitsandbytes-foundation/bitsandbytes/blob/6d714a5cce3db5bd7f577bc447becc7a92d5ccc7/tests/test_linear4bit.py#L52-L71

        I'm not sure why this was not included in the original `Linear4bit` implementation.
        """

        weight = state_dict.pop(prefix + "weight")
        bias = state_dict.pop(prefix + "bias", None)
        # During serialization, the quant_state is stored as subkeys of "weight.".
        # We expect the remaining keys to be quant_state keys. We validate that they at least have the correct prefix.
        quant_state_sd = state_dict
        assert all(k.startswith(prefix + "weight.") for k in quant_state_sd.keys())

        if len(quant_state_sd) > 0:
            # We are loading a quantized state dict.
            self.weight = bnb.nn.Params4bit.from_prequantized(
                data=weight, quantized_stats=quant_state_sd, device=weight.device
            )
            self.bias = bias if bias is None else torch.nn.Parameter(bias, requires_grad=False)

        else:
            # We are loading a non-quantized state dict.

            # We could simply call the `super()._load_from_state_dict` method here, but then we wouldn't be able to load
            # into from a state_dict into a model on the "meta" device. By initializing a new `Params4bit` object, we
            # work around this issue.
            self.weight = bnb.nn.Params4bit(
                data=weight,
                requires_grad=self.weight.requires_grad,
                compress_statistics=self.weight.compress_statistics,
                quant_type=self.weight.quant_type,
                quant_storage=self.weight.quant_storage,
                module=self,
            )
            self.bias = bias if bias is None else torch.nn.Parameter(bias)


class Invoke2Linear8bitLt(torch.nn.Linear):
    """This class is the base module for the [LLM.int8()](https://arxiv.org/abs/2208.07339) algorithm."""

    def __init__(
        self,
        input_features: int,
        output_features: int,
        bias=True,
        has_fp16_weights=True,
        memory_efficient_backward=False,
        threshold=0.0,
        index=None,
        device=None,
    ):
        """
        Initialize Linear8bitLt class.

        Args:
            input_features (`int`):
                Number of input features of the linear layer.
            output_features (`int`):
                Number of output features of the linear layer.
            bias (`bool`, defaults to `True`):
                Whether the linear class uses the bias term as well.
        """
        super().__init__(input_features, output_features, bias, device)
        assert not memory_efficient_backward, "memory_efficient_backward is no longer required and the argument is deprecated in 0.37.0 and will be removed in 0.39.0"
        self.state = bnb.MatmulLtState()
        self.index = index

        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.weight = Int8Params(self.weight.data, has_fp16_weights=has_fp16_weights, requires_grad=has_fp16_weights)
        self._register_load_state_dict_pre_hook(maybe_rearrange_weight)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)

        # we only need to save SCB as extra data, because CB for quantized weights is already stored in weight.data
        scb_name = "SCB"

        # case 1: .cuda was called, SCB is in self.weight
        param_from_weight = getattr(self.weight, scb_name)
        # case 2: self.init_8bit_state was called, SCB is in self.state
        param_from_state = getattr(self.state, scb_name)
        # case 3: SCB is in self.state, weight layout reordered after first forward()
        layout_reordered = self.state.CxB is not None

        key_name = prefix + f"{scb_name}"
        format_name = prefix + "weight_format"

        if not self.state.has_fp16_weights:
            if param_from_weight is not None:
                destination[key_name] = param_from_weight if keep_vars else param_from_weight.detach()
                destination[format_name] = torch.tensor(0, dtype=torch.uint8)
            elif param_from_state is not None and not layout_reordered:
                destination[key_name] = param_from_state if keep_vars else param_from_state.detach()
                destination[format_name] = torch.tensor(0, dtype=torch.uint8)
            elif param_from_state is not None:
                destination[key_name] = param_from_state if keep_vars else param_from_state.detach()
                weights_format = self.state.formatB
                # At this point `weights_format` is an str
                if weights_format not in LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING:
                    raise ValueError(f"Unrecognized weights format {weights_format}")

                weights_format = LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING[weights_format]

                destination[format_name] = torch.tensor(weights_format, dtype=torch.uint8)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        unexpected_copy = list(unexpected_keys)

        for key in unexpected_copy:
            input_name = key[len(prefix) :]
            if input_name == "SCB":
                if self.weight.SCB is None:
                    # buffers not yet initialized, can't access them directly without quantizing first
                    raise RuntimeError(
                        "Loading a quantized checkpoint into non-quantized Linear8bitLt is "
                        "not supported. Please call module.cuda() before module.load_state_dict()",
                    )

                input_param = state_dict[key]
                self.weight.SCB.copy_(input_param)

                if self.state.SCB is not None:
                    self.state.SCB = self.weight.SCB

                unexpected_keys.remove(key)

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x: torch.Tensor):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

        if not self.state.has_fp16_weights:
            if self.state.CB is not None and self.state.CxB is not None:
                # we converted 8-bit row major to turing/ampere format in the first inference pass
                # we no longer need the row-major weight
                del self.state.CB
                self.weight.data = self.state.CxB
        return out


class InvokeLinear8bitLt(bnb.nn.Linear8bitLt):
    """Wraps `bnb.nn.Linear8bitLt` and adds the following functionality:
    - enables instantiation directly on the device
    - re-quantizaton when loading the state dict
    """

    def __init__(
        self, *args: Any, device: Optional[torch.device] = None, threshold: float = 6.0, **kwargs: Any
    ) -> None:
        super().__init__(*args, device=device, threshold=threshold, **kwargs)
        # If the device is CUDA or we are under a CUDA context manager, quantize the weight here, so we don't end up
        # filling the device memory with float32 weights which could lead to OOM
        # if torch.tensor(0, device=device).device.type == "cuda":
        #     self.quantize_()
        # self._register_load_state_dict_pre_hook(partial(_quantize_on_load_hook, self.quantize_))
        # self.register_load_state_dict_post_hook(_ignore_missing_weights_hook)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        unexpected_copy = list(unexpected_keys)

        for key in unexpected_copy:
            input_name = key[len(prefix) :]
            if input_name == "SCB":
                if self.weight.SCB is None:
                    # buffers not yet initialized, can't access them directly without quantizing first
                    raise RuntimeError(
                        "Loading a quantized checkpoint into non-quantized Linear8bitLt is "
                        "not supported. Please call module.cuda() before module.load_state_dict()",
                    )

                input_param = state_dict[key]
                self.weight.SCB.copy_(input_param)

                if self.state.SCB is not None:
                    self.state.SCB = self.weight.SCB

                unexpected_keys.remove(key)

    def quantize_(self, weight: Optional[torch.Tensor] = None, device: Optional[torch.device] = None) -> None:
        """Inplace quantize."""
        if weight is None:
            weight = self.weight.data
        if weight.data.dtype == torch.int8:
            # already quantized
            return
        assert isinstance(self.weight, bnb.nn.Int8Params)
        self.weight = self.quantize(self.weight, weight, device)

    @staticmethod
    def quantize(
        int8params: bnb.nn.Int8Params, weight: torch.Tensor, device: Optional[torch.device]
    ) -> bnb.nn.Int8Params:
        device = device or torch.device("cuda")
        if device.type != "cuda":
            raise RuntimeError(f"Unexpected device type: {device.type}")
        # https://github.com/TimDettmers/bitsandbytes/blob/0.41.0/bitsandbytes/nn/modules.py#L291-L302
        B = weight.contiguous().to(device=device, dtype=torch.float16)
        if int8params.has_fp16_weights:
            int8params.data = B
        else:
            CB, CBt, SCB, SCBt, _ = bnb.functional.double_quant(B)
            del CBt
            del SCBt
            int8params.data = CB
            int8params.CB = CB
            int8params.SCB = SCB
        return int8params


# class _Linear4bit(bnb.nn.Linear4bit):
#     """Wraps `bnb.nn.Linear4bit` to enable: instantiation directly on the device, re-quantizaton when loading the
#     state dict, meta-device initialization, and materialization."""

#     def __init__(self, *args: Any, device: Optional[torch.device] = None, **kwargs: Any) -> None:
#         super().__init__(*args, device=device, **kwargs)
#         self.weight = cast(bnb.nn.Params4bit, self.weight)  # type: ignore[has-type]
#         self.bias = cast(Optional[torch.nn.Parameter], self.bias)  # type: ignore[has-type]
#         # if the device is CUDA or we are under a CUDA context manager, quantize the weight here, so we don't end up
#         # filling the device memory with float32 weights which could lead to OOM
#         if torch.tensor(0, device=device).device.type == "cuda":
#             self.quantize_()
#         self._register_load_state_dict_pre_hook(partial(_quantize_on_load_hook, self.quantize_))
#         self.register_load_state_dict_post_hook(_ignore_missing_weights_hook)

#     def quantize_(self, weight: Optional[torch.Tensor] = None, device: Optional[torch.device] = None) -> None:
#         """Inplace quantize."""
#         if weight is None:
#             weight = self.weight.data
#         if weight.data.dtype == torch.uint8:
#             # already quantized
#             return
#         assert isinstance(self.weight, bnb.nn.Params4bit)
#         self.weight = self.quantize(self.weight, weight, device)

#     @staticmethod
#     def quantize(
#         params4bit: bnb.nn.Params4bit, weight: torch.Tensor, device: Optional[torch.device]
#     ) -> bnb.nn.Params4bit:
#         device = device or torch.device("cuda")
#         if device.type != "cuda":
#             raise RuntimeError(f"Unexpected device type: {device.type}")
#         # https://github.com/TimDettmers/bitsandbytes/blob/0.41.0/bitsandbytes/nn/modules.py#L156-L159
#         w = weight.contiguous().to(device=device, dtype=torch.half)
#         w_4bit, quant_state = bnb.functional.quantize_4bit(
#             w,
#             blocksize=params4bit.blocksize,
#             compress_statistics=params4bit.compress_statistics,
#             quant_type=params4bit.quant_type,
#         )
#         return _replace_param(params4bit, w_4bit, quant_state)

#     def to_empty(self, *, device: _DEVICE, recurse: bool = True) -> Self:
#         if self.weight.dtype == torch.uint8:  # was quantized
#             # cannot init the quantized params directly
#             weight = torch.empty(self.weight.quant_state.shape, device=device, dtype=torch.half)
#         else:
#             weight = torch.empty_like(self.weight.data, device=device)
#         device = torch.device(device)
#         if device.type == "cuda":  # re-quantize
#             self.quantize_(weight, device)
#         else:
#             self.weight = _replace_param(self.weight, weight)
#         if self.bias is not None:
#             self.bias = _replace_param(self.bias, torch.empty_like(self.bias, device=device))
#         return self


def convert_model_to_bnb_llm_int8(model: torch.nn.Module, ignore_modules: set[str]):
    linear_cls = InvokeLinear8bitLt
    _convert_linear_layers(model, linear_cls, ignore_modules)

    # TODO(ryand): Is this necessary?
    # set the compute dtype if necessary
    # for m in model.modules():
    #     if isinstance(m, bnb.nn.Linear4bit):
    #         m.compute_dtype = self.dtype
    #         m.compute_type_is_set = False


# class BitsandbytesPrecision(Precision):
#     """Plugin for quantizing weights with `bitsandbytes <https://github.com/TimDettmers/bitsandbytes>`__.

#     .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

#     .. note::
#         The optimizer is not automatically replaced with ``bitsandbytes.optim.Adam8bit`` or equivalent 8-bit optimizers.

#     Args:
#         mode: The quantization mode to use.
#         dtype: The compute dtype to use.
#         ignore_modules: The submodules whose Linear layers should not be replaced, for example. ``{"lm_head"}``.
#             This might be desirable for numerical stability. The string will be checked in as a prefix, so a value like
#             "transformer.blocks" will ignore all linear layers in all of the transformer blocks.
#     """

#     def __init__(
#         self,
#         mode: Literal["nf4", "nf4-dq", "fp4", "fp4-dq", "int8", "int8-training"],
#         dtype: Optional[torch.dtype] = None,
#         ignore_modules: Optional[Set[str]] = None,
#     ) -> None:
#         if dtype is None:
#             # try to be smart about the default selection
#             if mode.startswith("int8"):
#                 dtype = torch.float16
#             else:
#                 dtype = (
#                     torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
#                 )
#         if mode.startswith("int8") and dtype is not torch.float16:
#             # this limitation is mentioned in https://huggingface.co/blog/hf-bitsandbytes-integration#usage
#             raise ValueError(f"{mode!r} only works with `dtype=torch.float16`, but you chose `{dtype}`")

#         globals_ = globals()
#         mode_to_cls = {
#             "nf4": globals_["_NF4Linear"],
#             "nf4-dq": globals_["_NF4DQLinear"],
#             "fp4": globals_["_FP4Linear"],
#             "fp4-dq": globals_["_FP4DQLinear"],
#             "int8-training": globals_["_Linear8bitLt"],
#             "int8": globals_["_Int8LinearInference"],
#         }
#         self._linear_cls = mode_to_cls[mode]
#         self.dtype = dtype
#         self.ignore_modules = ignore_modules or set()

#     @override
#     def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
#         # avoid naive users thinking they quantized their model
#         if not any(isinstance(m, torch.nn.Linear) for m in module.modules()):
#             raise TypeError(
#                 "You are using the bitsandbytes precision plugin, but your model has no Linear layers. This plugin"
#                 " won't work for your model."
#             )

#         # convert modules if they haven't been converted already
#         if not any(isinstance(m, (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)) for m in module.modules()):
#             # this will not quantize the model but only replace the layer classes
#             _convert_layers(module, self._linear_cls, self.ignore_modules)

#         # set the compute dtype if necessary
#         for m in module.modules():
#             if isinstance(m, bnb.nn.Linear4bit):
#                 m.compute_dtype = self.dtype
#                 m.compute_type_is_set = False
#         return module


# def _quantize_on_load_hook(quantize_fn: Callable[[torch.Tensor], None], state_dict: OrderedDict, *_: Any) -> None:
#     # There is only one key that ends with `*.weight`, the other one is the bias
#     weight_key = next((name for name in state_dict if name.endswith("weight")), None)
#     if weight_key is None:
#         return
#     # Load the weight from the state dict and re-quantize it
#     weight = state_dict.pop(weight_key)
#     quantize_fn(weight)


# def _ignore_missing_weights_hook(module: torch.nn.Module, incompatible_keys: _IncompatibleKeys) -> None:
#     # since we manually loaded the weight in the `_quantize_on_load_hook` hook, we need to avoid this missing key false
#     # positive
#     for key in reversed(incompatible_keys.missing_keys):
#         if key.endswith("weight"):
#             incompatible_keys.missing_keys.remove(key)


def _replace_param(
    param: torch.nn.Parameter, data: torch.Tensor, quant_state: Optional[Tuple] = None
) -> torch.nn.Parameter:
    # doing `param.data = weight` raises a RuntimeError if param.data was on meta-device, so
    # we need to re-create the parameters instead of overwriting the data
    if param.device.type == "meta":
        if isinstance(param, bnb.nn.Params4bit):
            return bnb.nn.Params4bit(
                data,
                requires_grad=data.requires_grad,
                quant_state=quant_state,
                compress_statistics=param.compress_statistics,
                quant_type=param.quant_type,
            )
        return torch.nn.Parameter(data, requires_grad=data.requires_grad)
    param.data = data
    if isinstance(param, bnb.nn.Params4bit):
        param.quant_state = quant_state
    return param


def _convert_linear_layers(
    module: torch.nn.Module, linear_cls: Type, ignore_modules: Set[str], prefix: str = ""
) -> None:
    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear) and not any(fullname.startswith(s) for s in ignore_modules):
            has_bias = child.bias is not None
            # since we are going to copy over the child's data, the device doesn't matter. I chose CPU
            # to avoid spiking CUDA memory even though initialization is slower
            # 4bit layers support quantizing from meta-device params so this is only relevant for 8-bit
            _Linear4bit = globals()["_Linear4bit"]
            device = torch.device("meta" if issubclass(linear_cls, _Linear4bit) else "cpu")
            replacement = linear_cls(
                child.in_features,
                child.out_features,
                bias=has_bias,
                device=device,
            )
            if has_bias:
                replacement.bias = _replace_param(replacement.bias, child.bias.data.clone())
            state = {"quant_state": replacement.weight.quant_state if issubclass(linear_cls, _Linear4bit) else None}
            replacement.weight = _replace_param(replacement.weight, child.weight.data.clone(), **state)
            module.__setattr__(name, replacement)
        else:
            _convert_linear_layers(child, linear_cls, ignore_modules, prefix=fullname)


def _convert_linear_layers_to_llm_8bit(module: torch.nn.Module, ignore_modules: Set[str], prefix: str = "") -> None:
    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear) and not any(fullname.startswith(s) for s in ignore_modules):
            has_bias = child.bias is not None
            replacement = InvokeLinear8bitLt(
                child.in_features,
                child.out_features,
                bias=has_bias,
                has_fp16_weights=False,
                # device=device,
            )
            replacement.weight.data = child.weight.data
            if has_bias:
                replacement.bias.data = child.bias.data
            replacement.requires_grad_(False)
            module.__setattr__(name, replacement)
        else:
            _convert_linear_layers_to_llm_8bit(child, ignore_modules, prefix=fullname)


def _convert_linear_layers_to_nf4(
    module: torch.nn.Module, ignore_modules: Set[str], compute_dtype: torch.dtype, prefix: str = ""
) -> None:
    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear) and not any(fullname.startswith(s) for s in ignore_modules):
            has_bias = child.bias is not None
            replacement = InvokeLinearNF4(
                child.in_features,
                child.out_features,
                bias=has_bias,
                compute_dtype=torch.float16,
                # TODO(ryand): Test compress_statistics=True.
                # compress_statistics=True,
            )
            # replacement.weight.data = child.weight.data
            # if has_bias:
            #     replacement.bias.data = child.bias.data
            if has_bias:
                replacement.bias = _replace_param(replacement.bias, child.bias.data)
            replacement.weight = _replace_param(
                replacement.weight, child.weight.data, quant_state=replacement.weight.quant_state
            )
            replacement.requires_grad_(False)
            module.__setattr__(name, replacement)
        else:
            _convert_linear_layers_to_nf4(child, ignore_modules, compute_dtype=compute_dtype, prefix=fullname)


# def _replace_linear_layers(
#     model: torch.nn.Module,
#     linear_layer_type: Literal["Linear8bitLt", "Linear4bit"],
#     modules_to_not_convert: set[str],
#     current_key_name: str | None = None,
# ):
#     has_been_replaced = False
#     for name, module in model.named_children():
#         if current_key_name is None:
#             current_key_name = []
#         current_key_name.append(name)
#         if isinstance(module, torch.nn.Linear) and name not in modules_to_not_convert:
#             # Check if the current key is not in the `modules_to_not_convert`
#             current_key_name_str = ".".join(current_key_name)
#             proceed = True
#             for key in modules_to_not_convert:
#                 if (
#                     (key in current_key_name_str) and (key + "." in current_key_name_str)
#                 ) or key == current_key_name_str:
#                     proceed = False
#                     break
#             if proceed:
#                 # Load bnb module with empty weight and replace ``nn.Linear` module
#                 if bnb_quantization_config.load_in_8bit:
#                     bnb_module = bnb.nn.Linear8bitLt(
#                         module.in_features,
#                         module.out_features,
#                         module.bias is not None,
#                         has_fp16_weights=False,
#                         threshold=bnb_quantization_config.llm_int8_threshold,
#                     )
#                 elif bnb_quantization_config.load_in_4bit:
#                     bnb_module = bnb.nn.Linear4bit(
#                         module.in_features,
#                         module.out_features,
#                         module.bias is not None,
#                         bnb_quantization_config.bnb_4bit_compute_dtype,
#                         compress_statistics=bnb_quantization_config.bnb_4bit_use_double_quant,
#                         quant_type=bnb_quantization_config.bnb_4bit_quant_type,
#                     )
#                 else:
#                     raise ValueError("load_in_8bit and load_in_4bit can't be both False")
#                 bnb_module.weight.data = module.weight.data
#                 if module.bias is not None:
#                     bnb_module.bias.data = module.bias.data
#                 bnb_module.requires_grad_(False)
#                 setattr(model, name, bnb_module)
#                 has_been_replaced = True
#         if len(list(module.children())) > 0:
#             _, _has_been_replaced = _replace_with_bnb_layers(
#                 module, bnb_quantization_config, modules_to_not_convert, current_key_name
#             )
#             has_been_replaced = has_been_replaced | _has_been_replaced
#         # Remove the last key for recursion
#         current_key_name.pop(-1)
#     return model, has_been_replaced


def get_parameter_device(parameter: torch.nn.Module):
    return next(parameter.parameters()).device


def quantize_model_llm_int8(model: torch.nn.Module, modules_to_not_convert: set[str]):
    """Apply bitsandbytes LLM.8bit() quantization to the model."""
    model_device = get_parameter_device(model)
    if model_device.type != "meta":
        # Note: This is not strictly required, but I can't think of a good reason to quantize a model that's not on the
        # meta device, so we enforce it for now.
        raise RuntimeError("The model should be on the meta device to apply LLM.8bit() quantization.")

    with accelerate.init_empty_weights():
        _convert_linear_layers_to_llm_8bit(module=model, ignore_modules=modules_to_not_convert)

    return model


def quantize_model_nf4(model: torch.nn.Module, modules_to_not_convert: set[str], compute_dtype: torch.dtype):
    """Apply bitsandbytes nf4 quantization to the model."""
    # model_device = get_parameter_device(model)
    # if model_device.type != "meta":
    #     # Note: This is not strictly required, but I can't think of a good reason to quantize a model that's not on the
    #     # meta device, so we enforce it for now.
    #     raise RuntimeError("The model should be on the meta device to apply LLM.8bit() quantization.")

    # with accelerate.init_empty_weights():
    _convert_linear_layers_to_nf4(module=model, ignore_modules=modules_to_not_convert, compute_dtype=compute_dtype)

    return model
