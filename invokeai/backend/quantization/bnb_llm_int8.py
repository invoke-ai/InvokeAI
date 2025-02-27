import bitsandbytes as bnb
import torch

# This file contains utils for working with models that use bitsandbytes LLM.int8() quantization.
# The utils in this file are partially inspired by:
# https://github.com/Lightning-AI/pytorch-lightning/blob/1551a16b94f5234a4a78801098f64d0732ef5cb5/src/lightning/fabric/plugins/precision/bitsandbytes.py


# NOTE(ryand): All of the custom state_dict manipulation logic in this file is pretty hacky. This could be made much
# cleaner by re-implementing bnb.nn.Linear8bitLt with proper use of buffers and less magic. But, for now, we try to
# stick close to the bitsandbytes classes to make interoperability easier with other models that might use bitsandbytes.


class InvokeInt8Params(bnb.nn.Int8Params):
    """We override cuda() to avoid re-quantizing the weights in the following cases:
    - We loaded quantized weights from a state_dict on the cpu, and then moved the model to the gpu.
    - We are moving the model back-and-forth between the cpu and gpu.
    """

    def cuda(self, device):
        if self.has_fp16_weights:
            return super().cuda(device)
        elif self.CB is not None and self.SCB is not None:
            self.data = self.data.cuda()
            self.CB = self.data
            self.SCB = self.SCB.cuda()
        else:
            # We quantize the weight and store in 8bit row-major
            B = self.data.contiguous().half().cuda(device)
            CB, SCB, _ = bnb.functional.int8_vectorwise_quant(B)
            self.data = CB
            self.CB = CB
            self.SCB = SCB

        return self


class InvokeLinear8bitLt(bnb.nn.Linear8bitLt):
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
        weight = state_dict.pop(prefix + "weight")
        bias = state_dict.pop(prefix + "bias", None)

        # See `bnb.nn.Linear8bitLt._save_to_state_dict()` for the serialization logic of SCB and weight_format.
        scb = state_dict.pop(prefix + "SCB", None)

        weight_format = state_dict.pop(prefix + "weight_format", None)
        if weight_format is not None:
            # Currently, we only support weight_format=0.
            assert weight_format == 0

        # TODO(ryand): Technically, we should be using `strict`, `missing_keys`, `unexpected_keys`, and `error_msgs`
        # rather than raising an exception to correctly implement this API.
        assert len(state_dict) == 0

        if scb is not None:
            # We are loading a pre-quantized state dict.
            self.weight = InvokeInt8Params(
                data=weight,
                requires_grad=self.weight.requires_grad,
                has_fp16_weights=False,
                # Note: After quantization, CB is the same as weight.
                CB=weight,
                SCB=scb,
            )
            self.bias = bias if bias is None else torch.nn.Parameter(bias)
        else:
            # We are loading a non-quantized state dict.

            # We could simply call the `super()._load_from_state_dict()` method here, but then we wouldn't be able to
            # load from a state_dict into a model on the "meta" device. Attempting to load into a model on the "meta"
            # device requires setting `assign=True`, doing this with the default `super()._load_from_state_dict()`
            # implementation causes `Params4Bit` to be replaced by a `torch.nn.Parameter`. By initializing a new
            # `Params4bit` object, we work around this issue. It's a bit hacky, but it gets the job done.
            self.weight = InvokeInt8Params(
                data=weight,
                requires_grad=self.weight.requires_grad,
                has_fp16_weights=False,
                CB=None,
                SCB=None,
            )
            self.bias = bias if bias is None else torch.nn.Parameter(bias)

        # Reset the state. The persisted fields are based on the initialization behaviour in
        # `bnb.nn.Linear8bitLt.__init__()`.
        new_state = bnb.MatmulLtState()
        new_state.threshold = self.state.threshold
        new_state.has_fp16_weights = False
        new_state.use_pool = self.state.use_pool
        self.state = new_state

    def forward(self, x: torch.Tensor):
        # The state management in the base bnb.nn.Linear8bitLt is very convoluted. We override the forward method to
        # try to simplify the state management a bit. We initialize a new MatmulLtState object for each forward pass.
        # By avoiding persistent state, it is easier to move the layer between devices without worrying about keeping
        # references to weights on the old device (e.g. self.state.CB).
        matmul_state = bnb.MatmulLtState()
        matmul_state.threshold = self.state.threshold
        matmul_state.has_fp16_weights = self.state.has_fp16_weights
        matmul_state.use_pool = self.state.use_pool
        matmul_state.is_training = self.training
        # The underlying InvokeInt8Params weight must already be quantized.
        assert self.weight.CB is not None
        matmul_state.CB = self.weight.CB
        matmul_state.SCB = self.weight.SCB

        # weights are cast automatically as Int8Params, but the bias has to be cast manually.
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        return bnb.matmul(x, self.weight, bias=self.bias, state=matmul_state)


def _convert_linear_layers_to_llm_8bit(
    module: torch.nn.Module, ignore_modules: set[str], outlier_threshold: float, prefix: str = ""
) -> None:
    """Convert all linear layers in the module to bnb.nn.Linear8bitLt layers."""
    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear) and not any(fullname.startswith(s) for s in ignore_modules):
            has_bias = child.bias is not None
            replacement = InvokeLinear8bitLt(
                child.in_features,
                child.out_features,
                bias=has_bias,
                has_fp16_weights=False,
                threshold=outlier_threshold,
            )
            replacement.weight.data = child.weight.data
            if has_bias:
                replacement.bias.data = child.bias.data
            replacement.requires_grad_(False)
            module.__setattr__(name, replacement)
        else:
            _convert_linear_layers_to_llm_8bit(
                child, ignore_modules, outlier_threshold=outlier_threshold, prefix=fullname
            )


def quantize_model_llm_int8(model: torch.nn.Module, modules_to_not_convert: set[str], outlier_threshold: float = 6.0):
    """Apply bitsandbytes LLM.8bit() quantization to the model."""
    _convert_linear_layers_to_llm_8bit(
        module=model, ignore_modules=modules_to_not_convert, outlier_threshold=outlier_threshold
    )

    return model
