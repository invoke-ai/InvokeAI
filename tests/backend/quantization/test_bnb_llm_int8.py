import logging
import warnings

import pytest
import torch

try:
    from invokeai.backend.quantization.bnb_llm_int8 import InvokeLinear8bitLt, silence_int8_cast_notice
except ImportError:
    pass

_CAST_NOTICE = "MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"


def test_invoke_linear_8bit_lt_quantization():
    """Test quantization with InvokeLinear8bitLt."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Set the seed for reproducibility since we are using a pretty tight atol.
    torch.manual_seed(3)

    orig_layer = torch.nn.Linear(32, 64)
    orig_layer_state_dict = orig_layer.state_dict()

    # Initialize a InvokeLinear8bitLt layer (it is not quantized yet).
    quantized_layer = InvokeLinear8bitLt(input_features=32, output_features=64, has_fp16_weights=False)

    # Load the non-quantized layer's state dict into the quantized layer.
    quantized_layer.load_state_dict(orig_layer_state_dict)

    # Move the InvokeLinear8bitLt layer to the GPU. This triggers quantization.
    quantized_layer.to("cuda")

    # Assert that the InvokeLinear8bitLt layer is quantized.
    assert quantized_layer.weight.CB is not None
    assert quantized_layer.weight.SCB is not None
    assert quantized_layer.weight.CB.dtype == torch.int8

    # Run inference on both the original and quantized layers.
    x = torch.randn(1, 32)
    y = orig_layer(x)
    y_quantized = quantized_layer(x.to("cuda"))
    assert y.shape == y_quantized.shape
    # All within ~20% of each other.
    assert torch.allclose(y, y_quantized.to("cpu"), atol=0.05)


def test_invoke_linear_8bit_lt_state_dict_roundtrip():
    """Test that we can roundtrip the state dict of a quantized InvokeLinear8bitLt layer."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Set the seed for reproducibility since we are using a pretty tight atol.
    torch.manual_seed(3)

    orig_layer = torch.nn.Linear(32, 64)
    orig_layer_state_dict = orig_layer.state_dict()

    # Run inference on the original layer.
    x = torch.randn(1, 32)
    y = orig_layer(x)

    # Prepare a quantized InvokeLinear8bitLt layer.
    quantized_layer_1 = InvokeLinear8bitLt(input_features=32, output_features=64, has_fp16_weights=False)
    quantized_layer_1.load_state_dict(orig_layer_state_dict)
    quantized_layer_1.to("cuda")

    # Assert that the InvokeLinear8bitLt layer is quantized.
    assert quantized_layer_1.weight.CB is not None
    assert quantized_layer_1.weight.SCB is not None
    assert quantized_layer_1.weight.CB.dtype == torch.int8

    # Run inference on the quantized layer.
    y_quantized_1 = quantized_layer_1(x.to("cuda"))

    # Save the state dict of the quantized layer.
    quantized_layer_1_state_dict = quantized_layer_1.state_dict()

    # Load the state dict of the quantized layer into a new quantized layer.
    quantized_layer_2 = InvokeLinear8bitLt(input_features=32, output_features=64, has_fp16_weights=False)
    quantized_layer_2.load_state_dict(quantized_layer_1_state_dict)
    quantized_layer_2.to("cuda")

    # Run inference on the new quantized layer.
    y_quantized_2 = quantized_layer_2(x.to("cuda"))

    # Assert that the inference results are the same.
    assert torch.allclose(y, y_quantized_1.to("cpu"), atol=0.05)
    assert torch.allclose(y_quantized_1, y_quantized_2, atol=1e-5)


def test_int8_cast_notice_is_silenced_as_a_warning_and_as_a_log_record(caplog):
    """bitsandbytes up to 0.49 warns with `warnings`; from 0.50 it logs. Other notices from either path still show."""
    pytest.importorskip("bitsandbytes")
    bnb_logger = logging.getLogger("bitsandbytes.autograd._functions")
    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING, logger=bnb_logger.name):
        warnings.simplefilter("always")
        silence_int8_cast_notice()
        silence_int8_cast_notice()  # idempotent: one log filter, not two

        warnings.warn(_CAST_NOTICE, UserWarning, stacklevel=1)
        warnings.warn("an unrelated bitsandbytes warning", UserWarning, stacklevel=1)
        bnb_logger.warning("MatMul8bitLt: inputs will be cast from %s to float16 during quantization", torch.bfloat16)
        bnb_logger.warning("an unrelated bitsandbytes log record")

    assert [str(w.message) for w in caught] == ["an unrelated bitsandbytes warning"]
    assert [r.getMessage() for r in caplog.records] == ["an unrelated bitsandbytes log record"]
    assert sum(type(f).__name__ == "_DropInt8CastNotice" for f in bnb_logger.filters) == 1


def test_a_bf16_int8_matmul_reports_no_cast_notice(caplog):
    """The installed bitsandbytes, not a stand-in: a renamed logger or message would show up here."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    layer = InvokeLinear8bitLt(input_features=32, output_features=64, has_fp16_weights=False)
    layer.load_state_dict(torch.nn.Linear(32, 64).state_dict())
    layer.to("cuda")

    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING):
        warnings.simplefilter("always")
        silence_int8_cast_notice()
        layer(torch.randn(4, 32, dtype=torch.bfloat16, device="cuda"))
        layer(torch.randn(4, 32, dtype=torch.bfloat16, device="cuda"))

    assert not [w for w in caught if "MatMul8bitLt" in str(w.message)]
    assert not [r for r in caplog.records if "MatMul8bitLt" in r.getMessage()]
