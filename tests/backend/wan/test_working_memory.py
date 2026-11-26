"""Which fitted line the Wan denoise estimate prices each kind of forward on.

The shipped-data test holds each line to its measured points, from their recorded features; this holds the nodes'
inputs to the right line. The expected bytes are the shipped `data/wan.json` coefficients, copied out.
"""

import gguf
import numpy as np
import pytest
import torch

from invokeai.backend.model_manager.taxonomy import ModelFormat, WanVariantType
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.wan.working_memory import estimate_wan_denoise_working_memory


@pytest.mark.parametrize(
    ("variant", "latent_shape", "token_timesteps", "dtype", "slope", "base"),
    [
        # 1280x720, 81 frames: 21 * 45 * 80 = 75600 tokens.
        (WanVariantType.T2V_A14B, (1, 16, 21, 90, 160), False, torch.bfloat16, 210461, 176160768),
        # The I2V expert's 36 input channels change no token count.
        (WanVariantType.I2V_A14B, (1, 16, 21, 90, 160), False, torch.float32, 310138, 553648128),
        # 1280x704, 121 frames: 31 * 22 * 40 = 27280 tokens.
        (WanVariantType.TI2V_5B, (1, 48, 31, 44, 80), False, torch.bfloat16, 150234, 308281344),
        (WanVariantType.TI2V_5B, (1, 48, 31, 44, 80), True, torch.bfloat16, 312975, 307232768),
        (WanVariantType.TI2V_5B, (1, 48, 31, 44, 80), True, torch.float32, 403217, 628097024),
    ],
    ids=["a14b-bf16", "i2v-a14b-fp32", "ti2v-bf16", "ti2v-token-timesteps-bf16", "ti2v-token-timesteps-fp32"],
)
def test_each_forward_is_priced_on_its_own_line(variant, latent_shape, token_timesteps, dtype, slope, base) -> None:
    estimate = estimate_wan_denoise_working_memory(
        torch.nn.Module(),
        variant=variant,
        latent_shape=torch.Size(latent_shape),
        token_timesteps=token_timesteps,
        model_format=ModelFormat.Diffusers,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    _, _, frames, height, width = latent_shape
    fitted = {term.name: term.bytes for term in estimate.terms if term.coefficient is not None}
    assert fitted == {"activations": frames * (height // 2) * (width // 2) * slope, "base": base}


def test_a_dtype_without_a_line_is_refused() -> None:
    with pytest.raises(ValueError, match="wan.denoise"):
        estimate_wan_denoise_working_memory(
            torch.nn.Module(),
            variant=WanVariantType.T2V_A14B,
            latent_shape=torch.Size((1, 16, 1, 8, 8)),
            model_format=ModelFormat.Diffusers,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )


@pytest.mark.parametrize(("token_timesteps", "rows"), [(False, 1), (True, 27280)], ids=["per-sample", "per-token"])
def test_a_lora_on_the_timestep_projection_sees_a_row_per_timestep(token_timesteps: bool, rows: int) -> None:
    """diffusers flattens per-token timesteps before the condition embedder, so its linears then see one row per
    token, not one per sample."""
    linear = torch.nn.Linear(256, 512, bias=False)
    raw = gguf.quantize(np.zeros((512, 256), np.float32), gguf.GGMLQuantizationType.Q8_0)
    linear.weight = torch.nn.Parameter(
        GGMLTensor(torch.from_numpy(raw), gguf.GGMLQuantizationType.Q8_0, linear.weight.shape, torch.bfloat16),
        requires_grad=False,
    )
    transformer = torch.nn.Module()
    transformer.condition_embedder = torch.nn.Module()
    transformer.condition_embedder.time_proj = linear
    lora = ModelPatchRaw(
        layers={
            "lora_transformer-condition_embedder.time_proj": LoRALayer(
                torch.zeros(512, 4), None, torch.zeros(4, 256), None, None
            )
        }
    )

    def sidecar(loras: list[ModelPatchRaw]) -> int:
        estimate = estimate_wan_denoise_working_memory(
            transformer,
            variant=WanVariantType.TI2V_5B,
            latent_shape=torch.Size((1, 48, 31, 44, 80)),
            token_timesteps=token_timesteps,
            loras=loras,
            model_format=ModelFormat.GGUFQuantized,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        return next(term.bytes for term in estimate.terms if term.name == "lora_sidecar")

    assert sidecar([]) == 0
    assert sidecar([lora]) == (rows * (512 + 4) + 512 * 4 + 4 * 256) * torch.bfloat16.itemsize
