"""Ideogram 4 starter models."""

from invokeai.backend.model_manager.starter_models.flux2 import flux2_vae
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
)

# region Ideogram 4
# Self-contained diffusers pipelines (both transformers + Qwen3-VL text encoder + VAE in one folder), so
# no separate dependencies. Gated, non-commercial license: the license must be accepted on the
# HuggingFace model page and a HuggingFace token configured before the download will succeed — same as
# FLUX.1 dev.
ideogram_4_nf4 = StarterModel(
    name="Ideogram 4 (nf4)",
    base=BaseModelType.Ideogram4,
    source="ideogram-ai/ideogram-4-nf4",
    description="Ideogram 4 text-to-image in nf4-quantized Diffusers format (CUDA only). Structured JSON "
    "prompting with regional layout control. Non-commercial license — accept it on HuggingFace first. ~16GB",
    type=ModelType.Main,
)

ideogram_4_fp8 = StarterModel(
    name="Ideogram 4 (fp8)",
    base=BaseModelType.Ideogram4,
    source="ideogram-ai/ideogram-4-fp8",
    description="Ideogram 4 text-to-image in fp8-quantized Diffusers format (runs on any device, higher "
    "memory use). Non-commercial license — accept it on HuggingFace first. ~26GB",
    type=ModelType.Main,
)

# Comfy-Org's single-file release. Each file is ONE of the two branches, both of which run at every
# step, so the conditional transformer pulls the unconditional one in as a dependency along with the
# encoder and the VAE. The repo is ungated, unlike the diffusers pipelines above, but the same
# non-commercial license applies.
#
# The build is named on both branches of both pairs, because the user picks the unconditional branch
# by name in Components and any conditional accepts any unconditional. Dropping it from one name is
# how a user ends up guiding an int8 branch against an fp8 one without meaning to.
ideogram_4_unconditional_single_file = StarterModel(
    name="Ideogram 4 Unconditional (single file, fp8)",
    previous_names=["Ideogram 4 Unconditional (single file)"],
    base=BaseModelType.Ideogram4,
    source="https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors",
    description="The unconditional branch of Ideogram 4, in ComfyUI scaled fp8. Useless on its own — "
    "Ideogram 4 guides its conditional branch against this one. ~8.6GB",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
)

ideogram_4_qwen3_vl_encoder_8b = StarterModel(
    name="Qwen3-VL 8B Encoder (Ideogram 4)",
    base=BaseModelType.Any,
    source="https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/text_encoders/qwen3vl_8b_fp8_scaled.safetensors",
    description="Qwen3-VL 8B text encoder for Ideogram 4, in ComfyUI scaled fp8. Distinct from the 4B "
    "encoder Krea-2 uses; Ideogram 4 taps 13 of its layers and the two are not interchangeable. ~9.9GB",
    type=ModelType.Qwen3VLEncoder,
    format=ModelFormat.Checkpoint,
)

ideogram_4_unconditional_int8 = StarterModel(
    name="Ideogram 4 Unconditional (single file, int8)",
    base=BaseModelType.Ideogram4,
    source="https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_unconditional_int8_convrot.safetensors",
    description="The unconditional branch of Ideogram 4, in ComfyUI int8_tensorwise. Useless on its "
    "own — Ideogram 4 guides its conditional branch against this one. ~8.9GB",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
)

ideogram_4_int8 = StarterModel(
    name="Ideogram 4 (single file, int8)",
    base=BaseModelType.Ideogram4,
    source="https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_int8_convrot.safetensors",
    description="Comfy-Org single-file Ideogram 4 in ComfyUI int8_tensorwise. Unlike the fp8 build "
    "it stays at its download size in memory on every device — 8.9GB per branch rather than ~17GB — "
    "because the weights are dequantized per forward instead of on load. Installs the unconditional "
    "branch, the Qwen3-VL 8B encoder and the VAE with it. Non-commercial license. ~28GB total",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    dependencies=[ideogram_4_unconditional_int8, ideogram_4_qwen3_vl_encoder_8b, flux2_vae],
)

ideogram_4_single_file = StarterModel(
    name="Ideogram 4 (single file, fp8)",
    previous_names=["Ideogram 4 (single file)"],
    base=BaseModelType.Ideogram4,
    source="https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_fp8_scaled.safetensors",
    description="Comfy-Org single-file Ideogram 4 in ComfyUI scaled fp8. Installs the unconditional "
    "branch, the Qwen3-VL 8B encoder and the shared 32-channel VAE with it; both transformer branches "
    "stay resident during generation. Non-commercial license. ~27GB total",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    dependencies=[ideogram_4_unconditional_single_file, ideogram_4_qwen3_vl_encoder_8b, flux2_vae],
)

# Community GGUF conversions of the same per-branch files, paired the same way. They are the smallest
# Ideogram 4 builds that run on any device: the Linear weights stay packed and dequantize per forward,
# so the download size is roughly the resident size. Pinned to a commit: none of these files carries
# metadata, so the branch is read from the filename, and a reupload under the same path could change
# what a pinned name means.
#
# Q8_0 is left out on purpose: at ~10.1GB per branch it is larger than the fp8 build it would replace.
_MOLBAL_GGUF = "https://huggingface.co/molbal/ideogram-4-gguf/resolve/83e58701001a85d11774c13a8b2baf1c77da3f27"
# The only K-quant pair converted with one recipe for both branches. It packs the norms and the
# embedding as well, which the loader dequantizes once at load.
_RECTANGLEWORM_GGUF = (
    "https://huggingface.co/rectangleworm/ideogram-4-gguf/resolve/7b353b17986757de25ecc1be87406adf09d01d21/diffusion"
)

ideogram_4_unconditional_gguf_q4_0 = StarterModel(
    name="Ideogram 4 Unconditional (GGUF, Q4_0)",
    base=BaseModelType.Ideogram4,
    source=f"{_MOLBAL_GGUF}/ideogram4-unconditional_transformer-q4_0.gguf",
    description="The unconditional branch of Ideogram 4, in GGUF Q4_0. Useless on its own — Ideogram 4 "
    "guides its conditional branch against this one. ~5.6GB",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
)

ideogram_4_gguf_q4_0 = StarterModel(
    name="Ideogram 4 (GGUF, Q4_0)",
    base=BaseModelType.Ideogram4,
    source=f"{_MOLBAL_GGUF}/ideogram4-transformer-q4_0.gguf",
    description="Community GGUF of Ideogram 4 in Q4_0, the smallest build: 5.6GB per branch in memory on "
    "every device, against 8.7GB for fp8 with FP8 Storage (about 17GB without). Installs the unconditional "
    "branch, the Qwen3-VL 8B encoder and the VAE with it. Non-commercial license. ~21GB total",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[ideogram_4_unconditional_gguf_q4_0, ideogram_4_qwen3_vl_encoder_8b, flux2_vae],
)

ideogram_4_unconditional_gguf_q5_k = StarterModel(
    name="Ideogram 4 Unconditional (GGUF, Q5_K)",
    base=BaseModelType.Ideogram4,
    source=f"{_RECTANGLEWORM_GGUF}/uncond/ideogram4_unconditional_Q5_K.gguf",
    description="The unconditional branch of Ideogram 4, in GGUF Q5_K. Useless on its own — Ideogram 4 "
    "guides its conditional branch against this one. ~6.4GB",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
)

ideogram_4_gguf_q5_k = StarterModel(
    name="Ideogram 4 (GGUF, Q5_K)",
    base=BaseModelType.Ideogram4,
    source=f"{_RECTANGLEWORM_GGUF}/cond/ideogram4_Q5_K.gguf",
    description="Community GGUF of Ideogram 4 in Q5_K: 6.4GB per branch in memory, on every device. "
    "Installs the unconditional branch, the Qwen3-VL 8B encoder and the VAE with it. Non-commercial "
    "license. ~23GB total",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[ideogram_4_unconditional_gguf_q5_k, ideogram_4_qwen3_vl_encoder_8b, flux2_vae],
)

ideogram_4_unconditional_gguf_q5_1 = StarterModel(
    name="Ideogram 4 Unconditional (GGUF, Q5_1)",
    base=BaseModelType.Ideogram4,
    source=f"{_MOLBAL_GGUF}/ideogram4-unconditional_transformer-q5_1.gguf",
    description="The unconditional branch of Ideogram 4, in GGUF Q5_1. Useless on its own — Ideogram 4 "
    "guides its conditional branch against this one. ~7.3GB",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
)

ideogram_4_gguf_q5_1 = StarterModel(
    name="Ideogram 4 (GGUF, Q5_1)",
    base=BaseModelType.Ideogram4,
    source=f"{_MOLBAL_GGUF}/ideogram4-transformer-q5_1.gguf",
    description="Community GGUF of Ideogram 4 in Q5_1, the largest of these starters: 7.3GB per branch "
    "in memory, on every device. Installs the unconditional branch, the Qwen3-VL 8B "
    "encoder and the VAE with it. Non-commercial license. ~25GB total",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[ideogram_4_unconditional_gguf_q5_1, ideogram_4_qwen3_vl_encoder_8b, flux2_vae],
)
