"""Wan starter models."""

from invokeai.backend.model_manager.starter_models.common import wan_22_t5_encoder
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
    WanVariantType,
)

wan_22_a14b_vae = StarterModel(
    name="Wan 2.2 A14B VAE",
    base=BaseModelType.Wan,
    source="Wan-AI/Wan2.2-T2V-A14B-Diffusers::vae/diffusion_pytorch_model.safetensors",
    description="Wan 2.2 A14B VAE (16-channel). Shared between T2V and I2V A14B variants. "
    "Not interchangeable with the TI2V-5B VAE. (~250MB)",
    type=ModelType.VAE,
    format=ModelFormat.Checkpoint,
)

wan_22_5b_vae = StarterModel(
    name="Wan 2.2 TI2V-5B VAE",
    base=BaseModelType.Wan,
    source="Wan-AI/Wan2.2-TI2V-5B-Diffusers::vae/diffusion_pytorch_model.safetensors",
    description="Wan 2.2 TI2V-5B VAE (48-channel). Required for the TI2V-5B model family. "
    "Not interchangeable with the A14B VAE. (~400MB)",
    type=ModelType.VAE,
    format=ModelFormat.Checkpoint,
)

# T2V A14B — full Diffusers + GGUF expert pairs (Q4_K_M and Q8_0).
# The high-noise GGUF is the "main" entry the user picks; the low-noise GGUF
# is wired as the partner expert via the Advanced panel. Each high-noise entry
# lists its low-noise partner plus the shared VAE/encoder as dependencies so
# the bundle/dependency installer pulls everything together.
wan_22_t2v_a14b_diffusers = StarterModel(
    name="Wan 2.2 T2V A14B (Diffusers)",
    base=BaseModelType.Wan,
    source="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    description="Full Diffusers Wan 2.2 T2V A14B model — both expert transformers, VAE, and UMT5-XXL "
    "encoder in a single folder. No additional components needed. (~80GB)",
    type=ModelType.Main,
    format=ModelFormat.Diffusers,
    variant=WanVariantType.T2V_A14B,
)

wan_22_t2v_a14b_low_gguf_q4_k_m = StarterModel(
    name="Wan 2.2 T2V A14B Low Noise (Q4_K_M)",
    base=BaseModelType.Wan,
    source="https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/resolve/main/LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf",
    description="Wan 2.2 T2V A14B low-noise expert transformer (Q4_K_M). Paired with the high-noise "
    "expert; selected via the Advanced 'Transformer (Low Noise)' field. (~9.7GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=WanVariantType.T2V_A14B,
)

wan_22_t2v_a14b_gguf_q4_k_m = StarterModel(
    name="Wan 2.2 T2V A14B High Noise (Q4_K_M)",
    base=BaseModelType.Wan,
    source="https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/resolve/main/HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf",
    description="Wan 2.2 T2V A14B high-noise expert transformer (Q4_K_M). Pick this as the main model; "
    "the low-noise partner is wired in Advanced. Good quality/size balance. (~9.7GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=WanVariantType.T2V_A14B,
    dependencies=[wan_22_a14b_vae, wan_22_t5_encoder, wan_22_t2v_a14b_low_gguf_q4_k_m],
)

wan_22_t2v_a14b_low_gguf_q8_0 = StarterModel(
    name="Wan 2.2 T2V A14B Low Noise (Q8_0)",
    base=BaseModelType.Wan,
    source="https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/resolve/main/LowNoise/Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf",
    description="Wan 2.2 T2V A14B low-noise expert transformer (Q8_0). Highest quality quantization. (~15.4GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=WanVariantType.T2V_A14B,
)

wan_22_t2v_a14b_gguf_q8_0 = StarterModel(
    name="Wan 2.2 T2V A14B High Noise (Q8_0)",
    base=BaseModelType.Wan,
    source="https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/resolve/main/HighNoise/Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf",
    description="Wan 2.2 T2V A14B high-noise expert transformer (Q8_0). Pick as the main; pair with the "
    "low-noise Q8_0 partner in Advanced. Highest quality quantization. (~15.4GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=WanVariantType.T2V_A14B,
    dependencies=[wan_22_a14b_vae, wan_22_t5_encoder, wan_22_t2v_a14b_low_gguf_q8_0],
)

# T2V Lightning LoRAs — V1.1 Seko rank-64 pair (4-step inference).
wan_22_t2v_lightning_high = StarterModel(
    name="Wan 2.2 T2V Lightning High Noise (4-step, V1.1)",
    base=BaseModelType.Wan,
    source="https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
    description="Lightning distillation LoRA for the Wan 2.2 T2V A14B high-noise expert — enables "
    "4-step generation. Use together with the low-noise variant. Settings: Steps=4, CFG=1.",
    type=ModelType.LoRA,
)

wan_22_t2v_lightning_low = StarterModel(
    name="Wan 2.2 T2V Lightning Low Noise (4-step, V1.1)",
    base=BaseModelType.Wan,
    source="https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
    description="Lightning distillation LoRA for the Wan 2.2 T2V A14B low-noise expert — enables "
    "4-step generation. Use together with the high-noise variant. Settings: Steps=4, CFG=1.",
    type=ModelType.LoRA,
)

# I2V A14B — full Diffusers + GGUF expert pairs (Q4_K_M and Q8_0).
wan_22_i2v_a14b_diffusers = StarterModel(
    name="Wan 2.2 I2V A14B (Diffusers)",
    base=BaseModelType.Wan,
    source="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    description="Full Diffusers Wan 2.2 I2V A14B model — both expert transformers, VAE, and UMT5-XXL "
    "encoder. Use the Reference Images panel to provide the conditioning image. (~80GB)",
    type=ModelType.Main,
    format=ModelFormat.Diffusers,
    variant=WanVariantType.I2V_A14B,
)

wan_22_i2v_a14b_low_gguf_q4_k_m = StarterModel(
    name="Wan 2.2 I2V A14B Low Noise (Q4_K_M)",
    base=BaseModelType.Wan,
    source="https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf",
    description="Wan 2.2 I2V A14B low-noise expert transformer (Q4_K_M). (~9.7GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=WanVariantType.I2V_A14B,
)

wan_22_i2v_a14b_gguf_q4_k_m = StarterModel(
    name="Wan 2.2 I2V A14B High Noise (Q4_K_M)",
    base=BaseModelType.Wan,
    source="https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
    description="Wan 2.2 I2V A14B high-noise expert transformer (Q4_K_M). Pick as the main; pair with "
    "the low-noise partner in Advanced. Use the Reference Images panel for the conditioning image. (~9.7GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=WanVariantType.I2V_A14B,
    dependencies=[wan_22_a14b_vae, wan_22_t5_encoder, wan_22_i2v_a14b_low_gguf_q4_k_m],
)

wan_22_i2v_a14b_low_gguf_q8_0 = StarterModel(
    name="Wan 2.2 I2V A14B Low Noise (Q8_0)",
    base=BaseModelType.Wan,
    source="https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf",
    description="Wan 2.2 I2V A14B low-noise expert transformer (Q8_0). Highest quality quantization. (~15.4GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=WanVariantType.I2V_A14B,
)

wan_22_i2v_a14b_gguf_q8_0 = StarterModel(
    name="Wan 2.2 I2V A14B High Noise (Q8_0)",
    base=BaseModelType.Wan,
    source="https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/HighNoise/Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf",
    description="Wan 2.2 I2V A14B high-noise expert transformer (Q8_0). Highest quality quantization. (~15.4GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=WanVariantType.I2V_A14B,
    dependencies=[wan_22_a14b_vae, wan_22_t5_encoder, wan_22_i2v_a14b_low_gguf_q8_0],
)

# I2V Lightning LoRAs — Seko rank-64 pair (4-step inference). Currently only V1.
wan_22_i2v_lightning_high = StarterModel(
    name="Wan 2.2 I2V Lightning High Noise (4-step, V1)",
    base=BaseModelType.Wan,
    source="https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
    description="Lightning distillation LoRA for the Wan 2.2 I2V A14B high-noise expert — enables "
    "4-step image-to-image generation. Use together with the low-noise variant. Settings: Steps=4, CFG=1.",
    type=ModelType.LoRA,
)

wan_22_i2v_lightning_low = StarterModel(
    name="Wan 2.2 I2V Lightning Low Noise (4-step, V1)",
    base=BaseModelType.Wan,
    source="https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
    description="Lightning distillation LoRA for the Wan 2.2 I2V A14B low-noise expert — enables "
    "4-step image-to-image generation. Use together with the high-noise variant. Settings: Steps=4, CFG=1.",
    type=ModelType.LoRA,
)

# TI2V-5B — single-transformer model (no expert pair). Uses its own 48-channel VAE.
wan_22_ti2v_5b_diffusers = StarterModel(
    name="Wan 2.2 TI2V-5B (Diffusers)",
    base=BaseModelType.Wan,
    source="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    description="Full Diffusers Wan 2.2 TI2V-5B model — single 5B transformer, 48-channel VAE, and "
    "UMT5-XXL encoder. Smaller and faster than A14B; runs on consumer GPUs. (~20GB)",
    type=ModelType.Main,
    format=ModelFormat.Diffusers,
    variant=WanVariantType.TI2V_5B,
)

wan_22_ti2v_5b_gguf_q4_k_m = StarterModel(
    name="Wan 2.2 TI2V-5B (Q4_K_M)",
    base=BaseModelType.Wan,
    source="https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/main/Wan2.2-TI2V-5B-Q4_K_M.gguf",
    description="Wan 2.2 TI2V-5B transformer (Q4_K_M). Single-expert model — no low-noise partner needed. (~3.4GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=WanVariantType.TI2V_5B,
    dependencies=[wan_22_5b_vae, wan_22_t5_encoder],
)

wan_22_ti2v_5b_gguf_q8_0 = StarterModel(
    name="Wan 2.2 TI2V-5B (Q8_0)",
    base=BaseModelType.Wan,
    source="https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/main/Wan2.2-TI2V-5B-Q8_0.gguf",
    description="Wan 2.2 TI2V-5B transformer (Q8_0). Highest quality quantization. (~5.4GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=WanVariantType.TI2V_5B,
    dependencies=[wan_22_5b_vae, wan_22_t5_encoder],
)
