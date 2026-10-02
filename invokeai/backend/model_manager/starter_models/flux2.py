"""FLUX.2 starter models."""

from invokeai.backend.model_manager.starter_models.common import (
    flux2_dev_cow_mistral_q4,
    flux2_dev_cow_mistral_q8,
    flux2_klein_qwen3_4b_encoder,
    flux2_klein_qwen3_8b_encoder,
    gemma2_2b_encoder,
)
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
    PiDDecoderVariantType,
)

# FLUX.2 Klein shares one 32-channel VAE across the 4B and 9B variants, so a single decoder per preset covers both.
# The 128-channel packed latent is unambiguous (unlike the 16ch FLUX/SD3 case), so no directory-name disambiguation
# is needed for the config probe.
pid_decoder_flux2_2k = StarterModel(
    name="PiD Decoder FLUX.2 (2K)",
    base=BaseModelType.Flux2,
    source="nvidia/PiD::checkpoints/PiD_res2k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth",
    description="NVIDIA PiD 4x super-resolution decoder for FLUX.2 Klein latents, 2K target preset (e.g. 512 -> 2048). ~5GB",
    type=ModelType.PiDDecoder,
    format=ModelFormat.Checkpoint,
    variant=PiDDecoderVariantType.Res2k_Sr4x,
    dependencies=[gemma2_2b_encoder],
)

pid_decoder_flux2_2kto4k = StarterModel(
    name="PiD Decoder FLUX.2 (2K to 4K)",
    base=BaseModelType.Flux2,
    source="nvidia/PiD::checkpoints_deprecated/PiD_res2kto4k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth",
    description="NVIDIA PiD 4x super-resolution decoder for FLUX.2 Klein latents, 2K-to-4K preset (legacy architecture; NVIDIA's newer v1.5 checkpoint uses a different network that is not yet supported). ~5GB",
    type=ModelType.PiDDecoder,
    format=ModelFormat.Checkpoint,
    variant=PiDDecoderVariantType.Res2kTo4k_Sr4x,
    dependencies=[gemma2_2b_encoder],
)

# region FLUX.2 Klein
flux2_vae = StarterModel(
    name="FLUX.2 VAE",
    base=BaseModelType.Flux2,
    source="black-forest-labs/FLUX.2-klein-4B::vae",
    description="FLUX.2 VAE (16-channel, same architecture as FLUX.1 VAE). ~168MB",
    type=ModelType.VAE,
)

flux2_klein_4b = StarterModel(
    name="FLUX.2 Klein 4B (Diffusers)",
    base=BaseModelType.Flux2,
    source="black-forest-labs/FLUX.2-klein-4B",
    description="FLUX.2 Klein 4B in Diffusers format - includes transformer, VAE and Qwen3 encoder. ~16GB",
    type=ModelType.Main,
)

flux2_klein_4b_single = StarterModel(
    name="FLUX.2 Klein 4B",
    base=BaseModelType.Flux2,
    source="https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/flux-2-klein-4b.safetensors",
    description="FLUX.2 Klein 4B standalone transformer. Installs with VAE and Qwen3 4B encoder. ~8GB",
    type=ModelType.Main,
    dependencies=[flux2_vae, flux2_klein_qwen3_4b_encoder],
)

flux2_klein_4b_fp8 = StarterModel(
    name="FLUX.2 Klein 4B (FP8)",
    base=BaseModelType.Flux2,
    source="https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors",
    description="FLUX.2 Klein 4B FP8 quantized - smaller and faster. Installs with VAE and Qwen3 4B encoder. ~4GB",
    type=ModelType.Main,
    dependencies=[flux2_vae, flux2_klein_qwen3_4b_encoder],
)

flux2_klein_9b = StarterModel(
    name="FLUX.2 Klein 9B (Diffusers)",
    base=BaseModelType.Flux2,
    source="black-forest-labs/FLUX.2-klein-9B",
    description="FLUX.2 Klein 9B in Diffusers format - includes transformer, VAE and Qwen3 encoder. ~35GB",
    type=ModelType.Main,
)

flux2_klein_9b_fp8 = StarterModel(
    name="FLUX.2 Klein 9B (FP8)",
    base=BaseModelType.Flux2,
    source="https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/resolve/main/flux-2-klein-9b-fp8.safetensors",
    description="FLUX.2 Klein 9B FP8 quantized - more efficient than full precision. Installs with VAE and Qwen3 8B encoder. ~9.5GB",
    type=ModelType.Main,
    dependencies=[flux2_vae, flux2_klein_qwen3_8b_encoder],
)

# A community repack rather than a BFL or Comfy-Org release -- neither publishes an int8 build of
# FLUX.2. It is listed because it is the only FLUX.2 build whose download size is also its resident
# size on every device: `Int8ConvrotLinear` keeps the stored codes and dequantizes per forward,
# where the fp8 weights above expand to bf16 on load unless `fp8_compute` is available.
#
# Pinned to a commit rather than `main`, which for this entry is a correctness matter and not just
# reproducibility: the decode is driven by the file's own `comfy_quant` marker, which today says
# `int8_tensorwise` with no `convrot` flag. A reupload at the same path that added `"convrot": true`
# -- or that was genuinely rotated -- would be derotated with a Hadamard that was never applied, and
# the result loads cleanly, logs "kept 144 layer(s) in int8" and generates noise.
flux2_klein_9b_int8 = StarterModel(
    name="FLUX.2 Klein 9B (int8)",
    base=BaseModelType.Flux2,
    source="https://huggingface.co/Winnougan/Klein9b-Distilled-Base-INT8-Convrot/resolve/0373bf363446b8aa1e77245658dc2163875d7f08/flux-2-klein-9b-int8-convrot.safetensors",
    description="FLUX.2 Klein 9B in ComfyUI int8_tensorwise, from a community repack (no first-party "
    "int8 build of FLUX.2 exists). Unlike the FP8 build it stays at its download size in memory on "
    "every supported GPU - 9.5GB rather than ~18GB - because the weights are dequantized per forward "
    "instead of on load. Quality matches FP8 closely. Installs with VAE and Qwen3 8B encoder. ~9.5GB",
    type=ModelType.Main,
    dependencies=[flux2_vae, flux2_klein_qwen3_8b_encoder],
)

flux2_klein_4b_sdnq = StarterModel(
    name="FLUX.2 Klein 4B (SDNQ dynamic 4-bit)",
    base=BaseModelType.Flux2,
    source="Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
    description="FLUX.2 Klein 4B quantized via SDNQ to dynamic uint4/int5 mixed precision. "
    "Full self-contained Flux2KleinPipeline (transformer + Qwen3 4B + AutoencoderKLFlux2). ~5GB",
    type=ModelType.Main,
    format=ModelFormat.SDNQQuantized,
)

flux2_klein_9b_sdnq = StarterModel(
    name="FLUX.2 Klein 9B (SDNQ dynamic 4-bit + SVD)",
    base=BaseModelType.Flux2,
    source="Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32",
    description="FLUX.2 Klein 9B quantized via SDNQ to dynamic uint4/int5 + SVD rank 32. "
    "Full self-contained Flux2KleinPipeline. ~13GB",
    type=ModelType.Main,
    format=ModelFormat.SDNQQuantized,
)

flux2_klein_4b_gguf_q4 = StarterModel(
    name="FLUX.2 Klein 4B (GGUF Q4)",
    base=BaseModelType.Flux2,
    source="https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q4_K_M.gguf",
    description="FLUX.2 Klein 4B GGUF Q4_K_M quantized - runs on 6-8GB VRAM. Installs with VAE and Qwen3 4B encoder. ~2.6GB",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[flux2_vae, flux2_klein_qwen3_4b_encoder],
)

flux2_klein_4b_gguf_q8 = StarterModel(
    name="FLUX.2 Klein 4B (GGUF Q8)",
    base=BaseModelType.Flux2,
    source="https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q8_0.gguf",
    description="FLUX.2 Klein 4B GGUF Q8_0 quantized - higher quality than Q4. Installs with VAE and Qwen3 4B encoder. ~4.3GB",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[flux2_vae, flux2_klein_qwen3_4b_encoder],
)

flux2_klein_9b_gguf_q4 = StarterModel(
    name="FLUX.2 Klein 9B (GGUF Q4)",
    base=BaseModelType.Flux2,
    source="https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/resolve/main/flux-2-klein-9b-Q4_K_M.gguf",
    description="FLUX.2 Klein 9B GGUF Q4_K_M quantized - runs on 12GB+ VRAM. Installs with VAE and Qwen3 8B encoder. ~5.8GB",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[flux2_vae, flux2_klein_qwen3_8b_encoder],
)

flux2_klein_9b_gguf_q8 = StarterModel(
    name="FLUX.2 Klein 9B (GGUF Q8)",
    base=BaseModelType.Flux2,
    source="https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/resolve/main/flux-2-klein-9b-Q8_0.gguf",
    description="FLUX.2 Klein 9B GGUF Q8_0 quantized - higher quality than Q4. Installs with VAE and Qwen3 8B encoder. ~10GB",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[flux2_vae, flux2_klein_qwen3_8b_encoder],
)

# --- Diffusers transformer ---
flux2_dev_diffusers = StarterModel(
    name="FLUX.2 [dev] (Diffusers)",
    base=BaseModelType.Flux2,
    source="black-forest-labs/FLUX.2-dev",
    description="FLUX.2 [dev] full Diffusers pipeline - includes transformer, VAE, and Mistral text encoder. ~80GB. Non-Commercial License.",
    type=ModelType.Main,
)

flux2_dev_diffusers_nf4 = StarterModel(
    name="FLUX.2 [dev] (Diffusers, NF4)",
    base=BaseModelType.Flux2,
    source="diffusers/FLUX.2-dev-bnb-4bit",
    description="FLUX.2 [dev] with NF4-quantized DiT and text encoder - runs on ~18GB VRAM with offload. Non-Commercial License.",
    type=ModelType.Main,
)

# --- GGUF transformers from gguf-org/flux2-dev-gguf (canonical repo) ---
# These are the GGUFs BFL/community curate for cow-paired inference. Default
# encoder dependency is cow Q4 to make starter installs work out of the box.
flux2_dev_gguf_q3_k_m = StarterModel(
    name="FLUX.2 [dev] Transformer (GGUF Q3_K_M)",
    base=BaseModelType.Flux2,
    source="https://huggingface.co/gguf-org/flux2-dev-gguf/resolve/main/flux2-dev-q3_k_m.gguf",
    description="FLUX.2 [dev] transformer Q3_K_M — fits ~12GB VRAM with offload. ~15.9GB",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[flux2_vae, flux2_dev_cow_mistral_q4],
)

flux2_dev_gguf_q4_k_m = StarterModel(
    name="FLUX.2 [dev] Transformer (GGUF Q4_K_M)",
    base=BaseModelType.Flux2,
    source="https://huggingface.co/gguf-org/flux2-dev-gguf/resolve/main/flux2-dev-q4_k_m.gguf",
    description="FLUX.2 [dev] transformer Q4_K_M — good quality / size tradeoff. ~20GB",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[flux2_vae, flux2_dev_cow_mistral_q4],
)

flux2_dev_gguf_q5_k_m = StarterModel(
    name="FLUX.2 [dev] Transformer (GGUF Q5_K_M)",
    base=BaseModelType.Flux2,
    source="https://huggingface.co/gguf-org/flux2-dev-gguf/resolve/main/flux2-dev-q5_k_m.gguf",
    description="FLUX.2 [dev] transformer Q5_K_M — higher fidelity than Q4. ~24GB",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[flux2_vae, flux2_dev_cow_mistral_q8],
)

flux2_dev_gguf_q6_k = StarterModel(
    name="FLUX.2 [dev] Transformer (GGUF Q6_K)",
    base=BaseModelType.Flux2,
    source="https://huggingface.co/gguf-org/flux2-dev-gguf/resolve/main/flux2-dev-q6_k.gguf",
    description="FLUX.2 [dev] transformer Q6_K — near-Q8 quality at lower size. ~27.9GB",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[flux2_vae, flux2_dev_cow_mistral_q8],
)

flux2_dev_gguf_q8_0 = StarterModel(
    name="FLUX.2 [dev] Transformer (GGUF Q8_0)",
    base=BaseModelType.Flux2,
    source="https://huggingface.co/gguf-org/flux2-dev-gguf/resolve/main/flux2-dev-q8_0.gguf",
    description="FLUX.2 [dev] transformer Q8_0 — highest GGUF fidelity. ~35.5GB",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[flux2_vae, flux2_dev_cow_mistral_q8],
)
