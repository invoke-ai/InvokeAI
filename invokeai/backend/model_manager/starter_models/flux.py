"""FLUX.1 starter models."""

from invokeai.backend.model_manager.starter_models.common import (
    clip_l_encoder,
    clip_vit_l_image_encoder,
    gemma2_2b_encoder,
    siglip,
    t5_8b_quantized_encoder,
    t5_base_encoder,
)
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
    PiDDecoderVariantType,
)

flux_vae = StarterModel(
    name="FLUX.1-schnell_ae",
    base=BaseModelType.Flux,
    source="black-forest-labs/FLUX.1-schnell::ae.safetensors",
    description="FLUX VAE compatible with both schnell and dev variants.",
    type=ModelType.VAE,
)

# NVIDIA PiD decoders (https://huggingface.co/nvidia/PiD). Code is Apache-2.0; weights are NSCLv1 (non-commercial /
# research). Each is a 4x super-resolution decoder that replaces the regular VAE decode and needs the Gemma-2 encoder.
pid_decoder_flux_2k = StarterModel(
    name="PiD Decoder FLUX (2K)",
    base=BaseModelType.Flux,
    source="nvidia/PiD::checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
    description="NVIDIA PiD 4x super-resolution decoder for FLUX latents, 2K target preset (e.g. 512 -> 2048). ~5GB",
    type=ModelType.PiDDecoder,
    format=ModelFormat.Checkpoint,
    variant=PiDDecoderVariantType.Res2k_Sr4x,
    dependencies=[gemma2_2b_encoder],
)

pid_decoder_flux_2kto4k = StarterModel(
    name="PiD Decoder FLUX (2K to 4K)",
    base=BaseModelType.Flux,
    source="nvidia/PiD::checkpoints_deprecated/PiD_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
    description="NVIDIA PiD 4x super-resolution decoder for FLUX latents, 2K-to-4K preset (legacy architecture; NVIDIA's newer v1.5 checkpoint uses a different network that is not yet supported). ~5GB",
    type=ModelType.PiDDecoder,
    format=ModelFormat.Checkpoint,
    variant=PiDDecoderVariantType.Res2kTo4k_Sr4x,
    dependencies=[gemma2_2b_encoder],
)

# region: Main
flux_schnell_quantized = StarterModel(
    name="FLUX.1 schnell (quantized)",
    base=BaseModelType.Flux,
    source="InvokeAI/flux_schnell::transformer/bnb_nf4/flux1-schnell-bnb_nf4.safetensors",
    description="FLUX schnell transformer quantized to bitsandbytes NF4 format. Total size with dependencies: ~12GB",
    type=ModelType.Main,
    dependencies=[t5_8b_quantized_encoder, flux_vae, clip_l_encoder],
)

flux_dev_quantized = StarterModel(
    name="FLUX.1 dev (quantized)",
    base=BaseModelType.Flux,
    source="InvokeAI/flux_dev::transformer/bnb_nf4/flux1-dev-bnb_nf4.safetensors",
    description="FLUX dev transformer quantized to bitsandbytes NF4 format. Total size with dependencies: ~12GB",
    type=ModelType.Main,
    dependencies=[t5_8b_quantized_encoder, flux_vae, clip_l_encoder],
)

flux_schnell = StarterModel(
    name="FLUX.1 schnell",
    base=BaseModelType.Flux,
    source="InvokeAI/flux_schnell::transformer/base/flux1-schnell.safetensors",
    description="FLUX schnell transformer in bfloat16. Total size with dependencies: ~33GB",
    type=ModelType.Main,
    dependencies=[t5_base_encoder, flux_vae, clip_l_encoder],
)

# A community repack -- neither BFL nor Comfy-Org publishes an int8 build of FLUX.1. Pinned to a
# commit rather than `main`, which here is a correctness matter and not just reproducibility: the
# decode follows the file's own `comfy_quant` marker, which says `convrot: true` at group size 256.
# A reupload at the same path that changed or dropped that flag would be derotated wrongly, and the
# result loads cleanly, logs its layer count and generates noise.
flux_dev_int8 = StarterModel(
    name="FLUX.1 dev (int8)",
    base=BaseModelType.Flux,
    source="https://huggingface.co/AX1Y2JP/FLUX.1-dev-INT8-ConvRot/resolve/01a17db182d52088b222d615ce54729fa090b378/flux1-dev-int8-convrot.safetensors",
    description="FLUX dev transformer in ComfyUI int8_tensorwise, from a community repack (no "
    "first-party int8 build of FLUX.1 exists). Half the memory of the bfloat16 transformer at 8 bits "
    "per weight, on every supported GPU - FLUX.1 dev (quantized) is smaller still, at 4. Total size "
    "with dependencies: ~18GB",
    type=ModelType.Main,
    dependencies=[t5_8b_quantized_encoder, flux_vae, clip_l_encoder],
)

flux_dev = StarterModel(
    name="FLUX.1 dev",
    base=BaseModelType.Flux,
    source="InvokeAI/flux_dev::transformer/base/flux1-dev.safetensors",
    description="FLUX dev transformer in bfloat16. Total size with dependencies: ~33GB",
    type=ModelType.Main,
    dependencies=[t5_base_encoder, flux_vae, clip_l_encoder],
)

flux_schnell_sdnq = StarterModel(
    name="FLUX.1 schnell (SDNQ uint4 + SVD)",
    base=BaseModelType.Flux,
    source="Disty0/FLUX.1-schnell-SDNQ-uint4-svd-r32",
    description="FLUX.1 schnell quantized via SDNQ to uint4 + SVD rank 32. Full self-contained "
    "Flux pipeline (transformer + T5 + CLIP + VAE). ~15GB",
    type=ModelType.Main,
    format=ModelFormat.SDNQQuantized,
)

flux_kontext = StarterModel(
    name="FLUX.1 Kontext dev",
    base=BaseModelType.Flux,
    source="https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev/resolve/main/flux1-kontext-dev.safetensors",
    description="FLUX.1 Kontext dev transformer in bfloat16. Total size with dependencies: ~33GB",
    type=ModelType.Main,
    dependencies=[t5_base_encoder, flux_vae, clip_l_encoder],
)

flux_kontext_quantized = StarterModel(
    name="FLUX.1 Kontext dev (quantized)",
    base=BaseModelType.Flux,
    source="https://huggingface.co/unsloth/FLUX.1-Kontext-dev-GGUF/resolve/main/flux1-kontext-dev-Q4_K_M.gguf",
    description="FLUX.1 Kontext dev quantized (q4_k_m). Total size with dependencies: ~12GB",
    type=ModelType.Main,
    dependencies=[t5_8b_quantized_encoder, flux_vae, clip_l_encoder],
)

flux_krea = StarterModel(
    name="FLUX.1 Krea dev",
    base=BaseModelType.Flux,
    source="https://huggingface.co/InvokeAI/FLUX.1-Krea-dev/resolve/main/flux1-krea-dev.safetensors",
    description="FLUX.1 Krea dev. Total size with dependencies: ~29GB",
    type=ModelType.Main,
    dependencies=[t5_8b_quantized_encoder, flux_vae, clip_l_encoder],
)

flux_krea_quantized = StarterModel(
    name="FLUX.1 Krea dev (quantized)",
    base=BaseModelType.Flux,
    source="https://huggingface.co/InvokeAI/FLUX.1-Krea-dev-GGUF/resolve/main/flux1-krea-dev-Q4_K_M.gguf",
    description="FLUX.1 Krea dev quantized (q4_k_m). Total size with dependencies: ~12GB",
    type=ModelType.Main,
    dependencies=[t5_8b_quantized_encoder, flux_vae, clip_l_encoder],
)

ip_adapter_flux = StarterModel(
    name="Standard Reference (XLabs FLUX IP-Adapter v2)",
    base=BaseModelType.Flux,
    source="https://huggingface.co/XLabs-AI/flux-ip-adapter-v2/resolve/main/ip_adapter.safetensors",
    description="References images with a more generalized/looser degree of precision.",
    type=ModelType.IPAdapter,
    dependencies=[clip_vit_l_image_encoder],
)

union_cnet_flux = StarterModel(
    name="FLUX.1-dev-Controlnet-Union",
    base=BaseModelType.Flux,
    source="InstantX/FLUX.1-dev-Controlnet-Union",
    description="A unified ControlNet for FLUX.1-dev model that supports 7 control modes, including canny (0), tile (1), depth (2), blur (3), pose (4), gray (5), low quality (6)",
    type=ModelType.ControlNet,
)

# endregion
# region Control LoRA
flux_canny_control_lora = StarterModel(
    name="Hard Edge Detection (Canny)",
    base=BaseModelType.Flux,
    source="black-forest-labs/FLUX.1-Canny-dev-lora::flux1-canny-dev-lora.safetensors",
    description="Uses detected edges in the image to control composition.",
    type=ModelType.ControlLoRa,
)

flux_depth_control_lora = StarterModel(
    name="Depth Map",
    base=BaseModelType.Flux,
    source="black-forest-labs/FLUX.1-Depth-dev-lora::flux1-depth-dev-lora.safetensors",
    description="Uses depth information in the image to control the depth in the generation.",
    type=ModelType.ControlLoRa,
)

# region FLUX Redux
flux_redux = StarterModel(
    name="FLUX Redux",
    base=BaseModelType.Flux,
    source="black-forest-labs/FLUX.1-Redux-dev::flux1-redux-dev.safetensors",
    description="FLUX Redux model (for image variation).",
    type=ModelType.FluxRedux,
    dependencies=[siglip],
)

# region FLUX Fill
flux_fill = StarterModel(
    name="FLUX Fill",
    base=BaseModelType.Flux,
    source="black-forest-labs/FLUX.1-Fill-dev::flux1-fill-dev.safetensors",
    description="FLUX Fill model (for inpainting).",
    type=ModelType.Main,
)
