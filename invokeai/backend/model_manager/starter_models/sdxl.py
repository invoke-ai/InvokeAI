"""SDXL starter models."""

from invokeai.backend.model_manager.starter_models.common import gemma2_2b_encoder, ip_adapter_sdxl_image_encoder
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
    PiDDecoderVariantType,
)

# region VAE
sdxl_fp16_vae_fix = StarterModel(
    name="sdxl-vae-fp16-fix",
    base=BaseModelType.StableDiffusionXL,
    source="madebyollin/sdxl-vae-fp16-fix",
    description="SDXL VAE that works with FP16.",
    type=ModelType.VAE,
)

# SDXL uses a 4-channel latent, which is unambiguous (no FLUX/SD3-style directory-name disambiguation needed).
# NVIDIA ships only the 2K-to-4K preset for SDXL (no plain 2K checkpoint).
pid_decoder_sdxl_2kto4k = StarterModel(
    name="PiD Decoder SDXL (2K to 4K)",
    base=BaseModelType.StableDiffusionXL,
    source="nvidia/PiD::checkpoints/PiD_res2kto4k_sr4x_official_sdxl_distill_4step/model_ema_bf16.pth",
    description="NVIDIA PiD 4x super-resolution decoder for SDXL latents, 2K-to-4K preset. ~5GB",
    type=ModelType.PiDDecoder,
    format=ModelFormat.Checkpoint,
    variant=PiDDecoderVariantType.Res2kTo4k_Sr4x,
    dependencies=[gemma2_2b_encoder],
)

juggernaut_sdxl = StarterModel(
    name="Juggernaut XL v9",
    base=BaseModelType.StableDiffusionXL,
    source="RunDiffusion/Juggernaut-XL-v9",
    description="Photograph-focused model.",
    type=ModelType.Main,
    dependencies=[sdxl_fp16_vae_fix],
)

dreamshaper_sdxl = StarterModel(
    name="Dreamshaper XL v2 Turbo",
    base=BaseModelType.StableDiffusionXL,
    source="Lykon/dreamshaper-xl-v2-turbo",
    description="For turbo, use CFG Scale 2, 4-8 steps, DPM++ SDE Karras. For non-turbo, use CFG Scale 6, 20-40 steps, DPM++ 2M SDE Karras.",
    type=ModelType.Main,
    dependencies=[sdxl_fp16_vae_fix],
)

archvis_sdxl = StarterModel(
    name="Architecture (RealVisXL5)",
    base=BaseModelType.StableDiffusionXL,
    source="SG161222/RealVisXL_V5.0",
    description="A photorealistic model, with architecture among its many use cases",
    type=ModelType.Main,
    dependencies=[sdxl_fp16_vae_fix],
)

# region LoRA
alien_lora_sdxl = StarterModel(
    name="Alien Style",
    base=BaseModelType.StableDiffusionXL,
    source="https://huggingface.co/RalFinger/alien-style-lora-sdxl/resolve/main/alienzkin-sdxl.safetensors",
    description="Futuristic, intricate alien styles. Trigger with 'alienzkin'.",
    type=ModelType.LoRA,
)

noodle_lora_sdxl = StarterModel(
    name="Noodles Style",
    base=BaseModelType.StableDiffusionXL,
    source="https://huggingface.co/RalFinger/noodles-lora-sdxl/resolve/main/noodlez-sdxl.safetensors",
    description="Never-ending, no-holds-barred, noodle nightmare. Trigger with 'noodlez'.",
    type=ModelType.LoRA,
)

ip_adapter_sdxl = StarterModel(
    name="Standard Reference (IP Adapter ViT-H)",
    base=BaseModelType.StableDiffusionXL,
    source="https://huggingface.co/InvokeAI/ip_adapter_sdxl_vit_h/resolve/main/ip-adapter_sdxl_vit-h.safetensors",
    description="References images with a higher degree of precision.",
    type=ModelType.IPAdapter,
    dependencies=[ip_adapter_sdxl_image_encoder],
    previous_names=["IP Adapter SDXL"],
)

ip_adapter_plus_sdxl = StarterModel(
    name="Precise Reference (IP Adapter Plus ViT-H)",
    base=BaseModelType.StableDiffusionXL,
    source="https://huggingface.co/InvokeAI/ip-adapter-plus_sdxl_vit-h/resolve/main/ip-adapter-plus_sdxl_vit-h.safetensors",
    description="References images with a higher degree of precision.",
    type=ModelType.IPAdapter,
    dependencies=[ip_adapter_sdxl_image_encoder],
    previous_names=["IP Adapter Plus SDXL"],
)

qr_code_cnet_sdxl = StarterModel(
    name="QRCode Monster (SDXL)",
    base=BaseModelType.StableDiffusionXL,
    source="monster-labs/control_v1p_sdxl_qrcode_monster",
    description="ControlNet model that generates scannable creative QR codes",
    type=ModelType.ControlNet,
)

canny_sdxl = StarterModel(
    name="Hard Edge Detection (canny)",
    base=BaseModelType.StableDiffusionXL,
    source="xinsir/controlNet-canny-sdxl-1.0",
    description="Uses detected edges in the image to control composition.",
    type=ModelType.ControlNet,
    previous_names=["canny-sdxl"],
)

depth_sdxl = StarterModel(
    name="Depth Map",
    base=BaseModelType.StableDiffusionXL,
    source="diffusers/controlNet-depth-sdxl-1.0",
    description="Uses depth information in the image to control the depth in the generation.",
    type=ModelType.ControlNet,
    previous_names=["depth-sdxl"],
)

softedge_sdxl = StarterModel(
    name="Soft Edge Detection (softedge)",
    base=BaseModelType.StableDiffusionXL,
    source="SargeZT/controlNet-sd-xl-1.0-softedge-dexined",
    description="Uses a soft edge detection map to control composition.",
    type=ModelType.ControlNet,
    previous_names=["softedge-dexined-sdxl"],
)

openpose_sdxl = StarterModel(
    name="Pose Detection (openpose)",
    base=BaseModelType.StableDiffusionXL,
    source="xinsir/controlNet-openpose-sdxl-1.0",
    description="Uses pose information to control the pose of human characters in the generation.",
    type=ModelType.ControlNet,
    previous_names=["openpose-sdxl", "controlnet-openpose-sdxl"],
)

scribble_sdxl = StarterModel(
    name="Contour Detection (scribble)",
    base=BaseModelType.StableDiffusionXL,
    source="xinsir/controlNet-scribble-sdxl-1.0",
    description="Uses edges, contours, or line art in the image to control composition.",
    type=ModelType.ControlNet,
    previous_names=["scribble-sdxl", "controlnet-scribble-sdxl"],
)

tile_sdxl = StarterModel(
    name="Tile",
    base=BaseModelType.StableDiffusionXL,
    source="xinsir/controlNet-tile-sdxl-1.0",
    description="Uses image data to replicate exact colors/structure in the resulting generation.",
    type=ModelType.ControlNet,
    previous_names=["tile-sdxl"],
)

union_cnet_sdxl = StarterModel(
    name="Multi-Guidance Detection (Union Pro)",
    base=BaseModelType.StableDiffusionXL,
    source="InvokeAI/Xinsir-SDXL_Controlnet_Union",
    description="A unified ControlNet for SDXL model that supports 10+ control types",
    type=ModelType.ControlNet,
)

t2i_canny_sdxl = StarterModel(
    name="Hard Edge Detection (canny)",
    base=BaseModelType.StableDiffusionXL,
    source="TencentARC/t2i-adapter-canny-sdxl-1.0",
    description="Uses detected edges in the image to control composition",
    type=ModelType.T2IAdapter,
    previous_names=["canny-sdxl"],
)

t2i_lineart_sdxl = StarterModel(
    name="Lineart",
    base=BaseModelType.StableDiffusionXL,
    source="TencentARC/t2i-adapter-lineart-sdxl-1.0",
    description="Uses lineart detection to guide the lighting of the composition.",
    type=ModelType.T2IAdapter,
    previous_names=["lineart-sdxl"],
)

t2i_sketch_sdxl = StarterModel(
    name="Sketch",
    base=BaseModelType.StableDiffusionXL,
    source="TencentARC/t2i-adapter-sketch-sdxl-1.0",
    description="Uses a sketch to control composition",
    type=ModelType.T2IAdapter,
    previous_names=["sketch-sdxl"],
)
