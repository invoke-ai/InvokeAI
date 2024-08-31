from typing import Optional

from pydantic import BaseModel

from invokeai.backend.model_manager.config import BaseModelType, ModelFormat, ModelType


class StarterModelWithoutDependencies(BaseModel):
    description: str
    source: str
    name: str
    base: BaseModelType
    type: ModelType
    format: Optional[ModelFormat] = None
    is_installed: bool = False


class StarterModel(StarterModelWithoutDependencies):
    # Optional list of model source dependencies that need to be installed before this model can be used
    dependencies: Optional[list[StarterModelWithoutDependencies]] = None


sdxl_fp16_vae_fix = StarterModel(
    name="sdxl-vae-fp16-fix",
    base=BaseModelType.StableDiffusionXL,
    source="madebyollin/sdxl-vae-fp16-fix",
    description="SDXL VAE that works with FP16.",
    type=ModelType.VAE,
)

ip_adapter_sd_image_encoder = StarterModel(
    name="IP Adapter SD1.5 Image Encoder",
    base=BaseModelType.StableDiffusion1,
    source="InvokeAI/ip_adapter_sd_image_encoder",
    description="IP Adapter SD Image Encoder",
    type=ModelType.CLIPVision,
)

ip_adapter_sdxl_image_encoder = StarterModel(
    name="IP Adapter SDXL Image Encoder",
    base=BaseModelType.StableDiffusionXL,
    source="InvokeAI/ip_adapter_sdxl_image_encoder",
    description="IP Adapter SDXL Image Encoder",
    type=ModelType.CLIPVision,
)

cyberrealistic_negative = StarterModel(
    name="CyberRealistic Negative v3",
    base=BaseModelType.StableDiffusion1,
    source="https://huggingface.co/cyberdelia/CyberRealistic_Negative/resolve/main/CyberRealistic_Negative_v3.pt",
    description="Negative embedding specifically for use with CyberRealistic.",
    type=ModelType.TextualInversion,
)

t5_base_encoder = StarterModel(
    name="t5_base_encoder",
    base=BaseModelType.Any,
    source="InvokeAI/t5-v1_1-xxl::bfloat16",
    description="T5-XXL text encoder (used in FLUX pipelines). ~8GB",
    type=ModelType.T5Encoder,
)

t5_8b_quantized_encoder = StarterModel(
    name="t5_bnb_int8_quantized_encoder",
    base=BaseModelType.Any,
    source="InvokeAI/t5-v1_1-xxl::bnb_llm_int8",
    description="T5-XXL text encoder with bitsandbytes LLM.int8() quantization (used in FLUX pipelines). ~5GB",
    type=ModelType.T5Encoder,
    format=ModelFormat.BnbQuantizedLlmInt8b,
)

clip_l_encoder = StarterModel(
    name="clip-vit-large-patch14",
    base=BaseModelType.Any,
    source="InvokeAI/clip-vit-large-patch14-text-encoder::bfloat16",
    description="CLIP-L text encoder (used in FLUX pipelines). ~250MB",
    type=ModelType.CLIPEmbed,
)

flux_vae = StarterModel(
    name="FLUX.1-schnell_ae",
    base=BaseModelType.Flux,
    source="black-forest-labs/FLUX.1-schnell::ae.safetensors",
    description="FLUX VAE compatible with both schnell and dev variants.",
    type=ModelType.VAE,
)


# List of starter models, displayed on the frontend.
# The order/sort of this list is not changed by the frontend - set it how you want it here.
STARTER_MODELS: list[StarterModel] = [
    # region: Main
    StarterModel(
        name="FLUX Schnell (Quantized)",
        base=BaseModelType.Flux,
        source="InvokeAI/flux_schnell::transformer/bnb_nf4/flux1-schnell-bnb_nf4.safetensors",
        description="FLUX schnell transformer quantized to bitsandbytes NF4 format. Total size with dependencies: ~12GB",
        type=ModelType.Main,
        dependencies=[t5_8b_quantized_encoder, flux_vae, clip_l_encoder],
    ),
    StarterModel(
        name="FLUX Dev (Quantized)",
        base=BaseModelType.Flux,
        source="InvokeAI/flux_dev::transformer/bnb_nf4/flux1-dev-bnb_nf4.safetensors",
        description="FLUX dev transformer quantized to bitsandbytes NF4 format. Total size with dependencies: ~12GB",
        type=ModelType.Main,
        dependencies=[t5_8b_quantized_encoder, flux_vae, clip_l_encoder],
    ),
    StarterModel(
        name="FLUX Schnell",
        base=BaseModelType.Flux,
        source="InvokeAI/flux_schnell::transformer/base/flux1-schnell.safetensors",
        description="FLUX schnell transformer in bfloat16. Total size with dependencies: ~33GB",
        type=ModelType.Main,
        dependencies=[t5_base_encoder, flux_vae, clip_l_encoder],
    ),
    StarterModel(
        name="FLUX Dev",
        base=BaseModelType.Flux,
        source="InvokeAI/flux_dev::transformer/base/flux1-dev.safetensors",
        description="FLUX dev transformer in bfloat16. Total size with dependencies: ~33GB",
        type=ModelType.Main,
        dependencies=[t5_base_encoder, flux_vae, clip_l_encoder],
    ),
    StarterModel(
        name="CyberRealistic v4.1",
        base=BaseModelType.StableDiffusion1,
        source="https://huggingface.co/cyberdelia/CyberRealistic/resolve/main/CyberRealistic_V4.1_FP16.safetensors",
        description="Photorealistic model. See other variants in HF repo 'cyberdelia/CyberRealistic'.",
        type=ModelType.Main,
        dependencies=[cyberrealistic_negative],
    ),
    StarterModel(
        name="ReV Animated",
        base=BaseModelType.StableDiffusion1,
        source="stablediffusionapi/rev-animated",
        description="Fantasy and anime style images.",
        type=ModelType.Main,
    ),
    StarterModel(
        name="Dreamshaper 8",
        base=BaseModelType.StableDiffusion1,
        source="Lykon/dreamshaper-8",
        description="Popular versatile model.",
        type=ModelType.Main,
    ),
    StarterModel(
        name="Dreamshaper 8 (inpainting)",
        base=BaseModelType.StableDiffusion1,
        source="Lykon/dreamshaper-8-inpainting",
        description="Inpainting version of Dreamshaper 8.",
        type=ModelType.Main,
    ),
    StarterModel(
        name="Deliberate v5",
        base=BaseModelType.StableDiffusion1,
        source="https://huggingface.co/XpucT/Deliberate/resolve/main/Deliberate_v5.safetensors",
        description="Popular versatile model",
        type=ModelType.Main,
    ),
    StarterModel(
        name="Deliberate v5 (inpainting)",
        base=BaseModelType.StableDiffusion1,
        source="https://huggingface.co/XpucT/Deliberate/resolve/main/Deliberate_v5-inpainting.safetensors",
        description="Inpainting version of Deliberate v5.",
        type=ModelType.Main,
    ),
    StarterModel(
        name="Juggernaut XL v9",
        base=BaseModelType.StableDiffusionXL,
        source="RunDiffusion/Juggernaut-XL-v9",
        description="Photograph-focused model.",
        type=ModelType.Main,
        dependencies=[sdxl_fp16_vae_fix],
    ),
    StarterModel(
        name="Dreamshaper XL v2 Turbo",
        base=BaseModelType.StableDiffusionXL,
        source="Lykon/dreamshaper-xl-v2-turbo",
        description="For turbo, use CFG Scale 2, 4-8 steps, DPM++ SDE Karras. For non-turbo, use CFG Scale 6, 20-40 steps, DPM++ 2M SDE Karras.",
        type=ModelType.Main,
        dependencies=[sdxl_fp16_vae_fix],
    ),
    StarterModel(
        name="SDXL Refiner",
        base=BaseModelType.StableDiffusionXLRefiner,
        source="stabilityai/stable-diffusion-xl-refiner-1.0",
        description="The OG Stable Diffusion XL refiner model.",
        type=ModelType.Main,
        dependencies=[sdxl_fp16_vae_fix],
    ),
    # endregion
    # region VAE
    sdxl_fp16_vae_fix,
    flux_vae,
    # endregion
    # region LoRA
    StarterModel(
        name="Alien Style",
        base=BaseModelType.StableDiffusionXL,
        source="https://huggingface.co/RalFinger/alien-style-lora-sdxl/resolve/main/alienzkin-sdxl.safetensors",
        description="Futuristic, intricate alien styles. Trigger with 'alienzkin'.",
        type=ModelType.LoRA,
    ),
    StarterModel(
        name="Noodles Style",
        base=BaseModelType.StableDiffusionXL,
        source="https://huggingface.co/RalFinger/noodles-lora-sdxl/resolve/main/noodlez-sdxl.safetensors",
        description="Never-ending, no-holds-barred, noodle nightmare. Trigger with 'noodlez'.",
        type=ModelType.LoRA,
    ),
    # endregion
    # region TI
    StarterModel(
        name="EasyNegative",
        base=BaseModelType.StableDiffusion1,
        source="https://huggingface.co/embed/EasyNegative/resolve/main/EasyNegative.safetensors",
        description="A textual inversion to use in the negative prompt to reduce bad anatomy",
        type=ModelType.TextualInversion,
    ),
    # endregion
    # region IP Adapter
    StarterModel(
        name="IP Adapter",
        base=BaseModelType.StableDiffusion1,
        source="https://huggingface.co/InvokeAI/ip_adapter_sd15/resolve/main/ip-adapter_sd15.safetensors",
        description="IP-Adapter for SD 1.5 models",
        type=ModelType.IPAdapter,
        dependencies=[ip_adapter_sd_image_encoder],
    ),
    StarterModel(
        name="IP Adapter Plus",
        base=BaseModelType.StableDiffusion1,
        source="https://huggingface.co/InvokeAI/ip_adapter_plus_sd15/resolve/main/ip-adapter-plus_sd15.safetensors",
        description="Refined IP-Adapter for SD 1.5 models",
        type=ModelType.IPAdapter,
        dependencies=[ip_adapter_sd_image_encoder],
    ),
    StarterModel(
        name="IP Adapter Plus Face",
        base=BaseModelType.StableDiffusion1,
        source="https://huggingface.co/InvokeAI/ip_adapter_plus_face_sd15/resolve/main/ip-adapter-plus-face_sd15.safetensors",
        description="Refined IP-Adapter for SD 1.5 models, adapted for faces",
        type=ModelType.IPAdapter,
        dependencies=[ip_adapter_sd_image_encoder],
    ),
    StarterModel(
        name="IP Adapter SDXL",
        base=BaseModelType.StableDiffusionXL,
        source="https://huggingface.co/InvokeAI/ip_adapter_sdxl_vit_h/resolve/main/ip-adapter_sdxl_vit-h.safetensors",
        description="IP-Adapter for SDXL models",
        type=ModelType.IPAdapter,
        dependencies=[ip_adapter_sdxl_image_encoder],
    ),
    # endregion
    # region ControlNet
    StarterModel(
        name="QRCode Monster v2 (SD1.5)",
        base=BaseModelType.StableDiffusion1,
        source="monster-labs/control_v1p_sd15_qrcode_monster::v2",
        description="ControlNet model that generates scannable creative QR codes",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="QRCode Monster (SDXL)",
        base=BaseModelType.StableDiffusionXL,
        source="monster-labs/control_v1p_sdxl_qrcode_monster",
        description="ControlNet model that generates scannable creative QR codes",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="canny",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15_canny",
        description="ControlNet weights trained on sd-1.5 with canny conditioning.",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="inpaint",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15_inpaint",
        description="ControlNet weights trained on sd-1.5 with canny conditioning, inpaint version",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="mlsd",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15_mlsd",
        description="ControlNet weights trained on sd-1.5 with canny conditioning, MLSD version",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="depth",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11f1p_sd15_depth",
        description="ControlNet weights trained on sd-1.5 with depth conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="normal_bae",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15_normalbae",
        description="ControlNet weights trained on sd-1.5 with normalbae image conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="seg",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15_seg",
        description="ControlNet weights trained on sd-1.5 with seg image conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="lineart",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15_lineart",
        description="ControlNet weights trained on sd-1.5 with lineart image conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="lineart_anime",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15s2_lineart_anime",
        description="ControlNet weights trained on sd-1.5 with anime image conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="openpose",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15_openpose",
        description="ControlNet weights trained on sd-1.5 with openpose image conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="scribble",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15_scribble",
        description="ControlNet weights trained on sd-1.5 with scribble image conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="softedge",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15_softedge",
        description="ControlNet weights trained on sd-1.5 with soft edge conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="shuffle",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11e_sd15_shuffle",
        description="ControlNet weights trained on sd-1.5 with shuffle image conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="tile",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11f1e_sd15_tile",
        description="ControlNet weights trained on sd-1.5 with tiled image conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="ip2p",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11e_sd15_ip2p",
        description="ControlNet weights trained on sd-1.5 with ip2p conditioning.",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="canny-sdxl",
        base=BaseModelType.StableDiffusionXL,
        source="xinsir/controlNet-canny-sdxl-1.0",
        description="ControlNet weights trained on sdxl-1.0 with canny conditioning, by Xinsir.",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="depth-sdxl",
        base=BaseModelType.StableDiffusionXL,
        source="diffusers/controlNet-depth-sdxl-1.0",
        description="ControlNet weights trained on sdxl-1.0 with depth conditioning.",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="softedge-dexined-sdxl",
        base=BaseModelType.StableDiffusionXL,
        source="SargeZT/controlNet-sd-xl-1.0-softedge-dexined",
        description="ControlNet weights trained on sdxl-1.0 with dexined soft edge preprocessing.",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="depth-16bit-zoe-sdxl",
        base=BaseModelType.StableDiffusionXL,
        source="SargeZT/controlNet-sd-xl-1.0-depth-16bit-zoe",
        description="ControlNet weights trained on sdxl-1.0 with Zoe's preprocessor (16 bits).",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="depth-zoe-sdxl",
        base=BaseModelType.StableDiffusionXL,
        source="diffusers/controlNet-zoe-depth-sdxl-1.0",
        description="ControlNet weights trained on sdxl-1.0 with Zoe's preprocessor (32 bits).",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="openpose-sdxl",
        base=BaseModelType.StableDiffusionXL,
        source="xinsir/controlNet-openpose-sdxl-1.0",
        description="ControlNet weights trained on sdxl-1.0 compatible with the DWPose processor by Xinsir.",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="scribble-sdxl",
        base=BaseModelType.StableDiffusionXL,
        source="xinsir/controlNet-scribble-sdxl-1.0",
        description="ControlNet weights trained on sdxl-1.0 compatible with various lineart processors and black/white sketches by Xinsir.",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="tile-sdxl",
        base=BaseModelType.StableDiffusionXL,
        source="xinsir/controlNet-tile-sdxl-1.0",
        description="ControlNet weights trained on sdxl-1.0 with tiled image conditioning",
        type=ModelType.ControlNet,
    ),
    # endregion
    # region T2I Adapter
    StarterModel(
        name="canny-sd15",
        base=BaseModelType.StableDiffusion1,
        source="TencentARC/t2iadapter_canny_sd15v2",
        description="T2I Adapter weights trained on sd-1.5 with canny conditioning.",
        type=ModelType.T2IAdapter,
    ),
    StarterModel(
        name="sketch-sd15",
        base=BaseModelType.StableDiffusion1,
        source="TencentARC/t2iadapter_sketch_sd15v2",
        description="T2I Adapter weights trained on sd-1.5 with sketch conditioning.",
        type=ModelType.T2IAdapter,
    ),
    StarterModel(
        name="depth-sd15",
        base=BaseModelType.StableDiffusion1,
        source="TencentARC/t2iadapter_depth_sd15v2",
        description="T2I Adapter weights trained on sd-1.5 with depth conditioning.",
        type=ModelType.T2IAdapter,
    ),
    StarterModel(
        name="zoedepth-sd15",
        base=BaseModelType.StableDiffusion1,
        source="TencentARC/t2iadapter_zoedepth_sd15v1",
        description="T2I Adapter weights trained on sd-1.5 with zoe depth conditioning.",
        type=ModelType.T2IAdapter,
    ),
    StarterModel(
        name="canny-sdxl",
        base=BaseModelType.StableDiffusionXL,
        source="TencentARC/t2i-adapter-canny-sdxl-1.0",
        description="T2I Adapter weights trained on sdxl-1.0 with canny conditioning.",
        type=ModelType.T2IAdapter,
    ),
    StarterModel(
        name="zoedepth-sdxl",
        base=BaseModelType.StableDiffusionXL,
        source="TencentARC/t2i-adapter-depth-zoe-sdxl-1.0",
        description="T2I Adapter weights trained on sdxl-1.0 with zoe depth conditioning.",
        type=ModelType.T2IAdapter,
    ),
    StarterModel(
        name="lineart-sdxl",
        base=BaseModelType.StableDiffusionXL,
        source="TencentARC/t2i-adapter-lineart-sdxl-1.0",
        description="T2I Adapter weights trained on sdxl-1.0 with lineart conditioning.",
        type=ModelType.T2IAdapter,
    ),
    StarterModel(
        name="sketch-sdxl",
        base=BaseModelType.StableDiffusionXL,
        source="TencentARC/t2i-adapter-sketch-sdxl-1.0",
        description="T2I Adapter weights trained on sdxl-1.0 with sketch conditioning.",
        type=ModelType.T2IAdapter,
    ),
    # endregion
    # region SpandrelImageToImage
    StarterModel(
        name="RealESRGAN_x4plus_anime_6B",
        base=BaseModelType.Any,
        source="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        description="A Real-ESRGAN 4x upscaling model (optimized for anime images).",
        type=ModelType.SpandrelImageToImage,
    ),
    StarterModel(
        name="RealESRGAN_x4plus",
        base=BaseModelType.Any,
        source="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        description="A Real-ESRGAN 4x upscaling model (general-purpose).",
        type=ModelType.SpandrelImageToImage,
    ),
    StarterModel(
        name="ESRGAN_SRx4_DF2KOST_official",
        base=BaseModelType.Any,
        source="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
        description="The official ESRGAN 4x upscaling model.",
        type=ModelType.SpandrelImageToImage,
    ),
    StarterModel(
        name="RealESRGAN_x2plus",
        base=BaseModelType.Any,
        source="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        description="A Real-ESRGAN 2x upscaling model (general-purpose).",
        type=ModelType.SpandrelImageToImage,
    ),
    StarterModel(
        name="SwinIR - realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN",
        base=BaseModelType.Any,
        source="https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN-with-dict-keys-params-and-params_ema.pth",
        description="A SwinIR 4x upscaling model.",
        type=ModelType.SpandrelImageToImage,
    ),
    # endregion
    # region TextEncoders
    t5_base_encoder,
    t5_8b_quantized_encoder,
    clip_l_encoder,
    # endregion
]

assert len(STARTER_MODELS) == len({m.source for m in STARTER_MODELS}), "Duplicate starter models"
