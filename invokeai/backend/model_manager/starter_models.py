from typing import Optional

from pydantic import BaseModel

from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType


class StarterModelWithoutDependencies(BaseModel):
    description: str
    source: str
    name: str
    base: BaseModelType
    type: ModelType
    format: Optional[ModelFormat] = None
    is_installed: bool = False
    # allows us to track what models a user has installed across name changes within starter models
    # if you update a starter model name, please add the old one to this list for that starter model
    previous_names: list[str] = []


class StarterModel(StarterModelWithoutDependencies):
    # Optional list of model source dependencies that need to be installed before this model can be used
    dependencies: Optional[list[StarterModelWithoutDependencies]] = None


class StarterModelBundles(BaseModel):
    name: str
    models: list[StarterModel]


cyberrealistic_negative = StarterModel(
    name="CyberRealistic Negative v3",
    base=BaseModelType.StableDiffusion1,
    source="https://huggingface.co/cyberdelia/CyberRealistic_Negative/resolve/main/CyberRealistic_Negative_v3.pt",
    description="Negative embedding specifically for use with CyberRealistic.",
    type=ModelType.TextualInversion,
)

# region CLIP Image Encoders
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
# Note: This model is installed from the same source as the CLIPEmbed model below. The model contains both the image
# encoder and the text encoder, but we need separate model entries so that they get loaded correctly.
clip_vit_l_image_encoder = StarterModel(
    name="clip-vit-large-patch14",
    base=BaseModelType.Any,
    source="InvokeAI/clip-vit-large-patch14",
    description="CLIP ViT-L Image Encoder",
    type=ModelType.CLIPVision,
)
# endregion

# region TextEncoders
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
# endregion

# region VAE
sdxl_fp16_vae_fix = StarterModel(
    name="sdxl-vae-fp16-fix",
    base=BaseModelType.StableDiffusionXL,
    source="madebyollin/sdxl-vae-fp16-fix",
    description="SDXL VAE that works with FP16.",
    type=ModelType.VAE,
)
flux_vae = StarterModel(
    name="FLUX.1-schnell_ae",
    base=BaseModelType.Flux,
    source="black-forest-labs/FLUX.1-schnell::ae.safetensors",
    description="FLUX VAE compatible with both schnell and dev variants.",
    type=ModelType.VAE,
)
# endregion


# region: Main
flux_schnell_quantized = StarterModel(
    name="FLUX Schnell (Quantized)",
    base=BaseModelType.Flux,
    source="InvokeAI/flux_schnell::transformer/bnb_nf4/flux1-schnell-bnb_nf4.safetensors",
    description="FLUX schnell transformer quantized to bitsandbytes NF4 format. Total size with dependencies: ~12GB",
    type=ModelType.Main,
    dependencies=[t5_8b_quantized_encoder, flux_vae, clip_l_encoder],
)
flux_dev_quantized = StarterModel(
    name="FLUX Dev (Quantized)",
    base=BaseModelType.Flux,
    source="InvokeAI/flux_dev::transformer/bnb_nf4/flux1-dev-bnb_nf4.safetensors",
    description="FLUX dev transformer quantized to bitsandbytes NF4 format. Total size with dependencies: ~12GB",
    type=ModelType.Main,
    dependencies=[t5_8b_quantized_encoder, flux_vae, clip_l_encoder],
)
flux_schnell = StarterModel(
    name="FLUX Schnell",
    base=BaseModelType.Flux,
    source="InvokeAI/flux_schnell::transformer/base/flux1-schnell.safetensors",
    description="FLUX schnell transformer in bfloat16. Total size with dependencies: ~33GB",
    type=ModelType.Main,
    dependencies=[t5_base_encoder, flux_vae, clip_l_encoder],
)
flux_dev = StarterModel(
    name="FLUX Dev",
    base=BaseModelType.Flux,
    source="InvokeAI/flux_dev::transformer/base/flux1-dev.safetensors",
    description="FLUX dev transformer in bfloat16. Total size with dependencies: ~33GB",
    type=ModelType.Main,
    dependencies=[t5_base_encoder, flux_vae, clip_l_encoder],
)
sd35_medium = StarterModel(
    name="SD3.5 Medium",
    base=BaseModelType.StableDiffusion3,
    source="stabilityai/stable-diffusion-3.5-medium",
    description="Medium SD3.5 Model: ~15GB",
    type=ModelType.Main,
    dependencies=[],
)
sd35_large = StarterModel(
    name="SD3.5 Large",
    base=BaseModelType.StableDiffusion3,
    source="stabilityai/stable-diffusion-3.5-large",
    description="Large SD3.5 Model: ~19G",
    type=ModelType.Main,
    dependencies=[],
)
cyberrealistic_sd1 = StarterModel(
    name="CyberRealistic v4.1",
    base=BaseModelType.StableDiffusion1,
    source="https://huggingface.co/cyberdelia/CyberRealistic/resolve/main/CyberRealistic_V4.1_FP16.safetensors",
    description="Photorealistic model. See other variants in HF repo 'cyberdelia/CyberRealistic'.",
    type=ModelType.Main,
    dependencies=[cyberrealistic_negative],
)
rev_animated_sd1 = StarterModel(
    name="ReV Animated",
    base=BaseModelType.StableDiffusion1,
    source="stablediffusionapi/rev-animated",
    description="Fantasy and anime style images.",
    type=ModelType.Main,
)
dreamshaper_8_sd1 = StarterModel(
    name="Dreamshaper 8",
    base=BaseModelType.StableDiffusion1,
    source="Lykon/dreamshaper-8",
    description="Popular versatile model.",
    type=ModelType.Main,
)
dreamshaper_8_inpainting_sd1 = StarterModel(
    name="Dreamshaper 8 (inpainting)",
    base=BaseModelType.StableDiffusion1,
    source="Lykon/dreamshaper-8-inpainting",
    description="Inpainting version of Dreamshaper 8.",
    type=ModelType.Main,
)
deliberate_sd1 = StarterModel(
    name="Deliberate v5",
    base=BaseModelType.StableDiffusion1,
    source="https://huggingface.co/XpucT/Deliberate/resolve/main/Deliberate_v5.safetensors",
    description="Popular versatile model",
    type=ModelType.Main,
)
deliberate_inpainting_sd1 = StarterModel(
    name="Deliberate v5 (inpainting)",
    base=BaseModelType.StableDiffusion1,
    source="https://huggingface.co/XpucT/Deliberate/resolve/main/Deliberate_v5-inpainting.safetensors",
    description="Inpainting version of Deliberate v5.",
    type=ModelType.Main,
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

sdxl_refiner = StarterModel(
    name="SDXL Refiner",
    base=BaseModelType.StableDiffusionXLRefiner,
    source="stabilityai/stable-diffusion-xl-refiner-1.0",
    description="The OG Stable Diffusion XL refiner model.",
    type=ModelType.Main,
    dependencies=[sdxl_fp16_vae_fix],
)
# endregion

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
# endregion
# region TI
easy_neg_sd1 = StarterModel(
    name="EasyNegative",
    base=BaseModelType.StableDiffusion1,
    source="https://huggingface.co/embed/EasyNegative/resolve/main/EasyNegative.safetensors",
    description="A textual inversion to use in the negative prompt to reduce bad anatomy",
    type=ModelType.TextualInversion,
)
# endregion
# region IP Adapter
ip_adapter_sd1 = StarterModel(
    name="Standard Reference (IP Adapter)",
    base=BaseModelType.StableDiffusion1,
    source="https://huggingface.co/InvokeAI/ip_adapter_sd15/resolve/main/ip-adapter_sd15.safetensors",
    description="References images with a more generalized/looser degree of precision.",
    type=ModelType.IPAdapter,
    dependencies=[ip_adapter_sd_image_encoder],
    previous_names=["IP Adapter"],
)
ip_adapter_plus_sd1 = StarterModel(
    name="Precise Reference (IP Adapter Plus)",
    base=BaseModelType.StableDiffusion1,
    source="https://huggingface.co/InvokeAI/ip_adapter_plus_sd15/resolve/main/ip-adapter-plus_sd15.safetensors",
    description="References images with a higher degree of precision.",
    type=ModelType.IPAdapter,
    dependencies=[ip_adapter_sd_image_encoder],
    previous_names=["IP Adapter Plus"],
)
ip_adapter_plus_face_sd1 = StarterModel(
    name="Face Reference (IP Adapter Plus Face)",
    base=BaseModelType.StableDiffusion1,
    source="https://huggingface.co/InvokeAI/ip_adapter_plus_face_sd15/resolve/main/ip-adapter-plus-face_sd15.safetensors",
    description="References images with a higher degree of precision, adapted for faces",
    type=ModelType.IPAdapter,
    dependencies=[ip_adapter_sd_image_encoder],
    previous_names=["IP Adapter Plus Face"],
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
ip_adapter_flux = StarterModel(
    name="Standard Reference (XLabs FLUX IP-Adapter v2)",
    base=BaseModelType.Flux,
    source="https://huggingface.co/XLabs-AI/flux-ip-adapter-v2/resolve/main/ip_adapter.safetensors",
    description="References images with a more generalized/looser degree of precision.",
    type=ModelType.IPAdapter,
    dependencies=[clip_vit_l_image_encoder],
)
# endregion
# region ControlNet
qr_code_cnet_sd1 = StarterModel(
    name="QRCode Monster v2 (SD1.5)",
    base=BaseModelType.StableDiffusion1,
    source="monster-labs/control_v1p_sd15_qrcode_monster::v2",
    description="ControlNet model that generates scannable creative QR codes",
    type=ModelType.ControlNet,
)
qr_code_cnet_sdxl = StarterModel(
    name="QRCode Monster (SDXL)",
    base=BaseModelType.StableDiffusionXL,
    source="monster-labs/control_v1p_sdxl_qrcode_monster",
    description="ControlNet model that generates scannable creative QR codes",
    type=ModelType.ControlNet,
)
canny_sd1 = StarterModel(
    name="Hard Edge Detection (canny)",
    base=BaseModelType.StableDiffusion1,
    source="lllyasviel/control_v11p_sd15_canny",
    description="Uses detected edges in the image to control composition.",
    type=ModelType.ControlNet,
    previous_names=["canny"],
)
inpaint_cnet_sd1 = StarterModel(
    name="Inpainting",
    base=BaseModelType.StableDiffusion1,
    source="lllyasviel/control_v11p_sd15_inpaint",
    description="ControlNet weights trained on sd-1.5 with canny conditioning, inpaint version",
    type=ModelType.ControlNet,
    previous_names=["inpaint"],
)
mlsd_sd1 = StarterModel(
    name="Line Drawing (mlsd)",
    base=BaseModelType.StableDiffusion1,
    source="lllyasviel/control_v11p_sd15_mlsd",
    description="Uses straight line detection for controlling the generation.",
    type=ModelType.ControlNet,
    previous_names=["mlsd"],
)
depth_sd1 = StarterModel(
    name="Depth Map",
    base=BaseModelType.StableDiffusion1,
    source="lllyasviel/control_v11f1p_sd15_depth",
    description="Uses depth information in the image to control the depth in the generation.",
    type=ModelType.ControlNet,
    previous_names=["depth"],
)
normal_bae_sd1 = StarterModel(
    name="Lighting Detection (Normals)",
    base=BaseModelType.StableDiffusion1,
    source="lllyasviel/control_v11p_sd15_normalbae",
    description="Uses detected lighting information to guide the lighting of the composition.",
    type=ModelType.ControlNet,
    previous_names=["normal_bae"],
)
seg_sd1 = StarterModel(
    name="Segmentation Map",
    base=BaseModelType.StableDiffusion1,
    source="lllyasviel/control_v11p_sd15_seg",
    description="Uses segmentation maps to guide the structure of the composition.",
    type=ModelType.ControlNet,
    previous_names=["seg"],
)
lineart_sd1 = StarterModel(
    name="Lineart",
    base=BaseModelType.StableDiffusion1,
    source="lllyasviel/control_v11p_sd15_lineart",
    description="Uses lineart detection to guide the lighting of the composition.",
    type=ModelType.ControlNet,
    previous_names=["lineart"],
)
lineart_anime_sd1 = StarterModel(
    name="Lineart Anime",
    base=BaseModelType.StableDiffusion1,
    source="lllyasviel/control_v11p_sd15s2_lineart_anime",
    description="Uses anime lineart detection to guide the lighting of the composition.",
    type=ModelType.ControlNet,
    previous_names=["lineart_anime"],
)
openpose_sd1 = StarterModel(
    name="Pose Detection (openpose)",
    base=BaseModelType.StableDiffusion1,
    source="lllyasviel/control_v11p_sd15_openpose",
    description="Uses pose information to control the pose of human characters in the generation.",
    type=ModelType.ControlNet,
    previous_names=["openpose"],
)
scribble_sd1 = StarterModel(
    name="Contour Detection (scribble)",
    base=BaseModelType.StableDiffusion1,
    source="lllyasviel/control_v11p_sd15_scribble",
    description="Uses edges, contours, or line art in the image to control composition.",
    type=ModelType.ControlNet,
    previous_names=["scribble"],
)
softedge_sd1 = StarterModel(
    name="Soft Edge Detection (softedge)",
    base=BaseModelType.StableDiffusion1,
    source="lllyasviel/control_v11p_sd15_softedge",
    description="Uses a soft edge detection map to control composition.",
    type=ModelType.ControlNet,
    previous_names=["softedge"],
)
shuffle_sd1 = StarterModel(
    name="Remix (shuffle)",
    base=BaseModelType.StableDiffusion1,
    source="lllyasviel/control_v11e_sd15_shuffle",
    description="ControlNet weights trained on sd-1.5 with shuffle image conditioning",
    type=ModelType.ControlNet,
    previous_names=["shuffle"],
)
tile_sd1 = StarterModel(
    name="Tile",
    base=BaseModelType.StableDiffusion1,
    source="lllyasviel/control_v11f1e_sd15_tile",
    description="Uses image data to replicate exact colors/structure in the resulting generation.",
    type=ModelType.ControlNet,
    previous_names=["tile"],
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
# endregion
# region T2I Adapter
t2i_canny_sd1 = StarterModel(
    name="Hard Edge Detection (canny)",
    base=BaseModelType.StableDiffusion1,
    source="TencentARC/t2iadapter_canny_sd15v2",
    description="Uses detected edges in the image to control composition",
    type=ModelType.T2IAdapter,
    previous_names=["canny-sd15"],
)
t2i_sketch_sd1 = StarterModel(
    name="Sketch",
    base=BaseModelType.StableDiffusion1,
    source="TencentARC/t2iadapter_sketch_sd15v2",
    description="Uses a sketch to control composition",
    type=ModelType.T2IAdapter,
    previous_names=["sketch-sd15"],
)
t2i_depth_sd1 = StarterModel(
    name="Depth Map",
    base=BaseModelType.StableDiffusion1,
    source="TencentARC/t2iadapter_depth_sd15v2",
    description="Uses depth information in the image to control the depth in the generation.",
    type=ModelType.T2IAdapter,
    previous_names=["depth-sd15"],
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
# endregion
# region SpandrelImageToImage
realesrgan_anime = StarterModel(
    name="RealESRGAN_x4plus_anime_6B",
    base=BaseModelType.Any,
    source="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    description="A Real-ESRGAN 4x upscaling model (optimized for anime images).",
    type=ModelType.SpandrelImageToImage,
)
realesrgan_x4 = StarterModel(
    name="RealESRGAN_x4plus",
    base=BaseModelType.Any,
    source="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    description="A Real-ESRGAN 4x upscaling model (general-purpose).",
    type=ModelType.SpandrelImageToImage,
)
esrgan_srx4 = StarterModel(
    name="ESRGAN_SRx4_DF2KOST_official",
    base=BaseModelType.Any,
    source="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
    description="The official ESRGAN 4x upscaling model.",
    type=ModelType.SpandrelImageToImage,
)
realesrgan_x2 = StarterModel(
    name="RealESRGAN_x2plus",
    base=BaseModelType.Any,
    source="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    description="A Real-ESRGAN 2x upscaling model (general-purpose).",
    type=ModelType.SpandrelImageToImage,
)
swinir = StarterModel(
    name="SwinIR - realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN",
    base=BaseModelType.Any,
    source="https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN-with-dict-keys-params-and-params_ema.pth",
    description="A SwinIR 4x upscaling model.",
    type=ModelType.SpandrelImageToImage,
)

# endregion

# region CogView4
cogview4 = StarterModel(
    name="CogView4",
    base=BaseModelType.CogView4,
    source="THUDM/CogView4-6B",
    description="The base CogView4 model (~29GB).",
    type=ModelType.Main,
)
# endregion

# region SigLIP
siglip = StarterModel(
    name="SigLIP - google/siglip-so400m-patch14-384",
    base=BaseModelType.Any,
    source="google/siglip-so400m-patch14-384",
    description="A SigLIP model (used by FLUX Redux).",
    type=ModelType.SigLIP,
)
# endregion

# region FLUX Redux
flux_redux = StarterModel(
    name="FLUX Redux",
    base=BaseModelType.Flux,
    source="black-forest-labs/FLUX.1-Redux-dev::flux1-redux-dev.safetensors",
    description="FLUX Redux model (for image variation).",
    type=ModelType.FluxRedux,
    dependencies=[siglip],
)
# endregion

# region LlavaOnevisionModel
llava_onevision = StarterModel(
    name="LLaVA Onevision Qwen2 0.5B",
    base=BaseModelType.Any,
    source="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    description="LLaVA Onevision VLLM model",
    type=ModelType.LlavaOnevision,
)
# endregion

# region FLUX Fill
flux_fill = StarterModel(
    name="FLUX Fill",
    base=BaseModelType.Flux,
    source="black-forest-labs/FLUX.1-Fill-dev::flux1-fill-dev.safetensors",
    description="FLUX Fill model (for inpainting).",
    type=ModelType.Main,
)
# endregion

# List of starter models, displayed on the frontend.
# The order/sort of this list is not changed by the frontend - set it how you want it here.
STARTER_MODELS: list[StarterModel] = [
    flux_schnell_quantized,
    flux_dev_quantized,
    flux_schnell,
    flux_dev,
    sd35_medium,
    sd35_large,
    cyberrealistic_sd1,
    rev_animated_sd1,
    dreamshaper_8_sd1,
    dreamshaper_8_inpainting_sd1,
    deliberate_sd1,
    deliberate_inpainting_sd1,
    juggernaut_sdxl,
    dreamshaper_sdxl,
    archvis_sdxl,
    sdxl_refiner,
    sdxl_fp16_vae_fix,
    flux_vae,
    alien_lora_sdxl,
    noodle_lora_sdxl,
    easy_neg_sd1,
    ip_adapter_sd1,
    ip_adapter_plus_sd1,
    ip_adapter_plus_face_sd1,
    ip_adapter_sdxl,
    ip_adapter_plus_sdxl,
    ip_adapter_flux,
    qr_code_cnet_sd1,
    qr_code_cnet_sdxl,
    canny_sd1,
    inpaint_cnet_sd1,
    mlsd_sd1,
    depth_sd1,
    normal_bae_sd1,
    seg_sd1,
    lineart_sd1,
    lineart_anime_sd1,
    openpose_sd1,
    scribble_sd1,
    softedge_sd1,
    shuffle_sd1,
    tile_sd1,
    canny_sdxl,
    depth_sdxl,
    softedge_sdxl,
    openpose_sdxl,
    scribble_sdxl,
    tile_sdxl,
    union_cnet_sdxl,
    union_cnet_flux,
    flux_canny_control_lora,
    flux_depth_control_lora,
    t2i_canny_sd1,
    t2i_sketch_sd1,
    t2i_depth_sd1,
    t2i_canny_sdxl,
    t2i_lineart_sdxl,
    t2i_sketch_sdxl,
    realesrgan_x4,
    realesrgan_anime,
    realesrgan_x2,
    swinir,
    t5_base_encoder,
    t5_8b_quantized_encoder,
    clip_l_encoder,
    siglip,
    flux_redux,
    llava_onevision,
    flux_fill,
    cogview4,
]

sd1_bundle: list[StarterModel] = [
    dreamshaper_8_sd1,
    easy_neg_sd1,
    ip_adapter_sd1,
    ip_adapter_plus_sd1,
    ip_adapter_plus_face_sd1,
    canny_sd1,
    inpaint_cnet_sd1,
    mlsd_sd1,
    depth_sd1,
    normal_bae_sd1,
    seg_sd1,
    lineart_sd1,
    lineart_anime_sd1,
    openpose_sd1,
    scribble_sd1,
    softedge_sd1,
    shuffle_sd1,
    tile_sd1,
    swinir,
]

sdxl_bundle: list[StarterModel] = [
    juggernaut_sdxl,
    sdxl_fp16_vae_fix,
    ip_adapter_sdxl,
    ip_adapter_plus_sdxl,
    canny_sdxl,
    depth_sdxl,
    softedge_sdxl,
    openpose_sdxl,
    scribble_sdxl,
    tile_sdxl,
    swinir,
]

flux_bundle: list[StarterModel] = [
    flux_schnell_quantized,
    flux_dev_quantized,
    flux_vae,
    t5_8b_quantized_encoder,
    clip_l_encoder,
    union_cnet_flux,
    ip_adapter_flux,
    flux_canny_control_lora,
    flux_depth_control_lora,
    flux_redux,
    flux_fill,
]

STARTER_BUNDLES: dict[str, list[StarterModel]] = {
    BaseModelType.StableDiffusion1: sd1_bundle,
    BaseModelType.StableDiffusionXL: sdxl_bundle,
    BaseModelType.Flux: flux_bundle,
}

assert len(STARTER_MODELS) == len({m.source for m in STARTER_MODELS}), "Duplicate starter models"
