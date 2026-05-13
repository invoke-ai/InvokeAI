from typing import Optional

from pydantic import BaseModel

from invokeai.backend.model_manager.configs.external_api import (
    ExternalApiModelDefaultSettings,
    ExternalImageSize,
    ExternalModelCapabilities,
    ExternalModelPanelSchema,
    ExternalResolutionPreset,
)
from invokeai.backend.model_manager.taxonomy import (
    AnyVariant,
    BaseModelType,
    ModelFormat,
    ModelType,
    QwenImageVariantType,
)


class StarterModelWithoutDependencies(BaseModel):
    description: str
    source: str
    name: str
    base: BaseModelType
    type: ModelType
    format: Optional[ModelFormat] = None
    variant: Optional[AnyVariant] = None
    is_installed: bool = False
    capabilities: ExternalModelCapabilities | None = None
    default_settings: ExternalApiModelDefaultSettings | None = None
    panel_schema: ExternalModelPanelSchema | None = None
    # allows us to track what models a user has installed across name changes within starter models
    # if you update a starter model name, please add the old one to this list for that starter model
    previous_names: list[str] = []


class StarterModel(StarterModelWithoutDependencies):
    # Optional list of model source dependencies that need to be installed before this model can be used
    dependencies: Optional[list[StarterModelWithoutDependencies]] = None


class StarterModelBundle(BaseModel):
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

# This is CLIP-ViT-H-14-laion2B-s32B-b79K
ip_adapter_sd_image_encoder = StarterModel(
    name="IP Adapter SD1.5 Image Encoder",
    base=BaseModelType.Any,
    source="InvokeAI/ip_adapter_sd_image_encoder",
    description="IP Adapter SD Image Encoder",
    type=ModelType.CLIPVision,
)

# This is CLIP-ViT-bigG-14-laion2B-39B-b160k
ip_adapter_sdxl_image_encoder = StarterModel(
    name="IP Adapter SDXL Image Encoder",
    base=BaseModelType.Any,
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
    description="T5-XXL text encoder (used in FLUX pipelines). ~9.5GB",
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
flux_dev = StarterModel(
    name="FLUX.1 dev",
    base=BaseModelType.Flux,
    source="InvokeAI/flux_dev::transformer/base/flux1-dev.safetensors",
    description="FLUX dev transformer in bfloat16. Total size with dependencies: ~33GB",
    type=ModelType.Main,
    dependencies=[t5_base_encoder, flux_vae, clip_l_encoder],
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
sd35_medium = StarterModel(
    name="SD3.5 Medium",
    base=BaseModelType.StableDiffusion3,
    source="stabilityai/stable-diffusion-3.5-medium",
    description="Medium SD3.5 Model: ~16GB",
    type=ModelType.Main,
    dependencies=[],
)
sd35_large = StarterModel(
    name="SD3.5 Large",
    base=BaseModelType.StableDiffusion3,
    source="stabilityai/stable-diffusion-3.5-large",
    description="Large SD3.5 Model: ~28GB",
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
animesharp_v4_rcan = StarterModel(
    name="2x-AnimeSharpV4_RCAN",
    base=BaseModelType.Any,
    source="https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV4/2x-AnimeSharpV4_RCAN.safetensors",
    description="A 2x upscaling model (optimized for anime images).",
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
    description="The base CogView4 model (~31GB).",
    type=ModelType.Main,
)
# endregion

# region Qwen Image components (shared between Edit and txt2img variants)
qwen_image_vae = StarterModel(
    name="Qwen Image VAE",
    base=BaseModelType.QwenImage,
    source="Qwen/Qwen-Image-Edit-2511::vae/diffusion_pytorch_model.safetensors",
    description="Qwen Image VAE (AutoencoderKLQwenImage), shared between the Edit and txt2img variants. "
    "Use with GGUF transformers to avoid downloading the full ~40GB Diffusers pipeline. (~250MB)",
    type=ModelType.VAE,
    format=ModelFormat.Checkpoint,
)

qwen_vl_encoder_fp8 = StarterModel(
    name="Qwen2.5-VL Encoder (fp8 scaled)",
    base=BaseModelType.Any,
    source="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    description="ComfyUI's single-file FP8-scaled Qwen2.5-VL 7B encoder. Bundles the language model and "
    "visual tower; tokenizer/processor are fetched from HuggingFace on first use. (~7GB)",
    type=ModelType.QwenVLEncoder,
    format=ModelFormat.Checkpoint,
)

qwen_vl_encoder_diffusers = StarterModel(
    name="Qwen2.5-VL Encoder (Diffusers)",
    base=BaseModelType.Any,
    source="Qwen/Qwen-Image-Edit-2511::text_encoder+tokenizer+processor",
    description="Full-precision Qwen2.5-VL 7B encoder in Diffusers folder layout (text_encoder + tokenizer + processor). "
    "Larger than the fp8 variant but no on-the-fly dequantization. (~16GB)",
    type=ModelType.QwenVLEncoder,
    format=ModelFormat.QwenVLEncoder,
)
# endregion

# region Qwen Image Edit
qwen_image_edit = StarterModel(
    name="Qwen Image Edit 2511",
    base=BaseModelType.QwenImage,
    source="Qwen/Qwen-Image-Edit-2511",
    description="Qwen Image Edit 2511 full diffusers model. Supports text-guided image editing with multiple reference images. (~40GB)",
    type=ModelType.Main,
    variant=QwenImageVariantType.Edit,
)

qwen_image_edit_gguf_q4_k_m = StarterModel(
    name="Qwen Image Edit 2511 (Q4_K_M)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q4_K_M.gguf",
    description="Qwen Image Edit 2511 - Q4_K_M quantized transformer. Good quality/size balance. (~13GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=QwenImageVariantType.Edit,
    dependencies=[qwen_image_vae, qwen_vl_encoder_fp8],
)

qwen_image_edit_gguf_q2_k = StarterModel(
    name="Qwen Image Edit 2511 (Q2_K)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q2_K.gguf",
    description="Qwen Image Edit 2511 - Q2_K heavily quantized transformer. Smallest size, lower quality. (~7.5GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=QwenImageVariantType.Edit,
    dependencies=[qwen_image_vae, qwen_vl_encoder_fp8],
)

qwen_image_edit_gguf_q6_k = StarterModel(
    name="Qwen Image Edit 2511 (Q6_K)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q6_K.gguf",
    description="Qwen Image Edit 2511 - Q6_K quantized transformer. Near-lossless quality. (~17GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=QwenImageVariantType.Edit,
    dependencies=[qwen_image_vae, qwen_vl_encoder_fp8],
)

qwen_image_edit_gguf_q8_0 = StarterModel(
    name="Qwen Image Edit 2511 (Q8_0)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q8_0.gguf",
    description="Qwen Image Edit 2511 - Q8_0 quantized transformer. Highest quality quantization. (~22GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    variant=QwenImageVariantType.Edit,
    dependencies=[qwen_image_vae, qwen_vl_encoder_fp8],
)

qwen_image_edit_lightning_4step = StarterModel(
    name="Qwen Image Edit Lightning (4-step, bf16)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    description="Lightning distillation LoRA for Qwen Image Edit — enables generation in just 4 steps. "
    "Settings: Steps=4, CFG=1, Shift Override=3.",
    type=ModelType.LoRA,
)

qwen_image_edit_lightning_8step = StarterModel(
    name="Qwen Image Edit Lightning (8-step, bf16)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
    description="Lightning distillation LoRA for Qwen Image Edit — enables generation in 8 steps with better quality. "
    "Settings: Steps=8, CFG=1, Shift Override=3.",
    type=ModelType.LoRA,
)

# Qwen Image (txt2img)
qwen_image = StarterModel(
    name="Qwen Image 2512",
    base=BaseModelType.QwenImage,
    source="Qwen/Qwen-Image-2512",
    description="Qwen Image 2512 full diffusers model. High-quality text-to-image generation. (~40GB)",
    type=ModelType.Main,
)

qwen_image_gguf_q4_k_m = StarterModel(
    name="Qwen Image 2512 (Q4_K_M)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q4_K_M.gguf",
    description="Qwen Image 2512 - Q4_K_M quantized transformer. Good quality/size balance. (~13GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[qwen_image_vae, qwen_vl_encoder_fp8],
)

qwen_image_gguf_q2_k = StarterModel(
    name="Qwen Image 2512 (Q2_K)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q2_K.gguf",
    description="Qwen Image 2512 - Q2_K heavily quantized transformer. Smallest size, lower quality. (~7.5GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[qwen_image_vae, qwen_vl_encoder_fp8],
)

qwen_image_gguf_q6_k = StarterModel(
    name="Qwen Image 2512 (Q6_K)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q6_K.gguf",
    description="Qwen Image 2512 - Q6_K quantized transformer. Near-lossless quality. (~17GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[qwen_image_vae, qwen_vl_encoder_fp8],
)

qwen_image_gguf_q8_0 = StarterModel(
    name="Qwen Image 2512 (Q8_0)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q8_0.gguf",
    description="Qwen Image 2512 - Q8_0 quantized transformer. Highest quality quantization. (~22GB)",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[qwen_image_vae, qwen_vl_encoder_fp8],
)

qwen_image_lightning_4step = StarterModel(
    name="Qwen Image Lightning (4-step, V2.0, bf16)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors",
    description="Lightning distillation LoRA for Qwen Image — enables generation in just 4 steps. "
    "Settings: Steps=4, CFG=1, Shift Override=3.",
    type=ModelType.LoRA,
)

qwen_image_lightning_8step = StarterModel(
    name="Qwen Image Lightning (8-step, V2.0, bf16)",
    base=BaseModelType.QwenImage,
    source="https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors",
    description="Lightning distillation LoRA for Qwen Image — enables generation in 8 steps with better quality. "
    "Settings: Steps=8, CFG=1, Shift Override=3.",
    type=ModelType.LoRA,
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

# region LlavaOnevisionModel (vision-language models for Image-to-Prompt)
llava_onevision = StarterModel(
    name="LLaVA Onevision Qwen2 0.5B",
    base=BaseModelType.Any,
    source="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    description="LLaVA Onevision vision-language model (~1 GB). Lightweight default for the Image-to-Prompt feature.",
    type=ModelType.LlavaOnevision,
)

llava_onevision_7b = StarterModel(
    name="LLaVA Onevision Qwen2 7B",
    base=BaseModelType.Any,
    source="llava-hf/llava-onevision-qwen2-7b-ov-hf",
    description="LLaVA Onevision 7B vision-language model. Larger, higher-quality alternative for Image-to-Prompt. (~16 GB)",
    type=ModelType.LlavaOnevision,
)
# endregion

# region TextLLM (causal language models for Prompt Expansion)
qwen2_5_1_5b_instruct = StarterModel(
    name="Qwen2.5-1.5B-Instruct",
    base=BaseModelType.Any,
    source="Qwen/Qwen2.5-1.5B-Instruct",
    description="Qwen2.5 1.5B instruction-tuned LLM. Recommended default for the Prompt Expansion feature — small and fast. (~3 GB)",
    type=ModelType.TextLLM,
)

qwen2_5_3b_instruct = StarterModel(
    name="Qwen2.5-3B-Instruct",
    base=BaseModelType.Any,
    source="Qwen/Qwen2.5-3B-Instruct",
    description="Qwen2.5 3B instruction-tuned LLM. Better prompt expansion quality at the cost of more VRAM. (~6 GB)",
    type=ModelType.TextLLM,
)

smollm2_1_7b_instruct = StarterModel(
    name="SmolLM2-1.7B-Instruct",
    base=BaseModelType.Any,
    source="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    description="SmolLM2 1.7B instruction-tuned LLM (Apache-2.0). Alternative to Qwen for prompt expansion. (~3 GB)",
    type=ModelType.TextLLM,
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

# region FLUX.2 Klein
flux2_vae = StarterModel(
    name="FLUX.2 VAE",
    base=BaseModelType.Flux2,
    source="black-forest-labs/FLUX.2-klein-4B::vae",
    description="FLUX.2 VAE (16-channel, same architecture as FLUX.1 VAE). ~168MB",
    type=ModelType.VAE,
)

flux2_klein_qwen3_4b_encoder = StarterModel(
    name="FLUX.2 Klein Qwen3 4B Encoder",
    base=BaseModelType.Any,
    source="black-forest-labs/FLUX.2-klein-4B::text_encoder+tokenizer",
    description="Qwen3 4B text encoder for FLUX.2 Klein 4B (also compatible with Z-Image). ~8GB",
    type=ModelType.Qwen3Encoder,
)

flux2_klein_qwen3_8b_encoder = StarterModel(
    name="FLUX.2 Klein Qwen3 8B Encoder",
    base=BaseModelType.Any,
    source="black-forest-labs/FLUX.2-klein-9B::text_encoder+tokenizer",
    description="Qwen3 8B text encoder for FLUX.2 Klein 9B models. ~16GB",
    type=ModelType.Qwen3Encoder,
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
# endregion

# region Z-Image
z_image_qwen3_encoder = StarterModel(
    name="Z-Image Qwen3 Text Encoder",
    base=BaseModelType.Any,
    source="Tongyi-MAI/Z-Image-Turbo::text_encoder+tokenizer",
    description="Qwen3 4B text encoder with tokenizer for Z-Image (full precision). ~8GB",
    type=ModelType.Qwen3Encoder,
)

z_image_qwen3_encoder_quantized = StarterModel(
    name="Z-Image Qwen3 Text Encoder (quantized)",
    base=BaseModelType.Any,
    source="https://huggingface.co/worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF/resolve/main/Qwen_3_4b-Q6_K.gguf",
    description="Qwen3 4B text encoder for Z-Image quantized to GGUF Q6_K format. ~3.3GB",
    type=ModelType.Qwen3Encoder,
    format=ModelFormat.GGUFQuantized,
)

z_image_turbo = StarterModel(
    name="Z-Image Turbo",
    base=BaseModelType.ZImage,
    source="Tongyi-MAI/Z-Image-Turbo",
    description="Z-Image Turbo - fast 6B parameter text-to-image model with 8 inference steps. Supports bilingual prompts (English & Chinese). ~33GB",
    type=ModelType.Main,
)

z_image_turbo_quantized = StarterModel(
    name="Z-Image Turbo (quantized)",
    base=BaseModelType.ZImage,
    source="https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q4_K.gguf",
    description="Z-Image Turbo quantized to GGUF Q4_K format. Requires standalone Qwen3 text encoder and Flux VAE. ~4GB",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[z_image_qwen3_encoder_quantized, flux_vae],
)

z_image_turbo_q8 = StarterModel(
    name="Z-Image Turbo (Q8)",
    base=BaseModelType.ZImage,
    source="https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q8_0.gguf",
    description="Z-Image Turbo quantized to GGUF Q8_0 format. Higher quality, larger size. Requires standalone Qwen3 text encoder and Flux VAE. ~6.6GB",
    type=ModelType.Main,
    format=ModelFormat.GGUFQuantized,
    dependencies=[z_image_qwen3_encoder_quantized, flux_vae],
)

z_image_controlnet_union = StarterModel(
    name="Z-Image ControlNet Union",
    base=BaseModelType.ZImage,
    source="https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors",
    description="Unified ControlNet for Z-Image Turbo supporting Canny, HED, Depth, Pose, MLSD, and Inpainting modes.",
    type=ModelType.ControlNet,
)

z_image_controlnet_tile = StarterModel(
    name="Z-Image ControlNet Tile",
    base=BaseModelType.ZImage,
    source="https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1/resolve/main/Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.safetensors",
    description="Dedicated Tile ControlNet for Z-Image Turbo. Useful for upscaling and adding detail. ~6.7GB",
    type=ModelType.ControlNet,
)
# endregion

# region External API
GEMINI_3_IMAGE_ALLOWED_ASPECT_RATIOS = [
    "1:1",
    "1:4",
    "1:8",
    "2:3",
    "3:2",
    "3:4",
    "4:1",
    "4:3",
    "4:5",
    "5:4",
    "8:1",
    "9:16",
    "16:9",
    "21:9",
]
GEMINI_3_IMAGE_MAX_SIZE = ExternalImageSize(width=4096, height=4096)


def _gemini_3_resolution_presets(
    image_sizes: list[str],
    aspect_ratios: list[str] | None = None,
) -> list[ExternalResolutionPreset]:
    """Build resolution presets for Gemini 3 models.

    Each preset combines an aspect ratio with an image size preset (512/1K/2K/4K).
    Pixel dimensions are approximations based on the preset name (longest side).
    """
    if aspect_ratios is None:
        aspect_ratios = GEMINI_3_IMAGE_ALLOWED_ASPECT_RATIOS
    base_pixels = {"512": 512, "1K": 1024, "2K": 2048, "4K": 4096}
    presets: list[ExternalResolutionPreset] = []
    for image_size in image_sizes:
        base = base_pixels[image_size]
        for ratio_str in aspect_ratios:
            w_part, h_part = (int(x) for x in ratio_str.split(":"))
            if w_part >= h_part:
                w = base
                h = max(1, round(base * h_part / w_part))
            else:
                h = base
                w = max(1, round(base * w_part / h_part))
            presets.append(
                ExternalResolutionPreset(
                    label=f"{ratio_str} ({image_size}) — {w}\u00d7{h}",
                    aspect_ratio=ratio_str,
                    image_size=image_size,
                    width=w,
                    height=h,
                )
            )
    return presets


GEMINI_3_PRO_RESOLUTION_PRESETS = _gemini_3_resolution_presets(["1K", "2K", "4K"])
GEMINI_3_1_FLASH_RESOLUTION_PRESETS = _gemini_3_resolution_presets(["512", "1K", "2K", "4K"])

gemini_flash_image = StarterModel(
    name="Gemini 2.5 Flash Image",
    base=BaseModelType.External,
    source="external://gemini/gemini-2.5-flash-image",
    description="Google Gemini 2.5 Flash image generation model (external API). Requires a configured Gemini API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img"],
        supports_seed=True,
        supports_reference_images=True,
        max_images_per_request=1,
        allowed_aspect_ratios=[
            "1:1",
            "2:3",
            "3:2",
            "3:4",
            "4:3",
            "4:5",
            "5:4",
            "9:16",
            "16:9",
            "21:9",
        ],
        aspect_ratio_sizes={
            "1:1": ExternalImageSize(width=1024, height=1024),
            "2:3": ExternalImageSize(width=832, height=1248),
            "3:2": ExternalImageSize(width=1248, height=832),
            "3:4": ExternalImageSize(width=864, height=1184),
            "4:3": ExternalImageSize(width=1184, height=864),
            "4:5": ExternalImageSize(width=896, height=1152),
            "5:4": ExternalImageSize(width=1152, height=896),
            "9:16": ExternalImageSize(width=768, height=1344),
            "16:9": ExternalImageSize(width=1344, height=768),
            "21:9": ExternalImageSize(width=1536, height=672),
        },
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1),
    panel_schema=ExternalModelPanelSchema(prompts=[{"name": "reference_images"}], image=[{"name": "dimensions"}]),
)
gemini_pro_image_preview = StarterModel(
    name="Gemini 3 Pro Image Preview",
    base=BaseModelType.External,
    source="external://gemini/gemini-3-pro-image-preview",
    description="Google Gemini 3 Pro image generation preview model (external API). Supports up to 14 reference images, including up to 6 object references and up to 5 character references. Supports 1K/2K/4K resolution presets. Requires a configured Gemini API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img"],
        supports_seed=True,
        supports_reference_images=True,
        max_reference_images=14,
        max_images_per_request=1,
        max_image_size=GEMINI_3_IMAGE_MAX_SIZE,
        allowed_aspect_ratios=GEMINI_3_IMAGE_ALLOWED_ASPECT_RATIOS,
        resolution_presets=GEMINI_3_PRO_RESOLUTION_PRESETS,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1),
    panel_schema=ExternalModelPanelSchema(prompts=[{"name": "reference_images"}], image=[{"name": "dimensions"}]),
)
gemini_3_1_flash_image_preview = StarterModel(
    name="Gemini 3.1 Flash Image Preview",
    base=BaseModelType.External,
    source="external://gemini/gemini-3.1-flash-image-preview",
    description="Google Gemini 3.1 Flash image generation preview model (external API). Supports up to 14 reference images, including up to 10 object references and up to 4 character references. Supports 512/1K/2K/4K resolution presets. Requires a configured Gemini API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img"],
        supports_seed=True,
        supports_reference_images=True,
        max_reference_images=14,
        max_images_per_request=1,
        max_image_size=GEMINI_3_IMAGE_MAX_SIZE,
        allowed_aspect_ratios=GEMINI_3_IMAGE_ALLOWED_ASPECT_RATIOS,
        resolution_presets=GEMINI_3_1_FLASH_RESOLUTION_PRESETS,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1),
    panel_schema=ExternalModelPanelSchema(prompts=[{"name": "reference_images"}], image=[{"name": "dimensions"}]),
)
QWEN_IMAGE_2_ALLOWED_ASPECT_RATIOS = ["1:1", "4:3", "3:4", "16:9", "9:16"]
QWEN_IMAGE_MAX_ALLOWED_ASPECT_RATIOS = ["1:1", "4:3", "3:4", "16:9", "9:16"]
WAN_V2_ALLOWED_ASPECT_RATIOS = ["1:1", "4:3", "3:4", "16:9", "9:16"]

alibabacloud_qwen_image_2_pro = StarterModel(
    name="Qwen Image 2.0 Pro",
    base=BaseModelType.External,
    source="external://alibabacloud/qwen-image-2.0-pro",
    description="Alibaba Cloud Qwen Image 2.0 Pro model (external API). Best quality text-to-image with excellent bilingual text rendering. Requires a configured Alibaba Cloud DashScope API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img"],
        supports_negative_prompt=False,
        supports_seed=True,
        max_images_per_request=4,
        allowed_aspect_ratios=QWEN_IMAGE_2_ALLOWED_ASPECT_RATIOS,
        aspect_ratio_sizes={
            "1:1": ExternalImageSize(width=2048, height=2048),
            "4:3": ExternalImageSize(width=2368, height=1728),
            "3:4": ExternalImageSize(width=1728, height=2368),
            "16:9": ExternalImageSize(width=2688, height=1536),
            "9:16": ExternalImageSize(width=1536, height=2688),
        },
    ),
    default_settings=ExternalApiModelDefaultSettings(width=2048, height=2048, num_images=1),
    panel_schema=ExternalModelPanelSchema(image=[{"name": "dimensions"}]),
)
alibabacloud_qwen_image_2 = StarterModel(
    name="Qwen Image 2.0",
    base=BaseModelType.External,
    source="external://alibabacloud/qwen-image-2.0",
    description="Alibaba Cloud Qwen Image 2.0 model (external API). Fast text-to-image with good bilingual text rendering. Requires a configured Alibaba Cloud DashScope API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img"],
        supports_negative_prompt=False,
        supports_seed=True,
        max_images_per_request=4,
        allowed_aspect_ratios=QWEN_IMAGE_2_ALLOWED_ASPECT_RATIOS,
        aspect_ratio_sizes={
            "1:1": ExternalImageSize(width=2048, height=2048),
            "4:3": ExternalImageSize(width=2368, height=1728),
            "3:4": ExternalImageSize(width=1728, height=2368),
            "16:9": ExternalImageSize(width=2688, height=1536),
            "9:16": ExternalImageSize(width=1536, height=2688),
        },
    ),
    default_settings=ExternalApiModelDefaultSettings(width=2048, height=2048, num_images=1),
    panel_schema=ExternalModelPanelSchema(image=[{"name": "dimensions"}]),
)
alibabacloud_qwen_image_max = StarterModel(
    name="Qwen Image Max",
    base=BaseModelType.External,
    source="external://alibabacloud/qwen-image-max",
    description="Alibaba Cloud Qwen Image Max model (external API). High quality text-to-image generation. Requires a configured Alibaba Cloud DashScope API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img"],
        supports_negative_prompt=False,
        supports_seed=True,
        max_images_per_request=4,
        allowed_aspect_ratios=QWEN_IMAGE_MAX_ALLOWED_ASPECT_RATIOS,
        aspect_ratio_sizes={
            "1:1": ExternalImageSize(width=1328, height=1328),
            "4:3": ExternalImageSize(width=1472, height=1104),
            "3:4": ExternalImageSize(width=1104, height=1472),
            "16:9": ExternalImageSize(width=1664, height=928),
            "9:16": ExternalImageSize(width=928, height=1664),
        },
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1328, height=1328, num_images=1),
    panel_schema=ExternalModelPanelSchema(image=[{"name": "dimensions"}]),
)
alibabacloud_wan26_t2i = StarterModel(
    name="Wan 2.6 Text-to-Image",
    base=BaseModelType.External,
    source="external://alibabacloud/wan2.6-t2i",
    description="Alibaba Cloud Wan 2.6 text-to-image model (external API). Photorealistic image generation. Requires a configured Alibaba Cloud DashScope API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img"],
        supports_negative_prompt=False,
        supports_seed=True,
        max_images_per_request=4,
        allowed_aspect_ratios=WAN_V2_ALLOWED_ASPECT_RATIOS,
        aspect_ratio_sizes={
            "1:1": ExternalImageSize(width=1024, height=1024),
            "4:3": ExternalImageSize(width=1440, height=1080),
            "3:4": ExternalImageSize(width=1080, height=1440),
            "16:9": ExternalImageSize(width=1440, height=810),
            "9:16": ExternalImageSize(width=810, height=1440),
        },
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1),
    panel_schema=ExternalModelPanelSchema(image=[{"name": "dimensions"}]),
)
alibabacloud_qwen_image_edit_max = StarterModel(
    name="Qwen Image Edit Max",
    base=BaseModelType.External,
    source="external://alibabacloud/qwen-image-edit-max",
    description="Alibaba Cloud Qwen Image Edit Max model (external API). Image editing with industrial design and geometric reasoning, driven by up to 3 reference images. Requires a configured Alibaba Cloud DashScope API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img"],
        supports_negative_prompt=False,
        supports_reference_images=True,
        supports_seed=True,
        max_images_per_request=4,
        allowed_aspect_ratios=QWEN_IMAGE_2_ALLOWED_ASPECT_RATIOS,
        aspect_ratio_sizes={
            "1:1": ExternalImageSize(width=2048, height=2048),
            "4:3": ExternalImageSize(width=2368, height=1728),
            "3:4": ExternalImageSize(width=1728, height=2368),
            "16:9": ExternalImageSize(width=2688, height=1536),
            "9:16": ExternalImageSize(width=1536, height=2688),
        },
    ),
    default_settings=ExternalApiModelDefaultSettings(width=2048, height=2048, num_images=1),
    panel_schema=ExternalModelPanelSchema(prompts=[{"name": "reference_images"}], image=[{"name": "dimensions"}]),
)
OPENAI_GPT_IMAGE_ASPECT_RATIOS = ["1:1", "3:2", "2:3"]
OPENAI_GPT_IMAGE_ASPECT_RATIO_SIZES = {
    "1:1": ExternalImageSize(width=1024, height=1024),
    "3:2": ExternalImageSize(width=1536, height=1024),
    "2:3": ExternalImageSize(width=1024, height=1536),
}
OPENAI_GPT_IMAGE_PANEL_SCHEMA = ExternalModelPanelSchema(
    prompts=[{"name": "reference_images"}], image=[{"name": "dimensions"}]
)

openai_gpt_image_1_5 = StarterModel(
    name="GPT Image 1.5",
    base=BaseModelType.External,
    source="external://openai/gpt-image-1.5",
    description="OpenAI GPT-Image-1.5 image generation model. Fastest and most affordable GPT image model. Requires a configured OpenAI API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img", "img2img"],
        supports_reference_images=True,
        max_images_per_request=10,
        allowed_aspect_ratios=OPENAI_GPT_IMAGE_ASPECT_RATIOS,
        aspect_ratio_sizes=OPENAI_GPT_IMAGE_ASPECT_RATIO_SIZES,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1),
    panel_schema=OPENAI_GPT_IMAGE_PANEL_SCHEMA,
)
openai_gpt_image_1 = StarterModel(
    name="GPT Image 1",
    base=BaseModelType.External,
    source="external://openai/gpt-image-1",
    description="OpenAI GPT-Image-1 image generation model. High quality image generation. Requires a configured OpenAI API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img", "img2img"],
        supports_reference_images=True,
        max_images_per_request=10,
        allowed_aspect_ratios=OPENAI_GPT_IMAGE_ASPECT_RATIOS,
        aspect_ratio_sizes=OPENAI_GPT_IMAGE_ASPECT_RATIO_SIZES,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1),
    panel_schema=OPENAI_GPT_IMAGE_PANEL_SCHEMA,
)
openai_gpt_image_1_mini = StarterModel(
    name="GPT Image 1 Mini",
    base=BaseModelType.External,
    source="external://openai/gpt-image-1-mini",
    description="OpenAI GPT-Image-1-Mini image generation model. Cost-efficient option, 80%% cheaper than GPT-Image-1. Requires a configured OpenAI API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img", "img2img"],
        supports_reference_images=True,
        max_images_per_request=10,
        allowed_aspect_ratios=OPENAI_GPT_IMAGE_ASPECT_RATIOS,
        aspect_ratio_sizes=OPENAI_GPT_IMAGE_ASPECT_RATIO_SIZES,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1),
    panel_schema=OPENAI_GPT_IMAGE_PANEL_SCHEMA,
)
openai_dall_e_3 = StarterModel(
    name="DALL-E 3",
    base=BaseModelType.External,
    source="external://openai/dall-e-3",
    description="OpenAI DALL-E 3 image generation model. Supports vivid and natural styles. Only text-to-image, no editing. Requires a configured OpenAI API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img"],
        max_images_per_request=1,
        allowed_aspect_ratios=["1:1", "7:4", "4:7"],
        aspect_ratio_sizes={
            "1:1": ExternalImageSize(width=1024, height=1024),
            "7:4": ExternalImageSize(width=1792, height=1024),
            "4:7": ExternalImageSize(width=1024, height=1792),
        },
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1),
    panel_schema=ExternalModelPanelSchema(image=[{"name": "dimensions"}]),
)
SEEDREAM_ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]
SEEDREAM_2K_SIZES = {
    "1:1": ExternalImageSize(width=2048, height=2048),
    "3:4": ExternalImageSize(width=1728, height=2304),
    "4:3": ExternalImageSize(width=2304, height=1728),
    "16:9": ExternalImageSize(width=2848, height=1600),
    "9:16": ExternalImageSize(width=1600, height=2848),
    "3:2": ExternalImageSize(width=2496, height=1664),
    "2:3": ExternalImageSize(width=1664, height=2496),
    "21:9": ExternalImageSize(width=3136, height=1344),
}
SEEDREAM_1K_SIZES = {
    "1:1": ExternalImageSize(width=1024, height=1024),
    "3:4": ExternalImageSize(width=864, height=1152),
    "4:3": ExternalImageSize(width=1152, height=864),
    "16:9": ExternalImageSize(width=1312, height=736),
    "9:16": ExternalImageSize(width=736, height=1312),
    "2:3": ExternalImageSize(width=832, height=1248),
    "3:2": ExternalImageSize(width=1248, height=832),
    "21:9": ExternalImageSize(width=1568, height=672),
}
SEEDREAM_PANEL_SCHEMA = ExternalModelPanelSchema(prompts=[{"name": "reference_images"}], image=[{"name": "dimensions"}])
seedream_5_0 = StarterModel(
    name="Seedream 5.0",
    base=BaseModelType.External,
    source="external://seedream/seedream-5-0-260128",
    description="BytePlus Seedream 5.0 flagship image generation model (external API). Supports 2K and 4K resolutions, txt2img and img2img with multi-image reference input.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img", "img2img"],
        supports_reference_images=True,
        max_reference_images=14,
        max_images_per_request=15,
        allowed_aspect_ratios=SEEDREAM_ASPECT_RATIOS,
        aspect_ratio_sizes=SEEDREAM_2K_SIZES,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=2048, height=2048, num_images=1),
    panel_schema=SEEDREAM_PANEL_SCHEMA,
)
seedream_5_0_lite = StarterModel(
    name="Seedream 5.0 Lite",
    base=BaseModelType.External,
    source="external://seedream/seedream-5-0-lite-260128",
    description="BytePlus Seedream 5.0 Lite image generation model (external API). Supports 2K and 4K resolutions, txt2img and img2img with multi-image reference input.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img", "img2img"],
        supports_reference_images=True,
        max_reference_images=14,
        max_images_per_request=15,
        allowed_aspect_ratios=SEEDREAM_ASPECT_RATIOS,
        aspect_ratio_sizes=SEEDREAM_2K_SIZES,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=2048, height=2048, num_images=1),
    panel_schema=SEEDREAM_PANEL_SCHEMA,
)
seedream_4_5 = StarterModel(
    name="Seedream 4.5",
    base=BaseModelType.External,
    source="external://seedream/seedream-4-5-251128",
    description="BytePlus Seedream 4.5 image generation model (external API). Supports 2K and 4K resolutions, txt2img, img2img, batch generation, and multi-image reference input.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img", "img2img"],
        supports_reference_images=True,
        max_reference_images=14,
        max_images_per_request=15,
        allowed_aspect_ratios=SEEDREAM_ASPECT_RATIOS,
        aspect_ratio_sizes=SEEDREAM_2K_SIZES,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=2048, height=2048, num_images=1),
    panel_schema=SEEDREAM_PANEL_SCHEMA,
)
seedream_4_0 = StarterModel(
    name="Seedream 4.0",
    base=BaseModelType.External,
    source="external://seedream/seedream-4-0-250828",
    description="BytePlus Seedream 4.0 image generation model (external API). Supports 1K, 2K, and 4K resolutions, txt2img, img2img, batch generation, and multi-image reference input.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img", "img2img"],
        supports_reference_images=True,
        max_reference_images=14,
        max_images_per_request=15,
        allowed_aspect_ratios=SEEDREAM_ASPECT_RATIOS,
        aspect_ratio_sizes=SEEDREAM_2K_SIZES,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=2048, height=2048, num_images=1),
    panel_schema=SEEDREAM_PANEL_SCHEMA,
)
# Seedream 3.0 T2I (seedream-3-0-t2i-250415) removed — deprecated by BytePlus, replaced by seedream-4-0-250828.

# DALL-E 2 removed — deprecated by OpenAI, shutdown May 12, 2026.
# region Anima
anima_qwen3_encoder = StarterModel(
    name="Anima Qwen3 0.6B Text Encoder",
    base=BaseModelType.Any,
    source="https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors",
    description="Qwen3 0.6B text encoder for Anima. ~1.2GB",
    type=ModelType.Qwen3Encoder,
    format=ModelFormat.Checkpoint,
)

anima_vae = StarterModel(
    name="Anima QwenImage VAE",
    base=BaseModelType.Anima,
    source="https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors",
    description="QwenImage VAE for Anima (fine-tuned Wan 2.1 VAE, 16 latent channels). ~200MB",
    type=ModelType.VAE,
    format=ModelFormat.Checkpoint,
)

anima_preview3 = StarterModel(
    name="Anima Preview 3",
    base=BaseModelType.Anima,
    source="https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-preview3-base.safetensors",
    description="Anima Preview 3 - 2B parameter anime-focused text-to-image model built on Cosmos Predict2 DiT. ~4.5GB",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    dependencies=[anima_qwen3_encoder, anima_vae, t5_base_encoder],
)
# endregion

# List of starter models, displayed on the frontend.
# The order/sort of this list is not changed by the frontend - set it how you want it here.
STARTER_MODELS: list[StarterModel] = [
    flux_kontext_quantized,
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
    animesharp_v4_rcan,
    realesrgan_x2,
    swinir,
    t5_base_encoder,
    t5_8b_quantized_encoder,
    clip_l_encoder,
    siglip,
    flux_redux,
    llava_onevision,
    llava_onevision_7b,
    qwen2_5_1_5b_instruct,
    qwen2_5_3b_instruct,
    smollm2_1_7b_instruct,
    flux_fill,
    flux2_vae,
    flux2_klein_4b,
    flux2_klein_4b_single,
    flux2_klein_4b_fp8,
    flux2_klein_9b,
    flux2_klein_9b_fp8,
    flux2_klein_4b_gguf_q4,
    flux2_klein_4b_gguf_q8,
    flux2_klein_9b_gguf_q4,
    flux2_klein_9b_gguf_q8,
    flux2_klein_qwen3_4b_encoder,
    flux2_klein_qwen3_8b_encoder,
    cogview4,
    qwen_image_vae,
    qwen_vl_encoder_fp8,
    qwen_vl_encoder_diffusers,
    qwen_image_edit,
    qwen_image_edit_gguf_q2_k,
    qwen_image_edit_gguf_q4_k_m,
    qwen_image_edit_gguf_q6_k,
    qwen_image_edit_gguf_q8_0,
    qwen_image_edit_lightning_4step,
    qwen_image_edit_lightning_8step,
    qwen_image,
    qwen_image_gguf_q2_k,
    qwen_image_gguf_q4_k_m,
    qwen_image_gguf_q6_k,
    qwen_image_gguf_q8_0,
    qwen_image_lightning_4step,
    qwen_image_lightning_8step,
    flux_krea,
    flux_krea_quantized,
    z_image_turbo,
    z_image_turbo_quantized,
    z_image_turbo_q8,
    z_image_qwen3_encoder,
    z_image_qwen3_encoder_quantized,
    z_image_controlnet_union,
    z_image_controlnet_tile,
    gemini_flash_image,
    gemini_pro_image_preview,
    gemini_3_1_flash_image_preview,
    openai_gpt_image_1_5,
    openai_gpt_image_1,
    openai_gpt_image_1_mini,
    openai_dall_e_3,
    seedream_5_0,
    seedream_5_0_lite,
    seedream_4_5,
    seedream_4_0,
    alibabacloud_qwen_image_2_pro,
    alibabacloud_qwen_image_2,
    alibabacloud_qwen_image_max,
    alibabacloud_wan26_t2i,
    alibabacloud_qwen_image_edit_max,
    anima_preview3,
    anima_qwen3_encoder,
    anima_vae,
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
    flux_kontext_quantized,
    flux_krea_quantized,
]

zimage_bundle: list[StarterModel] = [
    z_image_turbo_quantized,
    z_image_qwen3_encoder_quantized,
    z_image_controlnet_union,
    z_image_controlnet_tile,
    flux_vae,
]

flux2_klein_bundle: list[StarterModel] = [
    flux2_klein_4b_gguf_q4,
    flux2_vae,
    flux2_klein_qwen3_4b_encoder,
]

qwen_image_bundle: list[StarterModel] = [
    qwen_image_vae,
    qwen_vl_encoder_fp8,
    qwen_image_edit,
    qwen_image_edit_gguf_q4_k_m,
    qwen_image_edit_gguf_q8_0,
    qwen_image_edit_lightning_4step,
    qwen_image_edit_lightning_8step,
    qwen_image,
    qwen_image_gguf_q4_k_m,
    qwen_image_gguf_q8_0,
    qwen_image_lightning_4step,
    qwen_image_lightning_8step,
]

anima_bundle: list[StarterModel] = [
    anima_preview3,
    anima_qwen3_encoder,
    anima_vae,
    t5_base_encoder,
]

STARTER_BUNDLES: dict[str, StarterModelBundle] = {
    BaseModelType.StableDiffusion1: StarterModelBundle(name="Stable Diffusion 1.5", models=sd1_bundle),
    BaseModelType.StableDiffusionXL: StarterModelBundle(name="SDXL", models=sdxl_bundle),
    BaseModelType.Flux: StarterModelBundle(name="FLUX.1 dev", models=flux_bundle),
    BaseModelType.Flux2: StarterModelBundle(name="FLUX.2 Klein", models=flux2_klein_bundle),
    BaseModelType.ZImage: StarterModelBundle(name="Z-Image Turbo", models=zimage_bundle),
    BaseModelType.QwenImage: StarterModelBundle(name="Qwen Image", models=qwen_image_bundle),
    BaseModelType.Anima: StarterModelBundle(name="Anima", models=anima_bundle),
}

assert len(STARTER_MODELS) == len({m.source for m in STARTER_MODELS}), "Duplicate starter models"
