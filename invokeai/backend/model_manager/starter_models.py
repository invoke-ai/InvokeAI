from dataclasses import dataclass
from typing import Optional

from invokeai.backend.model_manager.config import BaseModelType, ModelType


@dataclass
class StarterModel:
    description: str
    source: str
    name: str
    base: BaseModelType
    type: ModelType
    # Optional list of model source dependencies that need to be installed before this model can be used
    dependencies: Optional[list[str]] = None
    is_installed: bool = False


# List of starter models, displayed on the frontend.
# The order/sort of this list is not changed by the frontend - set it how you want it here.
STARTER_MODELS: list[StarterModel] = [
    # region: Main
    StarterModel(
        name="SD 1.5 (base)",
        base=BaseModelType.StableDiffusion1,
        source="runwayml/stable-diffusion-v1-5",
        description="Stable Diffusion version 1.5 diffusers model (4.27 GB)",
        type=ModelType.Main,
    ),
    StarterModel(
        name="SD 1.5 (inpainting)",
        base=BaseModelType.StableDiffusion1,
        source="runwayml/stable-diffusion-inpainting",
        description="RunwayML SD 1.5 model optimized for inpainting, diffusers version (4.27 GB)",
        type=ModelType.Main,
    ),
    StarterModel(
        name="Analog Diffusion",
        base=BaseModelType.StableDiffusion1,
        source="wavymulder/Analog-Diffusion",
        description="An SD-1.5 model trained on diverse analog photographs (2.13 GB)",
        type=ModelType.Main,
    ),
    StarterModel(
        name="Deliberate v5",
        base=BaseModelType.StableDiffusion1,
        source="https://huggingface.co/XpucT/Deliberate/resolve/main/Deliberate_v5.safetensors",
        description="Versatile model that produces detailed images up to 768px (4.27 GB)",
        type=ModelType.Main,
    ),
    StarterModel(
        name="Dungeons and Diffusion",
        base=BaseModelType.StableDiffusion1,
        source="0xJustin/Dungeons-and-Diffusion",
        description="Dungeons & Dragons characters (2.13 GB)",
        type=ModelType.Main,
    ),
    StarterModel(
        name="dreamlike photoreal v2",
        base=BaseModelType.StableDiffusion1,
        source="dreamlike-art/dreamlike-photoreal-2.0",
        description="A photorealistic model trained on 768 pixel images based on SD 1.5 (2.13 GB)",
        type=ModelType.Main,
    ),
    StarterModel(
        name="Inkpunk Diffusion",
        base=BaseModelType.StableDiffusion1,
        source="Envvi/Inkpunk-Diffusion",
        description='Stylized illustrations inspired by Gorillaz, FLCL and Shinkawa; prompt with "nvinkpunk" (4.27 GB)',
        type=ModelType.Main,
    ),
    StarterModel(
        name="OpenJourney",
        base=BaseModelType.StableDiffusion1,
        source="prompthero/openjourney",
        description='An SD 1.5 model fine tuned on Midjourney; prompt with "mdjrny-v4 style" (2.13 GB)',
        type=ModelType.Main,
    ),
    StarterModel(
        name="seek.art MEGA",
        base=BaseModelType.StableDiffusion1,
        source="coreco/seek.art_MEGA",
        description='A general use SD-1.5 "anything" model that supports multiple styles (2.1 GB)',
        type=ModelType.Main,
    ),
    StarterModel(
        name="TrinArt v2",
        base=BaseModelType.StableDiffusion1,
        source="naclbit/trinart_stable_diffusion_v2",
        description="An SD-1.5 model finetuned with ~40K assorted high resolution manga/anime-style images (2.13 GB)",
        type=ModelType.Main,
    ),
    StarterModel(
        name="SD 2.1 (base)",
        base=BaseModelType.StableDiffusion2,
        source="stabilityai/stable-diffusion-2-1",
        description="Stable Diffusion version 2.1 diffusers model, trained on 768 pixel images (5.21 GB)",
        type=ModelType.Main,
    ),
    StarterModel(
        name="SD 2.0 (inpainting)",
        base=BaseModelType.StableDiffusion2,
        source="stabilityai/stable-diffusion-2-inpainting",
        description="Stable Diffusion version 2.0 inpainting model (5.21 GB)",
        type=ModelType.Main,
    ),
    StarterModel(
        name="SDXL (base)",
        base=BaseModelType.StableDiffusionXL,
        source="stabilityai/stable-diffusion-xl-base-1.0",
        description="Stable Diffusion XL base model (12 GB)",
        type=ModelType.Main,
    ),
    StarterModel(
        name="SDXL Refiner",
        base=BaseModelType.StableDiffusionXLRefiner,
        source="stabilityai/stable-diffusion-xl-refiner-1.0",
        description="Stable Diffusion XL refiner model (12 GB)",
        type=ModelType.Main,
    ),
    # endregion
    # region VAE
    StarterModel(
        name="sdxl-vae-fp16-fix",
        base=BaseModelType.StableDiffusionXL,
        source="madebyollin/sdxl-vae-fp16-fix",
        description="Version of the SDXL-1.0 VAE that works in half precision mode",
        type=ModelType.VAE,
    ),
    # endregion
    # region LoRA
    StarterModel(
        name="FlatColor",
        base=BaseModelType.StableDiffusion1,
        source="https://civitai.com/models/6433/loraflatcolor",
        description="A LoRA that generates scenery using solid blocks of color",
        type=ModelType.LoRA,
    ),
    StarterModel(
        name="Ink scenery",
        base=BaseModelType.StableDiffusion1,
        source="https://civitai.com/api/download/models/83390",
        description="Generate india ink-like landscapes",
        type=ModelType.LoRA,
    ),
    # endregion
    # region IP Adapter
    StarterModel(
        name="IP Adapter",
        base=BaseModelType.StableDiffusion1,
        source="InvokeAI/ip_adapter_sd15",
        description="IP-Adapter for SD 1.5 models",
        type=ModelType.IPAdapter,
        dependencies=["InvokeAI/ip_adapter_sd_image_encoder"],
    ),
    StarterModel(
        name="IP Adapter Plus",
        base=BaseModelType.StableDiffusion1,
        source="InvokeAI/ip_adapter_plus_sd15",
        description="Refined IP-Adapter for SD 1.5 models",
        type=ModelType.IPAdapter,
        dependencies=["InvokeAI/ip_adapter_sd_image_encoder"],
    ),
    StarterModel(
        name="IP Adapter Plus Face",
        base=BaseModelType.StableDiffusion1,
        source="InvokeAI/ip_adapter_plus_face_sd15",
        description="Refined IP-Adapter for SD 1.5 models, adapted for faces",
        type=ModelType.IPAdapter,
        dependencies=["InvokeAI/ip_adapter_sd_image_encoder"],
    ),
    StarterModel(
        name="IP Adapter SDXL",
        base=BaseModelType.StableDiffusionXL,
        source="InvokeAI/ip_adapter_sdxl",
        description="IP-Adapter for SDXL models",
        type=ModelType.IPAdapter,
        dependencies=["InvokeAI/ip_adapter_sdxl_image_encoder"],
    ),
    # endregion
    # region ControlNet
    StarterModel(
        name="QRCode Monster",
        base=BaseModelType.StableDiffusion1,
        source="monster-labs/control_v1p_sd15_qrcode_monster",
        description="Controlnet model that generates scannable creative QR codes",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="canny",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15_canny",
        description="Controlnet weights trained on sd-1.5 with canny conditioning.",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="inpaint",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15_inpaint",
        description="Controlnet weights trained on sd-1.5 with canny conditioning, inpaint version",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="mlsd",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15_mlsd",
        description="Controlnet weights trained on sd-1.5 with canny conditioning, MLSD version",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="depth",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11f1p_sd15_depth",
        description="Controlnet weights trained on sd-1.5 with depth conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="normal_bae",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15_normalbae",
        description="Controlnet weights trained on sd-1.5 with normalbae image conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="seg",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15_seg",
        description="Controlnet weights trained on sd-1.5 with seg image conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="lineart",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15_lineart",
        description="Controlnet weights trained on sd-1.5 with lineart image conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="lineart_anime",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15s2_lineart_anime",
        description="Controlnet weights trained on sd-1.5 with anime image conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="openpose",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15_openpose",
        description="Controlnet weights trained on sd-1.5 with openpose image conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="scribble",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15_scribble",
        description="Controlnet weights trained on sd-1.5 with scribble image conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="softedge",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11p_sd15_softedge",
        description="Controlnet weights trained on sd-1.5 with soft edge conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="shuffle",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11e_sd15_shuffle",
        description="Controlnet weights trained on sd-1.5 with shuffle image conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="tile",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11f1e_sd15_tile",
        description="Controlnet weights trained on sd-1.5 with tiled image conditioning",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="ip2p",
        base=BaseModelType.StableDiffusion1,
        source="lllyasviel/control_v11e_sd15_ip2p",
        description="Controlnet weights trained on sd-1.5 with ip2p conditioning.",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="canny-sdxl",
        base=BaseModelType.StableDiffusionXL,
        source="diffusers/controlnet-canny-sdxl-1.0",
        description="Controlnet weights trained on sdxl-1.0 with canny conditioning.",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="depth-sdxl",
        base=BaseModelType.StableDiffusionXL,
        source="diffusers/controlnet-depth-sdxl-1.0",
        description="Controlnet weights trained on sdxl-1.0 with depth conditioning.",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="softedge-dexined-sdxl",
        base=BaseModelType.StableDiffusionXL,
        source="SargeZT/controlnet-sd-xl-1.0-softedge-dexined",
        description="Controlnet weights trained on sdxl-1.0 with dexined soft edge preprocessing.",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="depth-16bit-zoe-sdxl",
        base=BaseModelType.StableDiffusionXL,
        source="SargeZT/controlnet-sd-xl-1.0-depth-16bit-zoe",
        description="Controlnet weights trained on sdxl-1.0 with Zoe's preprocessor (16 bits).",
        type=ModelType.ControlNet,
    ),
    StarterModel(
        name="depth-zoe-sdxl",
        base=BaseModelType.StableDiffusionXL,
        source="diffusers/controlnet-zoe-depth-sdxl-1.0",
        description="Controlnet weights trained on sdxl-1.0 with Zoe's preprocessor (32 bits).",
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
    # region TI
    StarterModel(
        name="EasyNegative",
        base=BaseModelType.StableDiffusion1,
        source="https://huggingface.co/embed/EasyNegative/resolve/main/EasyNegative.safetensors",
        description="A textual inversion to use in the negative prompt to reduce bad anatomy",
        type=ModelType.TextualInversion,
    ),
    # endregion
]

assert len(STARTER_MODELS) == len({m.source for m in STARTER_MODELS}), "Duplicate starter models"
