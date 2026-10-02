"""Stable Diffusion 1.x starter models."""

from invokeai.backend.model_manager.starter_models.common import ip_adapter_sd_image_encoder
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelType,
)

cyberrealistic_negative = StarterModel(
    name="CyberRealistic Negative v3",
    base=BaseModelType.StableDiffusion1,
    source="https://huggingface.co/cyberdelia/CyberRealistic_Negative/resolve/main/CyberRealistic_Negative_v3.pt",
    description="Negative embedding specifically for use with CyberRealistic.",
    type=ModelType.TextualInversion,
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

# endregion
# region ControlNet
qr_code_cnet_sd1 = StarterModel(
    name="QRCode Monster v2 (SD1.5)",
    base=BaseModelType.StableDiffusion1,
    source="monster-labs/control_v1p_sd15_qrcode_monster::v2",
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
