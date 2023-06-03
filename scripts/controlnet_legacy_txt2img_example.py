import os
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers.utils import load_image
from diffusers.models.controlnet import ControlNetModel
from invokeai.backend.generator import Txt2Img
from invokeai.backend.model_management import ModelManager


print("loading 'Girl with a Pearl Earring' image")
image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)
image.show()

print("preprocessing image with Canny edge detection")
image_np = np.array(image)
low_threshold = 100
high_threshold = 200
canny_np = cv2.Canny(image_np, low_threshold, high_threshold)
canny_image = Image.fromarray(canny_np)
canny_image.show()

# using invokeai model management for base model
print("loading base model stable-diffusion-1.5")
model_config_path = os.getcwd() + "/../configs/models.yaml"
model_manager = ModelManager(model_config_path)
model = model_manager.get_model('stable-diffusion-1.5')

print("loading control model lllyasviel/sd-controlnet-canny")
canny_controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",
                                                   torch_dtype=torch.float16).to("cuda")

print("testing Txt2Img() constructor with control_model arg")
txt2img_canny = Txt2Img(model, control_model=canny_controlnet)

print("testing Txt2Img.generate() with control_image arg")
outputs = txt2img_canny.generate(
    prompt="old man",
    control_image=canny_image,
    control_weight=1.0,
    seed=0,
    num_steps=30,
    precision="float16",
)
generate_output = next(outputs)
out_image = generate_output.image
out_image.show()



