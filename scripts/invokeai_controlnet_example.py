###############################
#
# InvokeAI ControlNet example
#    using backend directly
#
###############################

import os
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers.utils import load_image
from diffusers.models.controlnet import ControlNetModel
from invokeai.backend.generator import Txt2Img
from invokeai.backend.model_management import ModelManager

print("loading original 'Girl With A Pearl Earring' image")
original_image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)

print("Creating Canny edge detection image based on original image")
original_image = np.array(original_image)
canny_array = cv2.Canny(original_image,
                        100,  # low Canny threshold
                        200)  # high Canny threshold
canny_image = Image.fromarray(canny_array)

# using invokeai model management for base model
print("loading base model stable-diffusion-1.5")
model_config_path = os.getcwd() + "/../configs/models.yaml"
model_manager = ModelManager(model_config_path)
model = model_manager.get_model('stable-diffusion-1.5')

# for now using diffusers model.from_pretrained to load ControlNetModel sidemodel
print("loading Canny control model lllyasviel/sd-controlnet-canny")
canny_controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",
                                                   torch_dtype=torch.float16).to("cuda")

print("testing Txt2Img() constructor with ControlNet arg")
txt2img_canny = Txt2Img(model, control_model=canny_controlnet)

print("testing Txt2Img generate() using a Canny edge detection ControlNet and image")
outputs = txt2img_canny.generate(prompt="old man",
                                 control_image=canny_image,
                                 seed=0,
                                 num_steps=20,
                                 control_scale=1.0,
                                 precision="float16")
generate_output = next(outputs)
out_image = generate_output.image

outname = "invokeai_controlnet_testout.png"
print("saving generated image in home directory as: " + outname)
out_image.save(os.path.expanduser("~") + "/" + outname)
