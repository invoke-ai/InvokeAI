"""
Implements optimized version of Stablediffusion txt2img inference
"""

import time
import importlib
import sys

from diffusers import OnnxStableDiffusionPipeline

utils = importlib.import_module('openvino.utils')
utils.add_openvino_libs_to_path()

#Implementation of class for optimized modules.
class txt2img:

    def __init__(self, height, width, num_images, steps):
        self.height = height
        self.width = width
        self.num_images_per_prompt = num_images
        self.num_inference_steps = steps

    def onnx_txt2img(self, prompt, model, precision, outdir):
        #model ="runwayml/stable-diffusion-v1-5" or "CompVis/stable-diffusion-v1-4"
        if precision == "cpu":
            onnx_pipe = OnnxStableDiffusionPipeline.from_pretrained(model, revision="onnx", provider="CPUExecutionProvider")
        else:
            onnx_pipe = OnnxStableDiffusionPipeline.from_pretrained(model, revision="onnx", provider="OpenVINOExecutionProvider")

        try:
            image = onnx_pipe(prompt, self.height, self.height, num_images_per_prompt=self.num_images_per_prompt, num_inference_steps=self.num_inference_steps).images[0]
            timestamp = int(time.time())
            image.save(f"./{outdir}/Inference_{timestamp}.png")

        except (FileNotFoundError, TypeError, AssertionError) as e:
            print("Execption: ", e)
            sys.exit(-1)
