"""
Implements optimized version of Stablediffusion txt2img inference
"""

import timeit
import time
import importlib

from diffusers import OnnxStableDiffusionPipeline

utils = importlib.import_module('openvino.utils')
utils.add_openvino_libs_to_path()

#Implementation of class for optimized modules.
class txt2img_Optimized:

    def __init__(self, height, width, steps, num_images):
        self.height = height
        self.width = width
        self.num_images_per_prompt = num_images
        self.num_inference_steps = steps

    def onnx_txt2img(self, prompt, model, precision):
        #model can be "runwayml/stable-diffusion-v1-5" or "CompVis/stable-diffusion-v1-4"
        if precision == "cpu":
            onnx_pipe = OnnxStableDiffusionPipeline.from_pretrained(model, revision="onnx", provider="CPUExecutionProvider")
        else:
            onnx_pipe = OnnxStableDiffusionPipeline.from_pretrained(model, revision="onnx", provider="OpenVINOExecutionProvider")

        t_0 = timeit.default_timer()
        image = onnx_pipe(prompt, self.height, self.height, num_images_per_prompt=self.num_images_per_prompt, num_inference_steps=self.num_inference_steps).images[0]
        t_1 = timeit.default_timer()
        elapsed_time = round((t_1 - t_0), 3)
        print(f"Elapsed time for inference: {elapsed_time}")
        timestamp = int(time.time())
        image.save(f"Inference_{timestamp}.png")
