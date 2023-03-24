from diffusers import OnnxStableDiffusionPipeline
from optimum.intel.openvino import OVStableDiffusionPipeline

import openvino.utils as utils
utils.add_openvino_libs_to_path()

import timeit
import subprocess

class Txt2Img_Optimized:

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.num_images_per_prompt = 1

    def install_requirements(self, command):
        # execute command
        subprocess.check_call(command)

    def onnx_txt2img(self, prompt, model):
        model = "CompVis/stable-diffusion-v1-4"
        device = "cpu_fp32"
        
        if device == "cpu":
            onnx_pipe = OnnxStableDiffusionPipeline.from_pretrained(model, revision="onnx", provider="CPUExecutionProvider")
        else:
            onnx_pipe = OnnxStableDiffusionPipeline.from_pretrained(model, revision="onnx", provider="OpenVINOExecutionProvider")

        t_0 = timeit.default_timer()
        image = onnx_pipe(prompt, self.height, self.height).images[0]
        t_1 = timeit.default_timer()
        elapsed_time = round((t_1 - t_0), 3)
        print(f"Elapsed time for inference: {elapsed_time}")
        image.save("ONNX_inference.png")

    def openvino_txt2img(self,prompt,model):
        batch_size = 1

        model_type = "openvino"
        model_opevino = "echarlaix/stable-diffusion-v1-5-openvino"
        model_pytorch = "runwayml/stable-diffusion-v1-5"
        if model_type == "openvino":
            stable_diffusion = OVStableDiffusionPipeline.from_pretrained(model_opevino)
        else:
            stable_diffusion = OVStableDiffusionPipeline.from_pretrained(model, export=True)
        stable_diffusion.compile()

        start = timeit.default_timer()
        #Running the session by passing in the input data of the model
        image = stable_diffusion(prompt).images[0]
        end = timeit.default_timer()
        inference_time = end - start
        image.save(f"OpenVINO_new_OUTPUT.png")
        print(f"Inference time for input: {inference_time}")

            

