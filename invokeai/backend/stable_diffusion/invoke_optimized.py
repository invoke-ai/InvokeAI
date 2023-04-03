"""
Implements optimized version of Stablediffusion txt2img inference
"""

import timeit
import subprocess
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

    def execute_command(self, command):
        # execute command
        subprocess.check_call(command)

    def onnx_txt2img(self, prompt, model):
        model = "runwayml/stable-diffusion-v1-5" #"CompVis/stable-diffusion-v1-4"
        device = "cpu_fp32"
        
        if device == "cpu":
            onnx_pipe = OnnxStableDiffusionPipeline.from_pretrained(model, num_images_per_prompt=self.num_images_per_prompt, num_inference_steps=self.num_inference_steps, revision="onnx", provider="CPUExecutionProvider")
        else:
            onnx_pipe = OnnxStableDiffusionPipeline.from_pretrained(model, num_images_per_prompt=self.num_images_per_prompt, num_inference_steps=self.num_inference_steps, revision="onnx", provider="OpenVINOExecutionProvider")

        t_0 = timeit.default_timer()
        image = onnx_pipe(prompt, self.height, self.height).images[0]
        t_1 = timeit.default_timer()
        elapsed_time = round((t_1 - t_0), 3)
        print(f"Elapsed time for inference: {elapsed_time}")
        image.save("ONNX_inference.png")

    """
    Implementation of openvino based optimization
    """
    def openvino_txt2img(self,prompt,model):

        #from optimum.intel.openvino import OVStableDiffusionPipeline
        ovsd = importlib.import_module('optimum.intel.openvino')

        model_type = "Pytorch"
        model_pytorch = "stabilityai/stable-diffusion-2-1"#runwayml/stable-diffusion-v1-5 #CompVis/stable-diffusion-v1-4
        model_opevino = "echarlaix/stable-diffusion-v1-5-openvino"
        if model_type == "IR":
            stable_diffusion = ovsd.OVStableDiffusionPipeline.from_pretrained(model_opevino)
        elif model_type == "Pytorch":
            stable_diffusion = ovsd.OVStableDiffusionPipeline.from_pretrained(model_pytorch, export=True)
        else:
            command = "python convert_stable_diffusion_checkpoint_to_onnx.py --model_path=model_opevino --output_path=models\\stable_diffusion_onnx"
            self.execute_command(command)
            command = "mo --input_model " + model_opevino + "\\unet\\model.onnx --progress --input_shape [2,4,64,64],[-1],[2,77,768] --use_legacy_frontend --input sample,timestep,encoder_hidden_states"
            self.execute_command(command)
        stable_diffusion.compile()

        start = timeit.default_timer()
        #Running the session by passing in the input data of the model
        image = stable_diffusion(prompt).images[0]
        end = timeit.default_timer()
        inference_time = end - start
        image.save(f"OpenVINO_inference_{end}.png")
        print(f"Inference time for input: {inference_time}")

            

