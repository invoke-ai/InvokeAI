from diffusers import OnnxStableDiffusionPipeline
from optimum.intel.openvino import OVStableDiffusionPipeline

import timeit
import sys
import time
import subprocess

class Txt2Img_ONNX:

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def install_requirements(self):
        # create command to install packages
        command = ['pip', 'install', '-r'
        , "requirements-win-cpu_onnx.txt"] 
        # execute command
        subprocess.check_call(command)

    def onnx_txt2img(self, prompt):

        num_inference_steps = 1 #num_inference_steps
        self.install_requirements()
        model = "CompVis/stable-diffusion-v1-4"
        device = "cpu"
        #prompt = "a photo of an astronaut riding a horse on mars"
        #negative_prompt="bad hands, blurry"
        if device == "cpu":
            pipe = OnnxStableDiffusionPipeline.from_pretrained(model, revision="onnx", provider="CPUExecutionProvider")
        else:
            pipe = OnnxStableDiffusionPipeline.from_pretrained(model, revision="onnx", provider="OpenVINOExecutionProvider")
        #image = pipe(prompt, height, width, num_inference_steps, guidance_scale, negative_prompt).images[0] 

        for i,sentence in enumerate(prompt):
            t_0 = timeit.default_timer()
            image = pipe(sentence, self.height, self.height, num_inference_steps).images[0]
            t_1 = timeit.default_timer()
            elapsed_time = round((t_1 - t_0), 3)
            print(i,elapsed_time)
            image.save(str(i)+".png")

    def openvino_txt2img(self,prompt,model_type):
        batch_size = 1
        num_images_per_prompt = 1
        height = 512
        width = 512

        model_opevino = "echarlaix/stable-diffusion-v1-5-openvino"
        model_pytorch = "runwayml/stable-diffusion-v1-5"
        if model_type == "openvino":
            stable_diffusion = OVStableDiffusionPipeline.from_pretrained(model_opevino)
        else:
            stable_diffusion = OVStableDiffusionPipeline.from_pretrained(model_pytorch,export=True)
        stable_diffusion.compile()

        for index, prompt in enumerate(prompt):
            start = time.time()
            #Running the session by passing in the input data of the model
            image = stable_diffusion(prompt).images[0]
            #out = sess.run(None, {prompt})
            #print("output", out)
            end = time.time()
            inference_time = end - start
            image.save(f"Optimum_new_OUTPUT_{1}.png")
            print(f"inference time for{index}: {inference_time}")

