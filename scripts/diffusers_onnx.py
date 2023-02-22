from diffusers import OnnxStableDiffusionPipeline
import timeit
import sys

class Txt2Img_ONNX:

  def __init__(self, height, width):
      self.height = height
      self.width = width

  def onnx_txt2img(self, prompt):

      num_inference_steps = 2 #num_inference_steps
      model = "CompVis/stable-diffusion-v1-4"
      device = "cpu"
      #prompt = "a photo of an astronaut riding a horse on mars"
      #negative_prompt="bad hands, blurry"
      if device == "cpu":
          pipe = OnnxStableDiffusionPipeline.from_pretrained(model, revision="onnx", provider="CPUExecutionProvider")
      else:
          pipe = OnnxStableDiffusionPipeline.from_pretrained(model, revision="onnx", provider="CUDAExecutionProvider")
      #image = pipe(prompt, height, width, num_inference_steps, guidance_scale, negative_prompt).images[0] 

      for i,sentence in enumerate(prompt):
        t_0 = timeit.default_timer()
        image = pipe(sentence, self.height, self.height, num_inference_steps).images[0]
        t_1 = timeit.default_timer()
        elapsed_time = round((t_1 - t_0), 3)
        print(i,elapsed_time)
        image.save(str(i)+".png")

