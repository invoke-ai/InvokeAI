"""
This class is derived from Inference Model class
Implements ONNX inference pipeline
"""
import traceback
import time
import sys
import importlib

from diffusers import OnnxStableDiffusionPipeline

utils = importlib.import_module('openvino.utils')
utils.add_openvino_libs_to_path()

from .baseModel import inferenceModel
from .stable_diffusion.onnx_pipeline import txt2img
from ..frontend.CLI.readline import Completer, get_completer

class ONNX(inferenceModel) :
    """
    Instantiation of Onnx model class
    """
    def __init__(
        self,
        model=None,
        sampler_name="k_lms",
        precision="auto",
        outdir="outputs/img-samples",
        num_images=1,
        steps=50,
    ):
        self.height = None
        self.width = None
        self.iterations = 1
        self.steps = 50
        self.sampler_name = sampler_name
        self.precision = precision
        self.model_type = "Onnx"
        #Set precision for ONNX inference
        self.outdir = outdir
        self.precision = "float32"
        fallback = "runwayml/stable-diffusion-v1-5"
        self.model = model or fallback
        self.model_name = model or fallback
        self.num_images_per_prompt = num_images
        self.num_inference_steps = steps

    def prompt2image(
        self,
        prompt,
        iterations=None,
        steps=None,
        image_callback=None,
        step_callback=None,
        width=None,
        height=None,
        sampler_name=None,
        model=None,
        precision=None,
        catch_interrupts=False,
        **args,
    ):
        print("Enter prompt2image of onnx")
        tic = time.time()
        try:
            #txt2img_onnx = txt2img(width, height, iterations, steps)
            #txt2img_onnx.onnx_txt2img(prompt, self.model, self.precision, self.outdir)
            if precision == "cpu":
                onnx_pipe = OnnxStableDiffusionPipeline.from_pretrained(model, revision="onnx", provider="CPUExecutionProvider")
            else:
                print(f"Model used: {model}")
                onnx_pipe = OnnxStableDiffusionPipeline.from_pretrained(model, revision="onnx", provider="OpenVINOExecutionProvider")

            image = onnx_pipe(prompt, self.height, self.height, num_images_per_prompt=self.num_images_per_prompt, num_inference_steps=self.num_inference_steps).images[0]
            timestamp = int(time.time())
            image.save(f"./{self.outdir}/Inference_{timestamp}.png")

        except KeyboardInterrupt:
            # Clear the CUDA cache on an exception
            self.clear_cuda_cache()

            if catch_interrupts:
                print("**Interrupted** Partial results will be returned.")
            else:
                raise KeyboardInterrupt
        except RuntimeError:
            # Clear the CUDA cache on an exception
            self.clear_cuda_cache()

            print(traceback.format_exc(), file=sys.stderr)
            print(">> Could not generate image.")
        toc = time.time()
        print("\n>> Usage stats:")
        print(f">> image(s) generated in", "%4.2fs" % (toc - tic))

    def getCompleter(self, opt):
        """
        Invocation of completer
        """
        return get_completer(opt, models=[])
