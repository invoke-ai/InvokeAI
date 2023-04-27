"""
This class is derived from Inference Model class
Implements ONNX inference pipeline
"""
import traceback
import time
import sys
import os
import importlib

from diffusers import OnnxStableDiffusionPipeline

utils = importlib.import_module('openvino.utils')
utils.add_openvino_libs_to_path()

from .inferencePipeline import inferencePipeline
from ..frontend.CLI.readline import Completer, get_completer

class ONNX(inferencePipeline) :
    """
    Instantiation of Onnx model class
    """
    def __init__(
        self,
        model=None,
        sampler_name="k_lms",
        precision="auto",
        outdir="outputs/",
        num_images=1,
        steps=50,
    ):
        self.height = 512
        self.width = 512
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
        outdir=None,
        width=None,
        height=None,
        sampler_name=None,
        model=None,
        precision=None,
        catch_interrupts=False,
        **args,
    ):
        steps = steps or self.steps
        width = width or self.width
        height = height or self.height
        iterations = iterations or self.iterations
        outdir = outdir or self.outdir
        sampler_name = self.sampler_name or sampler_name
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print("Output directory: ", outdir)
        tic = time.time()
        try:
            if precision == "cpu":
                onnxPipeline = OnnxStableDiffusionPipeline.from_pretrained(self.model, revision="onnx", provider="CPUExecutionProvider")
            else:
                onnxPipeline = OnnxStableDiffusionPipeline.from_pretrained(self.model, revision="onnx", provider="OpenVINOExecutionProvider")

            image = onnxPipeline(prompt, height=height, width=width, num_images_per_prompt=iterations, num_inference_steps=steps).images[0]
            timestamp = int(time.time())
            image.save(f"{outdir}/Inference_{timestamp}.png")

        except KeyboardInterrupt:

            if catch_interrupts:
                print("**Interrupted** Partial results will be returned.")
            else:
                raise KeyboardInterrupt
        except RuntimeError:

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
