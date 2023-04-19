"""
This class is derived from Inference Model class
Implements ONNX inference related operations
"""
import time

from baseModel import inferenceModel
from stable_diffusion.onnx_pipeline import txt2img

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
        opt = None,
        args = None,
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

    
    def prompt2image(
        self,
        prompt,
        iterations=None,
        steps=None,
        width=None,
        height=None,
        model=None,
        precision=None
    ):
        tic = time.time()
        txt2img_onnx = txt2img(width, height, iterations, steps)
        txt2img_onnx.onnx_txt2img(prompt, model, precision)
        toc = time.time()
        print("\n>> Usage stats:")
        print(f">> image(s) generated in", "%4.2fs" % (toc - tic))