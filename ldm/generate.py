# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)
# Derived from source code carrying the following copyrights
# Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich
# Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors

import gc
import importlib
import os
import random
import re
import sys
import time
import traceback

import cv2
import diffusers
import numpy as np
import skimage
import torch
import transformers
from PIL import Image, ImageOps
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, logging

import ldm.invoke.conditioning
from ldm.invoke.args import metadata_from_png
from ldm.invoke.concepts_lib import HuggingFaceConceptsLibrary
from ldm.invoke.conditioning import get_uc_and_c_and_ec
from ldm.invoke.devices import choose_torch_device, choose_precision
from ldm.invoke.generator.inpaint import infill_methods
from ldm.invoke.globals import global_cache_dir, Globals
from ldm.invoke.image_util import InitImageResizer
from ldm.invoke.model_manager import ModelManager
from ldm.invoke.pngwriter import PngWriter
from ldm.invoke.seamless import configure_model_padding
from ldm.invoke.txt2mask import Txt2Mask
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ksampler import KSampler
from ldm.models.diffusion.plms import PLMSSampler


def fix_func(orig):
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        def new_func(*args, **kw):
            device = kw.get("device", "mps")
            kw["device"]="cpu"
            return orig(*args, **kw).to(device)
        return new_func
    return orig

torch.rand = fix_func(torch.rand)
torch.rand_like = fix_func(torch.rand_like)
torch.randn = fix_func(torch.randn)
torch.randn_like = fix_func(torch.randn_like)
torch.randint = fix_func(torch.randint)
torch.randint_like = fix_func(torch.randint_like)
torch.bernoulli = fix_func(torch.bernoulli)
torch.multinomial = fix_func(torch.multinomial)

# this is fallback model in case no default is defined
FALLBACK_MODEL_NAME='stable-diffusion-1.5'

"""Simplified text to image API for stable diffusion/latent diffusion

Example Usage:

from ldm.generate import Generate

# Create an object with default values
gr = Generate('stable-diffusion-1.4')

# do the slow model initialization
gr.load_model()

# Do the fast inference & image generation. Any options passed here
# override the default values assigned during class initialization
# Will call load_model() if the model was not previously loaded and so
# may be slow at first.
# The method returns a list of images. Each row of the list is a sub-list of [filename,seed]
results = gr.prompt2png(prompt     = "an astronaut riding a horse",
                         outdir     = "./outputs/samples",
                         iterations = 3)

for row in results:
    print(f'filename={row[0]}')
    print(f'seed    ={row[1]}')

# Same thing, but using an initial image.
results = gr.prompt2png(prompt   = "an astronaut riding a horse",
                         outdir   = "./outputs/,
                         iterations = 3,
                         init_img = "./sketches/horse+rider.png")

for row in results:
    print(f'filename={row[0]}')
    print(f'seed    ={row[1]}')

# Same thing, but we return a series of Image objects, which lets you manipulate them,
# combine them, and save them under arbitrary names

results = gr.prompt2image(prompt   = "an astronaut riding a horse"
                           outdir   = "./outputs/")
for row in results:
    im   = row[0]
    seed = row[1]
    im.save(f'./outputs/samples/an_astronaut_riding_a_horse-{seed}.png')
    im.thumbnail(100,100).save('./outputs/samples/astronaut_thumb.jpg')

Note that the old txt2img() and img2img() calls are deprecated but will
still work.

The full list of arguments to Generate() are:
gr = Generate(
          # these values are set once and shouldn't be changed
          conf:str          = path to configuration file ('configs/models.yaml')
          model:str         = symbolic name of the model in the configuration file
          precision:float   = float precision to be used
          safety_checker:bool = activate safety checker [False]

          # this value is sticky and maintained between generation calls
          sampler_name:str  = ['ddim', 'k_dpm_2_a', 'k_dpm_2', 'k_dpmpp_2', 'k_dpmpp_2_a', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms', 'plms']  // k_lms

          # these are deprecated - use conf and model instead
          weights     = path to model weights ('models/ldm/stable-diffusion-v1/model.ckpt')
          config      = path to model configuration ('configs/stable-diffusion/v1-inference.yaml')
          )

"""

class Generate:
    """Generate class
    Stores default values for multiple configuration items
    """

    def __init__(
            self,
            model = None,
            conf = 'configs/models.yaml',
            embedding_path = None,
            sampler_name = 'k_lms',
            ddim_eta = 0.0,  # deterministic
            full_precision = False,
            precision = 'auto',
            outdir = 'outputs/img-samples',
            gfpgan=None,
            codeformer=None,
            esrgan=None,
            free_gpu_mem: bool=False,
            safety_checker:bool=False,
            max_loaded_models:int=2,
            # these are deprecated; if present they override values in the conf file
            weights = None,
            config = None,
    ):
        mconfig             = OmegaConf.load(conf)
        self.height         = None
        self.width          = None
        self.model_manager    = None
        self.iterations     = 1
        self.steps          = 50
        self.cfg_scale      = 7.5
        self.sampler_name   = sampler_name
        self.ddim_eta       = ddim_eta    # same seed always produces same image
        self.precision      = precision
        self.strength       = 0.75
        self.seamless       = False
        self.seamless_axes  = {'x','y'}
        self.hires_fix      = False
        self.embedding_path = embedding_path
        self.model          = None     # empty for now
        self.model_hash     = None
        self.sampler        = None
        self.device         = None
        self.session_peakmem = None
        self.base_generator = None
        self.seed           = None
        self.outdir = outdir
        self.gfpgan = gfpgan
        self.codeformer = codeformer
        self.esrgan = esrgan
        self.free_gpu_mem = free_gpu_mem
        self.max_loaded_models = max_loaded_models,
        self.size_matters = True  # used to warn once about large image sizes and VRAM
        self.txt2mask = None
        self.safety_checker = None
        self.karras_max = None
        self.infill_method = None

        # Note that in previous versions, there was an option to pass the
        # device to Generate(). However the device was then ignored, so
        # it wasn't actually doing anything. This logic could be reinstated.
        device_type = choose_torch_device()
        print(f'>> Using device_type {device_type}')
        self.device = torch.device(device_type)
        if full_precision:
            if self.precision != 'auto':
              raise ValueError('Remove --full_precision / -F if using --precision')
            print('Please remove deprecated --full_precision / -F')
            print('If auto config does not work you can use --precision=float32')
            self.precision = 'float32'
        if self.precision == 'auto':
            self.precision = choose_precision(self.device)
        Globals.full_precision = self.precision=='float32'

        if is_xformers_available():
            if not Globals.disable_xformers:
                print('>> xformers memory-efficient attention is available and enabled')
            else:
                print('>> xformers memory-efficient attention is available but disabled')
        else:
            print('>> xformers not installed')

        # model caching system for fast switching
        self.model_manager = ModelManager(mconfig,self.device,self.precision,max_loaded_models=max_loaded_models)
        # don't accept invalid models
        fallback = self.model_manager.default_model() or FALLBACK_MODEL_NAME
        model = model or fallback
        if not self.model_manager.valid_model(model):
            print(f'** "{model}" is not a known model name; falling back to {fallback}.')
            model = None
        self.model_name  = model or fallback

        # for VRAM usage statistics
        self.session_peakmem = torch.cuda.max_memory_allocated() if self._has_cuda else None
        transformers.logging.set_verbosity_error()

        # gets rid of annoying messages about random seed
        logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

        # load safety checker if requested
        if safety_checker:
            try:
                print('>> Initializing safety checker')
                from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
                from transformers import AutoFeatureExtractor
                safety_model_id = "CompVis/stable-diffusion-safety-checker"
                safety_model_path = global_cache_dir("hub")
                self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id,
                                                                                   local_files_only=True,
                                                                                   cache_dir=safety_model_path,
                )
                self.safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id,
                                                                                     local_files_only=True,
                                                                                     cache_dir=safety_model_path,
                )
                self.safety_checker.to(self.device)
            except Exception:
                print('** An error was encountered while installing the safety checker:')
                print(traceback.format_exc())

    def prompt2png(self, prompt, outdir, **kwargs):
        """
        Takes a prompt and an output directory, writes out the requested number
        of PNG files, and returns an array of [[filename,seed],[filename,seed]...]
        Optional named arguments are the same as those passed to Generate and prompt2image()
        """
        results   = self.prompt2image(prompt, **kwargs)
        pngwriter = PngWriter(outdir)
        prefix    = pngwriter.unique_prefix()
        outputs   = []
        for image, seed in results:
            name = f'{prefix}.{seed}.png'
            path = pngwriter.save_image_and_prompt_to_png(
                image, dream_prompt=f'{prompt} -S{seed}', name=name)
            outputs.append([path, seed])
        return outputs

    def txt2img(self, prompt, **kwargs):
        outdir = kwargs.pop('outdir', self.outdir)
        return self.prompt2png(prompt, outdir, **kwargs)

    def img2img(self, prompt, **kwargs):
        outdir = kwargs.pop('outdir', self.outdir)
        assert (
            'init_img' in kwargs
        ), 'call to img2img() must include the init_img argument'
        return self.prompt2png(prompt, outdir, **kwargs)

    def prompt2image(
            self,
            # these are common
            prompt,
            iterations       = None,
            steps            = None,
            seed             = None,
            cfg_scale        = None,
            ddim_eta         = None,
            skip_normalize   = False,
            image_callback   = None,
            step_callback    = None,
            width            = None,
            height           = None,
            sampler_name     = None,
            seamless         = False,
            seamless_axes    = {'x','y'},
            log_tokenization = False,
            with_variations  = None,
            variation_amount = 0.0,
            threshold        = 0.0,
            perlin           = 0.0,
            karras_max       = None,
            outdir         = None,
            # these are specific to img2img and inpaint
            init_img         = None,
            init_mask        = None,
            text_mask        = None,
            invert_mask      = False,
            fit              = False,
            strength         = None,
            init_color       = None,
            # these are specific to embiggen (which also relies on img2img args)
            embiggen          = None,
            embiggen_tiles    = None,
            embiggen_strength = None,
            # these are specific to GFPGAN/ESRGAN
            gfpgan_strength=    0,
            facetool         = None,
            facetool_strength  = 0,
            codeformer_fidelity = None,
            save_original    = False,
            upscale          = None,
            # this is specific to inpainting and causes more extreme inpainting
            inpaint_replace  = 0.0,
            # This controls the size at which inpaint occurs (scaled up for inpaint, then back down for the result)
            inpaint_width    = None,
            inpaint_height   = None,
            # This will help match inpainted areas to the original image more smoothly
            mask_blur_radius: int = 8,
            # Set this True to handle KeyboardInterrupt internally
            catch_interrupts = False,
            hires_fix        = False,
            use_mps_noise    = False,
            # Seam settings for outpainting
            seam_size: int   = 0,
            seam_blur: int   = 0,
            seam_strength: float = 0.7,
            seam_steps: int  = 10,
            tile_size: int   = 32,
            infill_method = None,
            force_outpaint: bool = False,
            enable_image_debugging = False,

            **args,
    ):   # eat up additional cruft
        self.clear_cuda_stats()
        """
        ldm.generate.prompt2image() is the common entry point for txt2img() and img2img()
        It takes the following arguments:
           prompt                          // prompt string (no default)
           iterations                      // iterations (1); image count=iterations
           steps                           // refinement steps per iteration
           seed                            // seed for random number generator
           width                           // width of image, in multiples of 64 (512)
           height                          // height of image, in multiples of 64 (512)
           cfg_scale                       // how strongly the prompt influences the image (7.5) (must be >1)
           seamless                        // whether the generated image should tile
           hires_fix                        // whether the Hires Fix should be applied during generation
           init_img                        // path to an initial image
           init_mask                       // path to a mask for the initial image
           text_mask                       // a text string that will be used to guide clipseg generation of the init_mask
           invert_mask                     // boolean, if true invert the mask
           strength                        // strength for noising/unnoising init_img. 0.0 preserves image exactly, 1.0 replaces it completely
           facetool_strength               // strength for GFPGAN/CodeFormer. 0.0 preserves image exactly, 1.0 replaces it completely
           ddim_eta                        // image randomness (eta=0.0 means the same seed always produces the same image)
           step_callback                   // a function or method that will be called each step
           image_callback                  // a function or method that will be called each time an image is generated
           with_variations                 // a weighted list [(seed_1, weight_1), (seed_2, weight_2), ...] of variations which should be applied before doing any generation
           variation_amount                // optional 0-1 value to slerp from -S noise to random noise (allows variations on an image)
           threshold                       // optional value >=0 to add thresholding to latent values for k-diffusion samplers (0 disables)
           perlin                          // optional 0-1 value to add a percentage of perlin noise to the initial noise
           embiggen                        // scale factor relative to the size of the --init_img (-I), followed by ESRGAN upscaling strength (0-1.0), followed by minimum amount of overlap between tiles as a decimal ratio (0 - 1.0) or number of pixels
           embiggen_tiles                  // list of tiles by number in order to process and replace onto the image e.g. `0 2 4`
           embiggen_strength               // strength for embiggen. 0.0 preserves image exactly, 1.0 replaces it completely

        To use the step callback, define a function that receives two arguments:
        - Image GPU data
        - The step number

        To use the image callback, define a function of method that receives two arguments, an Image object
        and the seed. You can then do whatever you like with the image, including converting it to
        different formats and manipulating it. For example:

            def process_image(image,seed):
                image.save(f{'images/seed.png'})

        The code used to save images to a directory can be found in ldm/invoke/pngwriter.py.
        It contains code to create the requested output directory, select a unique informative
        name for each image, and write the prompt into the PNG metadata.
        """
        # TODO: convert this into a getattr() loop
        steps = steps or self.steps
        width = width or self.width
        height = height or self.height
        seamless = seamless or self.seamless
        seamless_axes = seamless_axes or self.seamless_axes
        hires_fix = hires_fix or self.hires_fix
        cfg_scale = cfg_scale or self.cfg_scale
        ddim_eta = ddim_eta or self.ddim_eta
        iterations = iterations or self.iterations
        strength = strength or self.strength
        outdir                = outdir     or self.outdir
        self.seed = seed
        self.log_tokenization = log_tokenization
        self.step_callback = step_callback
        self.karras_max = karras_max
        self.infill_method = infill_method or infill_methods()[0], # The infill method to use
        with_variations = [] if with_variations is None else with_variations

        # will instantiate the model or return it from cache
        model = self.set_model(self.model_name)

        # self.width and self.height are set by set_model()
        # to the width and height of the image training set
        width = width or self.width
        height = height or self.height

        if isinstance(model, DiffusionPipeline):
            configure_model_padding(model.unet, seamless, seamless_axes)
            configure_model_padding(model.vae, seamless, seamless_axes)
        else:
            configure_model_padding(model, seamless, seamless_axes)

        assert cfg_scale > 1.0, 'CFG_Scale (-C) must be >1.0'
        assert threshold >= 0.0, '--threshold must be >=0.0'
        assert (
            0.0 < strength <= 1.0
        ), 'img2img and inpaint strength can only work with 0.0 < strength < 1.0'
        assert (
            0.0 <= variation_amount <= 1.0
        ), '-v --variation_amount must be in [0.0, 1.0]'
        assert (
                0.0 <= perlin <= 1.0
        ), '--perlin must be in [0.0, 1.0]'
        assert (
            (embiggen == None and embiggen_tiles == None) or (
                (embiggen != None or embiggen_tiles != None) and init_img != None)
        ), 'Embiggen requires an init/input image to be specified'

        if len(with_variations) > 0 or variation_amount > 1.0:
            assert seed is not None,\
                'seed must be specified when using with_variations'
            if variation_amount == 0.0:
                assert iterations == 1,\
                    'when using --with_variations, multiple iterations are only possible when using --variation_amount'
            assert all(0 <= weight <= 1 for _, weight in with_variations),\
                f'variation weights must be in [0.0, 1.0]: got {[weight for _, weight in with_variations]}'

        width, height, _ = self._resolution_check(width, height, log=True)
        assert inpaint_replace >=0.0 and inpaint_replace <= 1.0,'inpaint_replace must be between 0.0 and 1.0'

        if sampler_name and (sampler_name != self.sampler_name):
            self.sampler_name = sampler_name
            self._set_sampler()

        # apply the concepts library to the prompt
        prompt = self.huggingface_concepts_library.replace_concepts_with_triggers(
            prompt,
            lambda concepts: self.load_huggingface_concepts(concepts),
            self.model.textual_inversion_manager.get_all_trigger_strings()
        )

        # bit of a hack to change the cached sampler's karras threshold to
        # whatever the user asked for
        if karras_max is not None and isinstance(self.sampler,KSampler):
            self.sampler.adjust_settings(karras_max=karras_max)

        tic = time.time()
        if self._has_cuda():
            torch.cuda.reset_peak_memory_stats()

        results = list()
        init_image = None
        mask_image = None

        try:
            if self.free_gpu_mem and self.model.cond_stage_model.device != self.model.device:
                self.model.cond_stage_model.device = self.model.device
                self.model.cond_stage_model.to(self.model.device)
        except AttributeError:
            print(">> Warning: '--free_gpu_mem' is not yet supported when generating image using model based on HuggingFace Diffuser.")
            pass

        try:
            uc, c, extra_conditioning_info = get_uc_and_c_and_ec(
                prompt, model =self.model,
                skip_normalize_legacy_blend=skip_normalize,
                log_tokens    =self.log_tokenization
            )

            init_image, mask_image = self._make_images(
                init_img,
                init_mask,
                width,
                height,
                fit=fit,
                text_mask=text_mask,
                invert_mask=invert_mask,
                force_outpaint=force_outpaint,
            )

            # TODO: Hacky selection of operation to perform. Needs to be refactored.
            generator = self.select_generator(init_image, mask_image, embiggen, hires_fix, force_outpaint)

            generator.set_variation(
                self.seed, variation_amount, with_variations
            )
            generator.use_mps_noise = use_mps_noise

            checker = {
                'checker':self.safety_checker,
                'extractor':self.safety_feature_extractor
            } if self.safety_checker else None

            results = generator.generate(
                prompt,
                iterations=iterations,
                seed=self.seed,
                sampler=self.sampler,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=(uc, c, extra_conditioning_info),
                ddim_eta=ddim_eta,
                image_callback=image_callback,  # called after the final image is generated
                step_callback=step_callback,    # called after each intermediate image is generated
                width=width,
                height=height,
                init_img=init_img,          # embiggen needs to manipulate from the unmodified init_img
                init_image=init_image,      # notice that init_image is different from init_img
                mask_image=mask_image,
                strength=strength,
                threshold=threshold,
                perlin=perlin,
                embiggen=embiggen,
                embiggen_tiles=embiggen_tiles,
                embiggen_strength=embiggen_strength,
                inpaint_replace=inpaint_replace,
                mask_blur_radius=mask_blur_radius,
                safety_checker=checker,
                seam_size = seam_size,
                seam_blur = seam_blur,
                seam_strength = seam_strength,
                seam_steps = seam_steps,
                tile_size = tile_size,
                infill_method = infill_method,
                force_outpaint = force_outpaint,
                inpaint_height = inpaint_height,
                inpaint_width = inpaint_width,
                enable_image_debugging = enable_image_debugging,
                free_gpu_mem=self.free_gpu_mem,
                clear_cuda_cache=self.clear_cuda_cache
            )

            if init_color:
                self.correct_colors(image_list           = results,
                                    reference_image_path = init_color,
                                    image_callback       = image_callback)

            if upscale is not None or facetool_strength > 0:
                self.upscale_and_reconstruct(results,
                                             upscale        = upscale,
                                             facetool       = facetool,
                                             strength       = facetool_strength,
                                             codeformer_fidelity = codeformer_fidelity,
                                             save_original  = save_original,
                                             image_callback = image_callback)

        except KeyboardInterrupt:
            # Clear the CUDA cache on an exception
            self.clear_cuda_cache()

            if catch_interrupts:
                print('**Interrupted** Partial results will be returned.')
            else:
                raise KeyboardInterrupt
        except RuntimeError:
            # Clear the CUDA cache on an exception
            self.clear_cuda_cache()

            print(traceback.format_exc(), file=sys.stderr)
            print('>> Could not generate image.')

        toc = time.time()
        print('\n>> Usage stats:')
        print(
            f'>>   {len(results)} image(s) generated in', '%4.2fs' % (
                toc - tic)
        )
        self.print_cuda_stats()
        return results

    def clear_cuda_cache(self):
        if self._has_cuda():
            self.max_memory_allocated = max(
                self.max_memory_allocated,
                torch.cuda.max_memory_allocated()
            )
            self.memory_allocated = max(
                self.memory_allocated,
                torch.cuda.memory_allocated()
            )
            self.session_peakmem = max(
                self.session_peakmem,
                torch.cuda.max_memory_allocated()
            )
            torch.cuda.empty_cache()

    def clear_cuda_stats(self):
        self.max_memory_allocated = 0
        self.memory_allocated = 0

    def print_cuda_stats(self):
        if self._has_cuda():
            print(
                '>>   Max VRAM used for this generation:',
                '%4.2fG.' % (self.max_memory_allocated / 1e9),
                'Current VRAM utilization:',
                '%4.2fG' % (self.memory_allocated / 1e9),
            )

            print(
                '>>   Max VRAM used since script start: ',
                '%4.2fG' % (self.session_peakmem / 1e9),
            )

    # this needs to be generalized to all sorts of postprocessors, which should be wrapped
    # in a nice harmonized call signature. For now we have a bunch of if/elses!
    def apply_postprocessor(
            self,
            image_path,
            tool                = 'gfpgan',  # one of 'upscale', 'gfpgan', 'codeformer', 'outpaint', or 'embiggen'
            facetool_strength   = 0.0,
            codeformer_fidelity = 0.75,
            upscale             = None,
            out_direction       = None,
            outcrop             = [],
            save_original       = True, # to get new name
            callback            = None,
            opt                 = None,
            ):
        # retrieve the seed from the image;
        seed   = None
        prompt = None

        args = metadata_from_png(image_path)
        seed = opt.seed or args.seed
        if seed is None or seed < 0:
            seed = random.randrange(0, np.iinfo(np.uint32).max)

        prompt = opt.prompt or args.prompt or ''
        print(f'>> using seed {seed} and prompt "{prompt}" for {image_path}')

        # try to reuse the same filename prefix as the original file.
        # we take everything up to the first period
        prefix = None
        m = re.match(r'^([^.]+)\.',os.path.basename(image_path))
        if m:
            prefix = m.groups()[0]

        # face fixers and esrgan take an Image, but embiggen takes a path
        image = Image.open(image_path)

        # used by multiple postfixers
        # todo: cross-attention control
        uc, c, extra_conditioning_info = get_uc_and_c_and_ec(
            prompt, model=self.model,
            skip_normalize_legacy_blend=opt.skip_normalize,
            log_tokens=ldm.invoke.conditioning.log_tokenization
        )

        if tool in ('gfpgan','codeformer','upscale'):
            if tool == 'gfpgan':
                facetool = 'gfpgan'
            elif tool == 'codeformer':
                facetool = 'codeformer'
            elif tool == 'upscale':
                facetool = 'gfpgan'   # but won't be run
                facetool_strength = 0
            return self.upscale_and_reconstruct(
                [[image,seed]],
                facetool = facetool,
                strength = facetool_strength,
                codeformer_fidelity = codeformer_fidelity,
                save_original = save_original,
                upscale = upscale,
                image_callback = callback,
                prefix = prefix,
            )

        elif tool == 'outcrop':
            from ldm.invoke.restoration.outcrop import Outcrop
            extend_instructions = {}
            for direction,pixels in _pairwise(opt.outcrop):
                try:
                    extend_instructions[direction]=int(pixels)
                except ValueError:
                    print('** invalid extension instruction. Use <directions> <pixels>..., as in "top 64 left 128 right 64 bottom 64"')

            opt.seed = seed
            opt.prompt = prompt

            if len(extend_instructions) > 0:
                restorer = Outcrop(image,self,)
                return restorer.process (
                    extend_instructions,
                    opt            = opt,
                    orig_opt       = args,
                    image_callback = callback,
                    prefix = prefix,
                )

        elif tool == 'embiggen':
            # fetch the metadata from the image
            generator = self.select_generator(embiggen=True)
            opt.strength = opt.embiggen_strength or 0.40
            print(f'>> Setting img2img strength to {opt.strength} for happy embiggening')
            generator.generate(
                prompt,
                sampler     = self.sampler,
                steps       = opt.steps,
                cfg_scale   = opt.cfg_scale,
                ddim_eta    = self.ddim_eta,
                conditioning= (uc, c, extra_conditioning_info),
                init_img    = image_path,  # not the Image! (sigh)
                init_image  = image,       # embiggen wants both! (sigh)
                strength    = opt.strength,
                width       = opt.width,
                height      = opt.height,
                embiggen    = opt.embiggen,
                embiggen_tiles = opt.embiggen_tiles,
                embiggen_strength = opt.embiggen_strength,
                image_callback = callback,
            )
        elif tool == 'outpaint':
            from ldm.invoke.restoration.outpaint import Outpaint
            restorer = Outpaint(image,self)
            return restorer.process(
                opt,
                args,
                image_callback = callback,
                prefix         = prefix
            )

        elif tool is None:
            print('* please provide at least one postprocessing option, such as -G or -U')
            return None
        else:
            print(f'* postprocessing tool {tool} is not yet supported')
            return None

    def select_generator(
            self,
            init_image:Image.Image=None,
            mask_image:Image.Image=None,
            embiggen:bool=False,
            hires_fix:bool=False,
            force_outpaint:bool=False,
    ):
        inpainting_model_in_use = self.sampler.uses_inpainting_model()

        if hires_fix:
            return self._make_txt2img2img()

        if embiggen is not None:
            return self._make_embiggen()

        if inpainting_model_in_use:
            return self._make_omnibus()

        if ((init_image is not None) and (mask_image is not None)) or force_outpaint:
            return self._make_inpaint()

        if init_image is not None:
            return self._make_img2img()

        return self._make_txt2img()

    def _make_images(
            self,
            img,
            mask,
            width,
            height,
            fit=False,
            text_mask=None,
            invert_mask=False,
            force_outpaint=False,
    ):
        init_image      = None
        init_mask       = None
        if not img:
            return None, None

        image = self._load_img(img)

        if image.width < self.width and image.height < self.height:
            print(f'>> WARNING: img2img and inpainting may produce unexpected results with initial images smaller than {self.width}x{self.height} in both dimensions')

        # if image has a transparent area and no mask was provided, then try to generate mask
        if self._has_transparency(image):
            self._transparency_check_and_warning(image, mask, force_outpaint)
            init_mask = self._create_init_mask(image, width, height, fit=fit)

        if (image.width * image.height) > (self.width * self.height) and self.size_matters:
            print(">> This input is larger than your defaults. If you run out of memory, please use a smaller image.")
            self.size_matters = False

        init_image   = self._create_init_image(image,width,height,fit=fit)

        if mask:
            mask_image = self._load_img(mask)
            init_mask = self._create_init_mask(mask_image,width,height,fit=fit)

        elif text_mask:
            init_mask = self._txt2mask(image, text_mask, width, height, fit=fit)

        if init_mask and invert_mask:
            init_mask = ImageOps.invert(init_mask)

        return init_image,init_mask

    def _make_base(self):
        return self._load_generator('','Generator')

    def _make_txt2img(self):
        return self._load_generator('.txt2img','Txt2Img')

    def _make_img2img(self):
        return self._load_generator('.img2img','Img2Img')

    def _make_embiggen(self):
        return self._load_generator('.embiggen','Embiggen')

    def _make_txt2img2img(self):
        return self._load_generator('.txt2img2img','Txt2Img2Img')

    def _make_inpaint(self):
        return self._load_generator('.inpaint','Inpaint')

    def _make_omnibus(self):
        return self._load_generator('.omnibus','Omnibus')

    def _load_generator(self, module, class_name):
        if self.is_legacy_model(self.model_name):
            mn = f'ldm.invoke.ckpt_generator{module}'
            cn = f'Ckpt{class_name}'
        else:
            mn = f'ldm.invoke.generator{module}'
            cn = class_name
        module = importlib.import_module(mn)
        constructor = getattr(module,cn)
        return constructor(self.model, self.precision)

    def load_model(self):
        '''
        preload model identified in self.model_name
        '''
        return self.set_model(self.model_name)

    def set_model(self,model_name):
        """
        Given the name of a model defined in models.yaml, will load and initialize it
        and return the model object. Previously-used models will be cached.

        If the passed model_name is invalid, raises a KeyError.
        If the model fails to load for some reason, will attempt to load the previously-
        loaded model (if any). If that fallback fails, will raise an AssertionError
        """
        if self.model_name == model_name and self.model is not None:
            return self.model

        previous_model_name = self.model_name

        # the model cache does the loading and offloading
        cache = self.model_manager
        if not cache.valid_model(model_name):
            raise KeyError(f'** "{model_name}" is not a known model name. Cannot change.')

        cache.print_vram_usage()

        # have to get rid of all references to model in order
        # to free it from GPU memory
        self.model = None
        self.sampler = None
        self.generators = {}
        gc.collect()
        try:
            model_data = cache.get_model(model_name)
        except Exception as e:
            print(f'** model {model_name} could not be loaded: {str(e)}')
            print(traceback.format_exc(), file=sys.stderr)
            if previous_model_name is None:
                raise e
            print(f'** trying to reload previous model')
            model_data = cache.get_model(previous_model_name) # load previous
            if model_data is None:
                raise e
            model_name = previous_model_name

        self.model = model_data['model']
        self.width = model_data['width']
        self.height= model_data['height']
        self.model_hash = model_data['hash']

        # uncache generators so they pick up new models
        self.generators = {}

        seed_everything(random.randrange(0, np.iinfo(np.uint32).max))
        if self.embedding_path is not None:
            for root, _, files in os.walk(self.embedding_path):
                for name in files:
                    ti_path = os.path.join(root, name)
                    self.model.textual_inversion_manager.load_textual_inversion(ti_path,
                                                                                defer_injecting_tokens=True)
            print(f'>> Textual inversions available: {", ".join(self.model.textual_inversion_manager.get_all_trigger_strings())}')

        self.model_name = model_name
        self._set_sampler()  # requires self.model_name to be set first
        return self.model

    def load_huggingface_concepts(self, concepts:list[str]):
        self.model.textual_inversion_manager.load_huggingface_concepts(concepts)

    @property
    def huggingface_concepts_library(self) -> HuggingFaceConceptsLibrary:
        return self.model.textual_inversion_manager.hf_concepts_library

    def correct_colors(self,
                       image_list,
                       reference_image_path,
                       image_callback = None):
        reference_image = Image.open(reference_image_path)
        correction_target = cv2.cvtColor(np.asarray(reference_image),
                                         cv2.COLOR_RGB2LAB)
        for r in image_list:
            image, seed = r
            image = cv2.cvtColor(np.asarray(image),
                                 cv2.COLOR_RGB2LAB)
            image = skimage.exposure.match_histograms(image,
                                                      correction_target,
                                                      channel_axis=2)
            image = Image.fromarray(
                cv2.cvtColor(image, cv2.COLOR_LAB2RGB).astype("uint8")
            )
            if image_callback is not None:
                image_callback(image, seed)
            else:
                r[0] = image

    def upscale_and_reconstruct(self,
                                image_list,
                                facetool      = 'gfpgan',
                                upscale       = None,
                                strength      =  0.0,
                                codeformer_fidelity = 0.75,
                                save_original = False,
                                image_callback = None,
                                prefix = None,
    ):

        for r in image_list:
            image, seed = r
            try:
                if strength > 0:
                    if self.gfpgan is not None or self.codeformer is not None:
                        if facetool == 'gfpgan':
                            if self.gfpgan is None:
                                print('>> GFPGAN not found. Face restoration is disabled.')
                            else:
                              image = self.gfpgan.process(image, strength, seed)
                        if facetool == 'codeformer':
                            if self.codeformer is None:
                                print('>> CodeFormer not found. Face restoration is disabled.')
                            else:
                                cf_device = 'cpu' if str(self.device) == 'mps' else self.device
                                image = self.codeformer.process(image=image, strength=strength, device=cf_device, seed=seed, fidelity=codeformer_fidelity)
                    else:
                        print(">> Face Restoration is disabled.")
                if upscale is not None:
                    if self.esrgan is not None:
                        if len(upscale) < 2:
                            upscale.append(0.75)
                        image = self.esrgan.process(
                            image, upscale[1], seed, int(upscale[0]))
                    else:
                        print(">> ESRGAN is disabled. Image not upscaled.")
            except Exception as e:
                print(
                    f'>> Error running RealESRGAN or GFPGAN. Your image was not upscaled.\n{e}'
                )

            if image_callback is not None:
                image_callback(image, seed, upscaled=True, use_prefix=prefix)
            else:
                r[0] = image

    def apply_textmask(self, image_path:str, prompt:str, callback, threshold:float=0.5):
        assert os.path.exists(image_path), f'** "{image_path}" not found. Please enter the name of an existing image file to mask **'
        basename,_ = os.path.splitext(os.path.basename(image_path))
        if self.txt2mask is None:
            self.txt2mask  = Txt2Mask(device = self.device, refined=True)
        segmented  = self.txt2mask.segment(image_path,prompt)
        trans = segmented.to_transparent()
        inverse = segmented.to_transparent(invert=True)
        mask = segmented.to_mask(threshold)

        path_filter = re.compile(r'[<>:"/\\|?*]')
        safe_prompt = path_filter.sub('_', prompt)[:50].rstrip(' .')

        callback(trans,f'{safe_prompt}.deselected',use_prefix=basename)
        callback(inverse,f'{safe_prompt}.selected',use_prefix=basename)
        callback(mask,f'{safe_prompt}.masked',use_prefix=basename)

    # to help WebGUI - front end to generator util function
    def sample_to_image(self, samples):
        return self._make_base().sample_to_image(samples)

    def sample_to_lowres_estimated_image(self, samples):
        return self._make_base().sample_to_lowres_estimated_image(samples)

    def is_legacy_model(self,model_name)->bool:
        return self.model_manager.is_legacy(model_name)

    def _set_sampler(self):
        if isinstance(self.model, DiffusionPipeline):
            return self._set_scheduler()
        else:
            return self._set_sampler_legacy()

    # very repetitive code - can this be simplified? The KSampler names are
    # consistent, at least
    def _set_sampler_legacy(self):
        msg = f'>> Setting Sampler to {self.sampler_name}'
        if self.sampler_name == 'plms':
            self.sampler = PLMSSampler(self.model, device=self.device)
        elif self.sampler_name == 'ddim':
            self.sampler = DDIMSampler(self.model, device=self.device)
        elif self.sampler_name == 'k_dpm_2_a':
            self.sampler = KSampler(self.model, 'dpm_2_ancestral', device=self.device)
        elif self.sampler_name == 'k_dpm_2':
            self.sampler = KSampler(self.model, 'dpm_2', device=self.device)
        elif self.sampler_name == 'k_dpmpp_2_a':
            self.sampler = KSampler(self.model, 'dpmpp_2s_ancestral', device=self.device)
        elif self.sampler_name == 'k_dpmpp_2':
            self.sampler = KSampler(self.model, 'dpmpp_2m', device=self.device)
        elif self.sampler_name == 'k_euler_a':
            self.sampler = KSampler(self.model, 'euler_ancestral', device=self.device)
        elif self.sampler_name == 'k_euler':
            self.sampler = KSampler(self.model, 'euler', device=self.device)
        elif self.sampler_name == 'k_heun':
            self.sampler = KSampler(self.model, 'heun', device=self.device)
        elif self.sampler_name == 'k_lms':
            self.sampler = KSampler(self.model, 'lms', device=self.device)
        else:
            msg = f'>> Unsupported Sampler: {self.sampler_name}, Defaulting to plms'
            self.sampler = PLMSSampler(self.model, device=self.device)

        print(msg)

    def _set_scheduler(self):
        default = self.model.scheduler

        # See https://github.com/huggingface/diffusers/issues/277#issuecomment-1371428672
        scheduler_map = dict(
            ddim=diffusers.DDIMScheduler,
            dpmpp_2=diffusers.DPMSolverMultistepScheduler,
            k_dpm_2=diffusers.KDPM2DiscreteScheduler,
            k_dpm_2_a=diffusers.KDPM2AncestralDiscreteScheduler,
            # DPMSolverMultistepScheduler is technically not `k_` anything, as it is neither
            # the k-diffusers implementation nor included in EDM (Karras 2022), but we can
            # provide an alias for compatibility.
            k_dpmpp_2=diffusers.DPMSolverMultistepScheduler,
            k_euler=diffusers.EulerDiscreteScheduler,
            k_euler_a=diffusers.EulerAncestralDiscreteScheduler,
            k_heun=diffusers.HeunDiscreteScheduler,
            k_lms=diffusers.LMSDiscreteScheduler,
            plms=diffusers.PNDMScheduler,
        )

        if self.sampler_name in scheduler_map:
            sampler_class = scheduler_map[self.sampler_name]
            msg = f'>> Setting Sampler to {self.sampler_name} ({sampler_class.__name__})'
            self.sampler = sampler_class.from_config(self.model.scheduler.config)
        else:
            msg = (f'>> Unsupported Sampler: {self.sampler_name} '
                  f'Defaulting to {default}')
            self.sampler = default

        print(msg)

        if not hasattr(self.sampler, 'uses_inpainting_model'):
            # FIXME: terrible kludge!
            self.sampler.uses_inpainting_model = lambda: False

    def _load_img(self, img)->Image:
        if isinstance(img, Image.Image):
            image = img
            print(
                f'>> using provided input image of size {image.width}x{image.height}'
            )
        elif isinstance(img, str):
            assert os.path.exists(img), f'>> {img}: File not found'

            image = Image.open(img)
            print(
                f'>> loaded input image of size {image.width}x{image.height} from {img}'
            )
        else:
            image = Image.open(img)
            print(
                f'>> loaded input image of size {image.width}x{image.height}'
            )
        image = ImageOps.exif_transpose(image)
        return image

    def _create_init_image(self, image: Image.Image, width, height, fit=True):
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        image = self._fit_image(image, (width, height)) if fit else self._squeeze_image(image)
        return image

    def _create_init_mask(self, image, width, height, fit=True):
        # convert into a black/white mask
        image = self._image_to_mask(image)
        image = image.convert('RGB')
        image = self._fit_image(image, (width, height)) if fit else self._squeeze_image(image)
        return image

    # The mask is expected to have the region to be inpainted
    # with alpha transparency. It converts it into a black/white
    # image with the transparent part black.
    def _image_to_mask(self, mask_image: Image.Image, invert=False) -> Image:
        # Obtain the mask from the transparency channel
        if mask_image.mode == 'L':
            mask = mask_image
        elif mask_image.mode in ('RGB', 'P'):
            mask = mask_image.convert('L')
        else:
            # Obtain the mask from the transparency channel
            mask = Image.new(mode="L", size=mask_image.size, color=255)
            mask.putdata(mask_image.getdata(band=3))
        if invert:
            mask = ImageOps.invert(mask)
        return mask

    def _txt2mask(self, image:Image, text_mask:list, width, height, fit=True) -> Image:
        prompt = text_mask[0]
        confidence_level = text_mask[1] if len(text_mask)>1 else 0.5
        if self.txt2mask is None:
            self.txt2mask = Txt2Mask(device = self.device)

        segmented = self.txt2mask.segment(image, prompt)
        mask = segmented.to_mask(float(confidence_level))
        mask = mask.convert('RGB')
        mask = self._fit_image(mask, (width, height)) if fit else self._squeeze_image(mask)
        return mask

    def _has_transparency(self, image):
        if image.info.get("transparency", None) is not None:
            return True
        if image.mode == "P":
            transparent = image.info.get("transparency", -1)
            for _, index in image.getcolors():
                if index == transparent:
                    return True
        elif image.mode == "RGBA":
            extrema = image.getextrema()
            if extrema[3][0] < 255:
                return True
        return False

    def _check_for_erasure(self, image:Image.Image)->bool:
        if image.mode not in ('RGBA','RGB'):
            return False
        width, height = image.size
        pixdata = image.load()
        colored = 0
        for y in range(height):
            for x in range(width):
                if pixdata[x, y][3] == 0:
                    r, g, b, _ = pixdata[x, y]
                    if (r, g, b) != (0, 0, 0) and \
                       (r, g, b) != (255, 255, 255):
                        colored += 1
        return colored == 0

    def _transparency_check_and_warning(self,image, mask, force_outpaint=False):
        if not mask:
            print(
                '>> Initial image has transparent areas. Will inpaint in these regions.')
            if (not force_outpaint) and self._check_for_erasure(image):
                print(
                    '>> WARNING: Colors underneath the transparent region seem to have been erased.\n',
                    '>>          Inpainting will be suboptimal. Please preserve the colors when making\n',
                    '>>          a transparency mask, or provide mask explicitly using --init_mask (-M).'
                )

    def _squeeze_image(self, image):
        x, y, resize_needed = self._resolution_check(image.width, image.height)
        if resize_needed:
            return InitImageResizer(image).resize(x, y)
        return image

    def _fit_image(self, image, max_dimensions):
        w, h = max_dimensions
        print(
            f'>> image will be resized to fit inside a box {w}x{h} in size.'
        )
        # note that InitImageResizer does the multiple of 64 truncation internally
        image = InitImageResizer(image).resize(width=w, height=h)
        print(
            f'>> after adjusting image dimensions to be multiples of 64, init image is {image.width}x{image.height}'
        )
        return image

    def _resolution_check(self, width, height, log=False):
        resize_needed = False
        w, h = map(
            lambda x: x - x % 64, (width, height)
        )  # resize to integer multiple of 64
        if h != height or w != width:
            if log:
                print(
                    f'>> Provided width and height must be multiples of 64. Auto-resizing to {w}x{h}'
                )
            height = h
            width = w
            resize_needed = True
        return width, height, resize_needed


    def _has_cuda(self):
        return self.device.type == 'cuda'

    def write_intermediate_images(self,modulus,path):
        counter = -1
        if not os.path.exists(path):
            os.makedirs(path)
        def callback(img):
            nonlocal counter
            counter += 1
            if counter % modulus != 0:
                return;
            image = self.sample_to_image(img)
            image.save(os.path.join(path,f'{counter:03}.png'),'PNG')
        return callback

def _pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)
