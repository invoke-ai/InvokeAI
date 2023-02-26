'''
Simple class hierarchy
'''
import copy
import dataclasses
import diffusers
import importlib
import traceback

from abc import ABCMeta, abstractmethod
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
from typing import List, Type
from dataclasses import dataclass
from diffusers.schedulers import SchedulerMixin as Scheduler

import invokeai.assets as image_assets
from ldm.invoke.globals import global_config_dir
from ldm.invoke.conditioning import get_uc_and_c_and_ec
from ldm.invoke.model_manager import ModelManager
from ldm.invoke.generator.diffusers_pipeline import StableDiffusionGeneratorPipeline
from ldm.invoke.devices import choose_torch_device

@dataclass
class RendererBasicParams:
    width: int=512
    height: int=512
    cfg_scale: int=7.5
    steps: int=20
    ddim_eta: float=0.0
    model: str='stable-diffusion-1.5'
    scheduler: int='ddim'
    precision: str='float16'

@dataclass
class RendererOutput:
    image: Image
    seed: int
    model_name: str
    model_hash: str
    params: RendererBasicParams

class InvokeAIRenderer(metaclass=ABCMeta):
    scheduler_map = dict(
        ddim=diffusers.DDIMScheduler,
        dpmpp_2=diffusers.DPMSolverMultistepScheduler,
        k_dpm_2=diffusers.KDPM2DiscreteScheduler,
        k_dpm_2_a=diffusers.KDPM2AncestralDiscreteScheduler,
        k_dpmpp_2=diffusers.DPMSolverMultistepScheduler,
        k_euler=diffusers.EulerDiscreteScheduler,
        k_euler_a=diffusers.EulerAncestralDiscreteScheduler,
        k_heun=diffusers.HeunDiscreteScheduler,
        k_lms=diffusers.LMSDiscreteScheduler,
        plms=diffusers.PNDMScheduler,
    )

    def __init__(self,
                 model_manager: ModelManager,
                 params: RendererBasicParams
                 ):
        self.model_manager=model_manager
        self.params=params

    def render(self,
               prompt: str='',
               callback: callable=None,
               step_callback: callable=None,
               **keyword_args,
               )->List[RendererOutput]:

        model_name = self.params.model or self.model_manager.current_model
        model_info: dict = self.model_manager.get_model(model_name)
        model:StableDiffusionGeneratorPipeline = model_info['model']
        model_hash = model_info['hash']
        scheduler: Scheduler = self.get_scheduler(
            model=model,
            scheduler_name=self.params.scheduler
        )
        uc, c, extra_conditioning_info = get_uc_and_c_and_ec(prompt,model=model)

        def _wrap_results(image: Image, seed: int, **kwargs):
            nonlocal results
            results.append(output)

        generator = self.load_generator(model, self._generator_name())
        while True:
            results = generator.generate(prompt,
                                         conditioning=(uc, c, extra_conditioning_info),
                                         sampler=scheduler,
                                         **dataclasses.asdict(self.params),
                                         **keyword_args
                                         )
            output = RendererOutput(
                image=results[0][0],
                seed=results[0][1],
                model_name = model_name,
                model_hash = model_hash,
                params=copy.copy(self.params)
            )
            if callback:
                callback(output)
            yield output

    def load_generator(self, model: StableDiffusionGeneratorPipeline, class_name: str):
        module_name = f'ldm.invoke.generator.{class_name.lower()}'
        module = importlib.import_module(module_name)
        constructor = getattr(module, class_name)
        return constructor(model, self.params.precision)
               
    def get_scheduler(self, scheduler_name:str, model: StableDiffusionGeneratorPipeline)->Scheduler:
        scheduler_class = self.scheduler_map.get(scheduler_name,'ddim')
        scheduler = scheduler_class.from_config(model.scheduler.config)
        # hack copied over from generate.py
        if not hasattr(scheduler, 'uses_inpainting_model'):
            scheduler.uses_inpainting_model = lambda: False
        return scheduler
        
    @abstractmethod
    def _generator_name(self)->str:
        '''
        In derived classes will return the name of the generator to use.
        '''
        pass

# ------------------------------------
class Txt2Img(InvokeAIRenderer):
    def _generator_name(self)->str:
        return 'Txt2Img'

# ------------------------------------
class Img2Img(InvokeAIRenderer):
    def render(self,
               init_image: Image,
               strength: float=0.75,
               **keyword_args
               )->List[RendererOutput]:
        return super().render(init_image=init_image,
                              strength=strength,
                              **keyword_args
                              )

    def _generator_name(self)->str:
        return 'Img2Img'

class RendererFactory(object):
    def __init__(self,
                 model_manager: ModelManager,
                 params: RendererBasicParams
                 ):
        self.model_manager = model_manager
        self.params = params
                 
    def renderer(self, rendererclass: Type[InvokeAIRenderer], **keyword_args)->InvokeAIRenderer:
        return rendererclass(self.model_manager,
                             self.params,
                             **keyword_args
                             )

# ---- testing ---
def main():
    config_file = Path(global_config_dir()) / "models.yaml"
    model_manager = ModelManager(OmegaConf.load(config_file),
                                 precision='float16',
                                 device_type=choose_torch_device(),
                                 )

    params = RendererBasicParams(
        model = 'stable-diffusion-1.5',
        steps = 30,
        scheduler = 'k_lms',
        cfg_scale = 8.0,
        height = 640,
        width = 640
        )
    factory = RendererFactory(model_manager, params)

    print ('=== TXT2IMG TEST ===')
    txt2img = factory.renderer(Txt2Img)
    outputs = txt2img.render(prompt='banana sushi')
    for i in range(3):
        output = next(outputs)
        print(f'image={output.image}, seed={output.seed}, model={output.model_name}, hash={output.model_hash}')


if __name__=='__main__':
    main()
