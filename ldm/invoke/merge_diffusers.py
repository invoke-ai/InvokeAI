'''
ldm.invoke.merge_diffusers exports a single function call merge_diffusion_models()
used to merge 2-3 models together and create a new InvokeAI-registered diffusion model.
'''
import os
from typing import List
from diffusers import DiffusionPipeline
from ldm.invoke.globals import global_config_file, global_models_dir, global_cache_dir
from ldm.invoke.model_manager import ModelManager
from omegaconf import OmegaConf

def merge_diffusion_models(models:List['str'],
                           merged_model_name:str,
                           alpha:float=0.5,
                           interp:str=None,
                           force:bool=False,
                           **kwargs):
    '''
    models - up to three models, designated by their InvokeAI models.yaml model name
    merged_model_name = name for new model
    alpha  - The interpolation parameter. Ranges from 0 to 1.  It affects the ratio in which the checkpoints are merged. A 0.8 alpha
               would mean that the first model checkpoints would affect the final result far less than an alpha of 0.2
    interp - The interpolation method to use for the merging. Supports "sigmoid", "inv_sigmoid", "add_difference" and None.
               Passing None uses the default interpolation which is weighted sum interpolation. For merging three checkpoints, only "add_difference" is supported.
    force  - Whether to ignore mismatch in model_config.json for the current models. Defaults to False.

    **kwargs - the default DiffusionPipeline.get_config_dict kwargs:
         cache_dir, resume_download, force_download, proxies, local_files_only, use_auth_token, revision, torch_dtype, device_map
    '''
    config_file = global_config_file()
    model_manager = ModelManager(OmegaConf.load(config_file))
    model_ids_or_paths = [model_manager.model_name_or_path(x) for x in models]

    pipe = DiffusionPipeline.from_pretrained(model_ids_or_paths[0],
                                             cache_dir=kwargs.get('cache_dir',global_cache_dir()),
                                             custom_pipeline='checkpoint_merger')
    merged_pipe = pipe.merge(pretrained_model_name_or_path_list=model_ids_or_paths,
                             alpha=alpha,
                             interp=interp,
                             force=force,
                             **kwargs)
    dump_path = global_models_dir() / 'merged_diffusers'
    os.makedirs(dump_path,exist_ok=True)
    dump_path = dump_path / merged_model_name
    merged_pipe.save_pretrained (
        dump_path,
        safe_serialization=1
    )
    model_manager.import_diffuser_model(
        dump_path,
        model_name = merged_model_name,
        description = f'Merge of models {", ".join(models)}'
    )
    print('REMINDER: When PR 2369 is merged, replace merge_diffusers.py line 56 with vae= argument to impormodel()')
    if vae := model_manager.config[models[0]].get('vae',None):
        print(f'>> Using configured VAE assigned to {models[0]}')
        model_manager.config[merged_model_name]['vae'] = vae

    model_manager.commit(config_file)
