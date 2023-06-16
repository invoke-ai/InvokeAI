'''
Migrate the models directory and models.yaml file from an existing
InvokeAI 2.3 installation to 3.0.0.
'''

import io
import os
import argparse
import shutil
import yaml

import transformers
import diffusers
import warnings
from pathlib import Path
from omegaconf import OmegaConf
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    AutoFeatureExtractor,
    BertTokenizerFast,
)

import invokeai.backend.util.logging as logger
from invokeai.backend.model_management import ModelManager
from invokeai.backend.model_management.model_probe import (
    ModelProbe, ModelType, BaseModelType, SchedulerPredictionType, ModelVariantInfo
    )

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
diffusers.logging.set_verbosity_error()

model_names = set()

def unique_name(name,info)->str:
    done = False
    key = ModelManager.create_key(name,info.base_type,info.model_type)
    unique_name = key
    counter = 1
    while not done:
        if unique_name in model_names:
            unique_name = f'{key}-{counter:0>2d}'
            counter += 1
        else:
            done = True
    model_names.add(unique_name)
    name,_,_ = ModelManager.parse_key(unique_name)
    return name

def create_directory_structure(dest: Path):
    for model_base in [BaseModelType.StableDiffusion1,BaseModelType.StableDiffusion2]:
        for model_type in [ModelType.Pipeline, ModelType.Vae, ModelType.Lora,
                           ModelType.ControlNet,ModelType.TextualInversion]:
            path = dest / model_base.value / model_type.value
            path.mkdir(parents=True, exist_ok=True)
    path = dest / 'core'
    path.mkdir(parents=True, exist_ok=True)

def copy_file(src:Path,dest:Path):
    logger.info(f'Copying {str(src)} to {str(dest)}')
    try:
        shutil.copy(src, dest)
    except Exception as e:
        logger.error(f'COPY FAILED: {str(e)}')

def copy_dir(src:Path,dest:Path):
    logger.info(f'Copying {str(src)} to {str(dest)}')
    try:
        shutil.copytree(src, dest)
    except Exception as e:
        logger.error(f'COPY FAILED: {str(e)}')

def migrate_models(src_dir: Path, dest_dir: Path):
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            # hack - don't copy raw learned_embeds.bin, let them
            # be copied as part of a tree copy operation
            if f == 'learned_embeds.bin':
                continue
            try:
                model = Path(root,f)
                info = ModelProbe().heuristic_probe(model)
                if not info:
                    continue
                dest = Path(dest_dir, info.base_type.value, info.model_type.value, f)
                copy_file(model, dest)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(str(e))
        for d in dirs:
            try:
                model = Path(root,d)
                info = ModelProbe().heuristic_probe(model)
                if not info:
                    continue
                dest = Path(dest_dir, info.base_type.value, info.model_type.value, model.name)
                copy_dir(model, dest)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(str(e))

def migrate_support_models(dest_directory: Path):
    if Path('./models/clipseg').exists():
        copy_dir(Path('./models/clipseg'),dest_directory / 'core/misc/clipseg')
    if Path('./models/realesrgan').exists():
        copy_dir(Path('./models/realesrgan'),dest_directory / 'core/upscaling/realesrgan')
    for d in ['codeformer','gfpgan']:
        path = Path('./models',d)
        if path.exists():
            copy_dir(path,dest_directory / f'core/face_restoration/{d}')

def migrate_conversion_models(dest_directory: Path):
    # These are needed for the conversion script
    kwargs = dict(
        cache_dir = Path('./models/hub'),
        #local_files_only = True
    )
    try:
        logger.info('Migrating core tokenizers and text encoders')
        target_dir = dest_directory / 'core' / 'convert'
        
        # bert
        bert = BertTokenizerFast.from_pretrained("bert-base-uncased", **kwargs)
        bert.save_pretrained(target_dir / 'bert-base-uncased', safe_serialization=True)
        
        # sd-1
        repo_id = 'openai/clip-vit-large-patch14'
        pipeline = CLIPTokenizer.from_pretrained(repo_id, **kwargs)
        pipeline.save_pretrained(target_dir / 'clip-vit-large-patch14' / 'tokenizer', safe_serialization=True)

        pipeline = CLIPTextModel.from_pretrained(repo_id, **kwargs)
        pipeline.save_pretrained(target_dir / 'clip-vit-large-patch14' / 'text_encoder', safe_serialization=True)

        # sd-2
        repo_id = "stabilityai/stable-diffusion-2"
        pipeline = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer", **kwargs)
        pipeline.save_pretrained(target_dir / 'stable-diffusion-2-clip' / 'tokenizer', safe_serialization=True)

        pipeline = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder", **kwargs)
        pipeline.save_pretrained(target_dir / 'stable-diffusion-2-clip' / 'text_encoder', safe_serialization=True)

        # VAE
        logger.info('Migrating stable diffusion VAE')
        vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse', **kwargs)
        vae.save_pretrained(target_dir / 'sd-vae-ft-mse', safe_serialization=True)

        # safety checking
        logger.info('Migrating safety checker')
        repo_id = "CompVis/stable-diffusion-safety-checker"
        pipeline = AutoFeatureExtractor.from_pretrained(repo_id,**kwargs)
        pipeline.save_pretrained(target_dir / 'stable-diffusion-safety-checker', safe_serialization=True)

        pipeline = StableDiffusionSafetyChecker.from_pretrained(repo_id,**kwargs)
        pipeline.save_pretrained(target_dir / 'stable-diffusion-safety-checker', safe_serialization=True)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(str(e))

def migrate_tuning_models(dest: Path):
    for subdir in ['embeddings','loras','controlnets']:
        src = Path('.',subdir)
        if not src.is_dir():
            logger.info(f'{subdir} directory not found; skipping')
            continue
        logger.info(f'Scanning {subdir}')
        migrate_models(src, dest)

def write_yaml(model_name: str, path:Path, info:ModelVariantInfo, dest_yaml: io.TextIOBase):
    name = unique_name(model_name, info)
    stanza = {
        f'{info.base_type.value}/{info.model_type.value}/{name}': {
            'name': model_name,
            'path': str(path),
            'description': f'diffusers model {model_name}',
            'format': 'diffusers',
            'image_size': info.image_size,
            'base': info.base_type.value,
            'variant': info.variant_type.value,
            'prediction_type': info.prediction_type.value,
            'upcast_attention': info.prediction_type == SchedulerPredictionType.VPrediction
        }
    }
    dest_yaml.write(yaml.dump(stanza))
    dest_yaml.flush()
    
def migrate_converted(dest_dir: Path, dest_yaml: io.TextIOBase):
    for sub_dir in [Path('./models/converted_ckpts'),Path('./models/optimize-ckpts')]:
        for model in sub_dir.glob('*'):
            if not model.is_dir():
                continue
            info = ModelProbe().heuristic_probe(model)
            if not info:
                continue
            dest = Path(dest_dir, info.base_type.value, info.model_type.value, model.name)
            try:
                copy_dir(model,dest)
                rel_path = Path('models',dest.relative_to(dest_dir))
                write_yaml(model.name,path=rel_path,info=info, dest_yaml=dest_yaml)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.warning(f'Could not migrate the converted diffusers {model.name}: {str(e)}. Skipping.')

def migrate_pipelines(dest_dir: Path, dest_yaml: io.TextIOBase):
    cache = Path('./models/hub')
    kwargs = dict(
        cache_dir = cache,
        safety_checker = None,
        # local_files_only = True,
    )
    for model in cache.glob('models--*'):
        if len(list(model.glob('snapshots/**/model_index.json')))==0:
            continue
        _,owner,repo_name=model.name.split('--')
        repo_id = f'{owner}/{repo_name}'
        revisions = [x.name for x in model.glob('refs/*')]
        
        # if an fp16 is available we use that
        revision = 'fp16' if len(revisions) > 1 and 'fp16' in revisions else revisions[0]
        logger.info(f'Migrating {repo_id}, revision {revision}')
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                repo_id,
                revision=revision,
                **kwargs)
            info = ModelProbe().heuristic_probe(pipeline)
            if not info:
                continue
            dest = Path(dest_dir, info.base_type.value, info.model_type.value, f'{repo_name}')
            pipeline.save_pretrained(dest, safe_serialization=True)
            rel_path = Path('models',dest.relative_to(dest_dir))
            write_yaml(repo_name, path=rel_path, info=info, dest_yaml=dest_yaml)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.warning(f'Could not load the "{revision}" version of {repo_id}. Skipping.')

def migrate_checkpoints(dest_dir: Path, dest_yaml: io.TextIOBase):
    # find any checkpoints referred to in old models.yaml
    conf = OmegaConf.load('./configs/models.yaml')
    orig_models_dir = Path.cwd() / 'models'
    for model_name, stanza in conf.items():
        if stanza.get('format') and stanza['format'] == 'ckpt':
            try:
                logger.info(f'Migrating checkpoint model {model_name}')
                weights = orig_models_dir.parent / stanza['weights']
                config = stanza['config']
                info = ModelProbe().heuristic_probe(weights)
                if not info:
                    continue
                
                # uh oh, weights is in the old models directory - move it into the new one
                if Path(weights).is_relative_to(orig_models_dir):
                    dest = Path(dest_dir, info.base_type.value, info.model_type.value,weights.name)
                    copy_file(weights,dest)
                    weights = Path('models', info.base_type.value, info.model_type.value,weights.name)
                model_name = unique_name(model_name, info)
                stanza = {
                    f'{info.base_type.value}/{info.model_type.value}/{model_name}':
                    {
                        'name': model_name,
                        'path': str(weights),
                        'description': f'checkpoint model {model_name}',
                        'format': 'checkpoint',
                        'image_size': info.image_size,
                        'base': info.base_type.value,
                        'variant': info.variant_type.value,
                        'config': config
                    }
                }
                print(yaml.dump(stanza),file=dest_yaml,end="")
                dest_yaml.flush()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(str(e))
    
def main():
    parser = argparse.ArgumentParser(description="Model directory migrator")
    parser.add_argument('root_directory',
                        help='Root directory (containing "models", "embeddings", "controlnets" and "loras")'
                        )
    parser.add_argument('--dest-directory',
                        default='./models-3.0',
                        help='Destination for new models directory',
                        )
    parser.add_argument('--dest-yaml',
                        default='./models.yaml-3.0',
                        help='Destination for new models.yaml file',
                        )
    args = parser.parse_args()
    root_directory = Path(args.root_directory)
    assert root_directory.is_dir(), f"{root_directory} is not a valid directory"
    assert (root_directory / 'models').is_dir(), f"{root_directory} does not contain a 'models' subdirectory"

    dest_directory = Path(args.dest_directory).resolve()
    dest_yaml = Path(args.dest_yaml).resolve()

    os.chdir(root_directory)
    with open(dest_yaml,'w') as yaml_file:
        yaml_file.write(yaml.dump({'__metadata__':
                                   {'version':'3.0.0'}
                                   }
                                  )
                        )
        create_directory_structure(dest_directory)
        migrate_support_models(dest_directory)
        migrate_conversion_models(dest_directory)
        migrate_tuning_models(dest_directory)
        migrate_converted(dest_directory,yaml_file)
        migrate_pipelines(dest_directory,yaml_file)
        migrate_checkpoints(dest_directory,yaml_file)

if __name__ == '__main__':
    main()

