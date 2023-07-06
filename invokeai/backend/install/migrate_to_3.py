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

from dataclasses import dataclass
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from typing import Union

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
    ModelProbe, ModelType, BaseModelType, SchedulerPredictionType, ModelProbeInfo
    )

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
diffusers.logging.set_verbosity_error()

# holder for paths that we will migrate
@dataclass
class ModelPaths:
    models: Path
    embeddings: Path
    loras: Path
    controlnets: Path

class MigrateTo3(object):
    def __init__(self,
                 root_directory: Path,
                 dest_models: Path,
                 yaml_file: io.TextIOBase,
                 src_paths: ModelPaths,
                 ):
        self.root_directory = root_directory
        self.dest_models = dest_models
        self.dest_yaml = yaml_file
        self.model_names = set()
        self.src_paths = src_paths
        
        self._initialize_yaml()

    def _initialize_yaml(self):
        self.dest_yaml.write(
            yaml.dump(
                {
                    '__metadata__':
                    {
                        'version':'3.0.0'}
                }
            )
        )
    
    def unique_name(self,name,info)->str:
        '''
        Create a unique name for a model for use within models.yaml.
        '''
        done = False
        
        # some model names have slashes in them, which really screws things up
        name = name.replace('/','_')
        
        key = ModelManager.create_key(name,info.base_type,info.model_type)
        unique_name = key
        counter = 1
        while not done:
            if unique_name in self.model_names:
                unique_name = f'{key}-{counter:0>2d}'
                counter += 1
            else:
                done = True
        self.model_names.add(unique_name)
        name,_,_ = ModelManager.parse_key(unique_name)
        return name

    def create_directory_structure(self):
        '''
        Create the basic directory structure for the models folder.
        '''
        for model_base in [BaseModelType.StableDiffusion1,BaseModelType.StableDiffusion2]:
            for model_type in [ModelType.Main, ModelType.Vae, ModelType.Lora,
                               ModelType.ControlNet,ModelType.TextualInversion]:
                path = self.dest_models / model_base.value / model_type.value
                path.mkdir(parents=True, exist_ok=True)
        path = self.dest_models / 'core'
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def copy_file(src:Path,dest:Path):
        '''
        copy a single file with logging
        '''
        if dest.exists():
            logger.info(f'Skipping existing {str(dest)}')
            return
        logger.info(f'Copying {str(src)} to {str(dest)}')
        try:
            shutil.copy(src, dest)
        except Exception as e:
            logger.error(f'COPY FAILED: {str(e)}')

    @staticmethod
    def copy_dir(src:Path,dest:Path):
        '''
        Recursively copy a directory with logging
        '''
        if dest.exists():
            logger.info(f'Skipping existing {str(dest)}')
            return
        
        logger.info(f'Copying {str(src)} to {str(dest)}')
        try:
            shutil.copytree(src, dest)
        except Exception as e:
            logger.error(f'COPY FAILED: {str(e)}')

    def migrate_models(self, src_dir: Path):
        '''
        Recursively walk through src directory, probe anything
        that looks like a model, and copy the model into the
        appropriate location within the destination models directory.
        '''
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
                    dest = self._model_probe_to_path(info) / f
                    self.copy_file(model, dest)
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
                    dest = self._model_probe_to_path(info) / model.name
                    self.copy_dir(model, dest)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(str(e))

    def migrate_support_models(self):
        '''
        Copy the clipseg, upscaler, and restoration models to their new
        locations.
        '''
        dest_directory = self.dest_models
        if (self.root_directory / 'models/clipseg').exists():
            self.copy_dir(self.root_directory / 'models/clipseg', dest_directory / 'core/misc/clipseg')
        if (self.root_directory / 'models/realesrgan').exists():
            self.copy_dir(self.root_directory / 'models/realesrgan', dest_directory / 'core/upscaling/realesrgan')
        for d in ['codeformer','gfpgan']:
            path = self.root_directory / 'models' / d
            if path.exists():
                self.copy_dir(path,dest_directory / f'core/face_restoration/{d}')

    def migrate_tuning_models(self):
        '''
        Migrate the embeddings, loras and controlnets directories to their new homes.
        '''
        for src in [self.src_paths.embeddings, self.src_paths.loras, self.src_paths.controlnets]:
            if not src:
                continue
            if src.is_dir():
                logger.info(f'Scanning {src}')
                self.migrate_models(src)
            else:
                logger.info(f'{src} directory not found; skipping')
                continue

    def migrate_conversion_models(self):
        '''
        Migrate all the models that are needed by the ckpt_to_diffusers conversion
        script.
        '''

        dest_directory = self.dest_models
        kwargs = dict(
            cache_dir = self.root_directory / 'models/hub',
            #local_files_only = True
        )
        try:
            logger.info('Migrating core tokenizers and text encoders')
            target_dir = dest_directory / 'core' / 'convert'

            self._migrate_pretrained(BertTokenizerFast,
                                     repo_id='bert-base-uncased',
                                     dest = target_dir / 'bert-base-uncased',
                                     **kwargs)

            # sd-1
            repo_id = 'openai/clip-vit-large-patch14'
            self._migrate_pretrained(CLIPTokenizer,
                                     repo_id= repo_id,
                                     dest= target_dir / 'clip-vit-large-patch14',
                                     **kwargs)
            self._migrate_pretrained(CLIPTextModel,
                                     repo_id = repo_id,
                                     dest = target_dir / 'clip-vit-large-patch14',
                                     force = True,
                                     **kwargs)

            # sd-2
            repo_id = "stabilityai/stable-diffusion-2"
            self._migrate_pretrained(CLIPTokenizer,
                                     repo_id = repo_id,
                                     dest = target_dir / 'stable-diffusion-2-clip' / 'tokenizer',
                                     **{'subfolder':'tokenizer',**kwargs}
                                     )
            self._migrate_pretrained(CLIPTextModel,
                                     repo_id = repo_id,
                                     dest = target_dir / 'stable-diffusion-2-clip' / 'text_encoder',
                                     **{'subfolder':'text_encoder',**kwargs}
                                     )

            # VAE
            logger.info('Migrating stable diffusion VAE')
            self._migrate_pretrained(AutoencoderKL,
                                     repo_id = 'stabilityai/sd-vae-ft-mse',
                                     dest = target_dir / 'sd-vae-ft-mse',
                                     **kwargs)
            
            # safety checking
            logger.info('Migrating safety checker')
            repo_id = "CompVis/stable-diffusion-safety-checker"
            self._migrate_pretrained(AutoFeatureExtractor,
                                     repo_id = repo_id,
                                     dest = target_dir / 'stable-diffusion-safety-checker',
                                     **kwargs)
            self._migrate_pretrained(StableDiffusionSafetyChecker,
                                     repo_id = repo_id,
                                     dest = target_dir / 'stable-diffusion-safety-checker',
                                     **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(str(e))

    def write_yaml(self, model_name: str, path:Path, info:ModelProbeInfo, **kwargs):
        '''
        Write a stanza for a moved model into the new models.yaml file.
        '''
        name = self.unique_name(model_name, info)
        stanza = {
            f'{info.base_type.value}/{info.model_type.value}/{name}': {
                'name': model_name,
                'path': str(path),
                'description': f'A {info.base_type.value} {info.model_type.value} model',
                'format': info.format,
                'image_size': info.image_size,
                'base': info.base_type.value,
                'variant': info.variant_type.value,
                'prediction_type': info.prediction_type.value,
                'upcast_attention': info.prediction_type == SchedulerPredictionType.VPrediction,
                **kwargs,
            }
        }
        self.dest_yaml.write(yaml.dump(stanza))
        self.dest_yaml.flush()
        
    def _model_probe_to_path(self, info: ModelProbeInfo)->Path:
        return Path(self.dest_models, info.base_type.value, info.model_type.value)

    def _migrate_pretrained(self, model_class, repo_id: str, dest: Path, force:bool=False, **kwargs):
        if dest.exists() and not force:
            logger.info(f'Skipping existing {dest}')
            return
        model = model_class.from_pretrained(repo_id, **kwargs)
        self._save_pretrained(model, dest, overwrite=force)

    def _save_pretrained(self, model, dest: Path, overwrite: bool=False):
        model_name = dest.name
        if overwrite:
            model.save_pretrained(dest, safe_serialization=True)
        else:
            download_path = dest.with_name(f'{model_name}.downloading')
            model.save_pretrained(download_path, safe_serialization=True)
            download_path.replace(dest)

    def _download_vae(self, repo_id: str, subfolder:str=None)->Path:
        vae = AutoencoderKL.from_pretrained(repo_id, cache_dir=self.root_directory / 'models/hub', subfolder=subfolder)
        info = ModelProbe().heuristic_probe(vae)
        _, model_name = repo_id.split('/')
        dest = self._model_probe_to_path(info) / self.unique_name(model_name, info)
        vae.save_pretrained(dest, safe_serialization=True)
        return dest

    def _vae_path(self, vae: Union[str,dict])->Path:
        '''
        Convert 2.3 VAE stanza to a straight path.
        '''
        vae_path = None
        
        # First get a path
        if isinstance(vae,str):
            vae_path = vae

        elif isinstance(vae,DictConfig):
            if p := vae.get('path'):
                vae_path = p
            elif repo_id := vae.get('repo_id'):
                if repo_id=='stabilityai/sd-vae-ft-mse':  # this guy is already downloaded
                    vae_path = 'models/core/convert/sd-vae-ft-mse'
                else:
                    vae_path = self._download_vae(repo_id, vae.get('subfolder'))

        assert vae_path is not None, "Couldn't find VAE for this model"

        # if the VAE is in the old models directory, then we must move it into the new
        # one. VAEs outside of this directory can stay where they are.
        vae_path = Path(vae_path)
        if vae_path.is_relative_to(self.src_paths.models):
            info = ModelProbe().heuristic_probe(vae_path)
            dest = self._model_probe_to_path(info) / vae_path.name
            if not dest.exists():
                self.copy_dir(vae_path,dest)
            vae_path = dest

        if vae_path.is_relative_to(self.dest_models):
            rel_path = vae_path.relative_to(self.dest_models)
            return Path('models',rel_path)
        else:
            return vae_path

    def migrate_repo_id(self, repo_id: str, model_name :str=None, **extra_config):
        '''
        Migrate a locally-cached diffusers pipeline identified with a repo_id
        '''
        dest_dir = self.dest_models
        
        cache = self.root_directory / 'models/hub'
        kwargs = dict(
            cache_dir = cache,
            safety_checker = None,
            # local_files_only = True,
        )

        owner,repo_name = repo_id.split('/')
        model_name = model_name or repo_name
        model = cache / '--'.join(['models',owner,repo_name])
        
        if len(list(model.glob('snapshots/**/model_index.json')))==0:
            return
        revisions = [x.name for x in model.glob('refs/*')]

        # if an fp16 is available we use that
        revision = 'fp16' if len(revisions) > 1 and 'fp16' in revisions else revisions[0]
        pipeline = StableDiffusionPipeline.from_pretrained(
            repo_id,
            revision=revision,
            **kwargs)

        info = ModelProbe().heuristic_probe(pipeline)
        if not info:
            return

        dest = self._model_probe_to_path(info) / repo_name
        self._save_pretrained(pipeline, dest)
            
        rel_path = Path('models',dest.relative_to(dest_dir))
        self.write_yaml(model_name, path=rel_path, info=info, **extra_config)

    def migrate_path(self, location: Path, model_name: str=None, **extra_config):
        '''
        Migrate a model referred to using 'weights' or 'path'
        '''

        # handle relative paths
        dest_dir = self.dest_models
        location = self.root_directory / location
        
        info = ModelProbe().heuristic_probe(location)
        if not info:
            return

        # uh oh, weights is in the old models directory - move it into the new one
        if Path(location).is_relative_to(self.src_paths.models):
            dest = Path(dest_dir, info.base_type.value, info.model_type.value, location.name)
            self.copy_dir(location,dest)
            location = Path('models', info.base_type.value, info.model_type.value, location.name)
        model_name = model_name or location.stem
        model_name = self.unique_name(model_name, info)
        self.write_yaml(model_name, path=location, info=info, **extra_config)

    def migrate_defined_models(self):
        '''
        Migrate models defined in models.yaml
        '''
        # find any models referred to in old models.yaml
        conf = OmegaConf.load(self.root_directory / 'configs/models.yaml')
        
        for model_name, stanza in conf.items():

            try:
                passthru_args = {}
                
                if vae := stanza.get('vae'):
                    try:
                        passthru_args['vae'] = str(self._vae_path(vae))
                    except Exception as e:
                        logger.warning(f'Could not find a VAE matching "{vae}" for model "{model_name}"')
                        logger.warning(str(e))

                if config := stanza.get('config'):
                    passthru_args['config'] = config
                
                if repo_id := stanza.get('repo_id'):
                    logger.info(f'Migrating diffusers model {model_name}')
                    self.migrate_repo_id(repo_id, model_name, **passthru_args)

                elif location := stanza.get('weights'):
                    logger.info(f'Migrating checkpoint model {model_name}')
                    self.migrate_path(Path(location), model_name, **passthru_args)
                    
                elif location := stanza.get('path'):
                    logger.info(f'Migrating diffusers model {model_name}')
                    self.migrate_path(Path(location), model_name, **passthru_args)
                    
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(str(e))
                    
    def migrate(self):
        self.create_directory_structure()
        # the configure script is doing this
        self.migrate_support_models()
        self.migrate_conversion_models()
        self.migrate_tuning_models()
        self.migrate_defined_models()

def _parse_legacy_initfile(root: Path, initfile: Path)->ModelPaths:
    '''
    Returns tuple of (embedding_path, lora_path, controlnet_path)
    '''
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument(
        '--embedding_directory',
        '--embedding_path',
        type=Path,
        dest='embedding_path',
        default=Path('embeddings'),
    )
    parser.add_argument(
        '--lora_directory',
        dest='lora_path',
        type=Path,
        default=Path('loras'),
    )
    opt,_ = parser.parse_known_args([f'@{str(initfile)}'])
    return ModelPaths(
        models = root / 'models',
        embeddings = root / str(opt.embedding_path).strip('"'),
        loras = root / str(opt.lora_path).strip('"'),
        controlnets = root / 'controlnets',
    )

def _parse_legacy_yamlfile(root: Path, initfile: Path)->ModelPaths:
    '''
    Returns tuple of (embedding_path, lora_path, controlnet_path)
    '''
    # Don't use the config object because it is unforgiving of version updates
    # Just use omegaconf directly
    opt = OmegaConf.load(initfile)
    paths = opt.InvokeAI.Paths
    models = paths.get('models_dir','models')
    embeddings = paths.get('embedding_dir','embeddings')
    loras = paths.get('lora_dir','loras')
    controlnets = paths.get('controlnet_dir','controlnets')
    return ModelPaths(
        models = root / models,
        embeddings = root / embeddings,
        loras = root /loras,
        controlnets = root / controlnets,
    )
    
def get_legacy_embeddings(root: Path) -> ModelPaths:
    path = root / 'invokeai.init'
    if path.exists():
        return _parse_legacy_initfile(root, path)
    path = root / 'invokeai.yaml'
    if path.exists():
        return _parse_legacy_yamlfile(root, path)

def do_migrate(src_directory: Path, dest_directory: Path):
    
    dest_models = dest_directory / 'models-3.0'
    dest_yaml = dest_directory / 'configs/models.yaml-3.0'

    paths = get_legacy_embeddings(src_directory)

    with open(dest_yaml,'w') as yaml_file:
        migrator = MigrateTo3(src_directory,
                              dest_models,
                              yaml_file,
                              src_paths = paths,
                              )
        migrator.migrate()

    shutil.rmtree(dest_directory / 'models.orig', ignore_errors=True)
    (dest_directory / 'models').replace(dest_directory / 'models.orig')
    dest_models.replace(dest_directory / 'models')

    (dest_directory /'configs/models.yaml').replace(dest_directory / 'configs/models.yaml.orig')
    dest_yaml.replace(dest_directory / 'configs/models.yaml')
    print(f"""Migration successful.
Original models directory moved to {dest_directory}/models.orig
Original models.yaml file moved to {dest_directory}/configs/models.yaml.orig
""")

def main():
    parser = argparse.ArgumentParser(prog="invokeai-migrate3",
                                     description="""
This will copy and convert the models directory and the configs/models.yaml from the InvokeAI 2.3 format 
'--from-directory' root to the InvokeAI 3.0 '--to-directory' root. These may be abbreviated '--from' and '--to'.a

The old models directory and config file will be renamed 'models.orig' and 'models.yaml.orig' respectively.
It is safe to provide the same directory for both arguments, but it is better to use the invokeai_configure
script, which will perform a full upgrade in place."""
                                     )
    parser.add_argument('--from-directory',
                        dest='root_directory',
                        type=Path,
                        required=True,
                        help='Source InvokeAI 2.3 root directory (containing "invokeai.init" or "invokeai.yaml")'
                        )
    parser.add_argument('--to-directory',
                        dest='dest_directory',
                        type=Path,
                        required=True,
                        help='Destination InvokeAI 3.0 directory (containing "invokeai.yaml")'
                        )
# TO DO: Implement full directory scanning
#    parser.add_argument('--all-models',
#                        action="store_true",
#                        help='Migrate all models found in `models` directory, not just those mentioned in models.yaml',
#                        )
    args = parser.parse_args()
    root_directory = args.root_directory
    assert root_directory.is_dir(), f"{root_directory} is not a valid directory"
    assert (root_directory / 'models').is_dir(), f"{root_directory} does not contain a 'models' subdirectory"
    assert (root_directory / 'invokeai.init').exists() or (root_directory / 'invokeai.yaml').exists(), f"{root_directory} does not contain an InvokeAI init file."

    dest_directory = args.dest_directory
    assert dest_directory.is_dir(), f"{dest_directory} is not a valid directory"

    # TODO: revisit
    # assert (dest_directory / 'models').is_dir(), f"{dest_directory} does not contain a 'models' subdirectory"
    # assert (dest_directory / 'invokeai.yaml').exists(), f"{dest_directory} does not contain an InvokeAI init file."

    do_migrate(root_directory,dest_directory)

if __name__ == '__main__':
    main()



