import os
import pytest

from omegaconf import OmegaConf
from pathlib import Path

os.environ['INVOKEAI_ROOT']='/tmp'
from invokeai.app.services.config import InvokeAIAppConfig, InvokeAISettings
from invokeai.app.invocations.generate import TextToImageInvocation

init1 = OmegaConf.create(
'''
globals:
  nsfw_checker: False
  max_loaded_models: 5

history:
  count: 100
 
txt2img:
  steps: 18
  scheduler: k_heun
  width: 768
 
img2img:
  width: 1024
  height: 1024
'''
)

init2 = OmegaConf.create(
'''
 globals:
   nsfw_checker: True
   max_loaded_models: 2

 history:
   count: 10
'''
)

def test_use_init():
    # note that we explicitly set omegaconf dict and argv here
    # so that the values aren't read from ~invokeai/invokeai.yaml and
    # sys.argv respectively.
    conf1 = InvokeAIAppConfig(init1,[])
    assert conf1
    assert conf1.max_loaded_models==5
    assert not conf1.nsfw_checker

    conf2 = InvokeAIAppConfig(init2,[])
    assert conf2
    assert conf2.nsfw_checker
    assert conf2.max_loaded_models==2
    assert not hasattr(conf2,'invalid_attribute')
    

def test_argv_override():
    conf = InvokeAIAppConfig(init1,['--nsfw_checker','--max_loaded=10'])
    assert conf.nsfw_checker
    assert conf.max_loaded_models==10
    assert conf.outdir==Path('outputs')  # this is the default
    
def test_env_override():
    # argv overrides 
    conf = InvokeAIAppConfig(conf=init1,argv=['--max_loaded=10'])
    assert conf.nsfw_checker==False
    
    os.environ['INVOKEAI_globals_nsfw_checker'] = 'True'
    conf = InvokeAIAppConfig(conf=init1,argv=['--max_loaded=10'])
    assert conf.nsfw_checker==True

    conf = InvokeAIAppConfig(conf=init1,argv=['--no-nsfw_checker','--max_loaded=10'])
    assert conf.nsfw_checker==False

    conf = InvokeAIAppConfig(conf=init1,argv=[],max_loaded_models=20)
    assert conf.max_loaded_models==20

    # have to comment this one out because of a race condition in setting same
    # environment variable in the CI test environment
    # assert conf.root==Path('/tmp')
    
def test_invocation():
    invocation = TextToImageInvocation(conf=init1,id='foobar')
    assert invocation.steps==18
    assert invocation.scheduler=='k_heun'
    assert invocation.height==512 # default

    invocation = TextToImageInvocation(conf=init1,id='foobar2',steps=30)
    assert invocation.steps==30

def test_type_coercion():
    conf = InvokeAIAppConfig(argv=['--root=/tmp/foobar'])
    assert conf.root==Path('/tmp/foobar')
    assert isinstance(conf.root,Path)
    conf = InvokeAIAppConfig(argv=['--root=/tmp/foobar'],root='/tmp/different')
    assert conf.root==Path('/tmp/different')
    assert isinstance(conf.root,Path)
