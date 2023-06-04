import os
import pytest
import sys

from omegaconf import OmegaConf
from pathlib import Path

os.environ['INVOKEAI_ROOT']='/tmp'
sys.argv = []  # to prevent config from trying to parse pytest arguments

from invokeai.app.services.config import InvokeAIAppConfig, InvokeAISettings
from invokeai.app.invocations.generate import TextToImageInvocation


init1 = OmegaConf.create(
'''
InvokeAI:
  Features:
    nsfw_checker: False
  Memory/Performance:
    max_loaded_models: 5
'''
)

init2 = OmegaConf.create(
'''
InvokeAI:
  Features:
    nsfw_checker: true
  Memory/Performance:
    max_loaded_models: 2
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
    
    os.environ['INVOKEAI_nsfw_checker'] = 'True'
    conf = InvokeAIAppConfig(conf=init1,argv=['--max_loaded=10'])
    assert conf.nsfw_checker==True

    # environment variables should be case insensitive
    os.environ['InvokeAI_Max_Loaded_Models'] = '15'
    conf = InvokeAIAppConfig(conf=init1)
    assert conf.max_loaded_models == 15

    conf = InvokeAIAppConfig(conf=init1,argv=['--no-nsfw_checker','--max_loaded=10'])
    assert conf.nsfw_checker==False
    assert conf.max_loaded_models==10

    conf = InvokeAIAppConfig(conf=init1,argv=[],max_loaded_models=20)
    assert conf.max_loaded_models==20

def test_type_coercion():
    conf = InvokeAIAppConfig(argv=['--root=/tmp/foobar'])
    assert conf.root==Path('/tmp/foobar')
    assert isinstance(conf.root,Path)
    conf = InvokeAIAppConfig(argv=['--root=/tmp/foobar'],root='/tmp/different')
    assert conf.root==Path('/tmp/different')
    assert isinstance(conf.root,Path)
