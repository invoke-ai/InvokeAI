import os
import pytest
import sys

from omegaconf import OmegaConf
from pathlib import Path

os.environ['INVOKEAI_ROOT']='/tmp'

from invokeai.app.services.config import InvokeAIAppConfig

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
    conf1 = InvokeAIAppConfig.get_config()
    assert conf1
    conf1.parse_args(conf=init1,argv=[])
    assert conf1.max_loaded_models==5
    assert not conf1.nsfw_checker

    conf2 = InvokeAIAppConfig.get_config()
    assert conf2
    conf2.parse_args(conf=init2,argv=[])
    assert conf2.nsfw_checker
    assert conf2.max_loaded_models==2
    assert not hasattr(conf2,'invalid_attribute')
    
def test_argv_override():
    conf = InvokeAIAppConfig.get_config()
    conf.parse_args(conf=init1,argv=['--nsfw_checker','--max_loaded=10'])
    assert conf.nsfw_checker
    assert conf.max_loaded_models==10
    assert conf.outdir==Path('outputs')  # this is the default
    
def test_env_override():
    # argv overrides 
    conf = InvokeAIAppConfig()
    conf.parse_args(conf=init1,argv=['--max_loaded=10'])
    assert conf.nsfw_checker==False
    os.environ['INVOKEAI_nsfw_checker'] = 'True'
    conf.parse_args(conf=init1,argv=['--max_loaded=10'])
    assert conf.nsfw_checker==True

    # environment variables should be case insensitive
    os.environ['InvokeAI_Max_Loaded_Models'] = '15'
    conf = InvokeAIAppConfig()
    conf.parse_args(conf=init1,argv=[])
    assert conf.max_loaded_models == 15

    conf = InvokeAIAppConfig()
    conf.parse_args(conf=init1,argv=['--no-nsfw_checker','--max_loaded=10'])
    assert conf.nsfw_checker==False
    assert conf.max_loaded_models==10

    conf = InvokeAIAppConfig.get_config(max_loaded_models=20)
    conf.parse_args(conf=init1,argv=[])
    assert conf.max_loaded_models==20

def test_type_coercion():
    conf = InvokeAIAppConfig().get_config()
    conf.parse_args(argv=['--root=/tmp/foobar'])
    assert conf.root==Path('/tmp/foobar')
    assert isinstance(conf.root,Path)
    conf = InvokeAIAppConfig.get_config(root='/tmp/different')
    conf.parse_args(argv=['--root=/tmp/foobar'])
    assert conf.root==Path('/tmp/different')
    assert isinstance(conf.root,Path)
