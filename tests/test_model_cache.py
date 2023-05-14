import pytest
import torch

from enum import Enum
from invokeai.backend.model_management.model_cache import ModelCache, MODEL_CLASSES

class DummyModelBase(object):
    '''Base class for dummy component of a diffusers model'''
    def __init__(self, repo_id):
        self.repo_id = repo_id
        self.device = torch.device('cpu')

    @classmethod
    def from_pretrained(cls,
                        repo_id:str,
                        revision:str=None,
                        subfolder:str=None,
                        cache_dir:str=None,
                        ):
        return cls(repo_id)
        
    def to(self, device):
        self.device = device

class DummyModelType1(DummyModelBase):
    pass

class DummyModelType2(DummyModelBase):
    pass

class DummyPipeline(DummyModelBase):
    '''Dummy pipeline object is a composite of several types'''
    def __init__(self,repo_id):
        super().__init__(repo_id)
        self.dummy_model_type1 = DummyModelType1('dummy/type1')
        self.dummy_model_type2 = DummyModelType2('dummy/type2')

class DMType(str, Enum):
    dummy_pipeline = 'dummy_pipeline'
    type1 = 'dummy_model_type1'
    type2 = 'dummy_model_type2'

MODEL_CLASSES.update(
    {
        DMType.dummy_pipeline: DummyPipeline,
        DMType.type1: DummyModelType1,
        DMType.type2: DummyModelType2,
    }
)

cache = ModelCache(max_cache_size=4)

def test_pipeline_fetch():
    assert cache.cache_size()==0
    with cache.get_model('dummy/pipeline1',DMType.dummy_pipeline) as pipeline1,\
         cache.get_model('dummy/pipeline1',DMType.dummy_pipeline) as pipeline1a,\
         cache.get_model('dummy/pipeline2',DMType.dummy_pipeline) as pipeline2:
        assert pipeline1 is not None, 'get_model() should not return None'
        assert pipeline1a is not None, 'get_model() should not return None'
        assert pipeline2 is not None, 'get_model() should not return None'
        assert type(pipeline1)==DummyPipeline,'get_model() did not return model of expected type'
        assert pipeline1==pipeline1a,'pipelines with the same repo_id should be the same'
        assert pipeline1!=pipeline2,'pipelines with different repo_ids should not be the same'
        assert len(cache.models)==2,'cache should uniquely cache models with same identity'
    with cache.get_model('dummy/pipeline3',DMType.dummy_pipeline) as pipeline3,\
         cache.get_model('dummy/pipeline4',DMType.dummy_pipeline) as pipeline4:
        assert len(cache.models)==4,'cache did not grow as expected'

def test_signatures():
    with cache.get_model('dummy/pipeline',DMType.dummy_pipeline,revision='main') as pipeline1,\
         cache.get_model('dummy/pipeline',DMType.dummy_pipeline,revision='fp16') as pipeline2,\
         cache.get_model('dummy/pipeline',DMType.dummy_pipeline,revision='main',subfolder='foo') as pipeline3:
        assert pipeline1 != pipeline2,'models are distinguished by their revision'
        assert pipeline1 != pipeline3,'models are distinguished by their subfolder'

def test_pipeline_device():
     with cache.get_model('dummy/pipeline1',DMType.type1) as model1:
         assert model1.device==torch.device('cuda'),'when in context, model device should be in GPU'
     with cache.get_model('dummy/pipeline1',DMType.type1, gpu_load=False) as model1:
         assert model1.device==torch.device('cpu'),'when gpu_load=False, model device should be CPU'

def test_submodel_fetch():
    with cache.get_model(repo_id_or_path='dummy/pipeline1',model_type=DMType.dummy_pipeline) as pipeline,\
         cache.get_model(repo_id_or_path='dummy/pipeline1',model_type=DMType.dummy_pipeline,submodel=DMType.type1) as part1,\
         cache.get_model(repo_id_or_path='dummy/pipeline2',model_type=DMType.dummy_pipeline,submodel=DMType.type1) as part2:
        assert type(part1)==DummyModelType1,'returned submodel is not of expected type'
        assert part1.device==torch.device('cuda'),'returned submodel should be in the GPU when in context'
        assert pipeline.dummy_model_type1==part1,'returned submodel should match the corresponding subpart of parent model'
        assert pipeline.dummy_model_type1!=part2,'returned submodel should not match the subpart of a different parent'

