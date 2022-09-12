import json
import string
from copy import deepcopy
from datetime import datetime, timezone

class DreamRequest():
  prompt: string
  initimg: string
  strength: float
  iterations: int
  steps: int
  width: int
  height: int
  fit = None
  cfgscale: float
  sampler_name: string
  gfpgan_strength: float
  upscale_level: int
  upscale_strength: float
  upscale: None
  progress_images = None
  seed: int
  time: int

  # TODO: use signals/events for progress instead
  start_callback = None
  progress_callback = None
  image_callback = None
  cancelled_callback = None
  done_callback = None

  def id(self, seed = None, upscaled = False) -> str:
    return f"{self.time}.{seed or self.seed}{'.u' if upscaled else ''}"

  # TODO: handle this more cleanly (probably by splitting this into a Job and Result class)
  # TODO: Set iterations to 1 or remove it from the dream result? And just keep it on the job?
  def clone_without_image(self, seed = None):
    data = deepcopy(self)
    #data = copy(self.__dict__)
    data.initimg = None
    data.start_callback = None
    data.progress_callback = None
    data.image_callback = None
    data.cancelled_callback = None
    data.done_callback = None
    if seed:
      data.seed = seed

    return data

  def to_json(self, seed: int = None):
    copy = self.clone_without_image(seed)
    return json.dumps(copy.__dict__)

  @staticmethod
  def from_json(j, newTime: bool = False):
    d = DreamRequest()
    d.prompt = j.get('prompt')
    d.initimg = j.get('initimg')
    d.strength = float(j.get('strength'))
    d.iterations = int(j.get('iterations'))
    d.steps = int(j.get('steps'))
    d.width = int(j.get('width'))
    d.height = int(j.get('height'))
    d.fit    = 'fit' in j
    d.seamless = 'seamless' in j
    d.cfgscale = float(j.get('cfgscale'))
    d.sampler_name  = j.get('sampler')
    d.variation_amount = float(j.get('variation_amount'))
    d.with_variations = j.get('with_variations')
    d.gfpgan_strength = float(j.get('gfpgan_strength'))
    d.upscale_level    = j.get('upscale_level')
    d.upscale_strength = j.get('upscale_strength')
    d.upscale = [int(d.upscale_level),float(d.upscale_strength)] if d.upscale_level != '' else None
    d.progress_images = 'progress_images' in j
    d.seed = int(j.get('seed'))
    d.time = int(datetime.now(timezone.utc).timestamp()) if newTime else int(j.get('time'))
    return d


class Signal():
  event: str
  data = None
  room: str = None
  broadcast: bool = False

  def __init__(self, event: str, data, room: str = None, broadcast: bool = False):
    self.event = event
    self.data = data
    self.room = room
    self.broadcast = broadcast

  @staticmethod
  def image_progress(jobId: str, dreamId: str, step: int, totalSteps: int, hasProgressImage: bool = False):
    return Signal('dream_progress', {
      'jobId': jobId,
      'dreamId': dreamId,
      'step': step,
      'totalSteps': totalSteps,
      'hasProgressImage': hasProgressImage
    }, room=jobId, broadcast=True)

  # TODO: use a result id or something? Like a sub-job
  @staticmethod
  def image_result(jobId: str, dreamId: str, dreamRequest: DreamRequest):
    return Signal('dream_result', {
      'jobId': jobId,
      'dreamId': dreamId,
      'dreamRequest': dreamRequest.__dict__
    }, room=jobId, broadcast=True)

  @staticmethod
  def job_started(jobId: str):
    return Signal('job_started', { 'jobId': jobId }, room=jobId, broadcast=True)
    
  @staticmethod
  def job_done(jobId: str):
    return Signal('job_done', { 'jobId': jobId }, room=jobId, broadcast=True)

  @staticmethod
  def job_canceled(jobId: str):
    return Signal('job_canceled', { 'jobId': jobId }, room=jobId, broadcast=True)
