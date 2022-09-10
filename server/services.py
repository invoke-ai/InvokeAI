import base64
import os
from queue import Queue
from threading import Thread
import time
from ldm.dream.pngwriter import PngWriter
from ldm.dream.server import CanceledException
from ldm.generate import Generate
from server.models import DreamRequest

class JobQueueService:
  __queue: Queue = Queue()

  def push(self, dreamRequest: DreamRequest):
    self.__queue.put(dreamRequest)

  def get(self, timeout: float = None) -> DreamRequest:
    return self.__queue.get(timeout= timeout)

# TODO: Name this better?
class LogService:
  __location: str
  __logFile: str

  def __init__(self, location:str, file:str):
    self.__location = location
    self.__logFile = file

  def log(self, dreamRequest: DreamRequest, seed = None, upscaled = False):
    with open(os.path.join(self.__location, self.__logFile), "a") as log:
      log.write(f"{dreamRequest.id(seed, upscaled)}: {dreamRequest.to_json(seed)}\n")


class ImageStorageService:
  __location: str
  __pngWriter: PngWriter

  def __init__(self, location):
    self.__location = location
    self.__pngWriter = PngWriter(self.__location)

  def __getName(self, dreamId: str, postfix: str = '') -> str:
    return f'{dreamId}{postfix}.png'

  def save(self, image, dreamRequest, seed = None, upscaled = False, postfix: str = '', metadataPostfix: str = '') -> str:
    name = self.__getName(dreamRequest.id(seed, upscaled), postfix)
    path = self.__pngWriter.save_image_and_prompt_to_png(image, f'{dreamRequest.prompt} -S{seed or dreamRequest.seed}{metadataPostfix}', name)
    return path

  def path(self, dreamId: str, postfix: str = '') -> str:
    name = self.__getName(dreamId, postfix)
    path = os.path.join(self.__location, name)
    return path


class GeneratorService:
  __model: Generate
  __queue: JobQueueService
  __imageStorage: ImageStorageService
  __intermediateStorage: ImageStorageService
  __log: LogService
  __thread: Thread
  __cancellationRequested: bool = False

  def __init__(self, model: Generate, queue: JobQueueService, imageStorage: ImageStorageService, intermediateStorage: ImageStorageService, log: LogService):
    self.__model = model
    self.__queue = queue
    self.__imageStorage = imageStorage
    self.__intermediateStorage = intermediateStorage
    self.__log = log

    # Create the background thread
    self.__thread = Thread(target=self.__process)
    self.__thread.daemon = True
    self.__thread.start()


  # Request cancellation of the current job
  def cancel(self):
    self.__cancellationRequested = True


  # TODO: Consider moving this to its own service if there's benefit in separating the generator
  def __process(self):
    # preload the model
    print('Preloading model')

    tic = time.time()
    self.__model.load_model()
    print(
      f'>> model loaded in', '%4.2fs' % (time.time() - tic)
    )

    print('Started queue processor')
    try:
      while True:
        dreamRequest = self.__queue.get()
        self.__generate(dreamRequest)

    except KeyboardInterrupt:
        print('Queue processor stopped')


  def __done(self, dreamRequest: DreamRequest, image, seed, upscaled=False):
    self.__imageStorage.save(image, dreamRequest, seed, upscaled)
    
    # TODO: get api path from Flask
    imgpath = f"/api/images/{dreamRequest.id(seed, upscaled)}"

    # TODO: handle upscaling logic better (this is appending data to log, but only on first generation)
    if not upscaled:
      self.__log.log(dreamRequest, seed, upscaled)
    
    dreamRequest.image_callback(imgpath, dreamRequest, seed, upscaled)


  def __progress(self, dreamRequest, sample, step):
    if self.__cancellationRequested:
      self.__cancellationRequested = False
      raise CanceledException

    imgpath = None
    if dreamRequest.progress_images and step % 5 == 0 and step < dreamRequest.steps - 1:
      image = self.__model._sample_to_image(sample)
      self.__intermediateStorage.save(image, dreamRequest, self.__model.seed, f'.{step}', f' [intermediate]')
      imgpath = f"/api/intermediates/{dreamRequest.id(self.__model.seed)}/{step}"
    
    dreamRequest.progress_callback(step, imgpath)
  

  def __generate(self, dreamRequest: DreamRequest):
    try:
      initimgfile = None
      if dreamRequest.initimg is not None:
        with open("./img2img-tmp.png", "wb") as f:
          initimg = dreamRequest.initimg.split(",")[1] # Ignore mime type
          f.write(base64.b64decode(initimg))
          initimgfile = "./img2img-tmp.png"

      # Get a random seed if we don't have one yet
      # TODO: handle "previous" seed usage?
      if dreamRequest.seed == -1:
        dreamRequest.seed = self.__model.seed

      # Zero gfpgan strength if the model doesn't exist
      # TODO: determine if this could be at the top now? Used to cause circular import
      from ldm.gfpgan.gfpgan_tools import gfpgan_model_exists
      if not gfpgan_model_exists:
        dreamRequest.gfpgan_strength = 0

      self.__model.prompt2image(
        prompt           = dreamRequest.prompt,
        init_img         = initimgfile, # TODO: ensure this works
        strength         = None if initimgfile is None else dreamRequest.strength,
        fit              = None if initimgfile is None else dreamRequest.fit,
        iterations       = dreamRequest.iterations,
        cfg_scale        = dreamRequest.cfgscale,
        width            = dreamRequest.width,
        height           = dreamRequest.height,
        seed             = dreamRequest.seed,
        steps            = dreamRequest.steps,
        variation_amount = dreamRequest.variation_amount,
        with_variations  = dreamRequest.with_variations,
        gfpgan_strength  = dreamRequest.gfpgan_strength,
        upscale          = dreamRequest.upscale,
        sampler_name     = dreamRequest.sampler_name,
        seamless         = dreamRequest.seamless,
        step_callback    = lambda sample, step: self.__progress(dreamRequest, sample, step),
        image_callback   = lambda image, seed, upscaled=False: self.__done(dreamRequest, image, seed, upscaled))

    except CanceledException:
      dreamRequest.cancelled_callback()

    finally:
      dreamRequest.done_callback()
      
      # Remove the temp file
      if (initimgfile is not None):
        os.remove("./img2img-tmp.png")
