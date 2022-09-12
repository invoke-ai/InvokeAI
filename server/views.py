"""Views module."""
import json
import os
from queue import Queue
from flask import current_app, jsonify, request, Response, send_from_directory, stream_with_context, url_for
from flask.views import MethodView
from dependency_injector.wiring import inject, Provide

from server.models import DreamRequest
from server.services import GeneratorService, ImageStorageService, JobQueueService
from server.containers import Container

class ApiJobs(MethodView):

  @inject
  def post(self, job_queue_service: JobQueueService = Provide[Container.generation_queue_service]):
    dreamRequest = DreamRequest.from_json(request.json, newTime = True)

    #self.canceled.clear()
    print(f">> Request to generate with prompt: {dreamRequest.prompt}")

    q = Queue()

    dreamRequest.start_callback = None
    dreamRequest.image_callback = None
    dreamRequest.progress_callback = None
    dreamRequest.cancelled_callback = None
    dreamRequest.done_callback = None

    # Push the request
    job_queue_service.push(dreamRequest)

    return { 'dreamId': dreamRequest.id() }
  

class WebIndex(MethodView):
  init_every_request = False
  __file: str = None
  
  def __init__(self, file):
    self.__file = file

  def get(self):
    return current_app.send_static_file(self.__file)

  @inject
  def post(self, job_queue_service: JobQueueService = Provide[Container.generation_queue_service]):
    dreamRequest = DreamRequest.from_json(request.json, newTime = True)

    #self.canceled.clear()
    print(f">> Request to generate with prompt: {dreamRequest.prompt}")

    q = Queue()

    images_generated = 0    # helps keep track of when upscaling is started
    images_upscaled = 0     # helps keep track of when upscaling is completed

    # if upscaling is requested, then this will be called twice, once when
    # the images are first generated, and then again when after upscaling
    # is complete. The upscaling replaces the original file, so the second
    # entry should not be inserted into the image list.
    def image_done(imgpath, dreamRequest, seed, upscaled=False):
        q.put({
          'type': 'result',
          'data': {'event': 'result', 'url': imgpath, 'seed': seed or dreamRequest.seed, 'config': dreamRequest.data_without_image(seed)}
        })

        # TODO: handle eventing around upscale differently
        # control state of the "postprocessing..." message
        upscaling_requested = dreamRequest.upscale or dreamRequest.gfpgan_strength>0
        nonlocal images_generated # NB: Is this bad python style? It is typical usage in a perl closure.
        nonlocal images_upscaled  # NB: Is this bad python style? It is typical usage in a perl closure.
        if upscaled:
            images_upscaled += 1
        else:
            images_generated +=1
        if upscaling_requested:
            action = None
            if images_generated >= dreamRequest.iterations:
                if images_upscaled < dreamRequest.iterations:
                    action = 'upscaling-started'
                else:
                    action = 'upscaling-done'
            if action:
                x = images_upscaled+1
                q.put({
                  'type': 'progress' if (action == 'upscaling-started') else 'done',
                  'data': {'event':action,'processed_file_cnt':f'{x}/{dreamRequest.iterations}'}
                })

    def image_progress(step, imgpath):
      # if self.canceled.is_set(): # TODO: Handle cancellation
      #     self.wfile.write(bytes(json.dumps({'event':'canceled'}) + '\n', 'utf-8'))
      #     raise CanceledException

      q.put({
        'type': 'progress',
        'data': {'event': 'step', 'step': step + 1, 'url': imgpath}
      })

    def image_canceled():
      q.put({
        'type': 'canceled',
        'data': {'event': 'canceled' }
      })

    def done():
      q.put({ 'type': 'done' })

    def start():
      q.put({ 'type': 'started' })


    dreamRequest.start_callback = start
    dreamRequest.image_callback = image_done
    dreamRequest.progress_callback = image_progress
    dreamRequest.cancelled_callback = image_canceled
    dreamRequest.done_callback = done

    # Push the request
    print('pushing job')
    job_queue_service.push(dreamRequest)
    print('job pushed')

    # Write responses
    def generateResponse():
      print('generating response')
      yield f"SEND"
      print('sent dummy event') 
      while True:
        event = q.get()

        if event['type'] == 'started':
          yield f"{json.dumps({'event': 'started', 'dreamId': dreamRequest.id() })}\n"

        if event['type'] == 'progress':
          yield f"{json.dumps(event['data'])}\n"

        elif event['type'] == 'result':
          yield f"{json.dumps(event['data'])}\n"

        elif event['type'] == 'canceled':
          yield f"{json.dumps(event['data'])}\n"
          break

        elif event['type'] == 'done':
          break

    return Response(stream_with_context(generateResponse()))
  

class WebConfig(MethodView):
  init_every_request = False

  def get(self):
    # unfortunately this import can't be at the top level, since that would cause a circular import
    from ldm.gfpgan.gfpgan_tools import gfpgan_model_exists
    config = {
        'gfpgan_model_exists': gfpgan_model_exists
    }
    js = f"let config = {json.dumps(config)};\n"
    return Response(js, mimetype="application/javascript")


class ApiCancel(MethodView):
  init_every_request = False
  
  @inject
  def get(self, generator_service: GeneratorService = Provide[Container.generator_service]):
    generator_service.cancel()
    return Response(status=204)


class ApiImages(MethodView):
  init_every_request = False
  __pathRoot = None
  __storage: ImageStorageService

  @inject
  def __init__(self, pathBase, storage: ImageStorageService = Provide[Container.image_storage_service]):
    self.__pathRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), pathBase))
    self.__storage = storage

  def get(self, dreamId):
    name = self.__storage.path(dreamId)
    fullpath=os.path.join(self.__pathRoot, name)
    return send_from_directory(os.path.dirname(fullpath), os.path.basename(fullpath))

class ApiIntermediates(MethodView):
  init_every_request = False
  __pathRoot = None
  __storage: ImageStorageService = Provide[Container.image_intermediates_storage_service]

  @inject
  def __init__(self, pathBase, storage: ImageStorageService = Provide[Container.image_intermediates_storage_service]):
    self.__pathRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), pathBase))
    self.__storage = storage

  def get(self, dreamId, step):
    name = self.__storage.path(dreamId, postfix=f'.{step}')
    fullpath=os.path.join(self.__pathRoot, name)
    return send_from_directory(os.path.dirname(fullpath), os.path.basename(fullpath))
