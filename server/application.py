"""Application module."""
import argparse
import os
import sys
from flask import Flask
from flask_cors import CORS
from omegaconf import OmegaConf
from dependency_injector.wiring import inject, Provide
from server import views
from server.containers import Container
from server.services import GeneratorService

# Initialize the generator so it warms up the model and starts processing from the queue
@inject
def initialize_app(generator_service: GeneratorService = Provide[Container.generator_service]) -> None:
  pass

def create_app(config) -> Flask:
  container = Container()
  container.config.from_dict(config)
  container.wire(modules=[__name__])

  app = Flask(__name__, static_url_path='')
  app.container = container

  # Set up CORS
  CORS(app, resources={r'/api/*': {'origins': '*'}})

  # Web Routes
  app.add_url_rule('/', view_func=views.WebIndex.as_view('web_index', 'index.html'))
  app.add_url_rule('/index.css', view_func=views.WebIndex.as_view('web_index_css', 'index.css'))
  app.add_url_rule('/index.js', view_func=views.WebIndex.as_view('web_index_js', 'index.js'))
  app.add_url_rule('/config.js', view_func=views.WebConfig.as_view('web_config'))

  # API Routes
  app.add_url_rule('/api/cancel', view_func=views.ApiCancel.as_view('api_cancel'))

  # TODO: Get storage root from config
  app.add_url_rule('/api/images/<string:dreamId>', view_func=views.ApiImages.as_view('api_images', '../'))
  app.add_url_rule('/api/intermediates/<string:dreamId>/<string:step>', view_func=views.ApiIntermediates.as_view('api_intermediates', '../'))

  app.static_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../static/dream_web/')) 

  # Initialize
  initialize_app()

  return app

def main():
  """Initialize command-line parsers and the diffusion model"""
  from scripts.dream import create_argv_parser
  arg_parser = create_argv_parser()
  opt = arg_parser.parse_args()

  if opt.laion400m:
    print('--laion400m flag has been deprecated. Please use --model laion400m instead.')
    sys.exit(-1)
  if opt.weights != 'model':
    print('--weights argument has been deprecated. Please configure ./configs/models.yaml, and call it using --model instead.')
    sys.exit(-1)
      
  try:
    models  = OmegaConf.load(opt.config)
    width   = models[opt.model].width
    height  = models[opt.model].height
    config  = models[opt.model].config
    weights = models[opt.model].weights
  except (FileNotFoundError, IOError, KeyError) as e:
    print(f'{e}. Aborting.')
    sys.exit(-1)

  print('* Initializing, be patient...\n')
  sys.path.append('.')
  from pytorch_lightning import logging

  # these two lines prevent a horrible warning message from appearing
  # when the frozen CLIP tokenizer is imported
  import transformers

  transformers.logging.set_verbosity_error()

  appConfig = {
    "model": {
      "width": width,
      "height": height,
      "sampler_name": opt.sampler_name,
      "weights": weights,
      "full_precision": opt.full_precision,
      "config": config,
      "grid": opt.grid,
      "latent_diffusion_weights": opt.laion400m,
      "embedding_path": opt.embedding_path,
      "device_type": opt.device
    }
  }

  # make sure the output directory exists
  if not os.path.exists(opt.outdir):
    os.makedirs(opt.outdir)

  # gets rid of annoying messages about random seed
  logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

  print('\n* starting api server...')
  # Change working directory to the stable-diffusion directory
  os.chdir(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
  )

  # Start server
  flask = create_app(appConfig)

  print(">> Started Stable Diffusion api server!")
  if opt.host == '0.0.0.0':
    print(f"Point your browser at http://localhost:{opt.port} or use the host's DNS name or IP address.")
  else:
    print(">> Default host address now 127.0.0.1 (localhost). Use --host 0.0.0.0 to bind any address.")
    print(f">> Point your browser at http://{opt.host}:{opt.port}.")

  try:
    flask.run(opt.host, opt.port)
  except KeyboardInterrupt:
    pass


if __name__ == '__main__':
  main()
