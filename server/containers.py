"""Containers module."""

from dependency_injector import containers, providers
from ldm.generate import Generate
from server import services

class Container(containers.DeclarativeContainer):
  wiring_config = containers.WiringConfiguration(packages=['server'])

  config = providers.Configuration()

  model_singleton = providers.Singleton(
    Generate,
    width = config.model.width,
    height = config.model.height,
    sampler_name = config.model.sampler_name,
    weights = config.model.weights,
    full_precision = config.model.full_precision,
    config = config.model.config,
    grid = config.model.grid,
    seamless = config.model.seamless,
    embedding_path = config.model.embedding_path,
    device_type = config.model.device_type
  )

  # TODO: get location from config
  image_storage_service = providers.Singleton(
    services.ImageStorageService,
    './outputs/img-samples/'
  )

  # TODO: get location from config
  image_intermediates_storage_service = providers.Singleton(
    services.ImageStorageService,
    './outputs/intermediates/'
  )

  queue_service = providers.Singleton(
    services.JobQueueService
  )

  # TODO: get locations from config
  log_service = providers.Singleton(
    services.LogService,
    './outputs/img-samples/',
    'dream_web_log.txt'
  )

  generator_service = providers.Singleton(
    services.GeneratorService,
    model = model_singleton,
    queue = queue_service,
    imageStorage = image_storage_service,
    intermediateStorage = image_intermediates_storage_service,
    log = log_service
  )
