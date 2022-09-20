# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

"""Application module."""
import argparse
import json
import os
import sys
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from omegaconf import OmegaConf
from dependency_injector.wiring import inject, Provide
from ldm.dream.args import Args
from ldm.dream.app.services.generation.services import GeneratorService
from ldm.dream.app.services.signaling.services import SignalService
from ldm.dream.app.api.containers import Container
from ldm.dream.app.api.views import (
    ApiCancel,
    ApiImages,
    ApiImagesList,
    ApiImagesMetadata,
    ApiIntermediates,
    ApiJobs,
    WebConfig,
    WebIndex,
)

# The socketio_service is injected here (rather than created in run_app) to initialize it
@inject
def initialize_app(
    app: Flask, socketio: SocketIO = Provide[Container.signaling_package.socketio]
) -> SocketIO:
    socketio.init_app(app)

    return socketio


# The signal and generator services are injected to warm up the processing queues
# TODO: Initialize these a better way?
@inject
def initialize_generator(
    signal_service: SignalService = Provide[Container.signaling_package.signal_service],
    generator_service: GeneratorService = Provide[
        Container.generator_package.generator_service
    ],
):
    pass


def add_routes(app: Flask):
    # Web Routes
    app.add_url_rule("/", view_func=WebIndex.as_view("web_index", "index.html"))
    app.add_url_rule(
        "/index.css", view_func=WebIndex.as_view("web_index_css", "index.css")
    )
    app.add_url_rule(
        "/index.js", view_func=WebIndex.as_view("web_index_js", "index.js")
    )
    app.add_url_rule("/config.js", view_func=WebConfig.as_view("web_config"))

    # API Routes
    app.add_url_rule("/api/jobs", view_func=ApiJobs.as_view("api_jobs"))
    app.add_url_rule("/api/cancel", view_func=ApiCancel.as_view("api_cancel"))

    # TODO: Get storage root from config
    app.add_url_rule(
        "/api/images/<string:dreamId>", view_func=ApiImages.as_view("api_images", "../../../../")
    )
    app.add_url_rule(
        "/api/images/<string:dreamId>/metadata",
        view_func=ApiImagesMetadata.as_view("api_images_metadata", "../../../../"),
    )
    app.add_url_rule("/api/images", view_func=ApiImagesList.as_view("api_images_list"))
    app.add_url_rule(
        "/api/intermediates/<string:dreamId>/<string:step>",
        view_func=ApiIntermediates.as_view("api_intermediates", "../../../../"),
    )

    app.static_folder = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../static/dream_web/")
    )


def configure_logging():
    # these two lines prevent a horrible warning message from appearing
    # when the frozen CLIP tokenizer is imported
    import transformers

    transformers.logging.set_verbosity_error()

    # gets rid of annoying messages about random seed
    from pytorch_lightning import logging

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def run_web_app(config) -> Flask:
    configure_logging()

    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

    # Change working directory to the stable-diffusion directory
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

    # Create the flask app
    app = Flask(__name__, static_url_path="")

    # Set up dependency injection container
    container = Container()
    container.config.from_dict(config)
    container.generator_package.config.from_dict(config)
    container.wire(modules=[__name__])
    app.container = container

    # Set up CORS
    cors = config.get("cors")
    if cors:
        print(f"Enabling CORS on origin {cors}")
        CORS(app, resources={r"/api/*": {"origins": cors}})

    # Add routes
    add_routes(app)

    # Initialize
    socketio = initialize_app(app)
    initialize_generator()

    host = config["host"]
    port = config["port"]

    print(">> Started Stable Diffusion api server!")
    if host == "0.0.0.0":
        print(
            f"Point your browser at http://localhost:{port} or use the host's DNS name or IP address."
        )
    else:
        print(
            ">> Default host address now 127.0.0.1 (localhost). Use --host 0.0.0.0 to bind any address."
        )
        print(f">> Point your browser at http://{host}:{port}.")

    # Run the app
    socketio.run(app, host, port)


if __name__ == "__main__":
    """Load configuration and run application"""
    arg_parser = Args()
    opt = arg_parser.parse_args()
    appConfig = opt.__dict__

    # Start server
    try:
        run_web_app(appConfig)
    except KeyboardInterrupt:
        pass
