import transformers
import json
import base64
import mimetypes
import os
import sys
import time
import eventlet
from threading import Event
from enum import Enum

from pathlib import Path
from pytorch_lightning import logging
from flask import Flask, send_from_directory, url_for, jsonify
from flask_socketio import SocketIO
from ldm.generate import Generate
from ldm.dream.pngwriter import PngWriter
from ldm.gfpgan.gfpgan_tools import gfpgan_model_exists


class CanceledException(Exception):
    pass


canceled = Event()

app = Flask(__name__, static_url_path='', static_folder='../frontend/dist/')

# serve generated images
app.config['OUTPUTS_FOLDER'] = "../outputs"


@app.route('/outputs/<path:filename>')
def outputs(filename):
    return send_from_directory(
        app.config['OUTPUTS_FOLDER'],
        filename
    )


# serve the vite build
@app.route("/", defaults={'path': ''})
def serve(path):
    return send_from_directory(app.static_folder, 'index.html')


host = 'localhost'
port = 9090

dev_mode = True

logger = True if dev_mode else False
engineio_logger = True if dev_mode else False
cors_allowed_origins = "*" if dev_mode else None

# default 1,000,000, needs to be higher for socketio to accept larger images
max_http_buffer_size = 10000000

socketio = SocketIO(app, cors_allowed_origins=cors_allowed_origins,
                    logger=logger, engineio_logger=logger, max_http_buffer_size=max_http_buffer_size)

transformers.logging.set_verbosity_error()

# initialize with defaults, we will populate all config
t2i = Generate()

# gets rid of annoying messages about random seed
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

tic = time.time()
t2i.load_model()
print(f'>> model loaded in', '%4.2fs' % (time.time() - tic))

print(f"\nServer online: http://{host}:{port}")


def make_reponse(status, message=None, data=None):
    response = {"status": status}
    if message is not None:
        response["message"] = message
    if data is not None:
        response["data"] = data
    return response


@socketio.on('cancel')
def handleCancel():
    canceled.set()
    return make_reponse("OK")


@socketio.on('generateImage')
def handle_generate_image(data):
    generate_image(data)
    return make_reponse("OK")


@socketio.on('requestAllImages')
def handle_request_all_images():
    paths = sorted(Path("outputs/img-samples").glob("*.png"),
                   key=os.path.getmtime)
    relative_paths = []
    for p in paths:
        relative_paths.append(str(p.relative_to('.')))
    return make_reponse("OK", data=relative_paths)


# TODO: I think this needs a safety mechanism.
@socketio.on('deleteImage')
def handle_delete_image(path):
    Path(path).unlink()
    return make_reponse("OK")


# TODO: I think this needs a safety mechanism.
@socketio.on('uploadInitialImage')
def handle_upload_initial_image(bytes, name):
    filePath = f'outputs/init-images/{name}'
    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    newFile = open(filePath, "wb")
    newFile.write(bytes)
    return make_reponse("OK", data=filePath)


# TODO: I think this needs a safety mechanism.
@socketio.on('uploadMask')
def handle_upload_initial_image(bytes, name):
    filePath = f'outputs/mask-images/{name}'
    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    newFile = open(filePath, "wb")
    newFile.write(bytes)
    return make_reponse("OK", data=filePath)


def generate_image(metadata):
    canceled.clear()
    prompt = metadata['prompt']
    strength = float(metadata['img2imgStrength'])
    iterations = int(metadata['imagesToGenerate'])
    steps = int(metadata['steps'])
    width = int(metadata['width'])
    height = int(metadata['height'])
    fit = False
    cfgscale = float(metadata['cfgScale'])
    sampler_name = metadata['sampler']
    gfpgan_strength = float(metadata['gfpganStrength']
                            ) if gfpgan_model_exists else 0
    upscale_level = metadata['upscalingLevel']
    upscale_strength = metadata['upscalingStrength']
    upscale = [int(upscale_level), float(upscale_strength)
               ] if upscale_level != 0 else None
    progress_images = False
    seed = t2i.seed if int(metadata['seed']) == -1 else int(metadata['seed'])
    init_img = metadata['initialImagePath']
    fit = metadata['shouldFitToWidthHeight']

    pngwriter = PngWriter("./outputs/img-samples/")
    prefix = pngwriter.unique_prefix()
    mask = metadata["maskPath"]
    seamless = metadata["seamless"]

    def image_progress(sample, step):
        if canceled.is_set():
            raise CanceledException
        socketio.emit('progress', {"step": step})
        eventlet.sleep(0)

    def image_done(image, seed, upscaled=False):
        name = f'{prefix}.{seed}.png'
        path = pngwriter.save_image_and_prompt_to_png(image, f'{prompt} -S{seed}', name)
        if not upscaled:
            socketio.emit(
                'result', {'url': os.path.relpath(path), 'metadata': metadata})
            eventlet.sleep(0)

            # params yet to support
            # ddim_eta       =    None,
            # skip_normalize =    False,
            # log_tokenization=  False,
            # with_variations =   None,
            # variation_amount =  0.0,
            # # these are specific to img2img
            # invert_mask    =    False,
            # # these are specific to GFPGAN/ESRGAN
            # save_original  =    False,

    t2i.prompt2image(prompt,
                     iterations=iterations,
                     init_img=init_img,
                     mask=mask,
                     cfg_scale=cfgscale,
                     width=width,
                     height=height,
                     seed=seed,
                     steps=steps,
                     gfpgan_strength=gfpgan_strength,
                     upscale=upscale,
                     sampler_name=sampler_name,
                     step_callback=image_progress,
                     strength=strength,
                     image_callback=image_done,
                     fit=fit,
                     seamless=seamless)


if __name__ == '__main__':
    socketio.run(app, host=host, port=port)
