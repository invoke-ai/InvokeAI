from ldm.gfpgan.gfpgan_tools import gfpgan_model_exists
from ldm.dream.pngwriter import PngWriter
from ldm.generate import Generate
from flask_socketio import SocketIO
from flask import Flask, send_from_directory, url_for, jsonify
import transformers
import json
import base64
import mimetypes
import os
import sys
import signal
import time
import eventlet
import traceback
from threading import Event
from enum import Enum

from pathlib import Path
from pytorch_lightning import logging
from parse_seed_weights import parse_seed_weights

# fix missing mimetypes on windows due to registry wonkiness
import mimetypes
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')


host = 'localhost'
port = 9090


app = Flask(__name__, static_url_path='', static_folder='../frontend/dist/')


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


dev_mode = False

logger = True if dev_mode else False
engineio_logger = True if dev_mode else False

# default 1,000,000, needs to be higher for socketio to accept larger images
max_http_buffer_size = 10000000

socketio = SocketIO(app,
                    logger=logger, engineio_logger=logger, max_http_buffer_size=max_http_buffer_size)


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


class CanceledException(Exception):
    pass


canceled = Event()

transformers.logging.set_verbosity_error()

# initialize with defaults, we will populate all config
model = Generate()

# gets rid of annoying messages about random seed
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

tic = time.time()
model.load_model()
print(f'>> model loaded in', '%4.2fs' % (time.time() - tic))

print(f"\nServer online: http://{host}:{port}")


def make_reponse(status, message=None, data=None):
    response = {"status": status}
    if message is not None:
        response["message"] = message
    if data is not None:
        response["data"] = data
    return response


def generate_image(data):
    canceled.clear()
    prompt = data['prompt']
    strength = float(data['img2imgStrength'])
    iterations = int(data['iterations'])
    steps = int(data['steps'])
    width = int(data['width'])
    height = int(data['height'])
    fit = False
    cfgscale = float(data['cfgScale'])
    sampler_name = data['sampler']
    gfpgan_strength = float(data['gfpganStrength']
                            ) if gfpgan_model_exists else 0
    upscale_level = data['upscalingLevel']
    upscale_strength = data['upscalingStrength']
    upscale = [int(upscale_level), float(upscale_strength)
               ] if upscale_level != 0 else None
    progress_images = False
    seed = model.seed if int(data['seed']) == -1 else int(data['seed'])
    init_img = data['initialImagePath']
    fit = data['shouldFitToWidthHeight']

    pngwriter = PngWriter("./outputs/img-samples/")
    prefix = pngwriter.unique_prefix()
    mask = data["maskPath"]
    seamless = data["seamless"]
    progress_images = data["shouldDisplayInProgress"]

    step_writer = PngWriter("outputs/intermediates")
    step_index = 1

    with_variations = None
    variation_amount = data["variantAmount"]

    seed_weights = parse_seed_weights(data["seedWeights"])

    if data["shouldGenerateVariations"] and seed_weights is not False:
        with_variations = seed_weights

    def image_progress(sample, step):
        if canceled.is_set():
            raise CanceledException

        nonlocal step_index
        if progress_images and step % 5 == 0 and step < steps - 1:
            image = model.sample_to_image(sample)
            name = f'{prefix}.{seed}.{step_index}.png'
            metadata = f'{prompt} -S{seed} [intermediate]'
            path = step_writer.save_image_and_prompt_to_png(
                image, metadata, name)
            step_index += 1
            socketio.emit('intermediateResult', {
                          'url': os.path.relpath(path), 'metadata': data})
        socketio.emit('progress', {"step": step + 1})
        eventlet.sleep(0)

    def image_done(image, seed, upscaled=False):
        filename = f'{prefix}.{seed}.png'
        path = pngwriter.save_image_and_prompt_to_png(image, f'{prompt} -S{seed}', filename)
        if not upscaled:
            # We may have passed -1 as seed to server, need actual seed for UI
            data["seed"] = seed
            socketio.emit(
                'result', {'url': os.path.relpath(path), 'metadata': data})
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

    try:
        print(with_variations)
        model.prompt2image(prompt,
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
                           strength=strength,
                           fit=fit,
                           seamless=seamless,
                           progress_images=progress_images,
                           with_variations=with_variations,
                           variation_amount=variation_amount,
                           step_callback=image_progress,
                           image_callback=image_done)

    except KeyboardInterrupt:
        raise
    except CanceledException:
        raise
    except Exception as e:
        socketio.emit('error', (str(e)))
        print("\n")
        traceback.print_exc()
        print("\n")


if __name__ == '__main__':
    socketio.run(app, host=host, port=port)
