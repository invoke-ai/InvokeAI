import mimetypes
import transformers
import json
import re
import os
import traceback
import glob
import eventlet

from flask_socketio import SocketIO
from flask import Flask, send_from_directory, url_for, jsonify
from pathlib import Path
from PIL import Image
from pytorch_lightning import logging
from threading import Event

from ldm.gfpgan.gfpgan_tools import real_esrgan_upscale
from ldm.gfpgan.gfpgan_tools import run_gfpgan
from ldm.generate import Generate

from modules.parameters import make_generation_parameters, make_esrgan_parameters, make_gfpgan_parameters


output_dir = "outputs/"  # Base output directory for images
host = 'localhost'  # Web & socket.io host
port = 9090  # Web & socket.io port

"""
Additional CORS origins to this list.
Useful if allowing other machines on your network to use the app.

Example:
additional_allowed_origins = ["192.168.1.8","192.168.1.24"]
"""
additional_allowed_origins = []


# fix missing mimetypes on windows due to registry wonkiness
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')

app = Flask(__name__, static_url_path='', static_folder='../frontend/dist/')


app.config['OUTPUTS_FOLDER'] = "../outputs"


@app.route('/outputs/<path:filename>')
def outputs(filename):
    return send_from_directory(
        app.config['OUTPUTS_FOLDER'],
        filename
    )


@app.route("/", defaults={'path': ''})
def serve(path):
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/socketio_config')
def socketio_config():
    return json.dumps({'host': host, 'port': port})


# True enables more logging from socket.io
dev_mode = False

logger = True if dev_mode else False
engineio_logger = True if dev_mode else False

# default 1,000,000, needs to be higher for socketio to accept larger images
max_http_buffer_size = 10000000


cors_allowed_origins = [f"http://{host}:{port}"] + additional_allowed_origins

socketio = SocketIO(app,
                    logger=logger,
                    engineio_logger=logger,
                    max_http_buffer_size=max_http_buffer_size,
                    cors_allowed_origins=cors_allowed_origins
                    )


class CanceledException(Exception):
    pass


canceled = Event()

# reduce logging outputs to error
transformers.logging.set_verbosity_error()
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

# Initialize and load model
model = Generate()
model.load_model()


# location for "finished" images
result_path = os.path.join(output_dir, 'img-samples/')

# temporary path for intermediates
intermediate_path = os.path.join(output_dir, 'intermediates/')

# path for user-uploaded init images and masks
init_path = os.path.join(output_dir, 'init/')
mask_path = os.path.join(output_dir, 'mask/')

# make all output paths
[os.makedirs(path, exist_ok=True)
 for path in [result_path, intermediate_path, init_path, mask_path]]


@socketio.on('requestAllImages')
def handle_request_all_images():
    paths = list(filter(os.path.isfile, glob.glob(result_path + "*.png")))
    paths.sort(key=lambda x: os.path.getmtime(x))
    return make_response("OK", data=paths)


@socketio.on('generateImage')
def handle_generate_image_event(data):
    generate_images(
        data
    )

    return make_response("OK")


@socketio.on('runESRGAN')
def handle_run_esrgan_event(data):
    parameters = make_esrgan_parameters(data)
    image = Image.open(data["imagePath"])

    image = real_esrgan_upscale(
        image=image,
        **parameters,
    )

    path = save_image(
        image=image,
        seed=parameters["seed"],
        output_dir=result_path,
        postprocessing="esrgan"
    )

    data["seed"] = parameters["seed"]
    socketio.emit(
        'result', {'url': os.path.relpath(path), 'metadata': data})
    eventlet.sleep(0)


@socketio.on('runGFPGAN')
def handle_run_gfpgan_event(data):
    parameters = make_gfpgan_parameters(data)
    image = Image.open(data["imagePath"])

    image = run_gfpgan(
        image=image,
        ** parameters,
        upsampler_scale=1
    )

    path = save_image(
        image=image,
        seed=parameters["seed"],
        output_dir=result_path,
        postprocessing="gfpgan"
    )

    data["seed"] = parameters["seed"]
    socketio.emit(
        'result', {'url': os.path.relpath(path), 'metadata': data})
    eventlet.sleep(0)


# TODO: I think this needs a safety mechanism.
@socketio.on('deleteImage')
def handle_delete_image(path):
    Path(path).unlink()
    return make_response("OK")


# TODO: I think this needs a safety mechanism.
@socketio.on('uploadInitialImage')
def handle_upload_initial_image(bytes, name):
    filePath = f'outputs/init-images/{name}'
    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    newFile = open(filePath, "wb")
    newFile.write(bytes)
    return make_response("OK", data=filePath)


# TODO: I think this needs a safety mechanism.
@socketio.on('uploadMask')
def handle_upload_initial_image(bytes, name):
    filePath = f'outputs/mask-images/{name}'
    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    newFile = open(filePath, "wb")
    newFile.write(bytes)
    return make_response("OK", data=filePath)


def make_response(status, message=None, data=None):
    response = {'status': status}
    if message is not None:
        response['message'] = message
    if data is not None:
        response['data'] = data
    return response


def save_image(image, seed, output_dir, step_index=None, postprocessing=None):
    # Prefix logic from `ldm.dream.pngwriter.unique_prefix`
    # sort reverse alphabetically until we find max+1
    dirlist = sorted(os.listdir(output_dir), reverse=True)
    # find the first filename that matches our pattern or return 000000.0.png
    existing_name = next(
        (f for f in dirlist if re.match('^(\d+)\..*\.png', f)),
        '0000000.0.png',
    )
    basecount = int(existing_name.split('.', 1)[0]) + 1
    prefix = f'{basecount:06}'

    filename = f'{prefix}.{seed}'

    if step_index:
        filename += f'.{step_index}'
    if postprocessing:
        filename += f'.{postprocessing}'

    filename += '.png'

    filepath = os.path.join(output_dir, filename)
    image.save(filepath, 'PNG')

    return filepath


def generate_images(data):
    canceled.clear()

    step_index = 1

    parameters = make_generation_parameters(data)

    should_run_esrgan = data["shouldRunESRGAN"]
    if should_run_esrgan:
        esrgan_parameters = make_esrgan_parameters(data)

    should_run_gfpgan = data["shouldRunGFPGAN"]
    if should_run_gfpgan:
        gfpgan_parameters = make_gfpgan_parameters(data)

    def image_progress(sample, step):
        if canceled.is_set():
            raise CanceledException
        nonlocal step_index
        if parameters["progress_images"] and step % 5 == 0 and step < steps - 1:
            image = model.sample_to_image(sample)
            path = save_image(image, data["seed"],
                              intermediate_path, step_index)
            step_index += 1
            socketio.emit('intermediateResult', {
                          'url': os.path.relpath(path), 'metadata': data})
        socketio.emit('progress', {'step': step + 1})
        eventlet.sleep(0)

    def image_done(image, seed, upscaled=False):
        if should_run_esrgan:
            esrgan_parameters['seed'] = seed
            image = real_esrgan_upscale(
                image=image,
                **esrgan_parameters,
            )
        if should_run_gfpgan:
            gfpgan_parameters['seed'] = seed
            image = run_gfpgan(
                image=image,
                **gfpgan_parameters,
                upsampler_scale=1,
            )
        path = save_image(image, seed, output_dir=result_path)
        data["seed"] = seed
        socketio.emit(
            'result', {'url': os.path.relpath(path), 'metadata': data})
        eventlet.sleep(0)

    try:
        model.prompt2image(
            **parameters,
            # In Web UI, we process GFPGAN and ESRGAN wholly separately, skip them here:
            step_callback=image_progress,
            image_callback=image_done
        )

    except KeyboardInterrupt:
        raise
    except CanceledException:
        raise
    except Exception as e:
        # socketio.emit('error', (str(e)))
        print("\n")
        traceback.print_exc()
        print("\n")


if __name__ == '__main__':
    socketio.run(app, host=host, port=port)
