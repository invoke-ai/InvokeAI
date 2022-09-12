from ldm.gfpgan.gfpgan_tools import gfpgan_model_exists
from ldm.gfpgan.gfpgan_tools import real_esrgan_upscale
from ldm.dream.pngwriter import PngWriter
from ldm.generate import Generate
from flask_socketio import SocketIO
from flask import Flask, send_from_directory, url_for, jsonify
from PIL import Image
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
from modules.parse_seed_weights import parse_seed_weights

# fix missing mimetypes on windows due to registry wonkiness
import mimetypes
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')

# Host and port to serve on
host = 'localhost'
port = 9090

# Output directory for images
output_dir = "web-outputs/"

"""
Additional CORS origins to this list.
Useful if allowing other machines on your network to use the app.

Example:
additional_allowed_origins = ["192.168.1.8","192.168.1.24"]
"""
additional_allowed_origins = []


def build_cors_allowed_origins(additional_allowed_origins):
    cors_allowed_origins = [f"http://{host}:{port}"]
    for origin in additional_allowed_origins:
        cors_allowed_origins.append(origin)
    return cors_allowed_origins


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


@app.route('/socketio_config')
def socketio_config():
    return json.dumps({'host': host, 'port': port})


# True enables more logging from socket.io
dev_mode = False

logger = True if dev_mode else False
engineio_logger = True if dev_mode else False

# default 1,000,000, needs to be higher for socketio to accept larger images
max_http_buffer_size = 10000000

cors_allowed_origins = build_cors_allowed_origins(additional_allowed_origins)

socketio = SocketIO(app,
                    logger=logger,
                    engineio_logger=logger,
                    max_http_buffer_size=max_http_buffer_size,
                    cors_allowed_origins=cors_allowed_origins)


@socketio.on('cancel')
def handleCancel():
    canceled.set()
    return make_reponse("OK")


@socketio.on('generateImage')
def handle_generate_image(data):
    generate_image(data)
    return make_reponse("OK")


@socketio.on('runESRGAN')
def handle_run_esrgan(data):
    image = Image.open(data["url"])
    strength = data["upscalingStrength"]
    upsampler_scale = data["upscalingLevel"]
    seed = 1
    outdir = "outputs/"
    image = real_esrgan_upscale(
        image=image,
        strength=strength,
        upsampler_scale=upsampler_scale,
        seed=seed,
        outdir=outdir)
    image.save("outputs/test/test.png")
    return make_reponse("OK")


@socketio.on('runGFPGAN')
def handle_run_gfpgan(data):
    image = Image.open(data["url"])
    strength = Image.open(data["gfpganStrength"])
    seed = Image.open(data["seed"])
    image = run_gfpgan(image, strength, seed, 1)
    image.save("")


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

model.load_model()

print(f"\nServer online: http://{host}:{port}")


# Logic from ldm.dream.pngwriter.unique_prefix
def get_unique_prefix(dir):
    # sort reverse alphabetically until we find max+1
    dirlist = sorted(os.listdir(dir), reverse=True)
    # find the first filename that matches our pattern or return 000000.0.png
    existing_name = next(
        (f for f in dirlist if re.match(r'^(\d+)\..*\.png', f)),
        '0000000.0.png',
    )
    basecount = int(existing_name.split('.', 1)[0]) + 1
    return f'{basecount: 06}'


# prefix = pngwriter.unique_prefix()

# def make_filename(upscaled, with_variations,):
#     if upscaled and opt.save_original:
#                         filename = f'{prefix}.{seed}.postprocessed.png'
#                     else:
#                         filename = f'{prefix}.{seed}.png'
#                     if opt.variation_amount > 0:
#                         iter_opt = argparse.Namespace(**vars(opt)) # copy
#                         this_variation = [[seed, opt.variation_amount]]
#                         if opt.with_variations is None:
#                             iter_opt.with_variations = this_variation
#                         else:
#                             iter_opt.with_variations = opt.with_variations + this_variation
#                         iter_opt.variation_amount = 0
#                         normalized_prompt = PromptFormatter(t2i, iter_opt).normalize_prompt()
#                         metadata_prompt = f'{normalized_prompt} -S{iter_opt.seed}'
#                     elif opt.with_variations is not None:
#                         normalized_prompt = PromptFormatter(t2i, opt).normalize_prompt()
#                         metadata_prompt = f'{normalized_prompt} -S{opt.seed}' # use the original seed - the per-iteration value is the last variation-seed
#                     else:
#                         normalized_prompt = PromptFormatter(t2i, opt).normalize_prompt()
#                         metadata_prompt = f'{normalized_prompt} -S{seed}'
#                     path = file_writer.save_image_and_prompt_to_png(image, metadata_prompt, filename)
#                     if (not upscaled) or opt.save_original:
#                         # only append to results if we didn't overwrite an earlier output
#                         results.append([path, metadata_prompt])


def make_reponse(status, message=None, data=None):
    response = {'status': status}
    if message is not None:
        response['message'] = message
    if data is not None:
        response['data'] = data
    return response


# set up writers for various image types
image_writer = PngWriter(os.path.join(output_dir, 'final_images'))
init_image_writer = PngWriter(os.path.join(output_dir, 'init_images'))
mask_image_writer = PngWriter(os.path.join(output_dir, 'mask_images'))
intermediate_writer = PngWriter(
    os.path.join(output_dir, 'intermediate_images'))

"""
Jobs:
- Generate image
    - At least from prompt
    - May use init
    - May use mask
    - May run ESRGAN
    - May run GFPGAN
"""

def generate_image(data):
    canceled.clear()

    prompt = data['prompt']
    strength = float(data['img2imgStrength'])
    iterations = int(data['iterations'])
    steps = int(data['steps'])
    width = int(data['width'])
    height = int(data['height'])
    fit = bool(data['shouldFitToWidthHeight'])
    cfgscale = float(data['cfgScale'])
    sampler_name = data['sampler']
    gfpgan_strength = float(data['gfpganStrength']
                            ) if gfpgan_model_exists else 0
    upscale_level = int(data['upscalingLevel'])
    upscale_strength = float(data['upscalingStrength'])
    upscale = [int(upscale_level), float(upscale_strength)
               ] if upscale_level != 0 else None

    # generate seed if set to random in UI
    seed = model.seed if int(data['seed']) == -1 else int(data['seed'])
    data['seed'] = seed

    init_img = data['initialImagePath']
    fit = data['shouldFitToWidthHeight']

    init_mask = data['maskPath']
    seamless = data['seamless']
    progress_images = data['shouldDisplayInProgress']

    step_index = 1

    with_variations = None
    variation_amount = data['variantAmount']

    seed_weights = parse_seed_weights(data['seedWeights'])

    if data['shouldGenerateVariations'] and seed_weights is not False:
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
        socketio.emit('progress', {'step': step + 1})
        eventlet.sleep(0)

    def image_done(image, seed, upscaled=False):
        prefix = get_unique_prefix()
        filename = f'{prefix}.{seed}.png'
        path = os.path.join(outdir, filename)
        image.save(path, 'PNG')
        if not upscaled:
            # We may have passed -1 as seed to server, need actual seed for UI
            socketio.emit(
                'result', {'url': os.path.relpath(path), 'metadata': data})
            eventlet.sleep(0)

    try:
        model.prompt2image(
            # Common generation parameters
            prompt,
            iterations=iterations,
            steps=steps,
            seed=seed,
            cfg_scale=cfgscale,
            # ddim_eta=None, # needs implementation
            # skip_normalize=False, # needs implementation
            width=width,
            height=height,
            sampler_name=sampler_name,
            seamless=seamless,
            with_variations=with_variations,
            variation_amount=variation_amount,

            # img2img & inpaint parameters
            init_img=init_img,
            init_mask=init_mask,
            fit=fit,
            strength=strength,

            # GFPGAN/ESRGAN parameters
            gfpgan_strength=gfpgan_strength,  # needs implementation
            # save_original=False, # needs implementation
            upscale=upscale,  # needs implementation

            # System parameters
            progress_images=progress_images,
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
