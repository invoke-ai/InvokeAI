import transformers
import json
import base64
import mimetypes
import os
import sys
import time
import eventlet
from pytorch_lightning import logging
from flask import Flask
from flask_socketio import SocketIO, emit, send
from ldm.simplet2i import T2I
from ldm.dream.pngwriter import PngWriter
from ldm.gfpgan.gfpgan_tools import gfpgan_model_exists

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

host = '127.0.0.1'
port = 5000

# CORS only for testing
socketio = SocketIO(app, host=host, port=port,
                    cors_allowed_origins="*", logger=True, engineio_logger=True)

transformers.logging.set_verbosity_error()

t2i = T2I()

# gets rid of annoying messages about random seed
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

tic = time.time()
t2i.load_model()
print(f'>> model loaded in', '%4.2fs' % (time.time() - tic))

print("\n* Initialization done! Ready to dream...")


@socketio.on('message')
def message(data):
    print(data)

# cancel in-progress image generation


@socketio.on('cancel')
def cancel():
    print('cancel received')

# generate an image


@socketio.on('generateImage')
def generateImage(data):
    print(data)
    prompt = data['prompt']
    initimg = None
    strength = float(data['img2imgStrength'])
    iterations = int(data['imagesToGenerate'])
    steps = int(data['steps'])
    width = int(data['width'])
    height = int(data['height'])
    fit = False
    initimg = None
    cfgscale = float(data['cfgScale'])
    sampler_name = data['sampler']
    gfpgan_strength = float(data['gfpganStrength']
                            ) if gfpgan_model_exists else 0
    upscale_level = data['upscalingLevel']
    upscale_strength = data['upscalingStrength']
    upscale = [int(upscale_level), float(upscale_strength)
               ] if upscale_level != 0 else None
    progress_images = False
    seed = t2i.seed if int(data['seed']) == -1 else int(data['seed'])
    seed = int(data['seed'])

    pngwriter = PngWriter("./outputs/img-samples/")
    prefix = pngwriter.unique_prefix()

    def image_progress(sample, step):
        socketio.emit('progress', (step+1) / steps)
        eventlet.sleep(0)

    def image_done(image, seed, upscaled=False):
        name = f'{prefix}.{seed}.png'
        path = pngwriter.save_image_and_prompt_to_png(image, f'{prompt} -S{seed}', name)
        # Append post_data to log, but only once!
        if not upscaled:
            socketio.emit('message', {'imgUrl': path})
            socketio.emit('progress', 0)
            eventlet.sleep(0)

    t2i.prompt2image(prompt,
                     iterations=iterations,
                     cfg_scale=cfgscale,
                     width=width,
                     height=height,
                     seed=seed,
                     steps=steps,
                     gfpgan_strength=gfpgan_strength,
                     upscale=upscale,
                     sampler_name=sampler_name,
                     step_callback=image_progress,
                     image_callback=image_done)


if __name__ == '__main__':
    socketio.run(app)
