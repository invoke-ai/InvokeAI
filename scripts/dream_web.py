import json
import base64
import mimetypes
import os
from pytorch_lightning import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

print("Loading model...")
from ldm.simplet2i import T2I
from ldm.dream.pngwriter import PngWriter
model = T2I(sampler_name='k_lms')

# to get rid of annoying warning messages from pytorch
import transformers
transformers.logging.set_verbosity_error()
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

print("Initializing model, be patient...")
model.load_model()

class DreamServer(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            with open("./static/dream_web/index.html", "rb") as content:
                self.wfile.write(content.read())
        elif os.path.exists("." + self.path):
            mime_type = mimetypes.guess_type(self.path)[0]
            if mime_type is not None:
                self.send_response(200)
                self.send_header("Content-type", mime_type)
                self.end_headers()
                with open("." + self.path, "rb") as content:
                    self.wfile.write(content.read())
            else:
                self.send_response(404)
        else:
            self.send_response(404)

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

        content_length = int(self.headers['Content-Length'])
        post_data = json.loads(self.rfile.read(content_length))
        prompt = post_data['prompt']
        initimg = post_data['initimg']
        iterations = int(post_data['iterations'])
        steps = int(post_data['steps'])
        width = int(post_data['width'])
        height = int(post_data['height'])
        cfgscale = float(post_data['cfgscale'])
        seed = None if int(post_data['seed']) == -1 else int(post_data['seed'])

        print(f"Request to generate with prompt: {prompt}")

        def image_done(image, seed):
            config = post_data.copy() # Shallow copy
            config['initimg'] = ''

            # Write PNGs
            pngwriter = PngWriter(
                "./outputs/img-samples/", config['prompt'], 1
            )
            # metadata_str = f'prompt2png({json.dumps(config)} seed={seed}' # gets written into the PNG
            pngwriter.write_image(image, seed)

            # Append post_data to log
            with open("./outputs/img-samples/dream_web_log.txt", "a") as log:
                for file_path, _ in pngwriter.files_written:
                    log.write(f"{file_path}: {json.dumps(config)}\n")

            self.wfile.write(bytes(json.dumps(
                {'event':'result', 'files':pngwriter.files_written, 'config':config}
            ) + '\n',"utf-8"))

        def image_progress(image, step):
            self.wfile.write(bytes(json.dumps(
                {'event':'step', 'step':step}
            ) + '\n',"utf-8"))

        # outputs = []
        if initimg is None:
            # Run txt2img
            model.prompt2image(prompt,
                               iterations=iterations,
                               cfg_scale = cfgscale,
                               width = width,
                               height = height,
                               seed = seed,
                               steps = steps,

                               step_callback=image_progress,
                               image_callback=image_done)
        else:
            # Decode initimg as base64 to temp file
            with open("./img2img-tmp.png", "wb") as f:
                initimg = initimg.split(",")[1] # Ignore mime type
                f.write(base64.b64decode(initimg))

                # Run img2img
                model.prompt2image(prompt,
                                   init_img = "./img2img-tmp.png",
                                   iterations = iterations,
                                   cfg_scale = cfgscale,
                                   seed = seed,
                                   steps = steps,

                                   step_callback=image_progress,
                                   image_callback=image_done)
            # Remove the temp file
            os.remove("./img2img-tmp.png")

        print(f"Prompt generated!")

if __name__ == "__main__":
    # Change working directory to the stable-diffusion directory
    os.chdir(
        os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
    )

    # Start server
    dream_server = ThreadingHTTPServer(("0.0.0.0", 9090), DreamServer)
    print("\n\n* Started Stable Diffusion dream server! Point your browser at http://localhost:9090 or use the host's DNS name or IP address. *")

    try:
        dream_server.serve_forever()
    except KeyboardInterrupt:
        pass

    dream_server.server_close()

