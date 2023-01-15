import base64
import glob
import io
import json
import math
import mimetypes
import os
import shutil
import traceback
from threading import Event
from uuid import uuid4

import eventlet
from PIL import Image
from PIL.Image import Image as ImageType
from flask import Flask, redirect, send_from_directory, request, make_response
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename

from backend.modules.get_canvas_generation_mode import (
    get_canvas_generation_mode,
)
from backend.modules.parameters import parameters_to_command
from ldm.generate import Generate
from ldm.invoke.args import Args, APP_ID, APP_VERSION, calculate_init_img_hash
from ldm.invoke.conditioning import get_tokens_for_prompt, get_prompt_structure
from ldm.invoke.generator.diffusers_pipeline import PipelineIntermediateState
from ldm.invoke.generator.inpaint import infill_methods
from ldm.invoke.globals import Globals
from ldm.invoke.pngwriter import PngWriter, retrieve_metadata
from ldm.invoke.prompt_parser import split_weighted_subprompts, Blend

# Loading Arguments
opt = Args()
args = opt.parse_args()

# Set the root directory for static files and relative paths
args.root_dir = os.path.expanduser(args.root_dir or "..")
if not os.path.isabs(args.outdir):
    args.outdir = os.path.join(args.root_dir, args.outdir)

# normalize the config directory relative to root
if not os.path.isabs(opt.conf):
    opt.conf = os.path.normpath(os.path.join(Globals.root,opt.conf))

class InvokeAIWebServer:
    def __init__(self, generate: Generate, gfpgan, codeformer, esrgan) -> None:
        self.host = args.host
        self.port = args.port

        self.generate = generate
        self.gfpgan = gfpgan
        self.codeformer = codeformer
        self.esrgan = esrgan

        self.canceled = Event()
        self.ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

    def allowed_file(self, filename: str) -> bool:
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in self.ALLOWED_EXTENSIONS
        )

    def run(self):
        self.setup_app()
        self.setup_flask()

    def setup_flask(self):
        # Fix missing mimetypes on Windows
        mimetypes.add_type("application/javascript", ".js")
        mimetypes.add_type("text/css", ".css")
        # Socket IO
        logger = True if args.web_verbose else False
        engineio_logger = True if args.web_verbose else False
        max_http_buffer_size = 10000000

        socketio_args = {
            "logger": logger,
            "engineio_logger": engineio_logger,
            "max_http_buffer_size": max_http_buffer_size,
            "ping_interval": (50, 50),
            "ping_timeout": 60,
        }

        if opt.cors:
            _cors = opt.cors
            # convert list back into comma-separated string,
            # be defensive here, not sure in what form this arrives
            if isinstance(_cors, list):
                _cors = ",".join(_cors)
            if "," in _cors:
                _cors = _cors.split(",")
            socketio_args["cors_allowed_origins"] = _cors

        frontend_path = self.find_frontend()
        self.app = Flask(
            __name__, static_url_path="", static_folder=frontend_path
        )

        self.socketio = SocketIO(self.app, **socketio_args)

        # Keep Server Alive Route
        @self.app.route("/flaskwebgui-keep-server-alive")
        def keep_alive():
            return {"message": "Server Running"}

        # Outputs Route
        self.app.config["OUTPUTS_FOLDER"] = os.path.abspath(args.outdir)

        @self.app.route("/outputs/<path:file_path>")
        def outputs(file_path):
            return send_from_directory(self.app.config["OUTPUTS_FOLDER"], file_path)

        # Base Route
        @self.app.route("/")
        def serve():
            if args.web_develop:
                return redirect("http://127.0.0.1:5173")
            else:
                return send_from_directory(self.app.static_folder, "index.html")

        @self.app.route("/upload", methods=["POST"])
        def upload():
            try:
                data = json.loads(request.form["data"])
                filename = ""
                # check if the post request has the file part
                if "file" in request.files:
                    file = request.files["file"]
                    # If the user does not select a file, the browser submits an
                    # empty file without a filename.
                    if file.filename == "":
                        return make_response("No file selected", 400)
                    filename = file.filename
                elif "dataURL" in data:
                    file = dataURL_to_bytes(data["dataURL"])
                    if "filename" not in data or data["filename"] == "":
                        return make_response("No filename provided", 400)
                    filename = data["filename"]
                else:
                    return make_response("No file or dataURL", 400)

                kind = data["kind"]

                if kind == "init":
                    path = self.init_image_path
                elif kind == "temp":
                    path = self.temp_image_path
                elif kind == "result":
                    path = self.result_path
                elif kind == "mask":
                    path = self.mask_image_path
                else:
                    return make_response(f"Invalid upload kind: {kind}", 400)

                if not self.allowed_file(filename):
                    return make_response(
                        f'Invalid file type, must be one of: {", ".join(self.ALLOWED_EXTENSIONS)}',
                        400,
                    )

                secured_filename = secure_filename(filename)

                uuid = uuid4().hex
                truncated_uuid = uuid[:8]

                split = os.path.splitext(secured_filename)
                name = f"{split[0]}.{truncated_uuid}{split[1]}"

                file_path = os.path.join(path, name)

                if "dataURL" in data:
                    with open(file_path, "wb") as f:
                        f.write(file)
                else:
                    file.save(file_path)

                mtime = os.path.getmtime(file_path)

                pil_image = Image.open(file_path)

                if "cropVisible" in data and data["cropVisible"] == True:
                    visible_image_bbox = pil_image.getbbox()
                    pil_image = pil_image.crop(visible_image_bbox)
                    pil_image.save(file_path)

                (width, height) = pil_image.size

                thumbnail_path = save_thumbnail(
                    pil_image, os.path.basename(file_path), self.thumbnail_image_path
                )

                response = {
                    "url": self.get_url_from_image_path(file_path),
                    "thumbnail": self.get_url_from_image_path(thumbnail_path),
                    "mtime": mtime,
                    "width": width,
                    "height": height,
                }

                return make_response(response, 200)

            except Exception as e:
                self.socketio.emit("error", {"message": (str(e))})
                print("\n")

                traceback.print_exc()
                print("\n")
                return make_response("Error uploading file", 500)

        self.load_socketio_listeners(self.socketio)

        if args.gui:
            print(">> Launching Invoke AI GUI")
            try:
                from flaskwebgui import FlaskUI

                FlaskUI(
                    app=self.app,
                    socketio=self.socketio,
                    server="flask_socketio",
                    width=1600,
                    height=1000,
                    port=self.port
                ).run()
            except KeyboardInterrupt:
                import sys

                sys.exit(0)
        else:
            useSSL = args.certfile or args.keyfile
            print(">> Started Invoke AI Web Server!")
            if self.host == "0.0.0.0":
                print(
                    f"Point your browser at http{'s' if useSSL else ''}://localhost:{self.port} or use the host's DNS name or IP address."
                )
            else:
                print(
                    ">> Default host address now 127.0.0.1 (localhost). Use --host 0.0.0.0 to bind any address."
                )
                print(
                    f">> Point your browser at http{'s' if useSSL else ''}://{self.host}:{self.port}"
                )
            if not useSSL:
                self.socketio.run(app=self.app, host=self.host, port=self.port)
            else:
                self.socketio.run(
                    app=self.app,
                    host=self.host,
                    port=self.port,
                    certfile=args.certfile,
                    keyfile=args.keyfile,
                )

    def find_frontend(self):
        my_dir = os.path.dirname(__file__)
        # LS: setup.py seems to put the frontend in different places on different systems, so
        # this is fragile and needs to be replaced with a better way of finding the front end.
        for candidate in (os.path.join(my_dir,'..','frontend','dist'),          # pip install -e .
                          os.path.join(my_dir,'../../../../frontend','dist'),   # pip install . (Linux, Mac)
                          os.path.join(my_dir,'../../../frontend','dist'),      # pip install . (Windows)
        ):
            if os.path.exists(candidate):
                return candidate
        assert "Frontend files cannot be found. Cannot continue"

    def setup_app(self):
        self.result_url = "outputs/"
        self.init_image_url = "outputs/init-images/"
        self.mask_image_url = "outputs/mask-images/"
        self.intermediate_url = "outputs/intermediates/"
        self.temp_image_url = "outputs/temp-images/"
        self.thumbnail_image_url = "outputs/thumbnails/"
        # location for "finished" images
        self.result_path = args.outdir
        # temporary path for intermediates
        self.intermediate_path = os.path.join(self.result_path, "intermediates/")
        # path for user-uploaded init images and masks
        self.init_image_path = os.path.join(self.result_path, "init-images/")
        self.mask_image_path = os.path.join(self.result_path, "mask-images/")
        # path for temp images e.g. gallery generations which are not committed
        self.temp_image_path = os.path.join(self.result_path, "temp-images/")
        # path for thumbnail images
        self.thumbnail_image_path = os.path.join(self.result_path, "thumbnails/")
        # txt log
        self.log_path = os.path.join(self.result_path, "invoke_log.txt")
        # make all output paths
        [
            os.makedirs(path, exist_ok=True)
            for path in [
                self.result_path,
                self.intermediate_path,
                self.init_image_path,
                self.mask_image_path,
                self.temp_image_path,
                self.thumbnail_image_path,
            ]
        ]

    def load_socketio_listeners(self, socketio):
        @socketio.on("requestSystemConfig")
        def handle_request_capabilities():
            print(f">> System config requested")
            config = self.get_system_config()
            config["model_list"] = self.generate.model_manager.list_models()
            config["infill_methods"] = infill_methods()
            socketio.emit("systemConfig", config)

        @socketio.on('searchForModels')
        def handle_search_models(search_folder: str):
            try:
                if not search_folder:
                    socketio.emit(
                    "foundModels",
                    {'search_folder': None, 'found_models': None},
                )
                else:
                    search_folder, found_models = self.generate.model_manager.search_models(search_folder)
                    socketio.emit(
                        "foundModels",
                        {'search_folder': search_folder, 'found_models': found_models},
                    )
            except Exception as e:
                self.socketio.emit("error", {"message": (str(e))})
                print("\n")

                traceback.print_exc()
                print("\n")

        @socketio.on("addNewModel")
        def handle_add_model(new_model_config: dict):
            try:
                model_name = new_model_config['name']
                del new_model_config['name']
                model_attributes = new_model_config
                if len(model_attributes['vae']) == 0:
                    del model_attributes['vae']
                update = False
                current_model_list = self.generate.model_manager.list_models()
                if model_name in current_model_list:
                    update = True

                print(f">> Adding New Model: {model_name}")

                self.generate.model_manager.add_model(
                    model_name=model_name, model_attributes=model_attributes, clobber=True)
                self.generate.model_manager.commit(opt.conf)

                new_model_list = self.generate.model_manager.list_models()
                socketio.emit(
                    "newModelAdded",
                    {"new_model_name": model_name,
                     "model_list": new_model_list, 'update': update},
                )
                print(f">> New Model Added: {model_name}")
            except Exception as e:
                self.socketio.emit("error", {"message": (str(e))})
                print("\n")

                traceback.print_exc()
                print("\n")

        @socketio.on("deleteModel")
        def handle_delete_model(model_name: str):
            try:
                print(f">> Deleting Model: {model_name}")
                self.generate.model_manager.del_model(model_name)
                self.generate.model_manager.commit(opt.conf)
                updated_model_list = self.generate.model_manager.list_models()
                socketio.emit(
                    "modelDeleted",
                    {"deleted_model_name": model_name,
                     "model_list": updated_model_list},
                )
                print(f">> Model Deleted: {model_name}")
            except Exception as e:
                self.socketio.emit("error", {"message": (str(e))})
                print("\n")

                traceback.print_exc()
                print("\n")

        @socketio.on("requestModelChange")
        def handle_set_model(model_name: str):
            try:
                print(f">> Model change requested: {model_name}")
                model = self.generate.set_model(model_name)
                model_list = self.generate.model_manager.list_models()
                if model is None:
                    socketio.emit(
                        "modelChangeFailed",
                        {"model_name": model_name, "model_list": model_list},
                    )
                else:
                    socketio.emit(
                        "modelChanged",
                        {"model_name": model_name, "model_list": model_list},
                    )
            except Exception as e:
                self.socketio.emit("error", {"message": (str(e))})
                print("\n")

                traceback.print_exc()
                print("\n")

        @socketio.on("requestEmptyTempFolder")
        def empty_temp_folder():
            try:
                temp_files = glob.glob(os.path.join(self.temp_image_path, "*"))
                for f in temp_files:
                    try:
                        os.remove(f)
                        thumbnail_path = os.path.join(
                            self.thumbnail_image_path,
                            os.path.splitext(os.path.basename(f))[0] + ".webp",
                        )
                        os.remove(thumbnail_path)
                    except Exception as e:
                        socketio.emit("error", {"message": f"Unable to delete {f}: {str(e)}"})
                        pass

                socketio.emit("tempFolderEmptied")
            except Exception as e:
                self.socketio.emit("error", {"message": (str(e))})
                print("\n")

                traceback.print_exc()
                print("\n")

        @socketio.on("requestSaveStagingAreaImageToGallery")
        def save_temp_image_to_gallery(url):
            try:
                image_path = self.get_image_path_from_url(url)
                new_path = os.path.join(self.result_path, os.path.basename(image_path))
                shutil.copy2(image_path, new_path)

                if os.path.splitext(new_path)[1] == ".png":
                    metadata = retrieve_metadata(new_path)
                else:
                    metadata = {}

                pil_image = Image.open(new_path)

                (width, height) = pil_image.size

                thumbnail_path = save_thumbnail(
                    pil_image, os.path.basename(new_path), self.thumbnail_image_path
                )

                image_array = [
                    {
                        "url": self.get_url_from_image_path(new_path),
                        "thumbnail": self.get_url_from_image_path(thumbnail_path),
                        "mtime": os.path.getmtime(new_path),
                        "metadata": metadata,
                        "width": width,
                        "height": height,
                        "category": "result",
                    }
                ]

                socketio.emit(
                    "galleryImages",
                    {"images": image_array, "category": "result"},
                )

            except Exception as e:
                self.socketio.emit("error", {"message": (str(e))})
                print("\n")

                traceback.print_exc()
                print("\n")

        @socketio.on("requestLatestImages")
        def handle_request_latest_images(category, latest_mtime):
            try:
                base_path = (
                    self.result_path if category == "result" else self.init_image_path
                )

                paths = []

                for ext in ("*.png", "*.jpg", "*.jpeg"):
                    paths.extend(glob.glob(os.path.join(base_path, ext)))

                image_paths = sorted(
                    paths, key=lambda x: os.path.getmtime(x), reverse=True
                )

                image_paths = list(
                    filter(
                        lambda x: os.path.getmtime(x) > latest_mtime,
                        image_paths,
                    )
                )

                image_array = []

                for path in image_paths:
                    try:
                        if os.path.splitext(path)[1] == ".png":
                            metadata = retrieve_metadata(path)
                        else:
                            metadata = {}

                        pil_image = Image.open(path)
                        (width, height) = pil_image.size

                        thumbnail_path = save_thumbnail(
                            pil_image, os.path.basename(path), self.thumbnail_image_path
                        )

                        image_array.append(
                            {
                                "url": self.get_url_from_image_path(path),
                                "thumbnail": self.get_url_from_image_path(
                                    thumbnail_path
                                ),
                                "mtime": os.path.getmtime(path),
                                "metadata": metadata.get("sd-metadata"),
                                "dreamPrompt": metadata.get("Dream"),
                                "width": width,
                                "height": height,
                                "category": category,
                            }
                        )
                    except Exception as e:
                        socketio.emit("error", {"message": f"Unable to load {path}: {str(e)}"})
                        pass

                socketio.emit(
                    "galleryImages",
                    {"images": image_array, "category": category},
                )
            except Exception as e:
                self.socketio.emit("error", {"message": (str(e))})
                print("\n")

                traceback.print_exc()
                print("\n")

        @socketio.on("requestImages")
        def handle_request_images(category, earliest_mtime=None):
            try:
                page_size = 50

                base_path = (
                    self.result_path if category == "result" else self.init_image_path
                )

                paths = []
                for ext in ("*.png", "*.jpg", "*.jpeg"):
                    paths.extend(glob.glob(os.path.join(base_path, ext)))

                image_paths = sorted(
                    paths, key=lambda x: os.path.getmtime(x), reverse=True
                )

                if earliest_mtime:
                    image_paths = list(
                        filter(
                            lambda x: os.path.getmtime(x) < earliest_mtime,
                            image_paths,
                        )
                    )

                areMoreImagesAvailable = len(image_paths) >= page_size
                image_paths = image_paths[slice(0, page_size)]

                image_array = []
                for path in image_paths:
                    try:
                        if os.path.splitext(path)[1] == ".png":
                            metadata = retrieve_metadata(path)
                        else:
                            metadata = {}

                        pil_image = Image.open(path)
                        (width, height) = pil_image.size

                        thumbnail_path = save_thumbnail(
                            pil_image, os.path.basename(path), self.thumbnail_image_path
                        )

                        image_array.append(
                            {
                                "url": self.get_url_from_image_path(path),
                                "thumbnail": self.get_url_from_image_path(
                                    thumbnail_path
                                ),
                                "mtime": os.path.getmtime(path),
                                "metadata": metadata.get("sd-metadata"),
                                "dreamPrompt": metadata.get("Dream"),
                                "width": width,
                                "height": height,
                                "category": category,
                            }
                        )
                    except Exception as e:
                        print(f">> Unable to load {path}")
                        socketio.emit("error", {"message": f"Unable to load {path}: {str(e)}"})
                        pass

                socketio.emit(
                    "galleryImages",
                    {
                        "images": image_array,
                        "areMoreImagesAvailable": areMoreImagesAvailable,
                        "category": category,
                    },
                )
            except Exception as e:
                self.socketio.emit("error", {"message": (str(e))})
                print("\n")

                traceback.print_exc()
                print("\n")

        @socketio.on("generateImage")
        def handle_generate_image_event(
            generation_parameters, esrgan_parameters, facetool_parameters
        ):
            try:
                # truncate long init_mask/init_img base64 if needed
                printable_parameters = {
                    **generation_parameters,
                }

                if "init_img" in generation_parameters:
                    printable_parameters["init_img"] = (
                        printable_parameters["init_img"][:64] + "..."
                    )

                if "init_mask" in generation_parameters:
                    printable_parameters["init_mask"] = (
                        printable_parameters["init_mask"][:64] + "..."
                    )

                print(
                    f">> Image generation requested: {printable_parameters}\nESRGAN parameters: {esrgan_parameters}\nFacetool parameters: {facetool_parameters}"
                )
                self.generate_images(
                    generation_parameters,
                    esrgan_parameters,
                    facetool_parameters,
                )
            except Exception as e:
                self.socketio.emit("error", {"message": (str(e))})
                print("\n")

                traceback.print_exc()
                print("\n")

        @socketio.on("runPostprocessing")
        def handle_run_postprocessing(original_image, postprocessing_parameters):
            try:
                print(
                    f'>> Postprocessing requested for "{original_image["url"]}": {postprocessing_parameters}'
                )

                progress = Progress()

                socketio.emit("progressUpdate", progress.to_formatted_dict())
                eventlet.sleep(0)

                original_image_path = self.get_image_path_from_url(
                    original_image["url"]
                )

                image = Image.open(original_image_path)

                try:
                    seed = original_image["metadata"]["image"]["seed"]
                except (KeyError) as e:
                    seed = "unknown_seed"
                    pass

                if postprocessing_parameters["type"] == "esrgan":
                    progress.set_current_status("common:statusUpscalingESRGAN")
                elif postprocessing_parameters["type"] == "gfpgan":
                    progress.set_current_status("common:statusRestoringFacesGFPGAN")
                elif postprocessing_parameters["type"] == "codeformer":
                    progress.set_current_status("common:statusRestoringFacesCodeFormer")

                socketio.emit("progressUpdate", progress.to_formatted_dict())
                eventlet.sleep(0)

                if postprocessing_parameters["type"] == "esrgan":
                    image = self.esrgan.process(
                        image=image,
                        upsampler_scale=postprocessing_parameters["upscale"][0],
                        strength=postprocessing_parameters["upscale"][1],
                        seed=seed,
                    )
                elif postprocessing_parameters["type"] == "gfpgan":
                    image = self.gfpgan.process(
                        image=image,
                        strength=postprocessing_parameters["facetool_strength"],
                        seed=seed,
                    )
                elif postprocessing_parameters["type"] == "codeformer":
                    image = self.codeformer.process(
                        image=image,
                        strength=postprocessing_parameters["facetool_strength"],
                        fidelity=postprocessing_parameters["codeformer_fidelity"],
                        seed=seed,
                        device="cpu"
                        if str(self.generate.device) == "mps"
                        else self.generate.device,
                    )
                else:
                    raise TypeError(
                        f'{postprocessing_parameters["type"]} is not a valid postprocessing type'
                    )

                progress.set_current_status("common:statusSavingImage")
                socketio.emit("progressUpdate", progress.to_formatted_dict())
                eventlet.sleep(0)

                postprocessing_parameters["seed"] = seed
                metadata = self.parameters_to_post_processed_image_metadata(
                    parameters=postprocessing_parameters,
                    original_image_path=original_image_path,
                )

                command = parameters_to_command(postprocessing_parameters)

                (width, height) = image.size

                path = self.save_result_image(
                    image,
                    command,
                    metadata,
                    self.result_path,
                    postprocessing=postprocessing_parameters["type"],
                )

                thumbnail_path = save_thumbnail(
                    image, os.path.basename(path), self.thumbnail_image_path
                )

                self.write_log_message(
                    f'[Postprocessed] "{original_image_path}" > "{path}": {postprocessing_parameters}'
                )

                progress.mark_complete()
                socketio.emit("progressUpdate", progress.to_formatted_dict())
                eventlet.sleep(0)

                socketio.emit(
                    "postprocessingResult",
                    {
                        "url": self.get_url_from_image_path(path),
                        "thumbnail": self.get_url_from_image_path(thumbnail_path),
                        "mtime": os.path.getmtime(path),
                        "metadata": metadata,
                        "dreamPrompt": command,
                        "width": width,
                        "height": height,
                    },
                )
            except Exception as e:
                self.socketio.emit("error", {"message": (str(e))})
                print("\n")

                traceback.print_exc()
                print("\n")

        @socketio.on("cancel")
        def handle_cancel():
            print(f">> Cancel processing requested")
            self.canceled.set()

        # TODO: I think this needs a safety mechanism.
        @socketio.on("deleteImage")
        def handle_delete_image(url, thumbnail, uuid, category):
            try:
                print(f'>> Delete requested "{url}"')
                from send2trash import send2trash

                path = self.get_image_path_from_url(url)
                thumbnail_path = self.get_image_path_from_url(thumbnail)

                send2trash(path)
                send2trash(thumbnail_path)

                socketio.emit(
                    "imageDeleted",
                    {"url": url, "uuid": uuid, "category": category},
                )
            except Exception as e:
                self.socketio.emit("error", {"message": (str(e))})
                print("\n")

                traceback.print_exc()
                print("\n")

    # App Functions
    def get_system_config(self):
        model_list: dict = self.generate.model_manager.list_models()
        active_model_name = None

        for model_name, model_dict in model_list.items():
            if model_dict["status"] == "active":
                active_model_name = model_name

        return {
            "model": "stable diffusion",
            "model_weights": active_model_name,
            "model_hash": self.generate.model_hash,
            "app_id": APP_ID,
            "app_version": APP_VERSION,
        }

    def generate_images(
        self, generation_parameters, esrgan_parameters, facetool_parameters
    ):
        try:
            self.canceled.clear()

            step_index = 1
            prior_variations = (
                generation_parameters["with_variations"]
                if "with_variations" in generation_parameters
                else []
            )

            actual_generation_mode = generation_parameters["generation_mode"]
            original_bounding_box = None

            progress = Progress(generation_parameters=generation_parameters)

            self.socketio.emit("progressUpdate", progress.to_formatted_dict())
            eventlet.sleep(0)

            """
            TODO:
            If a result image is used as an init image, and then deleted, we will want to be
            able to use it as an init image in the future. Need to handle this case.
            """

            """
            Prepare for generation based on generation_mode
            """
            if generation_parameters["generation_mode"] == "unifiedCanvas":
                """
                generation_parameters["init_img"] is a base64 image
                generation_parameters["init_mask"] is a base64 image

                So we need to convert each into a PIL Image.
                """

                truncated_outpaint_image_b64 = generation_parameters["init_img"][:64]
                truncated_outpaint_mask_b64 = generation_parameters["init_mask"][:64]

                init_img_url = generation_parameters["init_img"]

                original_bounding_box = generation_parameters["bounding_box"].copy()

                initial_image = dataURL_to_image(
                    generation_parameters["init_img"]
                ).convert("RGBA")

                """
                The outpaint image and mask are pre-cropped by the UI, so the bounding box we pass
                to the generator should be:
                    {
                        "x": 0,
                        "y": 0,
                        "width": original_bounding_box["width"],
                        "height": original_bounding_box["height"]
                    }
                """

                generation_parameters["bounding_box"]["x"] = 0
                generation_parameters["bounding_box"]["y"] = 0

                # Convert mask dataURL to an image and convert to greyscale
                mask_image = dataURL_to_image(
                    generation_parameters["init_mask"]
                ).convert("L")

                actual_generation_mode = get_canvas_generation_mode(
                    initial_image, mask_image
                )

                """
                Apply the mask to the init image, creating a "mask" image with
                transparency where inpainting should occur. This is the kind of
                mask that prompt2image() needs.
                """
                alpha_mask = initial_image.copy()
                alpha_mask.putalpha(mask_image)

                generation_parameters["init_img"] = initial_image
                generation_parameters["init_mask"] = alpha_mask

                # Remove the unneeded parameters for whichever mode we are doing
                if actual_generation_mode == "inpainting":
                    generation_parameters.pop("seam_size", None)
                    generation_parameters.pop("seam_blur", None)
                    generation_parameters.pop("seam_strength", None)
                    generation_parameters.pop("seam_steps", None)
                    generation_parameters.pop("tile_size", None)
                    generation_parameters.pop("force_outpaint", None)
                elif actual_generation_mode == "img2img":
                    generation_parameters["height"] = original_bounding_box["height"]
                    generation_parameters["width"] = original_bounding_box["width"]
                    generation_parameters.pop("init_mask", None)
                    generation_parameters.pop("seam_size", None)
                    generation_parameters.pop("seam_blur", None)
                    generation_parameters.pop("seam_strength", None)
                    generation_parameters.pop("seam_steps", None)
                    generation_parameters.pop("tile_size", None)
                    generation_parameters.pop("force_outpaint", None)
                    generation_parameters.pop("infill_method", None)
                elif actual_generation_mode == "txt2img":
                    generation_parameters["height"] = original_bounding_box["height"]
                    generation_parameters["width"] = original_bounding_box["width"]
                    generation_parameters.pop("strength", None)
                    generation_parameters.pop("fit", None)
                    generation_parameters.pop("init_img", None)
                    generation_parameters.pop("init_mask", None)
                    generation_parameters.pop("seam_size", None)
                    generation_parameters.pop("seam_blur", None)
                    generation_parameters.pop("seam_strength", None)
                    generation_parameters.pop("seam_steps", None)
                    generation_parameters.pop("tile_size", None)
                    generation_parameters.pop("force_outpaint", None)
                    generation_parameters.pop("infill_method", None)

            elif generation_parameters["generation_mode"] == "img2img":
                init_img_url = generation_parameters["init_img"]
                init_img_path = self.get_image_path_from_url(init_img_url)
                generation_parameters["init_img"] = Image.open(init_img_path).convert('RGB')

            def image_progress(sample, step):
                if self.canceled.is_set():
                    raise CanceledException

                nonlocal step_index
                nonlocal generation_parameters
                nonlocal progress

                generation_messages = {
                    "txt2img": "common:statusGeneratingTextToImage",
                    "img2img": "common:statusGeneratingImageToImage",
                    "inpainting": "common:statusGeneratingInpainting",
                    "outpainting": "common:statusGeneratingOutpainting",
                }

                progress.set_current_step(step + 1)
                progress.set_current_status(
                    f"{generation_messages[actual_generation_mode]}"
                )
                progress.set_current_status_has_steps(True)

                if (
                    generation_parameters["progress_images"]
                    and step % generation_parameters["save_intermediates"] == 0
                    and step < generation_parameters["steps"] - 1
                ):
                    image = self.generate.sample_to_image(sample)
                    metadata = self.parameters_to_generated_image_metadata(
                        generation_parameters
                    )
                    command = parameters_to_command(generation_parameters)

                    (width, height) = image.size

                    path = self.save_result_image(
                        image,
                        command,
                        metadata,
                        self.intermediate_path,
                        step_index=step_index,
                        postprocessing=False,
                    )

                    step_index += 1
                    self.socketio.emit(
                        "intermediateResult",
                        {
                            "url": self.get_url_from_image_path(path),
                            "mtime": os.path.getmtime(path),
                            "metadata": metadata,
                            "width": width,
                            "height": height,
                            "generationMode": generation_parameters["generation_mode"],
                            "boundingBox": original_bounding_box,
                        },
                    )


                if generation_parameters["progress_latents"]:
                    image = self.generate.sample_to_lowres_estimated_image(sample)
                    (width, height) = image.size
                    width *= 8
                    height *= 8
                    img_base64 = image_to_dataURL(image)
                    self.socketio.emit(
                        "intermediateResult",
                        {
                            "url": img_base64,
                            "isBase64": True,
                            "mtime": 0,
                            "metadata": {},
                            "width": width,
                            "height": height,
                            "generationMode": generation_parameters["generation_mode"],
                            "boundingBox": original_bounding_box,
                        },
                    )

                self.socketio.emit("progressUpdate", progress.to_formatted_dict())
                eventlet.sleep(0)

            def image_done(image, seed, first_seed, attention_maps_image=None):
                if self.canceled.is_set():
                    raise CanceledException

                nonlocal generation_parameters
                nonlocal esrgan_parameters
                nonlocal facetool_parameters
                nonlocal progress

                step_index = 1
                nonlocal prior_variations

                """
                Tidy up after generation based on generation_mode
                """
                # paste the inpainting image back onto the original
                if generation_parameters["generation_mode"] == "inpainting":
                    image = paste_image_into_bounding_box(
                        Image.open(init_img_path),
                        image,
                        **generation_parameters["bounding_box"],
                    )

                progress.set_current_status("common:statusGenerationComplete")

                self.socketio.emit("progressUpdate", progress.to_formatted_dict())
                eventlet.sleep(0)

                all_parameters = generation_parameters
                postprocessing = False

                if (
                    "variation_amount" in all_parameters
                    and all_parameters["variation_amount"] > 0
                ):
                    first_seed = first_seed or seed
                    this_variation = [[seed, all_parameters["variation_amount"]]]
                    all_parameters["with_variations"] = (
                        prior_variations + this_variation
                    )
                    all_parameters["seed"] = first_seed
                elif "with_variations" in all_parameters:
                    all_parameters["seed"] = first_seed
                else:
                    all_parameters["seed"] = seed

                if self.canceled.is_set():
                    raise CanceledException

                if esrgan_parameters:
                    progress.set_current_status("common:statusUpscaling")
                    progress.set_current_status_has_steps(False)
                    self.socketio.emit("progressUpdate", progress.to_formatted_dict())
                    eventlet.sleep(0)

                    image = self.esrgan.process(
                        image=image,
                        upsampler_scale=esrgan_parameters["level"],
                        strength=esrgan_parameters["strength"],
                        seed=seed,
                    )

                    postprocessing = True
                    all_parameters["upscale"] = [
                        esrgan_parameters["level"],
                        esrgan_parameters["strength"],
                    ]

                if self.canceled.is_set():
                    raise CanceledException

                if facetool_parameters:
                    if facetool_parameters["type"] == "gfpgan":
                        progress.set_current_status("common:statusRestoringFacesGFPGAN")
                    elif facetool_parameters["type"] == "codeformer":
                        progress.set_current_status("common:statusRestoringFacesCodeFormer")

                    progress.set_current_status_has_steps(False)
                    self.socketio.emit("progressUpdate", progress.to_formatted_dict())
                    eventlet.sleep(0)

                    if facetool_parameters["type"] == "gfpgan":
                        image = self.gfpgan.process(
                            image=image,
                            strength=facetool_parameters["strength"],
                            seed=seed,
                        )
                    elif facetool_parameters["type"] == "codeformer":
                        image = self.codeformer.process(
                            image=image,
                            strength=facetool_parameters["strength"],
                            fidelity=facetool_parameters["codeformer_fidelity"],
                            seed=seed,
                            device="cpu"
                            if str(self.generate.device) == "mps"
                            else self.generate.device,
                        )
                        all_parameters["codeformer_fidelity"] = facetool_parameters[
                            "codeformer_fidelity"
                        ]

                    postprocessing = True
                    all_parameters["facetool_strength"] = facetool_parameters[
                        "strength"
                    ]
                    all_parameters["facetool_type"] = facetool_parameters["type"]

                progress.set_current_status("common:statusSavingImage")
                self.socketio.emit("progressUpdate", progress.to_formatted_dict())
                eventlet.sleep(0)

                # restore the stashed URLS and discard the paths, we are about to send the result to client
                all_parameters["init_img"] = (
                    init_img_url
                    if generation_parameters["generation_mode"] == "img2img"
                    else ""
                )

                if "init_mask" in all_parameters:
                    all_parameters["init_mask"] = ""  # TODO: store the mask in metadata

                if generation_parameters["generation_mode"] == "unifiedCanvas":
                    all_parameters["bounding_box"] = original_bounding_box

                metadata = self.parameters_to_generated_image_metadata(all_parameters)

                command = parameters_to_command(all_parameters)

                (width, height) = image.size

                generated_image_outdir = (
                    self.result_path
                    if generation_parameters["generation_mode"]
                    in ["txt2img", "img2img"]
                    else self.temp_image_path
                )

                path = self.save_result_image(
                    image,
                    command,
                    metadata,
                    generated_image_outdir,
                    postprocessing=postprocessing,
                )

                thumbnail_path = save_thumbnail(
                    image, os.path.basename(path), self.thumbnail_image_path
                )

                print(f'>> Image generated: "{path}"')
                self.write_log_message(f'[Generated] "{path}": {command}')

                if progress.total_iterations > progress.current_iteration:
                    progress.set_current_step(1)
                    progress.set_current_status("common:statusIterationComplete")
                    progress.set_current_status_has_steps(False)
                else:
                    progress.mark_complete()

                self.socketio.emit("progressUpdate", progress.to_formatted_dict())
                eventlet.sleep(0)

                parsed_prompt, _ = get_prompt_structure(generation_parameters["prompt"])
                tokens = None if type(parsed_prompt) is Blend else \
                    get_tokens_for_prompt(self.generate.model, parsed_prompt)
                attention_maps_image_base64_url = None if attention_maps_image is None \
                    else image_to_dataURL(attention_maps_image)

                self.socketio.emit(
                    "generationResult",
                    {
                        "url": self.get_url_from_image_path(path),
                        "thumbnail": self.get_url_from_image_path(thumbnail_path),
                        "mtime": os.path.getmtime(path),
                        "metadata": metadata,
                        "dreamPrompt": command,
                        "width": width,
                        "height": height,
                        "boundingBox": original_bounding_box,
                        "generationMode": generation_parameters["generation_mode"],
                        "attentionMaps": attention_maps_image_base64_url,
                        "tokens": tokens,
                    },
                )
                eventlet.sleep(0)

                progress.set_current_iteration(progress.current_iteration + 1)

            print(generation_parameters)

            def diffusers_step_callback_adapter(*cb_args, **kwargs):
                if isinstance(cb_args[0], PipelineIntermediateState):
                    progress_state: PipelineIntermediateState = cb_args[0]
                    return image_progress(progress_state.latents, progress_state.step)
                else:
                    return image_progress(*cb_args, **kwargs)

            self.generate.prompt2image(
                **generation_parameters,
                step_callback=diffusers_step_callback_adapter,
                image_callback=image_done
            )

        except KeyboardInterrupt:
            self.socketio.emit("processingCanceled")
            raise
        except CanceledException:
            self.socketio.emit("processingCanceled")
            pass
        except Exception as e:
            print(e)
            self.socketio.emit("error", {"message": (str(e))})
            print("\n")

            traceback.print_exc()
            print("\n")

    def parameters_to_generated_image_metadata(self, parameters):
        try:
            # top-level metadata minus `image` or `images`
            metadata = self.get_system_config()
            # remove any image keys not mentioned in RFC #266
            rfc266_img_fields = [
                "type",
                "postprocessing",
                "sampler",
                "prompt",
                "seed",
                "variations",
                "steps",
                "cfg_scale",
                "threshold",
                "perlin",
                "step_number",
                "width",
                "height",
                "extra",
                "seamless",
                "hires_fix",
            ]

            rfc_dict = {}

            for item in parameters.items():
                key, value = item
                if key in rfc266_img_fields:
                    rfc_dict[key] = value

            postprocessing = []

            rfc_dict["type"] = parameters["generation_mode"]

            # 'postprocessing' is either null or an
            if "facetool_strength" in parameters:
                facetool_parameters = {
                    "type": str(parameters["facetool_type"]),
                    "strength": float(parameters["facetool_strength"]),
                }

                if parameters["facetool_type"] == "codeformer":
                    facetool_parameters["fidelity"] = float(
                        parameters["codeformer_fidelity"]
                    )

                postprocessing.append(facetool_parameters)

            if "upscale" in parameters:
                postprocessing.append(
                    {
                        "type": "esrgan",
                        "scale": int(parameters["upscale"][0]),
                        "strength": float(parameters["upscale"][1]),
                    }
                )

            rfc_dict["postprocessing"] = (
                postprocessing if len(postprocessing) > 0 else None
            )

            # semantic drift
            rfc_dict["sampler"] = parameters["sampler_name"]

            # display weighted subprompts (liable to change)
            subprompts = split_weighted_subprompts(
                parameters["prompt"], skip_normalize=True
            )
            subprompts = [{"prompt": x[0], "weight": x[1]} for x in subprompts]
            rfc_dict["prompt"] = subprompts

            # 'variations' should always exist and be an array, empty or consisting of {'seed': seed, 'weight': weight} pairs
            variations = []

            if "with_variations" in parameters:
                variations = [
                    {"seed": x[0], "weight": x[1]}
                    for x in parameters["with_variations"]
                ]

            rfc_dict["variations"] = variations

            print(parameters)

            if rfc_dict["type"] == "img2img":
                rfc_dict["strength"] = parameters["strength"]
                rfc_dict["fit"] = parameters["fit"]  # TODO: Noncompliant
                rfc_dict["orig_hash"] = calculate_init_img_hash(
                    self.get_image_path_from_url(parameters["init_img"])
                )
                rfc_dict["init_image_path"] = parameters[
                    "init_img"
                ]  # TODO: Noncompliant

            metadata["image"] = rfc_dict

            return metadata

        except Exception as e:
            self.socketio.emit("error", {"message": (str(e))})
            print("\n")

            traceback.print_exc()
            print("\n")

    def parameters_to_post_processed_image_metadata(
        self, parameters, original_image_path
    ):
        try:
            current_metadata = retrieve_metadata(original_image_path)["sd-metadata"]
            postprocessing_metadata = {}

            """
            if we don't have an original image metadata to reconstruct,
            need to record the original image and its hash
            """
            if "image" not in current_metadata:
                current_metadata["image"] = {}

                orig_hash = calculate_init_img_hash(
                    self.get_image_path_from_url(original_image_path)
                )

                postprocessing_metadata["orig_path"] = (original_image_path,)
                postprocessing_metadata["orig_hash"] = orig_hash

            if parameters["type"] == "esrgan":
                postprocessing_metadata["type"] = "esrgan"
                postprocessing_metadata["scale"] = parameters["upscale"][0]
                postprocessing_metadata["strength"] = parameters["upscale"][1]
            elif parameters["type"] == "gfpgan":
                postprocessing_metadata["type"] = "gfpgan"
                postprocessing_metadata["strength"] = parameters["facetool_strength"]
            elif parameters["type"] == "codeformer":
                postprocessing_metadata["type"] = "codeformer"
                postprocessing_metadata["strength"] = parameters["facetool_strength"]
                postprocessing_metadata["fidelity"] = parameters["codeformer_fidelity"]

            else:
                raise TypeError(f"Invalid type: {parameters['type']}")

            if "postprocessing" in current_metadata["image"] and isinstance(
                current_metadata["image"]["postprocessing"], list
            ):
                current_metadata["image"]["postprocessing"].append(
                    postprocessing_metadata
                )
            else:
                current_metadata["image"]["postprocessing"] = [postprocessing_metadata]

            return current_metadata

        except Exception as e:
            self.socketio.emit("error", {"message": (str(e))})
            print("\n")

            traceback.print_exc()
            print("\n")

    def save_result_image(
        self,
        image,
        command,
        metadata,
        output_dir,
        step_index=None,
        postprocessing=False,
    ):
        try:
            pngwriter = PngWriter(output_dir)

            number_prefix = pngwriter.unique_prefix()

            uuid = uuid4().hex
            truncated_uuid = uuid[:8]

            seed = "unknown_seed"

            if "image" in metadata:
                if "seed" in metadata["image"]:
                    seed = metadata["image"]["seed"]

            filename = f"{number_prefix}.{truncated_uuid}.{seed}"

            if step_index:
                filename += f".{step_index}"
            if postprocessing:
                filename += f".postprocessed"

            filename += ".png"

            path = pngwriter.save_image_and_prompt_to_png(
                image=image,
                dream_prompt=command,
                metadata=metadata,
                name=filename,
            )

            return os.path.abspath(path)

        except Exception as e:
            self.socketio.emit("error", {"message": (str(e))})
            print("\n")

            traceback.print_exc()
            print("\n")

    def make_unique_init_image_filename(self, name):
        try:
            uuid = uuid4().hex
            split = os.path.splitext(name)
            name = f"{split[0]}.{uuid}{split[1]}"
            return name
        except Exception as e:
            self.socketio.emit("error", {"message": (str(e))})
            print("\n")

            traceback.print_exc()
            print("\n")

    def calculate_real_steps(self, steps, strength, has_init_image):
        import math

        return math.floor(strength * steps) if has_init_image else steps

    def write_log_message(self, message):
        """Logs the filename and parameters used to generate or process that image to log file"""
        try:
            message = f"{message}\n"
            with open(self.log_path, "a", encoding="utf-8") as file:
                file.writelines(message)

        except Exception as e:
            self.socketio.emit("error", {"message": (str(e))})
            print("\n")

            traceback.print_exc()
            print("\n")

    def get_image_path_from_url(self, url):
        """Given a url to an image used by the client, returns the absolute file path to that image"""
        try:
            if "init-images" in url:
                return os.path.abspath(
                    os.path.join(self.init_image_path, os.path.basename(url))
                )
            elif "mask-images" in url:
                return os.path.abspath(
                    os.path.join(self.mask_image_path, os.path.basename(url))
                )
            elif "intermediates" in url:
                return os.path.abspath(
                    os.path.join(self.intermediate_path, os.path.basename(url))
                )
            elif "temp-images" in url:
                return os.path.abspath(
                    os.path.join(self.temp_image_path, os.path.basename(url))
                )
            elif "thumbnails" in url:
                return os.path.abspath(
                    os.path.join(self.thumbnail_image_path, os.path.basename(url))
                )
            else:
                return os.path.abspath(
                    os.path.join(self.result_path, os.path.basename(url))
                )
        except Exception as e:
            self.socketio.emit("error", {"message": (str(e))})
            print("\n")

            traceback.print_exc()
            print("\n")

    def get_url_from_image_path(self, path):
        """Given an absolute file path to an image, returns the URL that the client can use to load the image"""
        try:
            if "init-images" in path:
                return os.path.join(self.init_image_url, os.path.basename(path))
            elif "mask-images" in path:
                return os.path.join(self.mask_image_url, os.path.basename(path))
            elif "intermediates" in path:
                return os.path.join(self.intermediate_url, os.path.basename(path))
            elif "temp-images" in path:
                return os.path.join(self.temp_image_url, os.path.basename(path))
            elif "thumbnails" in path:
                return os.path.join(self.thumbnail_image_url, os.path.basename(path))
            else:
                return os.path.join(self.result_url, os.path.basename(path))
        except Exception as e:
            self.socketio.emit("error", {"message": (str(e))})
            print("\n")

            traceback.print_exc()
            print("\n")

    def save_file_unique_uuid_name(self, bytes, name, path):
        try:
            uuid = uuid4().hex
            truncated_uuid = uuid[:8]

            split = os.path.splitext(name)
            name = f"{split[0]}.{truncated_uuid}{split[1]}"

            file_path = os.path.join(path, name)

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            newFile = open(file_path, "wb")
            newFile.write(bytes)

            return file_path
        except Exception as e:
            self.socketio.emit("error", {"message": (str(e))})
            print("\n")

            traceback.print_exc()
            print("\n")


class Progress:
    def __init__(self, generation_parameters=None):
        self.current_step = 1
        self.total_steps = (
            self._calculate_real_steps(
                steps=generation_parameters["steps"],
                strength=generation_parameters["strength"]
                if "strength" in generation_parameters
                else None,
                has_init_image="init_img" in generation_parameters,
            )
            if generation_parameters
            else 1
        )
        self.current_iteration = 1
        self.total_iterations = (
            generation_parameters["iterations"] if generation_parameters else 1
        )
        self.current_status = "common:statusPreparing"
        self.is_processing = True
        self.current_status_has_steps = False
        self.has_error = False

    def set_current_step(self, current_step):
        self.current_step = current_step

    def set_total_steps(self, total_steps):
        self.total_steps = total_steps

    def set_current_iteration(self, current_iteration):
        self.current_iteration = current_iteration

    def set_total_iterations(self, total_iterations):
        self.total_iterations = total_iterations

    def set_current_status(self, current_status):
        self.current_status = current_status

    def set_is_processing(self, is_processing):
        self.is_processing = is_processing

    def set_current_status_has_steps(self, current_status_has_steps):
        self.current_status_has_steps = current_status_has_steps

    def set_has_error(self, has_error):
        self.has_error = has_error

    def mark_complete(self):
        self.current_status = "common:statusProcessingComplete"
        self.current_step = 0
        self.total_steps = 0
        self.current_iteration = 0
        self.total_iterations = 0
        self.is_processing = False

    def to_formatted_dict(
        self,
    ):
        return {
            "currentStep": self.current_step,
            "totalSteps": self.total_steps,
            "currentIteration": self.current_iteration,
            "totalIterations": self.total_iterations,
            "currentStatus": self.current_status,
            "isProcessing": self.is_processing,
            "currentStatusHasSteps": self.current_status_has_steps,
            "hasError": self.has_error,
        }

    def _calculate_real_steps(self, steps, strength, has_init_image):
        return math.floor(strength * steps) if has_init_image else steps


class CanceledException(Exception):
    pass


"""
Returns a copy an image, cropped to a bounding box.
"""


def copy_image_from_bounding_box(
    image: ImageType, x: int, y: int, width: int, height: int
) -> ImageType:
    with image as im:
        bounds = (x, y, x + width, y + height)
        im_cropped = im.crop(bounds)
        return im_cropped


"""
Converts a base64 image dataURL into an image.
The dataURL is split on the first commma.
"""


def dataURL_to_image(dataURL: str) -> ImageType:
    image = Image.open(
        io.BytesIO(
            base64.decodebytes(
                bytes(
                    dataURL.split(",", 1)[1],
                    "utf-8",
                )
            )
        )
    )
    return image

"""
Converts an image into a base64 image dataURL.
"""

def image_to_dataURL(image: ImageType) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = "data:image/png;base64," + base64.b64encode(
        buffered.getvalue()
    ).decode("UTF-8")
    return image_base64



"""
Converts a base64 image dataURL into bytes.
The dataURL is split on the first commma.
"""


def dataURL_to_bytes(dataURL: str) -> bytes:
    return base64.decodebytes(
        bytes(
            dataURL.split(",", 1)[1],
            "utf-8",
        )
    )


"""
Pastes an image onto another with a bounding box.
"""


def paste_image_into_bounding_box(
    recipient_image: ImageType,
    donor_image: ImageType,
    x: int,
    y: int,
    width: int,
    height: int,
) -> ImageType:
    with recipient_image as im:
        bounds = (x, y, x + width, y + height)
        im.paste(donor_image, bounds)
        return recipient_image


"""
Saves a thumbnail of an image, returning its path.
"""


def save_thumbnail(
    image: ImageType,
    filename: str,
    path: str,
    size: int = 256,
) -> str:
    base_filename = os.path.splitext(filename)[0]
    thumbnail_path = os.path.join(path, base_filename + ".webp")

    if os.path.exists(thumbnail_path):
        return thumbnail_path

    thumbnail_width = size
    thumbnail_height = round(size * (image.height / image.width))

    image_copy = image.copy()
    image_copy.thumbnail(size=(thumbnail_width, thumbnail_height))

    image_copy.save(thumbnail_path, "WEBP")

    return thumbnail_path
