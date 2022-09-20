# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import base64
import os
from threading import Event, Thread
import time
from PIL import Image
from ldm.generate import Generate
from ldm.dream.app.services.models import DreamResult, JobRequest, ProgressType, Signal
from ldm.dream.app.services.logging.services import LogService
from ldm.dream.app.services.signaling.services import SignalService
from ldm.dream.app.services.storage.services import ImageStorageService, JobQueueService


class CanceledException(Exception):
    pass


class GeneratorService:
    __model: Generate
    __queue: JobQueueService
    __imageStorage: ImageStorageService
    __intermediateStorage: ImageStorageService
    __log: LogService
    __thread: Thread
    __cancellationRequested: bool = False
    __signal_service: SignalService
    __ready_event: Event

    def __init__(
        self,
        model: Generate,
        queue: JobQueueService,
        imageStorage: ImageStorageService,
        intermediateStorage: ImageStorageService,
        log: LogService,
        signal_service: SignalService,
    ):
        self.__model = model
        self.__queue = queue
        self.__imageStorage = imageStorage
        self.__intermediateStorage = intermediateStorage
        self.__log = log
        self.__signal_service = signal_service

        # Create the background thread
        self.__event = Event()
        self.__thread = Thread(target=self.__process, name="GeneratorService")
        self.__thread.daemon = True
        self.__thread.start()

    # Request cancellation of the current job
    def cancel(self):
        self.__cancellationRequested = True

    def wait_for_ready(self, timeout: float = None):
        self.__event.wait(timeout=timeout)

    # TODO: Consider moving this to its own service if there's benefit in separating the generator
    def __process(self):
        # preload the model
        # TODO: support multiple models
        print(">> Preloading model")
        tic = time.time()
        self.__model.load_model()
        print(f">> model loaded in", "%4.2fs" % (time.time() - tic))

        self.__event.set()

        print(">> Started generation queue processor")
        try:
            while True:
                dreamRequest = self.__queue.get()
                self.__generate(dreamRequest)

        except KeyboardInterrupt:
            print(">> Generation queue processor stopped")

    def __on_start(self, jobRequest: JobRequest):
        self.__signal_service.emit(Signal.job_started(jobRequest.id))

    def __on_image_result(self, jobRequest: JobRequest, image, seed, upscaled=False):
        dreamResult = jobRequest.newDreamResult()
        dreamResult.seed = seed
        dreamResult.has_upscaled = upscaled
        dreamResult.iterations = 1
        jobRequest.results.append(dreamResult)
        # TODO: Separate status of GFPGAN?

        self.__imageStorage.save(image, dreamResult)

        # TODO: handle upscaling logic better (this is appending data to log, but only on first generation)
        if not upscaled:
            self.__log.log(dreamResult)

        # Send result signal
        self.__signal_service.emit(
            Signal.image_result(jobRequest.id, dreamResult.id, dreamResult)
        )

        upscaling_requested = dreamResult.enable_upscale or dreamResult.enable_gfpgan

        # Report upscaling status
        # TODO: this is very coupled to logic inside the generator. Fix that.
        if upscaling_requested and any(
            result.has_upscaled for result in jobRequest.results
        ):
            progressType = (
                ProgressType.UPSCALING_STARTED
                if len(jobRequest.results) < 2 * jobRequest.iterations
                else ProgressType.UPSCALING_DONE
            )
            upscale_count = sum(1 for i in jobRequest.results if i.has_upscaled)
            self.__signal_service.emit(
                Signal.image_progress(
                    jobRequest.id,
                    dreamResult.id,
                    upscale_count,
                    jobRequest.iterations,
                    progressType,
                )
            )

    def __on_progress(self, jobRequest: JobRequest, sample, step):
        if self.__cancellationRequested:
            self.__cancellationRequested = False
            raise CanceledException

        # TODO: Progress per request will be easier once the seeds (and ids) can all be pre-generated
        hasProgressImage = False
        s = str(len(jobRequest.results))
        if jobRequest.progress_images and step % 5 == 0 and step < jobRequest.steps - 1:
            image = self.__model._sample_to_image(sample)

            # TODO: clean this up, use a pre-defined dream result
            result = DreamResult()
            result.parse_json(jobRequest.__dict__, new_instance=False)
            self.__intermediateStorage.save(image, result, postfix=f".{s}.{step}")
            hasProgressImage = True

        self.__signal_service.emit(
            Signal.image_progress(
                jobRequest.id,
                f"{jobRequest.id}.{s}",
                step,
                jobRequest.steps,
                ProgressType.GENERATION,
                hasProgressImage,
            )
        )

    def __generate(self, jobRequest: JobRequest):
        try:
            # TODO: handle this file a file service for init images
            initimgfile = None  # TODO: support this on the model directly?
            if jobRequest.enable_init_image:
                if jobRequest.initimg is not None:
                    with open("./img2img-tmp.png", "wb") as f:
                        initimg = jobRequest.initimg.split(",")[1]  # Ignore mime type
                        f.write(base64.b64decode(initimg))
                        initimgfile = "./img2img-tmp.png"

            # Use previous seed if set to -1
            initSeed = jobRequest.seed
            if initSeed == -1:
                initSeed = self.__model.seed

            # Zero gfpgan strength if the model doesn't exist
            # TODO: determine if this could be at the top now? Used to cause circular import
            from ldm.gfpgan.gfpgan_tools import gfpgan_model_exists

            if not gfpgan_model_exists:
                jobRequest.enable_gfpgan = False

            # Signal start
            self.__on_start(jobRequest)

            # Generate in model
            # TODO: Split job generation requests instead of fitting all parameters here
            # TODO: Support no generation (just upscaling/gfpgan)

            upscale = None if not jobRequest.enable_upscale else jobRequest.upscale
            gfpgan_strength = (
                0 if not jobRequest.enable_gfpgan else jobRequest.gfpgan_strength
            )

            if not jobRequest.enable_generate:
                # If not generating, check if we're upscaling or running gfpgan
                if not upscale and not gfpgan_strength:
                    # Invalid settings (TODO: Add message to help user)
                    raise CanceledException()

                image = Image.open(initimgfile)
                # TODO: support progress for upscale?
                self.__model.upscale_and_reconstruct(
                    image_list=[[image, 0]],
                    upscale=upscale,
                    strength=gfpgan_strength,
                    save_original=False,
                    image_callback=lambda image, seed, upscaled=False: self.__on_image_result(
                        jobRequest, image, seed, upscaled
                    ),
                )

            else:
                # Generating - run the generation
                init_img = (
                    None
                    if (not jobRequest.enable_img2img or jobRequest.strength == 0)
                    else initimgfile
                )

                self.__model.prompt2image(
                    prompt=jobRequest.prompt,
                    init_img=init_img,  # TODO: ensure this works
                    strength=None if init_img is None else jobRequest.strength,
                    fit=None if init_img is None else jobRequest.fit,
                    iterations=jobRequest.iterations,
                    cfg_scale=jobRequest.cfg_scale,
                    width=jobRequest.width,
                    height=jobRequest.height,
                    seed=jobRequest.seed,
                    steps=jobRequest.steps,
                    variation_amount=jobRequest.variation_amount,
                    with_variations=jobRequest.with_variations,
                    gfpgan_strength=gfpgan_strength,
                    upscale=upscale,
                    sampler_name=jobRequest.sampler_name,
                    seamless=jobRequest.seamless,
                    embiggen=jobRequest.embiggen,
                    embiggen_tiles=jobRequest.embiggen_tiles,
                    step_callback=lambda sample, step: self.__on_progress(
                        jobRequest, sample, step
                    ),
                    image_callback=lambda image, seed, upscaled=False: self.__on_image_result(
                        jobRequest, image, seed, upscaled
                    ),
                )
                # TODO: add catch_interrupts as an option for console app (injected through options)

        except CanceledException:
            self.__signal_service.emit(Signal.job_canceled(jobRequest.id))

        finally:
            self.__signal_service.emit(Signal.job_done(jobRequest.id))

            # Remove the temp file
            if initimgfile is not None:
                os.remove("./img2img-tmp.png")
