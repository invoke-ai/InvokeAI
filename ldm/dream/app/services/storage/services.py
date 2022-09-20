# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from argparse import ArgumentParser
from glob import glob
import json
import os
from pathlib import Path
from queue import Queue
from shlex import shlex
from ldm.dream.args import Args
from PIL import Image

from ldm.dream.pngwriter import PngWriter
from ldm.dream.app.services.models import DreamResult, JobRequest, PaginatedItems, Signal


class JobQueueService:
    __queue: Queue = Queue()

    def push(self, jobRequest: JobRequest):
        self.__queue.put(jobRequest)

    def get(self, timeout: float = None) -> JobRequest:
        return self.__queue.get(timeout=timeout)


class SignalQueueService:
    __queue: Queue = Queue()

    def push(self, signal: Signal):
        self.__queue.put(signal)

    def get(self, block=False) -> Signal:
        return self.__queue.get(block=block)


class ImageStorageService:
    __location: str
    __pngWriter: PngWriter
    __legacyParser: ArgumentParser

    def __init__(self, location):
        self.__location = location
        self.__pngWriter = PngWriter(self.__location)
        self.__legacyParser = Args()  # TODO: inject this?

        # Create the storage directory if it doesn't exist
        Path(location).mkdir(parents=True, exist_ok=True)

    def __getName(self, dreamId: str, postfix: str = "") -> str:
        return f"{dreamId}{postfix}.png"

    def save(self, image, dreamResult: DreamResult, postfix: str = "") -> str:
        name = self.__getName(dreamResult.id, postfix)
        meta = (
            dreamResult.to_json()
        )  # TODO: make all methods consistent with writing metadata. Standardize metadata.
        path = self.__pngWriter.save_image_and_prompt_to_png(
            image, dream_prompt=None, metadata=meta, name=name
        )
        return path

    def path(self, dreamId: str, postfix: str = "") -> str:
        name = self.__getName(dreamId, postfix)
        path = os.path.join(self.__location, name)
        return path

    # Returns true if found, false if not found or error
    def delete(self, dreamId: str, postfix: str = "") -> bool:
        path = self.path(dreamId, postfix)
        if os.path.exists(path):
            os.remove(path)
            return True
        else:
            return False

    def getMetadata(self, dreamId: str, postfix: str = "") -> DreamResult:
        path = self.path(dreamId, postfix)
        image = Image.open(path)
        text = image.text
        if text.__contains__("sd-metadata") or text.get(
            "Dream"
        ):  # TODO: get this from PngWriter
            dreamMeta = text.get("sd-metadata") or text.get("Dream")
            try:
                j = json.loads(dreamMeta)
                return DreamResult.from_json(j)
            except ValueError:
                # Try to parse command-line format (legacy metadata format)
                try:
                    opt = self.__parseLegacyMetadata(dreamMeta)
                    optd = opt.__dict__
                    if (not "width" in optd) or (optd.get("width") is None):
                        optd["width"] = image.width
                    if (not "height" in optd) or (optd.get("height") is None):
                        optd["height"] = image.height
                    if (not "steps" in optd) or (optd.get("steps") is None):
                        optd[
                            "steps"
                        ] = 10  # No way around this unfortunately - seems like it wasn't storing this previously

                    optd["time"] = os.path.getmtime(
                        path
                    )  # Set timestamp manually (won't be exactly correct though)

                    return DreamResult.from_json(optd)

                except:
                    return None
        else:
            return None

    def __parseLegacyMetadata(self, command: str) -> DreamResult:
        # before splitting, escape single quotes so as not to mess
        # up the parser
        command = command.replace("'", "\\'")

        try:
            elements = shlex.split(command)
        except ValueError as e:
            return None

        # rearrange the arguments to mimic how it works in the Dream bot.
        switches = [""]
        switches_started = False

        for el in elements:
            if el[0] == "-" and not switches_started:
                switches_started = True
            if switches_started:
                switches.append(el)
            else:
                switches[0] += el
                switches[0] += " "
        switches[0] = switches[0][: len(switches[0]) - 1]

        try:
            opt = self.__legacyParser.parse_cmd(switches)
            return opt
        except SystemExit:
            return None

    def list_files(self, page: int, perPage: int) -> PaginatedItems:
        files = sorted(
            glob(os.path.join(self.__location, "*.png")),
            key=os.path.getmtime,
            reverse=True,
        )
        count = len(files)

        startId = page * perPage
        pageCount = int(count / perPage) + 1
        endId = min(startId + perPage, count)
        items = [] if startId >= count else files[startId:endId]

        items = list(map(lambda f: Path(f).stem, items))

        return PaginatedItems(items, page, pageCount, perPage, count)
