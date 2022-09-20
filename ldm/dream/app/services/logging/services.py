# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import os
from ldm.dream.app.services.models import DreamResult

# TODO: Name this better?
# TODO: Logging and signals should probably be event based (multiple listeners for an event)
class LogService:
    __location: str
    __logFile: str

    def __init__(self, location: str, file: str):
        self.__location = location
        self.__logFile = file

    def log(self, dreamResult: DreamResult, seed=None, upscaled=False):
        with open(os.path.join(self.__location, self.__logFile), "a") as log:
            log.write(f"{dreamResult.id}: {dreamResult.to_json()}\n")
