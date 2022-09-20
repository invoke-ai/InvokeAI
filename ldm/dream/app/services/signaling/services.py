# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from queue import Empty
from flask_socketio import SocketIO, join_room, leave_room
from ldm.dream.app.services.models import Signal
from ldm.dream.app.services.storage.services import SignalQueueService


class SignalService:
    __socketio: SocketIO
    __queue: SignalQueueService

    def __init__(self, socketio: SocketIO, queue: SignalQueueService):
        self.__socketio = socketio
        self.__queue = queue

        def on_join(data):
            room = data["room"]
            join_room(room)
            self.__socketio.emit("test", "something", room=room)

        def on_leave(data):
            room = data["room"]
            leave_room(room)

        self.__socketio.on_event("join_room", on_join)
        self.__socketio.on_event("leave_room", on_leave)

        self.__socketio.start_background_task(self.__process)

    def __process(self):
        # preload the model
        print("Started signal queue processor")
        try:
            while True:
                try:
                    signal = self.__queue.get()
                    self.__socketio.emit(
                        signal.event,
                        signal.data,
                        room=signal.job,
                        broadcast=signal.broadcast,
                    )
                except Empty:
                    pass
                finally:
                    self.__socketio.sleep(0.001)

        except KeyboardInterrupt:
            print("Signal queue processor stopped")

    def emit(self, signal: Signal):
        self.__queue.push(signal)
