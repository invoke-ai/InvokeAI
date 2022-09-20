# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

"""Containers module."""

from flask_socketio import SocketIO
from dependency_injector import containers, providers
from ldm.dream.app.services.signaling.services import SignalService


class SignalingContainer(containers.DeclarativeContainer):
    signal_queue_service = providers.Dependency()

    socketio = providers.ThreadSafeSingleton(SocketIO, app=None)

    signal_service = providers.ThreadSafeSingleton(
        SignalService, socketio=socketio, queue=signal_queue_service
    )
