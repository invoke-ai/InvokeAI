# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)


from .invocation_services import InvocationServices


class Invoker:
    """The invoker, used to execute invocations"""

    services: InvocationServices

    def __init__(self, services: InvocationServices):
        self.services = services
        self._start()

    def __start_service(self, service) -> None:
        # Call start() method on any services that have it
        start_op = getattr(service, "start", None)
        if callable(start_op):
            start_op(self)

    def __stop_service(self, service) -> None:
        # Call stop() method on any services that have it
        stop_op = getattr(service, "stop", None)
        if callable(stop_op):
            stop_op(self)

    def _start(self) -> None:
        """Starts the invoker. This is called automatically when the invoker is created."""
        for service in vars(self.services):
            self.__start_service(getattr(self.services, service))

    def stop(self) -> None:
        """Stops the invoker. A new invoker will have to be created to execute further."""
        # First stop all services
        for service in vars(self.services):
            self.__stop_service(getattr(self.services, service))
