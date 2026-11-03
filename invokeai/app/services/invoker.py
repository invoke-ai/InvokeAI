from invokeai.app.services.invocation_services import InvocationServices


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

    def _services_but_the_session_processor(self) -> list[object]:
        return [getattr(self.services, name) for name in vars(self.services) if name != "session_processor"]

    def _start(self) -> None:
        """Starts the invoker. This is called automatically when the invoker is created.

        The session processor starts after every other service: its workers dequeue at once, and an item left pending
        before the restart (the queue's start cancels those that were in progress) may reach any service, which must
        have been started by then.
        """
        for service in self._services_but_the_session_processor():
            self.__start_service(service)
        self.__start_service(self.services.session_processor)

    def stop(self) -> None:
        """Stops the invoker. A new invoker will have to be created to execute further."""
        # The session processor stops first: its workers stop taking queue items and cancel the sessions they run, and
        # its stop() waits for them to exit (bounded, see DefaultSessionProcessor.stop) before the services those
        # sessions use stop.
        self.__stop_service(self.services.session_processor)
        for service in self._services_but_the_session_processor():
            self.__stop_service(service)
