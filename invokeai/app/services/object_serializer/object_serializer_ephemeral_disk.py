import typing
from pathlib import Path
from typing import Optional, TypeVar

import torch

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.object_serializer.object_serializer_base import ObjectSerializerBase
from invokeai.app.services.object_serializer.object_serializer_common import ObjectNotFoundError
from invokeai.app.util.misc import uuid_string

T = TypeVar("T")


class ObjectSerializerEphemeralDisk(ObjectSerializerBase[T]):
    """Provides a disk-backed ephemeral storage for arbitrary python objects. The storage is cleared at startup.

    :param output_folder: The folder where the objects will be stored
    """

    def __init__(self, output_dir: Path):
        super().__init__()
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.__obj_class_name: Optional[str] = None

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._delete_all()

    def load(self, name: str) -> T:
        file_path = self._get_path(name)
        try:
            return torch.load(file_path)  # pyright: ignore [reportUnknownMemberType]
        except FileNotFoundError as e:
            raise ObjectNotFoundError(name) from e

    def save(self, obj: T) -> str:
        name = self._new_name()
        file_path = self._get_path(name)
        torch.save(obj, file_path)  # pyright: ignore [reportUnknownMemberType]
        return name

    def delete(self, name: str) -> None:
        file_path = self._get_path(name)
        file_path.unlink()

    @property
    def _obj_class_name(self) -> str:
        if not self.__obj_class_name:
            # `__orig_class__` is not available in the constructor for some technical, undoubtedly very pythonic reason
            self.__obj_class_name = typing.get_args(self.__orig_class__)[0].__name__  # pyright: ignore [reportUnknownMemberType, reportGeneralTypeIssues]
        return self.__obj_class_name

    def _get_path(self, name: str) -> Path:
        return self._output_dir / name

    def _new_name(self) -> str:
        return f"{self._obj_class_name}_{uuid_string()}"

    def _delete_all(self) -> None:
        """
        Deletes all objects from disk.
        Must be called after we have access to `self._invoker` (e.g. in `start()`).
        """

        # We could try using a temporary directory here, but they aren't cleared in the event of a crash, so we'd have
        # to manually clear them on startup anyways. This is a bit simpler and more reliable.

        if not self._invoker:
            raise ValueError("Invoker is not set. Must call `start()` first.")

        deleted_count = 0
        freed_space = 0
        for file in Path(self._output_dir).glob("*"):
            if file.is_file():
                freed_space += file.stat().st_size
                deleted_count += 1
                file.unlink()
        if deleted_count > 0:
            freed_space_in_mb = round(freed_space / 1024 / 1024, 2)
            self._invoker.services.logger.info(
                f"Deleted {deleted_count} {self._obj_class_name} files (freed {freed_space_in_mb}MB)"
            )
