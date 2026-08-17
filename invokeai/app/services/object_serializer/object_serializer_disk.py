import shutil
import tempfile
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Optional, TypeVar

import torch

from invokeai.app.services.object_serializer.object_serializer_base import ObjectSerializerBase
from invokeai.app.services.object_serializer.object_serializer_common import ObjectNotFoundError
from invokeai.app.util.misc import uuid_string

if TYPE_CHECKING:
    from invokeai.app.services.invoker import Invoker


T = TypeVar("T")


class ObjectSerializerDisk(ObjectSerializerBase[T]):
    """Disk-backed storage for arbitrary python objects. Serialization is handled by `torch.save` and `torch.load`.

    :param output_dir: The folder where the serialized objects will be stored
    :param safe_globals: A list of types to be added to the safe globals for torch serialization
    :param ephemeral: If True, objects will be stored in a temporary directory inside the given output_dir and cleaned up on exit
    """

    def __init__(
        self,
        output_dir: Path,
        safe_globals: list[type],
        ephemeral: bool = False,
    ) -> None:
        super().__init__()
        self._ephemeral = ephemeral
        self._base_output_dir = output_dir
        self._base_output_dir.mkdir(parents=True, exist_ok=True)

        if self._ephemeral:
            # Remove dangling tempdirs that might have been left over from an earlier unplanned shutdown.
            for temp_dir in filter(Path.is_dir, self._base_output_dir.glob("tmp*")):
                shutil.rmtree(temp_dir)

        # Must specify `ignore_cleanup_errors` to avoid fatal errors during cleanup on Windows
        self._tempdir = (
            tempfile.TemporaryDirectory(dir=self._base_output_dir, ignore_cleanup_errors=True) if ephemeral else None
        )
        self._output_dir = Path(self._tempdir.name) if self._tempdir else self._base_output_dir
        self.__obj_class_name: Optional[str] = None

        torch.serialization.add_safe_globals(safe_globals) if safe_globals else None

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
            self.__obj_class_name = typing.get_args(self.__orig_class__)[0].__name__  # pyright: ignore [reportUnknownMemberType, reportAttributeAccessIssue]
        return self.__obj_class_name

    def _get_path(self, name: str) -> Path:
        return self._output_dir / name

    def _new_name(self) -> str:
        return f"{self._obj_class_name}_{uuid_string()}"

    def _tempdir_cleanup(self) -> None:
        """Calls `cleanup` on the temporary directory, if it exists."""
        if self._tempdir:
            self._tempdir.cleanup()

    def __del__(self) -> None:
        # In case the service is not properly stopped, clean up the temporary directory when the class instance is GC'd.
        self._tempdir_cleanup()

    def stop(self, invoker: "Invoker") -> None:
        self._tempdir_cleanup()
