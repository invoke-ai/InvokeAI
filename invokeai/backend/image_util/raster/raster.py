import numpy as np

from invokeai.backend.image_util.raster.descriptor import (
    RasterConversionError,
    RasterDescriptor,
    RasterDescriptorError,
    ValueRange,
)

SUPPORTED_DTYPES = (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float16), np.dtype(np.float32))


def _root_nbytes(array: np.ndarray) -> int:
    """Size of the buffer that keeps `array` alive. Never less than the outermost ndarray in its base chain."""
    root = array
    while isinstance(root.base, np.ndarray):
        root = root.base
    owner = root.base
    if owner is None:
        return root.nbytes
    if hasattr(owner, "untyped_storage"):  # torch.Tensor: a slice keeps the whole storage alive
        return max(owner.untyped_storage().nbytes(), root.nbytes)
    try:
        return max(memoryview(owner).nbytes, root.nbytes)
    except TypeError:
        return root.nbytes


def _freeze_chain(array: np.ndarray) -> None:
    """Make `array` and every ndarray it is a view of read-only, so no view can be switched back to writable."""
    node = array
    while isinstance(node, np.ndarray):
        node.flags.writeable = False
        node = node.base


def _restore(array: np.ndarray, descriptor: RasterDescriptor) -> "RasterImage":
    return RasterImage.from_array(array, descriptor)


class RasterImage:
    """Immutable CPU raster: one NumPy (H, W, C) array plus the descriptor that says what the numbers mean.

    Nothing here normalizes values or changes dtype or color encoding. Use `from_array` (copy) unless a
    zero-copy view is deliberate (`borrow`). "Owned" means the raster holds its own C-contiguous array.
    """

    __slots__ = ("_descriptor", "_view", "_owns")

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError("create rasters with RasterImage.from_array() or RasterImage.borrow()")

    @classmethod
    def _create(cls, view: np.ndarray, descriptor: RasterDescriptor, owns: bool) -> "RasterImage":
        _validate(view, descriptor)
        view.flags.writeable = False
        raster = object.__new__(cls)
        raster._view = view
        raster._descriptor = descriptor
        raster._owns = owns
        return raster

    @classmethod
    def from_array(cls, array: np.ndarray, descriptor: RasterDescriptor) -> "RasterImage":
        """Copy `array` into a contiguous array the raster owns. Later changes to `array` do not show through."""
        _validate(array, descriptor)
        return cls._adopt(np.array(array, order="C", copy=True), descriptor)

    @classmethod
    def borrow(cls, array: np.ndarray, descriptor: RasterDescriptor) -> "RasterImage":
        """Zero-copy read-only view of `array`. The caller must not mutate `array` while the raster is in use."""
        _validate(array, descriptor)
        return cls._create(array.view(), descriptor, owns=False)

    @classmethod
    def _adopt(cls, array: np.ndarray, descriptor: RasterDescriptor) -> "RasterImage":
        """Take a freshly built, C-contiguous array nobody else references (conversion results) without copying."""
        # Freeze the owners too: a view of a writable array could be made writable again.
        _freeze_chain(array)
        return cls._create(array.view(), descriptor, owns=True)

    @property
    def descriptor(self) -> RasterDescriptor:
        return self._descriptor

    @property
    def dtype(self) -> np.dtype:
        return self._view.dtype

    @property
    def height(self) -> int:
        return self._view.shape[0]

    @property
    def width(self) -> int:
        return self._view.shape[1]

    @property
    def channel_count(self) -> int:
        return self._view.shape[2]

    @property
    def nbytes(self) -> int:
        return self._view.nbytes

    @property
    def retained_nbytes(self) -> int:
        """Bytes kept alive by this raster, including the parent buffer of a borrowed crop or slice."""
        return _root_nbytes(self._view)

    @property
    def is_owned(self) -> bool:
        return self._owns

    def view(self) -> np.ndarray:
        """Read-only HWC array, no copy."""
        return self._view

    def to_array(self) -> np.ndarray:
        """Writable contiguous copy."""
        return np.array(self._view, order="C", copy=True)

    def owned(self) -> "RasterImage":
        """Self when it already owns a compact array, else a compact copy (for caches and persistence)."""
        return self if self.is_owned else RasterImage.from_array(self._view, self._descriptor)

    def __reduce__(self):
        # Pickle and deepcopy rebuild through the read-only, owned path.
        return _restore, (self._view, self._descriptor)

    def __repr__(self) -> str:
        return (
            f"RasterImage({self.width}x{self.height}x{self.channel_count} {self.dtype}, "
            f"{self._descriptor.channels.value}/{self._descriptor.color.value}/{self._descriptor.value_range.value})"
        )


def _validate(array: np.ndarray, descriptor: RasterDescriptor) -> None:
    if not isinstance(array, np.ndarray):
        raise RasterDescriptorError(f"expected a numpy array, got {type(array).__name__}")
    if array.dtype not in SUPPORTED_DTYPES:
        raise RasterConversionError(f"unsupported raster dtype {array.dtype}")
    if array.ndim != 3:
        raise RasterDescriptorError(f"raster arrays must be (H, W, C), got shape {array.shape}")
    height, width, channels = array.shape
    if height < 1 or width < 1:
        raise RasterDescriptorError(f"raster must not be empty, got shape {array.shape}")
    if channels != descriptor.channel_count:
        raise RasterDescriptorError(f"array has {channels} channels, descriptor says {descriptor.channel_count}")
    if descriptor.value_range is ValueRange.UNBOUNDED and array.dtype.kind != "f":
        raise RasterDescriptorError("ValueRange.UNBOUNDED is only valid for float rasters")
