from typing import Any, Callable

import torch


def autocast(x: Any) -> Any:
    return x.get_autocasted_tensor() if isinstance(x, AutocastTensor) else x


def unwrap(x: Any) -> Any:
    return x.get_original_tensor() if isinstance(x, AutocastTensor) else x


def autocast_and_run(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Helper function that does the following:
    1. Autocasts all AutocastTensors torch.Tensors on their respective target devices.
    2. Applies the function to the autocasted tensors.
    """
    autocasted_args = [autocast(a) for a in args]
    autocasted_kwargs = {k: autocast(v) for k, v in kwargs.items()}
    return func(*autocasted_args, **autocasted_kwargs)


def apply_to_original_tensor(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Helper function that does the following:
    1. Unwraps all AutocastTensors to torch.Tensors.
    2. Applies the function to the unwrapped tensors.
    3. Wraps the result in an AutocastTensor.
    """
    devices = [a.device for a in args if isinstance(a, AutocastTensor)]
    # Assert that all target devices are the same.
    assert len(set(devices)) == 1

    unwrapped_args = [unwrap(a) for a in args]
    unwrapped_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
    result = func(*unwrapped_args, **unwrapped_kwargs)
    return AutocastTensor(result, devices[0])


AUTOCAST_TENSOR_OP_TABLE = {
    # Ops to run on the autocasted tensor.
    torch.ops.aten.t.default: autocast_and_run,  # pyright: ignore
    torch.ops.aten.addmm.default: autocast_and_run,  # pyright: ignore
    torch.ops.aten.mul.Tensor: autocast_and_run,  # pyright: ignore
    torch.ops.aten.add.Tensor: autocast_and_run,  # pyright: ignore
    # Ops to run on the original tensor.
    torch.ops.aten.detach.default: apply_to_original_tensor,  # pyright: ignore
    torch.ops.aten.view.default: apply_to_original_tensor,  # pyright: ignore
    torch.ops.aten.copy_.default: apply_to_original_tensor,  # pyright: ignore
}


class AutocastTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        target_device: torch.device,
    ):
        """The base torch.Tensor.__new__() method does some non-standard stuff to initialize the underlying tensor
        storage in C. We need to override it to ensure that it receives the expected arguments.
        """
        return torch.Tensor._make_subclass(cls, data, data.requires_grad)

    def __init__(
        self,
        data: torch.Tensor,
        target_device: torch.device,
    ):
        """Intialize an AutocastTensor.

        Args:
            data: The tensor to autocast.
            target_device: The device to autocast to.
        """
        super().__init__()
        self._original_tensor = data
        self._target_device = target_device

    def __repr__(self):
        return "AutocastTensor containing:\n" + super().__repr__()

    def get_autocasted_tensor(self):
        return self._original_tensor.to(device=self._target_device)

    def get_original_tensor(self):
        return self._original_tensor

    @property
    def device(self):
        return self._target_device

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        # We will likely hit cases here in the future where a new op is encountered that is not yet supported.
        # The new op simply needs to be added to the AUTOCAST_TENSOR_OP_TABLE.
        if func in AUTOCAST_TENSOR_OP_TABLE:
            return AUTOCAST_TENSOR_OP_TABLE[func](func, args, kwargs)
        return NotImplemented
