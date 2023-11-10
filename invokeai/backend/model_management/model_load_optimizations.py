from contextlib import contextmanager

import torch


def _no_op(*args, **kwargs):
    pass


@contextmanager
def skip_torch_weight_init():
    """A context manager that monkey-patches several of the common torch layers (torch.nn.Linear, torch.nn.Conv1d, etc.)
    to skip weight initialization.

    By default, `torch.nn.Linear` and `torch.nn.ConvNd` layers initialize their weights (according to a particular
    distribution) when __init__ is called. This weight initialization step can take a significant amount of time, and is
    completely unnecessary if the intent is to load checkpoint weights from disk for the layer. This context manager
    monkey-patches common torch layers to skip the weight initialization step.
    """
    torch_modules = [torch.nn.Linear, torch.nn.modules.conv._ConvNd, torch.nn.Embedding]
    saved_functions = [m.reset_parameters for m in torch_modules]

    try:
        for torch_module in torch_modules:
            torch_module.reset_parameters = _no_op

        yield None
    finally:
        for torch_module, saved_function in zip(torch_modules, saved_functions, strict=True):
            torch_module.reset_parameters = saved_function
