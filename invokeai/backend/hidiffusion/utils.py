import torch


def isinstance_str(x: object, cls_name: str, prefix: bool = False, contains: bool = False):
    """
    Checks whether x has any class equal to, prefixed with, or contains (cls_name) in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__.startswith(cls_name) and prefix:
            return True
        if contains and cls_name in _cls.__name__:
            return True
        if _cls.__name__ == cls_name:
            return True
    
    return False


def init_generator(device: torch.device, fallback: torch.Generator=None):
    """
    Forks the current default random generator given device.
    """
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback
    