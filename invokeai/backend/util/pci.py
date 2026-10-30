"""Where a CUDA or HIP device sits on the PCI bus: what links a torch device to the driver interfaces that know it by
location (D3DKMT adapters, NVML, DRM fdinfo)."""

import torch


def pci_location(index: int) -> tuple[int, int, int]:
    """``(domain, bus, device)`` of torch's CUDA or HIP device `index`."""
    props = torch.cuda.get_device_properties(index)
    return int(props.pci_domain_id), int(props.pci_bus_id), int(props.pci_device_id)


def pci_address(index: int) -> str:
    """The device's address as ``domain:bus:device.function`` in lower-case hex, e.g. ``0000:03:00.0``: how DRM
    fdinfo spells it, and one of the forms NVML parses."""
    domain, bus, device = pci_location(index)
    return f"{domain:04x}:{bus:02x}:{device:02x}.0"
