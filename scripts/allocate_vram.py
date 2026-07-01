import argparse

import torch


def display_vram_usage():
    """Displays the total, allocated, and free VRAM on the current CUDA device."""

    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device("cuda")

    total_vram = torch.cuda.get_device_properties(device).total_memory
    allocated_vram = torch.cuda.memory_allocated(device)
    free_vram = total_vram - allocated_vram

    print(f"Total VRAM: {total_vram / (1024 * 1024 * 1024):.2f} GB")
    print(f"Allocated VRAM: {allocated_vram / (1024 * 1024 * 1024):.2f} GB")
    print(f"Free VRAM: {free_vram / (1024 * 1024 * 1024):.2f} GB")


def allocate_vram(target_gb: float, target_free: bool = False):
    """Allocates VRAM on the current CUDA device. After allocation, the script will pause until the user presses Enter
    or ends the script, at which point the VRAM will be released.

    Args:
        target_gb (float): Amount of VRAM to allocate in GB.
        target_free (bool, optional): Instead of allocating <target_gb> VRAM, enough VRAM will be allocated so the system has <target_gb> of VRAM free. For example, if <target_gb> is 2 GB, the script will allocate VRAM until the free VRAM is 2 GB.
    """
    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device("cuda")

    if target_free:
        total_vram = torch.cuda.get_device_properties(device).total_memory
        free_vram = total_vram - torch.cuda.memory_allocated(device)
        target_free_bytes = target_gb * 1024 * 1024 * 1024
        bytes_to_allocate = free_vram - target_free_bytes

        if bytes_to_allocate <= 0:
            print(f"Already at or below the target free VRAM of {target_gb} GB")
            return
    else:
        bytes_to_allocate = target_gb * 1024 * 1024 * 1024

    # FloatTensor (4 bytes per element)
    _tensor = torch.empty(int(bytes_to_allocate / 4), dtype=torch.float, device="cuda")

    display_vram_usage()

    input("Press Enter to release VRAM allocation and exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Allocate VRAM for testing purposes. Only works on CUDA devices.")
    parser.add_argument("target_gb", type=float, help="Amount of VRAM to allocate in GB.")
    parser.add_argument(
        "--target-free",
        action="store_true",
        help="Instead of allocating <target_gb> VRAM, enough VRAM will be allocated so the system has <target_gb> of VRAM free. For example, if <target_gb> is 2 GB, the script will allocate VRAM until the free VRAM is 2 GB.",
    )

    args = parser.parse_args()

    allocate_vram(target_gb=args.target_gb, target_free=args.target_free)
