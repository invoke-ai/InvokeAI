import platform
import subprocess
from typing import List, Optional

import psutil
from pydantic import BaseModel


class GPUStat(BaseModel):
    id: int
    load: float
    memory: float
    memory_total: float


class SystemStats(BaseModel):
    cpu_usage: float
    ram_usage: float
    gpu_usage: Optional[List[GPUStat]]


# Function to fetch NVIDIA GPU stats (using nvidia-smi)
def get_nvidia_stats() -> Optional[List[GPUStat]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        output_lines = result.stdout.splitlines()

        gpu_usage = []
        for line in output_lines:
            parts = line.split(", ")
            if len(parts) == 4:
                gpu_id = int(parts[0])
                load = float(parts[1])
                memory_used = float(parts[2])
                memory_total = float(parts[3])
                gpu_usage.append(GPUStat(id=gpu_id, load=load, memory=memory_used, memory_total=memory_total))
        return gpu_usage if gpu_usage else None
    except Exception as e:
        print(f"Error fetching NVIDIA GPU stats: {e}")
        return None


# Function to fetch AMD GPU stats (using rocm-smi, needs testing/fixing by someone with AMD gpu)
def get_amd_stats() -> Optional[List[GPUStat]]:
    try:
        result = subprocess.run(["rocm-smi", "--showuse"], capture_output=True, text=True)
        output_lines = result.stdout.splitlines()

        gpu_usage = []
        for line in output_lines:
            if "GPU" in line:
                parts = line.split()
                if len(parts) >= 4:
                    gpu_id = int(parts[0])
                    load = float(parts[1])
                    memory_used = float(parts[2])
                    memory_total = float(parts[3])
                    gpu_usage.append(GPUStat(id=gpu_id, load=load, memory=memory_used, memory_total=memory_total))
        return gpu_usage if gpu_usage else None
    except Exception as e:
        print(f"Error fetching AMD GPU stats: {e}")
        return None


# Function to fetch Mac MPS GPU stats (placeholder, needs someone with Mac knowledge)
def get_mps_stats() -> Optional[List[GPUStat]]:
    try:
        # Using ioreg to get MPS stats on macOS
        result = subprocess.run(["ioreg", "-r", "-d", "AppleGPU"], capture_output=True, text=True)
        output_lines = result.stdout.splitlines()

        gpu_usage = []
        for line in output_lines:
            if "AppleGPU" in line:
                # Placeholder logic for parsing; needs to be implemented based on actual ioreg output
                gpu_id = len(gpu_usage)
                load = 60.0
                memory_used = 8192
                memory_total = 16384
                gpu_usage.append(GPUStat(id=gpu_id, load=load, memory=memory_used, memory_total=memory_total))
        return gpu_usage if gpu_usage else None
    except Exception as e:
        print(f"Error fetching MPS GPU stats: {e}")
        return None


def get_system_stats() -> SystemStats:
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().used / (1024**2)

    gpu_usage = None
    system_type = platform.system()

    if system_type in ["Windows", "Linux"]:
        gpu_usage = get_nvidia_stats()

        if gpu_usage is None:
            gpu_usage = get_amd_stats()

    elif system_type == "Darwin":
        gpu_usage = get_mps_stats()

    return SystemStats(cpu_usage=cpu_usage, ram_usage=ram_usage, gpu_usage=gpu_usage)


if __name__ == "__main__":
    stats = get_system_stats()
    print(stats)
