import psutil
import GPUtil
from pydantic import BaseModel
from typing import List

# Pydantic Models for System Stats
class GPUStat(BaseModel):
    id: int
    load: float
    memory: float
    memory_total: float

class SystemStats(BaseModel):
    cpu_usage: float
    ram_usage: float
    gpu_usage: List[GPUStat]

# Function to fetch system stats
def get_system_stats():
    # CPU stats
    cpu_usage = psutil.cpu_percent(interval=1)
    # RAM stats
    ram_usage = psutil.virtual_memory().percent
    # GPU stats (using GPUtil)
    gpus = GPUtil.getGPUs()
    gpu_usage = [{'id': gpu.id, 'load': gpu.load * 100, 'memory': gpu.memoryUsed, 'memory_total': gpu.memoryTotal} for gpu in gpus]

    return {
        'cpu_usage': cpu_usage,
        'ram_usage': ram_usage,
        'gpu_usage': gpu_usage
    }
