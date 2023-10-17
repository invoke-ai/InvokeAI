from dataclasses import dataclass, field
from typing import Dict

# size of GIG in bytes
GIG = 1073741824


@dataclass
class NodeStats:
    """Class for tracking execution stats of an invocation node"""

    calls: int = 0
    time_used: float = 0.0  # seconds
    max_vram: float = 0.0  # GB
    cache_hits: int = 0
    cache_misses: int = 0
    cache_high_watermark: int = 0


@dataclass
class NodeLog:
    """Class for tracking node usage"""

    # {node_type => NodeStats}
    nodes: Dict[str, NodeStats] = field(default_factory=dict)
