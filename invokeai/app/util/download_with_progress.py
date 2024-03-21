from pathlib import Path
from urllib import request

from tqdm import tqdm

from invokeai.backend.util.logging import InvokeAILogger


class ProgressBar:
    """Simple progress bar for urllib.request.urlretrieve using tqdm."""

    def __init__(self, model_name: str = "file"):
        self.pbar = None
        self.name = model_name

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if not self.pbar:
            self.pbar = tqdm(
                desc=self.name,
                initial=0,
                unit="iB",
                unit_scale=True,
                unit_divisor=1000,
                total=total_size,
            )
        self.pbar.update(block_size)


def download_with_progress_bar(name: str, url: str, dest_path: Path) -> bool:
    """Download a file from a URL to a destination path, with a progress bar.
    If the file already exists, it will not be downloaded again.

    Exceptions are not caught.

    Args:
        name (str): Name of the file being downloaded.
        url (str): URL to download the file from.
        dest_path (Path): Destination path to save the file to.

    Returns:
        bool: True if the file was downloaded, False if it already existed.
    """
    if dest_path.exists():
        return False  # already downloaded

    InvokeAILogger.get_logger().info(f"Downloading {name}...")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    request.urlretrieve(url, dest_path, ProgressBar(name))

    return True
