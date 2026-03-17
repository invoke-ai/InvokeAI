from abc import ABC, abstractmethod
from datetime import datetime

from invokeai.app.services.image_records.image_records_common import ImageCategory


class ImageSubfolderStrategy(ABC):
    """Base class for image subfolder strategies."""

    @abstractmethod
    def get_subfolder(self, image_name: str, image_category: ImageCategory, is_intermediate: bool) -> str:
        """Returns relative subfolder prefix (e.g. '2026/03/17', 'general'), or empty string for flat."""
        pass


class FlatStrategy(ImageSubfolderStrategy):
    """No subfolders - all images in one directory (default behavior)."""

    def get_subfolder(self, image_name: str, image_category: ImageCategory, is_intermediate: bool) -> str:
        return ""


class DateStrategy(ImageSubfolderStrategy):
    """Organize images by date: YYYY/MM/DD."""

    def get_subfolder(self, image_name: str, image_category: ImageCategory, is_intermediate: bool) -> str:
        now = datetime.now()
        return f"{now.year}/{now.month:02d}/{now.day:02d}"


class TypeStrategy(ImageSubfolderStrategy):
    """Organize images by category/type: general, intermediate, mask, control, etc."""

    def get_subfolder(self, image_name: str, image_category: ImageCategory, is_intermediate: bool) -> str:
        if is_intermediate:
            return "intermediate"
        return image_category.value


class HashStrategy(ImageSubfolderStrategy):
    """Organize images by UUID prefix for filesystem performance (first 2 characters)."""

    def get_subfolder(self, image_name: str, image_category: ImageCategory, is_intermediate: bool) -> str:
        return image_name[:2]


def create_subfolder_strategy(strategy_name: str) -> ImageSubfolderStrategy:
    """Factory function to create a subfolder strategy by name."""
    strategies: dict[str, type[ImageSubfolderStrategy]] = {
        "flat": FlatStrategy,
        "date": DateStrategy,
        "type": TypeStrategy,
        "hash": HashStrategy,
    }
    cls = strategies.get(strategy_name)
    if cls is None:
        raise ValueError(f"Unknown subfolder strategy: {strategy_name}. Valid options: {', '.join(strategies.keys())}")
    return cls()
