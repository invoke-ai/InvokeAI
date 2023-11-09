from invokeai.app.util.misc import uuid_string

from .names_base import NameServiceBase


class SimpleNameService(NameServiceBase):
    """Creates image names from UUIDs."""

    # TODO: Add customizable naming schemes
    def create_image_name(self) -> str:
        uuid_str = uuid_string()
        filename = f"{uuid_str}.png"
        return filename
