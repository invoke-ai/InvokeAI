from invokeai.app.services.names.names_base import NameServiceBase
from invokeai.app.util.misc import uuid_string


class SimpleNameService(NameServiceBase):
    """Creates image names from UUIDs."""

    # TODO: Add customizable naming schemes
    def create_image_name(self) -> str:
        uuid_str = uuid_string()
        filename = f"{uuid_str}.png"
        return filename

    def create_video_name(self) -> str:
        uuid_str = uuid_string()
        filename = f"{uuid_str}.mp4"
        return filename

    def create_canvas_project_name(self) -> str:
        # Canvas project names are bare UUIDs without an extension; the file-storage layer
        # appends `.invk` (and `.webp` for the thumbnail) on disk.
        return uuid_string()
