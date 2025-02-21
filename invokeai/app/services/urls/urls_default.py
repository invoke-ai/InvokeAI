import os

from invokeai.app.services.urls.urls_base import UrlServiceBase


class LocalUrlService(UrlServiceBase):
    def __init__(self, base_url: str = "api/v1", base_url_v2: str = "api/v2"):
        self._base_url = base_url
        self._base_url_v2 = base_url_v2

    def get_image_url(self, image_name: str, thumbnail: bool = False) -> str:
        image_basename = os.path.basename(image_name)

        # These paths are determined by the routes in invokeai/app/api/routers/images.py
        if thumbnail:
            return f"{self._base_url}/images/i/{image_basename}/thumbnail"

        return f"{self._base_url}/images/i/{image_basename}/full"

    def get_model_image_url(self, model_key: str) -> str:
        return f"{self._base_url_v2}/models/i/{model_key}/image"

    def get_style_preset_image_url(self, style_preset_id: str) -> str:
        return f"{self._base_url}/style_presets/i/{style_preset_id}/image"

    def get_workflow_thumbnail_url(self, workflow_id: str) -> str:
        return f"{self._base_url}/workflows/i/{workflow_id}/thumbnail"
