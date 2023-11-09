from typing import Any

from pydantic import BaseModel

"""
We want to exclude null values from objects that make their way to the client.

Unfortunately there is no built-in way to do this in pydantic, so we need to override the default
dict method to do this.

From https://github.com/tiangolo/fastapi/discussions/8882#discussioncomment-5154541
"""


class BaseModelExcludeNull(BaseModel):
    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        """
        Override the default dict method to exclude None values in the response
        """
        kwargs.pop("exclude_none", None)
        return super().model_dump(*args, exclude_none=True, **kwargs)

    pass
