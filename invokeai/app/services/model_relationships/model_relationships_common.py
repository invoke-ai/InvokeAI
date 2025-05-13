from datetime import datetime

from invokeai.app.util.model_exclude_null import BaseModelExcludeNull


class ModelRelationship(BaseModelExcludeNull):
    model_key_1: str
    model_key_2: str
    created_at: datetime
