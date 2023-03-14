import json
from invokeai.app.api_app import app
from fastapi.openapi.utils import get_openapi

openapi_doc = get_openapi(
    title=app.title,
    version=app.version,
    openapi_version=app.openapi_version,
    routes=app.routes,
)

with open("./openapi.json", "w") as f:
    json.dump(openapi_doc, f)
