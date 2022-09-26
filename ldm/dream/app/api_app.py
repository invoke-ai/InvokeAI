# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import schema_of
from pydantic.schema import schema
import uvicorn


from .invocations import *
from .invocations.baseinvocation import BaseInvocation
from .api.routers import invocation
from .api.dependencies import ApiDependencies
from ..args import Args

origins = []

# Create the app
app = FastAPI(
    title     = "Invoke AI",
    docs_url  = None,
    redoc_url = None
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = {}

# Add startup event to load dependencies
@app.on_event('startup')
async def startup_event():
    args = Args()
    config = args.parse_args()

    ApiDependencies.Initialize(config)

# Include all routers
app.include_router(
    invocation.invocation_router,
    prefix = '/api')

# Build a custom OpenAPI to include all outputs
# TODO: can outputs be included on metadata of invocation schemas somehow?
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title       = app.title,
        description = "An API for invoking AI image operations",
        version     = "1.0.0",
        routes      = app.routes
    )

    # Add all outputs
    for invoker in BaseInvocation.get_invocations():
        invoker_name = invoker.__name__
        name = f'{invoker_name}Outputs'
        output_schema = schema([invoker.Outputs], ref_prefix="#/components/schemas/")['definitions']['Outputs']
        output_schema["title"] = name
        openapi_schema["components"]["schemas"][name] = output_schema

        # Add a reference to the outputs to additionalProperties of the invoker schema
        invoker_schema = openapi_schema["components"]["schemas"][invoker_name]
        outputs_ref = { '$ref': f'#/components/schemas/{name}' }
        if 'additionalProperties' in invoker_schema:
            invoker_schema['additionalProperties']['outputs'] = outputs_ref
        else:
            invoker_additional_properties = {
                'outputs': {
                    '$ref': f'#/components/schemas/{name}'
                }
            }
            invoker_schema['additionalProperties'] = invoker_additional_properties
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Override API doc favicons
app.mount('/static', StaticFiles(directory='static/dream_web'), name='static')

@app.get("/docs", include_in_schema=False)
def overridden_swagger():
	return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title,
        swagger_favicon_url="/static/favicon.ico"
    )

@app.get("/redoc", include_in_schema=False)
def overridden_redoc():
	return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title,
        redoc_favicon_url="/static/favicon.ico"
    )

def invoke_api():
    # TODO: load configuration
    uvicorn.run(
        app,
        host = "0.0.0.0",
        port = 9090
    )


if __name__ == "__main__":
    invoke_api()
