"""FastAPI route for model relationship records."""

from typing import List

from fastapi import APIRouter, Body, HTTPException, Path, status
from pydantic import BaseModel, Field

from invokeai.app.api.dependencies import ApiDependencies

model_relationships_router = APIRouter(prefix="/v1/model_relationships", tags=["model_relationships"])

# === Schemas ===


class ModelRelationshipCreateRequest(BaseModel):
    model_key_1: str = Field(
        ...,
        description="The key of the first model in the relationship",
        examples=[
            "aa3b247f-90c9-4416-bfcd-aeaa57a5339e",
            "ac32b914-10ab-496e-a24a-3068724b9c35",
            "d944abfd-c7c3-42e2-a4ff-da640b29b8b4",
            "b1c2d3e4-f5a6-7890-abcd-ef1234567890",
            "12345678-90ab-cdef-1234-567890abcdef",
            "fedcba98-7654-3210-fedc-ba9876543210",
        ],
    )
    model_key_2: str = Field(
        ...,
        description="The key of the second model in the relationship",
        examples=[
            "3bb7c0eb-b6c8-469c-ad8c-4d69c06075e4",
            "f0c3da4e-d9ff-42b5-a45c-23be75c887c9",
            "38170dd8-f1e5-431e-866c-2c81f1277fcc",
            "c57fea2d-7646-424c-b9ad-c0ba60fc68be",
            "10f7807b-ab54-46a9-ab03-600e88c630a1",
            "f6c1d267-cf87-4ee0-bee0-37e791eacab7",
        ],
    )


class ModelRelationshipBatchRequest(BaseModel):
    model_keys: List[str] = Field(
        ...,
        description="List of model keys to fetch related models for",
        examples=[
            [
                "aa3b247f-90c9-4416-bfcd-aeaa57a5339e",
                "ac32b914-10ab-496e-a24a-3068724b9c35",
            ],
            [
                "b1c2d3e4-f5a6-7890-abcd-ef1234567890",
                "12345678-90ab-cdef-1234-567890abcdef",
                "fedcba98-7654-3210-fedc-ba9876543210",
            ],
            [
                "3bb7c0eb-b6c8-469c-ad8c-4d69c06075e4",
            ],
        ],
    )


# === Routes ===


@model_relationships_router.get(
    "/i/{model_key}",
    operation_id="get_related_models",
    response_model=list[str],
    responses={
        200: {
            "description": "A list of related model keys was retrieved successfully",
            "content": {
                "application/json": {
                    "example": [
                        "15e9eb28-8cfe-47c9-b610-37907a79fc3c",
                        "71272e82-0e5f-46d5-bca9-9a61f4bd8a82",
                        "a5d7cd49-1b98-4534-a475-aeee4ccf5fa2",
                    ]
                }
            },
        },
        404: {"description": "The specified model could not be found"},
        422: {"description": "Validation error"},
    },
)
async def get_related_models(
    model_key: str = Path(..., description="The key of the model to get relationships for"),
) -> list[str]:
    """
    Get a list of model keys related to a given model.
    """
    try:
        return ApiDependencies.invoker.services.model_relationships.get_related_model_keys(model_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@model_relationships_router.post(
    "/",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "The relationship was successfully created"},
        400: {"description": "Invalid model keys or self-referential relationship"},
        409: {"description": "The relationship already exists"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
    summary="Add Model Relationship",
    description="Creates a **bidirectional** relationship between two models, allowing each to reference the other as related.",
)
async def add_model_relationship(
    req: ModelRelationshipCreateRequest = Body(..., description="The model keys to relate"),
) -> None:
    """
    Add a relationship between two models.

    Relationships are bidirectional and will be accessible from both models.

    - Raises 400 if keys are invalid or identical.
    - Raises 409 if the relationship already exists.
    """
    try:
        if req.model_key_1 == req.model_key_2:
            raise HTTPException(status_code=400, detail="Cannot relate a model to itself.")

        ApiDependencies.invoker.services.model_relationships.add_model_relationship(
            req.model_key_1,
            req.model_key_2,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@model_relationships_router.delete(
    "/",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "The relationship was successfully removed"},
        400: {"description": "Invalid model keys or self-referential relationship"},
        404: {"description": "The relationship does not exist"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
    summary="Remove Model Relationship",
    description="Removes a **bidirectional** relationship between two models. The relationship must already exist.",
)
async def remove_model_relationship(
    req: ModelRelationshipCreateRequest = Body(..., description="The model keys to disconnect"),
) -> None:
    """
    Removes a bidirectional relationship between two model keys.

    - Raises 400 if attempting to unlink a model from itself.
    - Raises 404 if the relationship was not found.
    """
    try:
        if req.model_key_1 == req.model_key_2:
            raise HTTPException(status_code=400, detail="Cannot unlink a model from itself.")

        ApiDependencies.invoker.services.model_relationships.remove_model_relationship(
            req.model_key_1,
            req.model_key_2,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@model_relationships_router.post(
    "/batch",
    operation_id="get_related_models_batch",
    response_model=List[str],
    responses={
        200: {
            "description": "Related model keys retrieved successfully",
            "content": {
                "application/json": {
                    "example": [
                        "ca562b14-995e-4a42-90c1-9528f1a5921d",
                        "cc0c2b8a-c62e-41d6-878e-cc74dde5ca8f",
                        "18ca7649-6a9e-47d5-bc17-41ab1e8cec81",
                        "7c12d1b2-0ef9-4bec-ba55-797b2d8f2ee1",
                        "c382eaa3-0e28-4ab0-9446-408667699aeb",
                        "71272e82-0e5f-46d5-bca9-9a61f4bd8a82",
                        "a5d7cd49-1b98-4534-a475-aeee4ccf5fa2",
                    ]
                }
            },
        },
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
    summary="Get Related Model Keys (Batch)",
    description="Retrieves all **unique related model keys** for a list of given models. This is useful for contextual suggestions or filtering.",
)
async def get_related_models_batch(
    req: ModelRelationshipBatchRequest = Body(..., description="Model keys to check for related connections"),
) -> list[str]:
    """
    Accepts multiple model keys and returns a flat list of all unique related keys.

    Useful when working with multiple selections in the UI or cross-model comparisons.
    """
    try:
        all_related: set[str] = set()
        for key in req.model_keys:
            related = ApiDependencies.invoker.services.model_relationships.get_related_model_keys(key)
            all_related.update(related)
        return list(all_related)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
