from typing import Optional, Union

from dynamicprompts.generators import CombinatorialPromptGenerator, RandomPromptGenerator
from fastapi import Body
from fastapi.routing import APIRouter
from pydantic import BaseModel
from pyparsing import ParseException

utilities_router = APIRouter(prefix="/v1/utilities", tags=["utilities"])


class DynamicPromptsResponse(BaseModel):
    prompts: list[str]
    error: Optional[str] = None


@utilities_router.post(
    "/dynamicprompts",
    operation_id="parse_dynamicprompts",
    responses={
        200: {"model": DynamicPromptsResponse},
    },
)
async def parse_dynamicprompts(
    prompt: str = Body(description="The prompt to parse with dynamicprompts"),
    max_prompts: int = Body(ge=1, le=10000, default=1000, description="The max number of prompts to generate"),
    combinatorial: bool = Body(default=True, description="Whether to use the combinatorial generator"),
    seed: int | None = Body(None, description="The seed to use for random generation. Only used if not combinatorial"),
) -> DynamicPromptsResponse:
    """Creates a batch process"""
    max_prompts = min(max_prompts, 10000)
    generator: Union[RandomPromptGenerator, CombinatorialPromptGenerator]
    try:
        error: Optional[str] = None
        if combinatorial:
            generator = CombinatorialPromptGenerator()
            prompts = generator.generate(prompt, max_prompts=max_prompts)
        else:
            generator = RandomPromptGenerator(seed=seed)
            prompts = generator.generate(prompt, num_images=max_prompts)
    except ParseException as e:
        prompts = [prompt]
        error = str(e)
    return DynamicPromptsResponse(prompts=prompts if prompts else [""], error=error)
