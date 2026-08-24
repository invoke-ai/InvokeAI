from base64 import b64encode
from io import BytesIO
from typing import Any, Literal
from urllib.parse import urljoin

import httpx
from PIL.Image import Image

from invokeai.app.services.external_generation.errors import ExternalProviderRequestError
from invokeai.backend.text_llm_pipeline import DEFAULT_SYSTEM_PROMPT
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.invocation_api import (
    BaseInvocation,
    Classification,
    ImageField,
    InputField,
    InvocationContext,
    InvokeAIAppConfig,
    StringOutput,
    UIComponent,
    invocation,
)

DEEPSEEK_API_BASE = "https://api.deepseek.com/"

ResponseFormat = Literal["text", "json_object"]


def _auth_header(app_config: InvokeAIAppConfig) -> dict[str, str]:
    """Construct the Authorization header based on configured API key."""
    if not app_config.external_deepseek_api_key:
        raise ExternalProviderRequestError("DeepSeek API key is not configured")
    return {
        "Authorization": "Bearer " + app_config.external_deepseek_api_key,
    }


async def async_call_deepseek_llm(
    app_config: InvokeAIAppConfig,
    model_name: str,
    prompt: str,
    system_prompt: str,
    images: list[Image],
    max_tokens: int,
    temperature: float = 1.0,
    response_format: ResponseFormat = "text",
) -> str:
    # async method may be called by a fastapi handler without a new thread.
    request = _build_request(
        app_config, model_name, prompt, system_prompt, images, max_tokens, temperature, response_format
    )
    # httpx timeout default is 5s, too low for long responses. Increase read timeout while leaving other timeouts intact.
    async with httpx.AsyncClient(timeout=httpx.Timeout(5, read=2 * max_tokens)) as client:
        response = await client.send(request)
    return _extract_response_content(response)


def call_deepseek_llm(
    app_config: InvokeAIAppConfig,
    model_name: str,
    prompt: str,
    system_prompt: str,
    images: list[Image],
    max_tokens: int,
    temperature: float = 1.0,
    response_format: ResponseFormat = "text",
) -> str:
    # Apparently there's not a great way of using an async function from a synchronous one
    # https://discuss.python.org/t/calling-coroutines-from-sync-code-2/24093
    # so we have two functions that use different client implementations.
    request = _build_request(
        app_config, model_name, prompt, system_prompt, images, max_tokens, temperature, response_format
    )
    # httpx timeout default is 5s, too low for long responses. Increase read timeout while leaving other timeouts intact.
    with httpx.Client(timeout=httpx.Timeout(5, read=2 * max_tokens)) as client:
        response = client.send(request)
    return _extract_response_content(response)


def _build_request(
    app_config: InvokeAIAppConfig,
    model_name: str,
    prompt: str,
    system_prompt: str,
    images: list[Image],
    max_tokens: int,
    temperature: float = 1.0,
    response_format: ResponseFormat = "text",
) -> httpx.Request:
    url = urljoin(app_config.external_deepseek_base_url or DEEPSEEK_API_BASE, "/chat/completions")

    if not app_config.external_deepseek_api_key:
        raise ExternalProviderRequestError("DeepSeek API key is not configured")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    headers.update(_auth_header(app_config))

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"content": system_prompt, "role": "system"})
    if prompt or images:
        if not images:
            content: Any = prompt
        else:
            content = []
            if prompt:
                content.append({"type": "text", "text": prompt})
            for image in images:
                content.append({"type": "image_url", "image_url": {"url": encode_image(image)}})
        messages.append({"content": content, "role": "user"})

    payload = {
        "messages": messages,
        "model": model_name,
        "thinking": {"type": "disabled"},
        "reasoning_effort": "high",  # or "max" (there is no "low")
        "max_tokens": max_tokens,
        "temperature": temperature,  # range 0–2
        "top_p": 1,
        "response_format": {"type": response_format},
    }
    return httpx.Request("POST", url, headers=headers, json=payload)


def _extract_response_content(response: httpx.Response) -> str:
    response.raise_for_status()
    completion = response.json()
    if usage := completion.get("usage"):
        logger = InvokeAILogger.get_logger("DeepSeek")
        reasoning_tokens = (
            usage["completion_tokens_details"]["reasoning_tokens"] if "completion_tokens_details" in usage else 0
        )
        logger.info(
            "Prompt tokens: %d (%d cached). Response tokens: %d (%d reasoning).",
            usage["prompt_tokens"],
            usage["prompt_cache_hit_tokens"],
            usage["completion_tokens"],
            reasoning_tokens,
        )
    return completion["choices"][0]["message"]["content"]


def list_deepseek_models(app_config: InvokeAIAppConfig) -> list[str]:
    url = urljoin(app_config.external_deepseek_base_url or DEEPSEEK_API_BASE, "/models")
    headers = {
        "Accept": "application/json",
    }
    headers.update(_auth_header(app_config))
    response = httpx.get(url, headers=headers)
    response.raise_for_status()
    return [m["id"] for m in response.json()["data"]]


MAX_PIXELS = 800 * 800  # size for deepseek-v4-flash-vision-exp (August 2026)
WEBP_METHOD = 2  # 0–6


def downsample_image(image: Image, max_pixels: int = MAX_PIXELS) -> Image:
    # Does this code already exist somewhere?
    pixels = image.width * image.height
    if pixels <= max_pixels:
        return image
    scale = (max_pixels / pixels) ** 0.5
    return image.resize((round(image.width * scale), round(image.height * scale)))


def encode_image(image: Image) -> str:
    with BytesIO() as b:
        image.save(b, format="WebP", method=WEBP_METHOD)
        encoded = b64encode(b.getbuffer())
        return "data:image/webp;base64," + encoded.decode("ascii")


@invocation(
    "deepseek_text_llm",
    title="DeepSeek Text LLM (external API)",
    tags=["external", "llm", "text", "prompt"],
    category="llm",
    version="2.0.0",
    classification=Classification.Prototype,
)
class DeepSeekTextApiInvocation(BaseInvocation):
    """Call the DeepSeek API to generate text (e.g. for prompt expansion)."""

    prompt: str = InputField(
        default="",
        description="Input text prompt.",
        ui_component=UIComponent.Textarea,
    )
    system_prompt: str = InputField(
        default=DEFAULT_SYSTEM_PROMPT,
        description="System prompt that guides the model's behavior.",
        ui_component=UIComponent.Textarea,
    )
    images: list[ImageField] = InputField()
    # not hardcoding model names in the schema because the API provider may change them at any time
    model: str = InputField(
        title="DeepSeek Model", description="`deepseek-v4-flash` or `deepseek-v4-pro` or `deepseek-v4-flash-vision-exp`"
    )
    max_tokens: int = InputField(
        default=300,
        ge=1,
        le=10_000,
        description="Maximum number of tokens to generate.",
    )
    response_format: ResponseFormat = InputField()

    def invoke(self, context: InvocationContext) -> StringOutput:
        scaled_images = [downsample_image(context.images.get_pil(image.image_name)) for image in self.images]
        output = call_deepseek_llm(
            app_config=context.config.get(),
            model_name=self.model,
            prompt=self.prompt,
            system_prompt=self.system_prompt,
            images=scaled_images,
            max_tokens=self.max_tokens,
        )
        return StringOutput(value=output)
