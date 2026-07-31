from typing import Literal
from urllib.parse import urljoin

import httpx

from invokeai.app.services.external_generation.errors import ExternalProviderRequestError
from invokeai.backend.text_llm_pipeline import DEFAULT_SYSTEM_PROMPT
from invokeai.invocation_api import (
    BaseInvocation,
    Classification,
    InputField,
    InvocationContext,
    InvokeAIAppConfig,
    StringOutput,
    UIComponent,
    invocation,
)

DEEPSEEK_API_BASE = "https://api.deepseek.com/"


def _auth_header(app_config: InvokeAIAppConfig) -> dict[str, str]:
    """Construct the Authorization header based on configured API key."""
    if not app_config.external_deepseek_api_key:
        raise ExternalProviderRequestError("DeepSeek API key is not configured")
    return {
        "Authorization": "Bearer " + app_config.external_deepseek_api_key,
    }


def call_deepseek_llm(
    app_config: InvokeAIAppConfig,
    model_name: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float = 1.0,
) -> str:
    url = urljoin(app_config.external_deepseek_base_url or DEEPSEEK_API_BASE, "/chat/completions")

    if not app_config.external_deepseek_api_key:
        raise ExternalProviderRequestError("DeepSeek API key is not configured")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    headers.update(_auth_header(app_config))

    messages: list[dict[Literal["content", "role"], str]] = []
    if system_prompt:
        messages.append({"content": system_prompt, "role": "system"})
    if prompt:
        messages.append({"content": prompt, "role": "user"})

    payload = {
        "messages": messages,
        "model": model_name,
        "thinking": {"type": "disabled"},
        "reasoning_effort": "high",  # or "max" (there is no "low")
        "max_tokens": max_tokens,
        "temperature": temperature,  # range 0–2
        "top_p": 1,
    }
    response = httpx.post(url, headers=headers, json=payload)
    response.raise_for_status()
    completion = response.json()
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


@invocation(
    "deepseek_text_llm",
    title="DeepSeek Text LLM (external API)",
    tags=["external", "llm", "text", "prompt"],
    category="llm",
    version="1.0.0",
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
    # not hardcoding model names in the schema because the API provider may change them at any time
    model: str = InputField(title="DeepSeek Model", description="`deepseek-v4-flash` or `deepseek-v4-pro`")
    max_tokens: int = InputField(
        default=300,
        ge=1,
        le=2048,
        description="Maximum number of tokens to generate.",
    )

    def invoke(self, context: InvocationContext) -> StringOutput:
        output = call_deepseek_llm(
            app_config=context.config.get(),
            model_name=self.model,
            prompt=self.prompt,
            system_prompt=self.system_prompt,
            max_tokens=self.max_tokens,
        )
        return StringOutput(value=output)
