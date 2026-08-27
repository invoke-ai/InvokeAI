import json
from typing import Any

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import InputField
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation(
    "fal_generic_media",
    title="fal.ai Generic Media",
    tags=["external", "generation", "fal", "fal.ai", "generic"],
    category="media",
    version="1.0.0",
)
class FalGenericMediaInvocation(BaseInvocation):
    """Submit arbitrary JSON to any fal.ai endpoint and return its raw JSON result."""

    model_id: str = InputField(description="fal.ai endpoint ID, for example fal-ai/kling-video/v3/pro/text-to-video")
    input_json: str = InputField(default="{}", description="JSON object sent to the fal.ai endpoint")

    def invoke(self, context: InvocationContext) -> StringOutput:
        try:
            payload = json.loads(self.input_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"fal.ai input_json is not valid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError("fal.ai input_json must be a JSON object")

        result: dict[str, Any] = context._services.external_generation.generate_generic(
            provider_id="fal",
            model_id=self.model_id,
            payload=payload,
        )
        return StringOutput(value=json.dumps(result, ensure_ascii=False, sort_keys=True))
