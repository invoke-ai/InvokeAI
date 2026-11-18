"""External starter models must not advertise capabilities their provider's invocation node can't accept.

The frontend wires graph edges from model capabilities (e.g. a `negative_prompt` edge when
`supports_negative_prompt` is true). If the node has no such field, every generation fails graph validation.
"""

import pytest

from invokeai.app.invocations.external_image_generation import BaseExternalImageGenerationInvocation
from invokeai.backend.model_manager.starter_models import STARTER_MODELS


def _invocations_by_provider() -> dict[str, type[BaseExternalImageGenerationInvocation]]:
    found: dict[str, type[BaseExternalImageGenerationInvocation]] = {}
    pending = list(BaseExternalImageGenerationInvocation.__subclasses__())
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        provider_id = getattr(cls, "provider_id", None)
        if isinstance(provider_id, str) and provider_id:
            found[provider_id] = cls
    return found


EXTERNAL_STARTERS = [m for m in STARTER_MODELS if m.source.startswith("external://") and m.capabilities is not None]


@pytest.mark.parametrize("starter", EXTERNAL_STARTERS, ids=lambda m: m.source)
def test_negative_prompt_capability_matches_invocation_fields(starter) -> None:
    provider_id = starter.source.removeprefix("external://").split("/", 1)[0]
    invocation = _invocations_by_provider().get(provider_id)
    assert invocation is not None, f"no invocation node registered for provider {provider_id!r}"
    if starter.capabilities.supports_negative_prompt:
        assert "negative_prompt" in invocation.model_fields, (
            f"{starter.source} advertises negative prompts but {invocation.__name__} has no negative_prompt field"
        )
