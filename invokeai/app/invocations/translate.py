# Copyright (c) 2023 Lincoln D. Stein

from typing import Literal
from pydantic import Field
from .baseinvocation import (
    BaseInvocation,
    InvocationContext,
    InvocationConfig,
)
from .params import StringOutput

translate_available = False
try:
    import translators as ts

    translate_available = True
except:
    pass

DEFAULT_PROMPT = "" if translate_available else "To use this node, please 'pip install --upgrade translators'"


class TranslateInvocation(BaseInvocation):
    """Use the translators package to translate 330 languages into English prompts"""

    # fmt: off
    type: Literal["translate"] = "translate"

    # Inputs
    prompt: str = Field(default=DEFAULT_PROMPT, description="Prompt in any language")
    # fmt: on

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Translate", "tags": ["prompt", "translate", "translator"]},
        }

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(text=ts.translate_text(self.prompt))
