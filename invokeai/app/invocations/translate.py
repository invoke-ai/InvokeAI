# Copyright (c) 2023 Lincoln D. Stein

from typing import Literal, Union, List
from pydantic import Field
from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    InvocationConfig,
)

# from .params import StringOutput

translate_available = False
try:
    import translators as ts

    translate_available = True
except:
    # need dummy ts for regression tests to pass
    class DummyTranslator:
        @classmethod
        @property
        def translators_pool(cls) -> List[str]:
            pass

        @classmethod
        def translate_text(cls, **kwargs) -> Union[str, dict]:
            pass

    ts = DummyTranslator

DEFAULT_PROMPT = "" if translate_available else "To use this node, please 'pip install --upgrade translators'"


class TranslateOutput(BaseInvocationOutput):
    """Translated string output"""

    type: Literal["translated_string_output"] = "translated_string_output"
    prompt: str = Field(default=None, description="The translated prompt string")


class TranslateInvocation(BaseInvocation):
    """Use the translators package to translate 330 languages into English prompts"""

    # fmt: off
    type: Literal["translate"] = "translate"

    # Inputs
    text: str = Field(default=DEFAULT_PROMPT, description="Prompt in any language")
    translator: Literal[tuple(ts.translators_pool)] = Field(default="google", description="The translator service to use")
    # fmt: on

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Translate", "tags": ["prompt", "translate", "translator"]},
        }

    def invoke(self, context: InvocationContext) -> TranslateOutput:
        translation: str = ts.translate_text(self.text, translator=self.translator)
        return TranslateOutput(prompt=translation)
