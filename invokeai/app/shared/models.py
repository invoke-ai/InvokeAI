from pydantic import BaseModel, Field

from invokeai.app.invocations.fields import FieldDescriptions


class FreeUConfig(BaseModel):
    """
    Configuration for the FreeU hyperparameters.
    - https://huggingface.co/docs/diffusers/main/en/using-diffusers/freeu
    - https://github.com/ChenyangSi/FreeU
    """

    s1: float = Field(ge=-1, le=3, description=FieldDescriptions.freeu_s1)
    s2: float = Field(ge=-1, le=3, description=FieldDescriptions.freeu_s2)
    b1: float = Field(ge=-1, le=3, description=FieldDescriptions.freeu_b1)
    b2: float = Field(ge=-1, le=3, description=FieldDescriptions.freeu_b2)
