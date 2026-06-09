# توثيق ملف: controlnet.py

## مسار الملف الأصلي
```
invokeai/app/invocations/controlnet.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/invocations/controlnet.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **نماذج ControlNet** في InvokeAI. يحتوي على نماذج البيانات والندوات المعالجة المسبقة للصور.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 مكتبات Python الأساسية
```python
from typing import List, Union
```

### 2.2 Pydantic
```python
from pydantic import BaseModel, Field, field_validator, model_validator
```

### 2.3 مكتبات المشروع
```python
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation, BaseInvocationOutput, Classification, invocation, invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, InputField, OutputField
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.invocations.util import validate_begin_end_step, validate_weights
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.controlnet_utils import (
    CONTROLNET_MODE_VALUES, CONTROLNET_RESIZE_VALUES, heuristic_resize_fast,
)
from invokeai.backend.image_util.util import np_to_pil, pil_to_np
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 نماذج البيانات

#### ControlField
```python
class ControlField(BaseModel):
    image: ImageField = Field(description="The control image")
    control_model: ModelIdentifierField = Field(description="The ControlNet model to use")
    control_weight: Union[float, List[float]] = Field(default=1, description="The weight given to the ControlNet")
    begin_step_percent: float = Field(
        default=0, ge=0, le=1, description="When the ControlNet is first applied (% of total steps)"
    )
    end_step_percent: float = Field(
        default=1, ge=0, le=1, description="When the ControlNet is last applied (% of total steps)"
    )
    control_mode: CONTROLNET_MODE_VALUES = Field(default="balanced", description="The control mode to use")
    resize_mode: CONTROLNET_RESIZE_VALUES = Field(default="just_resize", description="The resize mode to use")

    @field_validator("control_weight")
    @classmethod
    def validate_control_weight(cls, v):
        validate_weights(v)
        return v

    @model_validator(mode="after")
    def validate_begin_end_step_percent(self):
        validate_begin_end_step(self.begin_step_percent, self.end_step_percent)
        return self
```

### 3.2 نماذج الإخراج

#### ControlOutput
```python
@invocation_output("control_output")
class ControlOutput(BaseInvocationOutput):
    """node output for ControlNet info"""
    control: ControlField = OutputField(description=FieldDescriptions.control)
```

### 3.3 النماذج

#### ControlNetInvocation
```python
@invocation(
    "controlnet", title="ControlNet - SD1.5, SD2, SDXL", tags=["controlnet"], category="conditioning", version="1.1.3"
)
class ControlNetInvocation(BaseInvocation):
    """Collects ControlNet info to pass to other nodes"""

    image: ImageField = InputField(description="The control image")
    control_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.controlnet_model,
        ui_model_base=[BaseModelType.StableDiffusion1, BaseModelType.StableDiffusion2, BaseModelType.StableDiffusionXL],
        ui_model_type=ModelType.ControlNet,
    )
    control_weight: Union[float, List[float]] = InputField(
        default=1.0, ge=-1, le=2, description="The weight given to the ControlNet"
    )
    begin_step_percent: float = InputField(
        default=0, ge=0, le=1, description="When the ControlNet is first applied (% of total steps)"
    )
    end_step_percent: float = InputField(
        default=1, ge=0, le=1, description="When the ControlNet is last applied (% of total steps)"
    )
    control_mode: CONTROLNET_MODE_VALUES = InputField(default="balanced", description="The control mode used")
    resize_mode: CONTROLNET_RESIZE_VALUES = InputField(default="just_resize", description="The resize mode used")

    @field_validator("control_weight")
    @classmethod
    def validate_control_weight(cls, v):
        validate_weights(v)
        return v

    @model_validator(mode="after")
    def validate_begin_end_step_percent(self) -> "ControlNetInvocation":
        validate_begin_end_step(self.begin_step_percent, self.end_step_percent)
        return self

    def invoke(self, context: InvocationContext) -> ControlOutput:
        return ControlOutput(
            control=ControlField(
                image=self.image,
                control_model=self.control_model,
                control_weight=self.control_weight,
                begin_step_percent=self.begin_step_percent,
                end_step_percent=self.end_step_percent,
                control_mode=self.control_mode,
                resize_mode=self.resize_mode,
            ),
        )
```

#### HeuristicResizeInvocation
```python
@invocation(
    "heuristic_resize",
    title="Heuristic Resize",
    tags=["image, controlnet"],
    category="controlnet_preprocessors",
    version="1.1.1",
    classification=Classification.Prototype,
)
class HeuristicResizeInvocation(BaseInvocation):
    """Resize an image using a heuristic method. Preserves edge maps."""

    image: ImageField = InputField(description="The image to resize")
    width: int = InputField(default=512, ge=1, description="The width to resize to (px)")
    height: int = InputField(default=512, ge=1, description="The height to resize to (px)")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name, "RGB")
        np_img = pil_to_np(image)
        np_resized = heuristic_resize_fast(np_img, (self.width, self.height))
        resized = np_to_pil(np_resized)
        image_dto = context.images.save(image=resized)
        return ImageOutput.build(image_dto)
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التحقق من الأوزان
```python
@field_validator("control_weight")
@classmethod
def validate_control_weight(cls, v):
    validate_weights(v)
    return v
```

### 4.2 التحقق من خطوات البدء والنهاية
```python
@model_validator(mode="after")
def validate_begin_end_step_percent(self):
    validate_begin_end_step(self.begin_step_percent, self.end_step_percent)
    return self
```

### 4.3 التعامل مع أحجام الصور المختلفة
```python
def heuristic_resize_fast(image, target_size):
    """Resize an image using a heuristic method that preserves edge maps."""
    pass
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **تحقق من صحة البيانات**: استخدام Pydantic للتحقق.
2. **flexibility**: دعم أنواع مختلفة من ControlNet.
3. **كفاءة الأداء**: استخدام NumPy للمعالجة.

### نقاط الضعف
1. **عدد كبير من النماذج**: قد يكون معقداً للصيانة.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              ControlNet Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ControlField                                               │
│       │                                                     │
│       ├── image: ImageField                                 │
│       ├── control_model: ModelIdentifierField               │
│       ├── control_weight: Union[float, List[float]]         │
│       ├── begin_step_percent: float                         │
│       ├── end_step_percent: float                           │
│       ├── control_mode: CONTROLNET_MODE_VALUES              │
│       └── resize_mode: CONTROLNET_RESIZE_VALUES             │
│       │                                                     │
│       ▼                                                     │
│  ControlNetInvocation                                       │
│       │                                                     │
│       ├── invoke(context)                                   │
│       │     └── Return ControlOutput                        │
│       │                                                     │
│       ▼                                                     │
│  HeuristicResizeInvocation                                  │
│       │                                                     │
│       ├── invoke(context)                                   │
│       │     ├── Get PIL image                               │
│       │     ├── Convert to numpy                            │
│       │     ├── Resize with heuristic method                │
│       │     ├── Convert back to PIL                         │
│       │     └── Return ImageOutput                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [ControlNet](https://arxiv.org/abs/2302.05543)
- [Image Processing](https://en.wikipedia.org/wiki/Digital_image_processing)
- [Edge Detection](https://en.wikipedia.org/wiki/Edge_detection)
