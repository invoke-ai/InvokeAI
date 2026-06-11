# توثيق ملف: compel.py

## مسار الملف الأصلي
```
invokeai/app/invocations/compel.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/invocations/compel.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **عقدة تحليل النص** (Text Prompt Parsing) التي تحوّل النص إلى تضمينات نصية (Text Embeddings) باستخدام مكتبة Compel. وهي تدعم نماذج SD1.5 و SDXL و SDXL Refiner.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 PyTorch
```python
import torch
```

### 2.2 Compel
```python
from compel import Compel, ReturnedEmbeddingsType, SplitLongTextMode
from compel.prompt_parser import (
    Blend, Conjunction, CrossAttentionControlSubstitute,
    FlattenedPrompt, Fragment
)
```
- **Compel**: مكتبة لتحويل النصوص إلى تضمينات نصية.
- **Blend**: مزج التضمينات من عدة نصوص.
- **Conjunction**: ربط التضمينات من عدة نصوص.
- **CrossAttentionControlSubstitute**: التحكم في الانتباه المتبادل.

### 2.3 Transformers
```python
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
```

### 2.4 مكتبات المشروع
```python
from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ConditioningField, FieldDescriptions, Input, InputField
from invokeai.app.invocations.model import CLIPField
from invokeai.app.invocations.primitives import ConditioningOutput
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    BasicConditioningInfo, ConditioningFieldData, SDXLConditioningInfo
)
from invokeai.backend.util.devices import TorchDevice
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 CompelInvocation (SD1.5)

#### الحقول
```python
@invocation("compel", title="Prompt - SD1.5", ...)
class CompelInvocation(BaseInvocation):
    prompt: str = InputField(default="", ...)
    clip: CLIPField = InputField(...)
    mask: Optional[TensorField] = InputField(default=None, ...)
```

#### دالة invoke()
```python
@torch.no_grad()
def invoke(self, context: InvocationContext) -> ConditioningOutput:
    # تحميل النموذج النصي
    text_encoder_info = context.models.load(self.clip.text_encoder)
    ti_list = generate_ti_list(self.prompt, text_encoder_info.config.base, context)

    with (
        text_encoder_info.model_on_device() as (cached_weights, text_encoder),
        context.models.load(self.clip.tokenizer) as tokenizer,
        LayerPatcher.apply_smart_model_patches(model=text_encoder, patches=_lora_loader(), ...),
        ModelPatcher.apply_clip_skip(text_encoder, self.clip.skipped_layers),
        ModelPatcher.apply_ti(tokenizer, text_encoder, ti_list) as (patched_tokenizer, ti_manager),
    ):
        # إنشاء كائن Compel
        compel = Compel(
            tokenizer=patched_tokenizer,
            text_encoder=text_encoder,
            textual_inversion_manager=ti_manager,
            dtype_for_device_getter=TorchDevice.choose_torch_dtype,
            truncate_long_prompts=False,
            split_long_text_mode=SplitLongTextMode.SENTENCES,
        )

        # تحليل النص
        conjunction = Compel.parse_prompt_string(self.prompt)
        c, _options = compel.build_conditioning_tensor_for_conjunction(conjunction)

    # حفظ البيانات
    conditioning_data = ConditioningFieldData(conditionings=[BasicConditioningInfo(embeds=c)])
    conditioning_name = context.conditioning.save(conditioning_data)

    return ConditioningOutput(
        conditioning=ConditioningField(conditioning_name=conditioning_name, mask=self.mask)
    )
```

### 3.2 SDXLPromptInvocationBase (SDXL)

#### دالة run_clip_compel()
```python
def run_clip_compel(self, context, clip_field, prompt, get_pooled, lora_prefix, zero_on_empty):
    text_encoder_info = context.models.load(clip_field.text_encoder)

    # إذا كان النص فارغاً، إرجاع صفريات
    if prompt == "" and zero_on_empty:
        c = torch.zeros((1, max_position_embeddings, hidden_size), dtype=cpu_text_encoder.dtype)
        if get_pooled:
            c_pooled = torch.zeros((1, hidden_size), dtype=c.dtype)
        return c, c_pooled

    # تحميل النموذج النصي
    ti_list = generate_ti_list(prompt, text_encoder_info.config.base, context)

    with (
        text_encoder_info.model_on_device() as (cached_weights, text_encoder),
        context.models.load(clip_field.tokenizer) as tokenizer,
        LayerPatcher.apply_smart_model_patches(...),
        ModelPatcher.apply_clip_skip(text_encoder, clip_field.skipped_layers),
        ModelPatcher.apply_ti(tokenizer, text_encoder, ti_list) as (patched_tokenizer, ti_manager),
    ):
        compel = Compel(
            tokenizer=patched_tokenizer,
            text_encoder=text_encoder,
            textual_inversion_manager=ti_manager,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=get_pooled,
            split_long_text_mode=SplitLongTextMode.SENTENCES,
        )

        conjunction = Compel.parse_prompt_string(prompt)
        c, _options = compel.build_conditioning_tensor_for_conjunction(conjunction)
        if get_pooled:
            c_pooled = compel.conditioning_provider.get_pooled_embeddings([prompt])

    return c, c_pooled
```

### 3.3 SDXLCompelPromptInvocation (SDXL)

```python
@invocation("sdxl_compel_prompt", title="Prompt - SDXL", ...)
class SDXLCompelPromptInvocation(BaseInvocation, SDXLPromptInvocationBase):
    prompt: str = InputField(default="", ...)
    style: str = InputField(default="", ...)
    original_width: int = InputField(default=1024, ...)
    original_height: int = InputField(default=1024, ...)
    crop_top: int = InputField(default=0, ...)
    crop_left: int = InputField(default=0, ...)
    target_width: int = InputField(default=1024, ...)
    target_height: int = InputField(default=1024, ...)
    clip: CLIPField = InputField(...)
    clip2: CLIPField = InputField(...)
    mask: Optional[TensorField] = InputField(default=None, ...)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        c1, c1_pooled = self.run_clip_compel(context, self.clip, self.prompt, False, "lora_te1_", zero_on_empty=True)
        if self.style.strip() == "":
            c2, c2_pooled = self.run_clip_compel(context, self.clip2, self.prompt, True, "lora_te2_", zero_on_empty=True)
        else:
            c2, c2_pooled = self.run_clip_compel(context, self.clip2, self.style, True, "lora_te2_", zero_on_empty=True)

        # محاذاة الأبعاد
        if c1.shape[1] < c2.shape[1]:
            c1 = torch.cat([c1, torch.zeros(...)], dim=1)
        elif c1.shape[1] > c2.shape[1]:
            c2 = torch.cat([c2, torch.zeros(...)], dim=1)

        # إنشاء التضمينات
        add_time_ids = torch.tensor([original_size + crop_coords + target_size])
        conditioning_data = ConditioningFieldData(
            conditionings=[SDXLConditioningInfo(
                embeds=torch.cat([c1, c2], dim=-1),
                pooled_embeds=c2_pooled,
                add_time_ids=add_time_ids
            )]
        )

        conditioning_name = context.conditioning.save(conditioning_data)
        return ConditioningOutput(conditioning=ConditioningField(conditioning_name=conditioning_name, mask=self.mask))
```

### 3.4 CLIPSkipInvocation

```python
@invocation("clip_skip", title="Apply CLIP Skip - SD1.5, SDXL", ...)
class CLIPSkipInvocation(BaseInvocation):
    clip: CLIPField = InputField(...)
    skipped_layers: int = InputField(default=0, ge=0, ...)

    def invoke(self, context: InvocationContext) -> CLIPSkipInvocationOutput:
        self.clip.skipped_layers += self.skipped_layers
        return CLIPSkipInvocationOutput(clip=self.clip)
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 النص الفارغ
```python
if prompt == "" and zero_on_empty:
    c = torch.zeros((1, max_position_embeddings, hidden_size), dtype=cpu_text_encoder.dtype)
    return c, c_pooled
```

### 4.2 محاذاة الأبعاد
```python
if c1.shape[1] < c2.shape[1]:
    c1 = torch.cat([c1, torch.zeros(...)], dim=1)
elif c1.shape[1] > c2.shape[1]:
    c2 = torch.cat([c2, torch.zeros(...)], dim=1)
```

### 4.3 التحقق من وجود pooled_embeds
```python
assert c2_pooled is not None
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **دعم متعدد النماذج**: SD1.5 و SDXL و SDXL Refiner.
2. **كفاءة الذاكرة**: استخدام `torch.no_grad()` و `detach().to("cpu")`.
3. **دعم LoRA**: إمكانية تطبيق LoRA على النموذج النصي.
4. **دعم Textual Inversion**: إمكانية تطبيق النماذج المدربة مسبقاً.

### نقاط الضعف
1. **تعقيد الكود**: وجود فئات متعددة مع تكرار في الكود.
2. **ال依赖 على Compel**: الاعتماد على مكتبة خارجية.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Compel Prompt Processing Flow                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CompelInvocation (SD1.5)                                   │
│       │                                                     │
│       ├── Load text_encoder                                 │
│       ├── Load tokenizer                                    │
│       ├── Apply LoRA patches                                │
│       ├── Apply CLIP Skip                                   │
│       ├── Apply Textual Inversion                           │
│       │                                                     │
│       ├── Compel.parse_prompt_string(prompt)                │
│       │     │                                               │
│       │     └── Parse prompt into Conjunction/Blend         │
│       │                                                     │
│       ├── compel.build_conditioning_tensor_for_conjunction()│
│       │     │                                               │
│       │     └── Generate text embeddings                    │
│       │                                                     │
│       └── Save conditioning_data                            │
│                                                             │
│  SDXLCompelPromptInvocation (SDXL)                          │
│       │                                                     │
│       ├── run_clip_compel(clip1, prompt, ...)               │
│       │     └── c1, c1_pooled                               │
│       │                                                     │
│       ├── run_clip_compel(clip2, style/prompt, ...)         │
│       │     └── c2, c2_pooled                               │
│       │                                                     │
│       ├── Align dimensions                                  │
│       │     └── Pad with zeros if needed                    │
│       │                                                     │
│       └── Concatenate embeddings                            │
│             └── SDXLConditioningInfo(embeds, pooled, time)  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [Compel Library](https://github.com/damian0815/compel)
- [CLIP Text Encoder](https://huggingface.co/docs/transformers/model_doc/clip)
- [Textual Inversion](https://arxiv.org/abs/2208.12242)
- [SDXL Architecture](https://arxiv.org/abs/2307.01952)
