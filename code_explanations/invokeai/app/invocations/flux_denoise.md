# توثيق ملف: flux_denoise.py

## مسار الملف الأصلي
```
invokeai/app/invocations/flux_denoise.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/invocations/flux_denoise.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **عملية التنقية** (Denoising Process) لنماذج FLUX في InvokeAI. يدير عملية إزالة الضوضاء لتوليد الصور.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 مكتبات Python الأساسية
```python
from contextlib import ExitStack
from typing import Callable, Iterator, Optional, Tuple, Union
```

### 2.2 NumPy و PyTorch
```python
import einops
import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms as tv_transforms
```

### 2.3 PIL
```python
from PIL import Image
from torchvision.transforms.functional import resize as tv_resize
```

### 2.4 Transformers
```python
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
```

### 2.5 مكتبات المشروع
```python
from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    DenoiseMaskField, FieldDescriptions, FluxConditioningField, FluxFillConditioningField,
    FluxKontextConditioningField, FluxReduxConditioningField, ImageField, Input, InputField, LatentsField,
)
from invokeai.app.invocations.flux_controlnet import FluxControlNetField
from invokeai.app.invocations.flux_vae_encode import FluxVaeEncodeInvocation
from invokeai.app.invocations.ip_adapter import IPAdapterField
from invokeai.app.invocations.latent_noise import validate_noise_tensor_shape
from invokeai.app.invocations.model import ControlLoRAField, LoRAField, TransformerField, VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.controlnet.instantx_controlnet_flux import InstantXControlNetFlux
from invokeai.backend.flux.controlnet.xlabs_controlnet_flux import XLabsControlNetFlux
from invokeai.backend.flux.denoise import denoise
from invokeai.backend.flux.dype.presets import (
    DYPE_PRESET_LABELS, DYPE_PRESET_OFF, DyPEPreset, get_dype_config_from_preset,
)
from invokeai.backend.flux.extensions.dype_extension import DyPEExtension
from invokeai.backend.flux.extensions.instantx_controlnet_extension import InstantXControlNetExtension
from invokeai.backend.flux.extensions.kontext_extension import KontextExtension
from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
from invokeai.backend.flux.extensions.xlabs_controlnet_extension import XLabsControlNetExtension
from invokeai.backend.flux.extensions.xlabs_ip_adapter_extension import XLabsIPAdapterExtension
from invokeai.backend.flux.ip_adapter.xlabs_ip_adapter_flux import XlabsIpAdapterFlux
from invokeai.backend.flux.model import Flux
from invokeai.backend.flux.sampling_utils import (
    clip_timestep_schedule_fractional, generate_img_ids, get_noise, get_schedule, pack, unpack,
)
from invokeai.backend.flux.schedulers import FLUX_SCHEDULER_LABELS, FLUX_SCHEDULER_MAP, FLUX_SCHEDULER_NAME_VALUES
from invokeai.backend.flux.text_conditioning import FluxReduxConditioning, FluxTextConditioning
from invokeai.backend.model_manager.taxonomy import BaseModelType, FluxVariantType, ModelFormat, ModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import FLUXConditioningInfo
from invokeai.backend.util.devices import TorchDevice
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 فئة FluxDenoiseInvocation

#### تعريف النموذج
```python
@invocation(
    "flux_denoise",
    title="FLUX Denoise",
    tags=["image", "flux"],
    category="latents",
    version="4.6.0",
)
class FluxDenoiseInvocation(BaseInvocation):
    """Run denoising process with a FLUX transformer model."""

    latents: Optional[LatentsField] = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    noise: Optional[LatentsField] = InputField(
        default=None,
        description=FieldDescriptions.noise,
        input=Input.Connection,
    )
    denoise_mask: Optional[DenoiseMaskField] = InputField(
        default=None,
        description=FieldDescriptions.denoise_mask,
        input=Input.Connection,
    )
    denoising_start: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.denoising_start)
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    add_noise: bool = InputField(default=True, description="Add noise based on denoising start.")
    transformer: TransformerField = InputField(
        description=FieldDescriptions.flux_model,
        input=Input.Connection,
        title="Transformer",
    )
    control_lora: Optional[ControlLoRAField] = InputField(
        description=FieldDescriptions.control_lora_model, input=Input.Connection, title="Control LoRA", default=None
    )
    positive_text_conditioning: FluxConditioningField | list[FluxConditioningField] = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_text_conditioning: FluxConditioningField | list[FluxConditioningField] | None = InputField(
        default=None,
        description="Negative conditioning tensor. Can be None if cfg_scale is 1.0.",
        input=Input.Connection,
    )
    redux_conditioning: FluxReduxConditioningField | list[FluxReduxConditioningField] | None = InputField(
        default=None,
        description="FLUX Redux conditioning tensor.",
        input=Input.Connection,
    )
    fill_conditioning: FluxFillConditioningField | None = InputField(
        default=None,
        description="FLUX Fill conditioning.",
        input=Input.Connection,
    )
    cfg_scale: float | list[float] = InputField(default=1.0, description=FieldDescriptions.cfg_scale, title="CFG Scale")
    cfg_scale_start_step: int = InputField(
        default=0,
        title="CFG Scale Start Step",
        description="Index of the first step to apply cfg_scale.",
    )
    cfg_scale_end_step: int = InputField(
        default=-1,
        title="CFG Scale End Step",
        description="Index of the last step to apply cfg_scale.",
    )
    width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")
    num_steps: int = InputField(
        default=4, description="Number of diffusion steps. Recommended values are schnell: 4, dev: 50."
    )
    scheduler: FLUX_SCHEDULER_NAME_VALUES = InputField(
        default="euler",
        description="Scheduler to use for denoising.",
    )
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التحقق من شكل الضوضاء
```python
validate_noise_tensor_shape(self.noise, self.latents, "denoise")
```

### 4.2 التعامل مع النطاق
```python
denoising_start: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.denoising_start)
denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
```

### 4.3 التعامل مع الأبعاد
```python
width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **دعم متعدد النماذج**: دعم FLUX ونماذج أخرى.
2. **.flexibility**: دعم خيارات متعددة للتنقية.
3. **كفاءة الأداء**: استخدام PyTorch للحوسبة.

### نقاط الضعف
1. **تعقيد الكود**: معقد نسبياً للفهم.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              FLUX Denoise Process                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  FluxDenoiseInvocation                                      │
│       │                                                     │
│       ├── Input:                                             │
│       │     ├── latents: Optional[LatentsField]             │
│       │     ├── noise: Optional[LatentsField]               │
│       │     ├── denoise_mask: Optional[DenoiseMaskField]    │
│       │     ├── transformer: TransformerField               │
│       │     ├── positive_text_conditioning                   │
│       │     ├── negative_text_conditioning                   │
│       │     ├── cfg_scale: float                            │
│       │     ├── width: int                                  │
│       │     ├── height: int                                 │
│       │     └── num_steps: int                              │
│       │                                                     │
│       ▼                                                     │
│  Denoising Process                                          │
│       │                                                     │
│       ├── Get noise schedule                                │
│       ├── For each step:                                    │
│       │     ├── Predict noise with transformer              │
│       │     ├── Apply guidance (CFG)                        │
│       │     └── Step scheduler                              │
│       │                                                     │
│       ▼                                                     │
│  Output                                                     │
│       └── LatentsOutput                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [FLUX Model](https://arxiv.org/abs/2311.15127)
- [Diffusion Models](https://arxiv.org/abs/2006.11239)
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598)
