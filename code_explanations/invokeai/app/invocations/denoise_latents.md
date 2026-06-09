# توثيق ملف: denoise_latents.py

## مسار الملف الأصلي
```
invokeai/app/invocations/denoise_latents.py
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/app/invocations/denoise_latents.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **عقدة التنقية اللاتينية** (Denoise Latents Node) وهي العقدة الأساسية لتوليد الصور في InvokeAI. وهي مسؤولة عن تحويل الضوضاء العشوائية إلى صور قابلة للفك تشفير عبر عملية التنقية التدريجية (Diffusion Process).

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 PyTorch و torchvision
```python
import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms.functional import resize as tv_resize
```

### 2.2 Diffusers
```python
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.adapter import T2IAdapter
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_dpmsolver_sde import DPMSolverSDEScheduler
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.schedulers.scheduling_tcd import TCDScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin as Scheduler
```

### 2.3 Transformers
```python
from transformers import CLIPVisionModelWithProjection
```

### 2.4 مكتبات المشروع
```python
from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    ConditioningField, DenoiseMaskField, FieldDescriptions, Input,
    InputField, LatentsField, UIType
)
from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    StableDiffusionGeneratorPipeline, ControlNetData, T2IAdapterData
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    BasicConditioningInfo, SDXLConditioningInfo, TextConditioningData,
    TextConditioningRegions, IPAdapterData, IPAdapterConditioningInfo, Range
)
from invokeai.backend.stable_diffusion.diffusion_backend import StableDiffusionBackend
from invokeai.backend.stable_diffusion.extensions_manager import ExtensionsManager
from invokeai.backend.util.devices import TorchDevice
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 دالة get_scheduler()
```python
def get_scheduler(
    context: InvocationContext,
    scheduler_info: ModelIdentifierField,
    scheduler_name: str,
    seed: int,
    unet_config: AnyModelConfig,
) -> Scheduler:
    scheduler_class, scheduler_extra_config = SCHEDULER_MAP.get(scheduler_name, SCHEDULER_MAP["ddim"])
    orig_scheduler_info = context.models.load(scheduler_info)

    with orig_scheduler_info as orig_scheduler:
        scheduler_config = orig_scheduler.config

    if "_backup" in scheduler_config:
        scheduler_config = scheduler_config["_backup"]
    scheduler_config = {**scheduler_config, **scheduler_extra_config, "_backup": scheduler_config}

    if hasattr(unet_config, "prediction_type"):
        scheduler_config["prediction_type"] = unet_config.prediction_type

    scheduler = scheduler_class.from_config(scheduler_config)
    return scheduler
```

### 3.2 فئة DenoiseLatentsInvocation

#### الحقول
```python
@invocation(
    "denoise_latents",
    title="Denoise - SD1.5, SDXL",
    tags=["latents", "denoise", "txt2img", "t2i", "t2l", "img2img", "i2i", "l2l"],
    category="latents",
    version="1.5.4",
)
class DenoiseLatentsInvocation(BaseInvocation):
    positive_conditioning: Union[ConditioningField, list[ConditioningField]] = InputField(...)
    negative_conditioning: Union[ConditioningField, list[ConditioningField]] = InputField(...)
    noise: Optional[LatentsField] = InputField(default=None, ...)
    steps: int = InputField(default=10, gt=0, ...)
    cfg_scale: Union[float, List[float]] = InputField(default=7.5, ...)
    denoising_start: float = InputField(default=0.0, ge=0, le=1, ...)
    denoising_end: float = InputField(default=1.0, ge=0, le=1, ...)
    scheduler: SCHEDULER_NAME_VALUES = InputField(default="euler", ...)
    unet: UNetField = InputField(...)
    control: Optional[Union[ControlField, list[ControlField]]] = InputField(default=None, ...)
    ip_adapter: Optional[Union[IPAdapterField, list[IPAdapterField]]] = InputField(default=None, ...)
    t2i_adapter: Optional[Union[T2IAdapterField, list[T2IAdapterField]]] = InputField(default=None, ...)
    cfg_rescale_multiplier: float = InputField(default=0, ...)
    latents: Optional[LatentsField] = InputField(default=None, ...)
    denoise_mask: Optional[DenoiseMaskField] = InputField(default=None, ...)
```

### 3.3 الدوال الرئيسية

#### _get_text_embeddings_and_masks()
```python
@staticmethod
def _get_text_embeddings_and_masks(cond_list, context, device, dtype):
    text_embeddings = []
    text_embeddings_masks = []
    for cond in cond_list:
        cond_data = context.conditioning.load(cond.conditioning_name)
        text_embeddings.append(cond_data.conditionings[0].to(device=device, dtype=dtype))
        mask = cond.mask
        if mask is not None:
            mask = context.tensors.load(mask.tensor_name)
        text_embeddings_masks.append(mask)
    return text_embeddings, text_embeddings_masks
```

#### _concat_regional_text_embeddings()
```python
@staticmethod
def _concat_regional_text_embeddings(text_conditionings, masks, latent_height, latent_width, dtype):
    text_embedding = []
    processed_masks = []
    embedding_ranges = []

    for prompt_idx, text_embedding_info in enumerate(text_conditionings):
        text_embedding.append(text_embedding_info.embeds)
        embedding_ranges.append(Range(start=cur_text_embedding_len, end=...))

    text_embedding = torch.cat(text_embedding, dim=1)

    if not all_masks_are_none:
        regions = TextConditioningRegions(masks=torch.cat(processed_masks, dim=1), ranges=embedding_ranges)

    return text_embedding, regions
```

#### get_conditioning_data()
```python
@staticmethod
def get_conditioning_data(context, positive_conditioning_field, negative_conditioning_field, ...):
    cond_text_embeddings, cond_text_embedding_masks = ...
    uncond_text_embeddings, uncond_text_embedding_masks = ...

    cond_text_embedding, cond_regions = ...
    uncond_text_embedding, uncond_regions = ...

    conditioning_data = TextConditioningData(
        uncond_text=uncond_text_embedding,
        cond_text=cond_text_embedding,
        guidance_scale=cfg_scale,
        guidance_rescale_multiplier=cfg_rescale_multiplier,
    )
    return conditioning_data
```

#### prep_control_data()
```python
@staticmethod
def prep_control_data(context, control_input, latents_shape, device, exit_stack, do_classifier_free_guidance):
    controlnet_data = []
    for control_info in control_list:
        control_model = exit_stack.enter_context(context.models.load(control_info.control_model))
        control_image = prepare_control_image(image=input_image, ...)
        control_item = ControlNetData(model=control_model, image_tensor=control_image, ...)
        controlnet_data.append(control_item)
    return controlnet_data
```

#### prepare_noise_and_latents()
```python
@staticmethod
def prepare_noise_and_latents(context, noise_field, latents_field):
    noise = None
    if noise_field is not None:
        noise = context.tensors.load(noise_field.latents_name)

    if latents_field is not None:
        latents = context.tensors.load(latents_field.latents_name)
    elif noise is not None:
        latents = torch.zeros_like(noise)
    else:
        raise ValueError("'latents' or 'noise' must be provided!")

    seed = noise_field.seed if noise_field and noise_field.seed else 0
    return seed, noise, latents
```

### 3.4 دالة invoke()
```python
def invoke(self, context: InvocationContext) -> LatentsOutput:
    if os.environ.get("USE_MODULAR_DENOISE", False):
        return self._new_invoke(context)
    else:
        return self._old_invoke(context)
```

### 3.5 _new_invoke() - المسار الجديد
```python
@torch.no_grad()
def _new_invoke(self, context: InvocationContext) -> LatentsOutput:
    ext_manager = ExtensionsManager(is_canceled=context.util.is_canceled)
    device = TorchDevice.choose_torch_device()
    dtype = TorchDevice.choose_torch_dtype()

    seed, noise, latents = self.prepare_noise_and_latents(context, self.noise, self.latents)
    conditioning_data = self.get_conditioning_data(...)
    scheduler = get_scheduler(...)
    timesteps, init_timestep, scheduler_step_kwargs = self.init_scheduler(...)

    # إضافة الامتدادات
    ext_manager.add_extension(PreviewExt(step_callback))
    if self.cfg_rescale_multiplier > 0:
        ext_manager.add_extension(RescaleCFGExt(self.cfg_rescale_multiplier))
    if self.unet.freeu_config:
        ext_manager.add_extension(FreeUExt(self.unet.freeu_config))
    # ... المزيد من الامتدادات

    # إنشاء سياق التنقية
    denoise_ctx = DenoiseContext(inputs=DenoiseInputs(...), unet=None, scheduler=scheduler)

    with ExitStack() as exit_stack:
        self.parse_controlnet_field(exit_stack, context, self.control, ext_manager)
        self.parse_t2i_adapter_field(exit_stack, context, self.t2i_adapter, ext_manager)

        with (
            context.models.load(self.unet.unet).model_on_device() as (cached_weights, unet),
            ModelPatcher.patch_unet_attention_processor(unet, denoise_ctx.inputs.attention_processor_cls),
            ext_manager.patch_extensions(denoise_ctx),
            ext_manager.patch_unet(unet, cached_weights),
        ):
            sd_backend = StableDiffusionBackend(unet, scheduler)
            result_latents = sd_backend.latents_from_embeddings(denoise_ctx, ext_manager)

    result_latents = result_latents.detach().to("cpu")
    TorchDevice.empty_cache()
    name = context.tensors.save(tensor=result_latents)
    return LatentsOutput.build(latents_name=name, latents=result_latents, seed=None)
```

### 3.6 _old_invoke() - المسار القديم
```python
@torch.no_grad()
def _old_invoke(self, context: InvocationContext) -> LatentsOutput:
    device = TorchDevice.choose_torch_device()
    seed, noise, latents = self.prepare_noise_and_latents(context, self.noise, self.latents)
    mask, masked_latents, gradient_mask = self.prep_inpaint_mask(context, latents)

    t2i_adapter_data = self.run_t2i_adapters(...)
    image_prompts = self.prep_ip_adapter_image_prompts(...)
    unet_config = context.models.get_config(self.unet.unet.key)

    with (
        ExitStack() as exit_stack,
        context.models.load(self.unet.unet).model_on_device() as (cached_weights, unet),
        ModelPatcher.apply_freeu(unet, self.unet.freeu_config),
        SeamlessExt.static_patch_model(unet, self.unet.seamless_axes),
        LayerPatcher.apply_smart_model_patches(model=unet, patches=_lora_loader(), ...),
    ):
        latents = latents.to(device=device, dtype=unet.dtype)
        scheduler = get_scheduler(...)
        pipeline = self.create_pipeline(unet, scheduler)
        conditioning_data = self.get_conditioning_data(...)
        controlnet_data = self.prep_control_data(...)
        ip_adapter_data = self.prep_ip_adapter_data(...)
        timesteps, init_timestep, scheduler_step_kwargs = self.init_scheduler(...)

        result_latents = pipeline.latents_from_embeddings(
            latents=latents, timesteps=timesteps, ..., callback=step_callback
        )

    result_latents = result_latents.to("cpu")
    TorchDevice.empty_cache()
    name = context.tensors.save(tensor=result_latents)
    return LatentsOutput.build(latents_name=name, latents=result_latents, seed=None)
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التحقق من الشكل
```python
if noise is not None and noise.shape[1:] != latents.shape[1:]:
    raise ValueError(f"Incompatible 'noise' and 'latents' shapes: {latents.shape=} {noise.shape=}")
```

### 4.2 التعامل مع القيم الافتراضية
```python
if latents_field is not None:
    latents = context.tensors.load(latents_field.latents_name)
elif noise is not None:
    latents = torch.zeros_like(noise)
else:
    raise ValueError("'latents' or 'noise' must be provided!")
```

### 4.3 التحقق من cfg_scale
```python
@field_validator("cfg_scale")
def ge_one(cls, v: Union[List[float], float]) -> Union[List[float], float]:
    if isinstance(v, list):
        for i in v:
            if i < 1:
                raise ValueError("cfg_scale must be greater than 1")
    else:
        if v < 1:
            raise ValueError("cfg_scale must be greater than 1")
    return v
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **مرونة التكامل**: دعم أنواع متعددة من التكييف (ControlNet, IP-Adapter, T2I-Adapter).
2. **كفاءة الذاكرة**: استخدام `torch.no_grad()` و `TorchDevice.empty_cache()`.
3. **امتدادية**: استخدام نمط الامتدادات (Extensions Pattern).
4. **تتبع**: دعم المعاينة وإحصائيات الأداء.

### نقاط الضعف
1. **تعقيد الكود**: الملف معقد جداً بأكثر من 1000 سطر.
2. **مساران منفصلان**: وجود `_new_invoke` و `_old_invoke` يزيد التعقيد.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Denoise Latents Flow                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  invoke(context)                                            │
│       │                                                     │
│       ├── USE_MODULAR_DENOISE?                              │
│       │     ├── Yes → _new_invoke()                         │
│       │     └── No → _old_invoke()                          │
│       │                                                     │
│  _new_invoke(context)                                       │
│       │                                                     │
│       ├── prepare_noise_and_latents()                       │
│       │     │                                               │
│       │     ├── Load noise from tensors                     │
│       │     ├── Load latents from tensors                   │
│       │     └── Create zeros_like if needed                 │
│       │                                                     │
│       ├── get_conditioning_data()                           │
│       │     │                                               │
│       │     ├── Load text embeddings                        │
│       │     ├── Concat regional embeddings                  │
│       │     └── Create TextConditioningData                 │
│       │                                                     │
│       ├── get_scheduler()                                   │
│       │     │                                               │
│       │     └── Configure scheduler with seed               │
│       │                                                     │
│       ├── init_scheduler()                                  │
│       │     │                                               │
│       │     ├── Set timesteps                               │
│       │     ├── Calculate start/end indices                 │
│       │     └── Prepare step kwargs                         │
│       │                                                     │
│       ├── Add Extensions                                    │
│       │     ├── PreviewExt                                  │
│       │     ├── RescaleCFGExt                               │
│       │     ├── FreeUExt                                    │
│       │     ├── LoRAExt                                     │
│       │     ├── SeamlessExt                                 │
│       │     ├── InpaintExt                                  │
│       │     ├── ControlNetExt                               │
│       │     └── T2IAdapterExt                               │
│       │                                                     │
│       ├── Load UNet                                         │
│       │     │                                               │
│       │     ├── Patch attention processor                   │
│       │     ├── Patch extensions                            │
│       │     └── Patch UNet (LoRA, FreeU, etc.)              │
│       │                                                     │
│       └── StableDiffusionBackend.latents_from_embeddings()  │
│             │                                               │
│             └── Run denoising loop                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [Diffusion Models](https://arxiv.org/abs/2006.11239)
- [Stable Diffusion](https://arxiv.org/abs/2112.10752)
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598)
- [ControlNet](https://arxiv.org/abs/2302.05543)
- [IP-Adapter](https://arxiv.org/abs/2308.06721)
