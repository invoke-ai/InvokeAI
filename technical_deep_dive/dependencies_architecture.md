# المستند الأول: تحليل البنية التحتية والمكتبات الأساسية
## Dependencies & Infrastructure Architecture Analysis

```
invokeai/
├── docs/
│   └── technical_deep_dive/
│       └── dependencies_architecture.md  <-- هذا الملف
```

---

## ملخص البحث

يُقدّم هذا المستند تحليلاً أكاديمياً شاملاً للبنية التحتية التقنية لمشروع InvokeAI، مع التركيز على الآليات البرمجية للمكتبات الأساسية التي تدعم تشغيل نماذج الذكاء الاصطناعي التوليدي على وحدات المعالجة المركزية (CPU). يعتمد التحليل على فحص مباشر لأكواد المشروع وملفات تكوين الحزم.

---

## أولاً: إطار إدارة الحزم ومتطلبات التشغيل

### 1.1. ملف pyproject.toml: الإطار التأسيسي

يُمثّل ملف `pyproject.toml` ([invokeai/pyproject.toml:34-84](pyproject.toml#L34-L84)) الوثيقة التأسيسية لإدارة حزم المشروع. يتطلّب المشروع إصدار بايثون في النطاق `>=3.11, <3.13`، وهو ما يتوافق مع متطلبات المكتبات الحديثة مثل PyTorch 2.7+ و Diffusers 0.37.

**المتطلبات الأساسية المسجلة:**

```toml
dependencies = [
    "accelerate",                    # تسريع تحميل النماذج
    "diffusers[torch]==0.37.0",      # خطوط أنابيب Stable Diffusion
    "torch~=2.7.0",                 # إطار التعلم العميق
    "torchvision",                   # معالجة الصور
    "transformers>=4.56.0",          # نماذج CLIP و T5
    "onnx==1.16.1",                 # تمثيل النماذج
    "onnxruntime==1.19.2",          # تنفيذ ONNX
    "safetensors",                   # تخزين الأوزان
    "xformers>=0.0.28.post1",       # تحسين الذاكرة
]
```

### 1.2. التكوينات الاختيارية للأجهزة

يُوفّر المشروع ثلاث تكوينات اختيارية ([pyproject.toml:86-99](pyproject.toml#L86-L99)) لدعم أجهزة المعالجة المختلفة:

| التكوين | الحزمة الأساسية | الهدف |
|---|---|---|
| `cpu` | `torch==2.7.1+cpu` | تشغيل على المعالج فقط |
| `cuda` | `torch==2.7.1+cu128` | أجهزة NVIDIA GPU |
| `rocm` | `torch==2.7.1+rocm6.3` | أجهزة AMD GPU |

هذا التصميم يُمكّن من عزل البيئة بشكل كامل عبر أدوات مثل `uv`، حيث يتم اختيار الحزم الصحيحة تلقائياً حسب نوع الجهاز المتاح.

---

## ثانياً: دور PyTorch في العمليات الحسابية على CPU

### 2.1. آلية اختيار الجهاز (Device Selection)

يُقدّم ملف `invokeai/backend/util/devices.py` ([devices.py:42-62](devices.py#L42-L62)) كلاس `TorchDevice` الذي يُمثّل طبقة التجريد للجهاز:

```python
class TorchDevice:
    @classmethod
    def choose_torch_device(cls) -> torch.device:
        app_config = get_config()
        if app_config.device != "auto":
            device = torch.device(app_config.device)
        elif torch.cuda.is_available():
            device = CUDA_DEVICE      # "cuda"
        elif torch.backends.mps.is_available():
            device = MPS_DEVICE       # "mps" (Apple Silicon)
        else:
            device = CPU_DEVICE       # "cpu"
        return cls.normalize(device)
```

**الآليات البرمجية للـ CPU:**

عندما لا يتوفر GPU، ينتقل النظام إلى وضع CPU مع تحسينات محددة:

1. **دقة الحساب (Precision):** يُختار `float32` بشكل افتراضي على CPU ([devices.py:87-88](devices.py#L87-L88))، لأن وحدات المعالجة المركزية لا تدعم float16 بشكل فعال مثل GPU
2. **الذاكرة:** يُستخدم الذاكرة العشوائية (RAM) كبديل لذاكرة GPU (VRAM)
3. **التسريع:** يعتمد على مكتبات BLAS/MKL المدمجة في PyTorch لتسريع المصفوفات

### 2.2. إدارة الدقة (Precision Management)

```python
@classmethod
def choose_torch_dtype(cls, device: Optional[torch.device] = None) -> torch.dtype:
    # ... (الكود المختصر)
    # CPU / safe fallback
    return cls._to_dtype("float32")  # خطأ أمان
```

هذا التصميم يضمن أن النموذج يعمل بشكل صحيح على CPU حتى لو لم تكن البطاقة تدعم الدقة المنخفضة.

### 2.3. إدارة الذاكرة (Memory Management)

يُوفر `TorchDevice` ([devices.py:105-110](devices.py#L105-L110)) دالة `empty_cache()` لمسح ذاكرة GPU:

```python
@classmethod
def empty_cache(cls) -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

على CPU، لا توجد خطوة مكافئة لأن الذاكرة العشوائية تُدار بواسطة نظام التشغيل مباشرة.

---

## ثالثاً: دور مكتبة Diffusers في إدارة خطوط الأنابيب

### 3.1. البنية الأساسية لخط الأنابيب

يُقدّم ملف `invokeai/backend/stable_diffusion/diffusers_pipeline.py` ([diffusers_pipeline.py:114-167](diffusers_pipeline.py#L114-L167)) كلاس `StableDiffusionGeneratorPipeline` الذي يمتد `StableDiffusionPipeline` من مكتبة Diffusers:

```python
class StableDiffusionGeneratorPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: Optional[StableDiffusionSafetyChecker],
        feature_extractor: Optional[CLIPImageProcessor],
    ):
        super().__init__(...)
        self.invokeai_diffuser = InvokeAIDiffuserComponent(self.unet, self._unet_forward)
```

### 3.2. تحسين الذاكرة على CPU

تُقدّم الدالة `_adjust_memory_efficient_attention()` ([diffusers_pipeline.py:170-243](diffusers_pipeline.py#L170-L243)) خوارزمية ذكية لاختيار أفضل طريقة لمعالجة الانتباه:

```python
def _adjust_memory_efficient_attention(self, latents: torch.Tensor):
    # على أجهزة 30xx/40xx، torch-sdp أسرع من xformers
    prefer_xformers = torch.cuda.is_available() and \
        torch.cuda.get_device_properties("cuda").major <= 7
    
    if config.attention_type == "xformers" and is_xformers_available():
        self.enable_xformers_memory_efficient_attention()
    elif config.attention_type == "sliced":
        slice_size = auto_detect_slice_size(latents)
        self.enable_attention_slicing(slice_size=slice_size)
    # ... باقي الخيارات
```

**الآليات على CPU:**

| الطريقة | الوصف | الاستخدام |
|---|---|---|
| `torch-sdp` | الانتباه المقنّن عبر PyTorch | الافتراضي |
| `sliced` | تقسيم الانتباه إلى شرائح | ذاكرة منخفضة |
| `xformers` | تسريع خارجي | GPU فقط |

### 3.3. دورة Denoising

تُقدّم الدالة `latents_from_embeddings()` ([diffusers_pipeline.py:274-435](diffusers_pipeline.py#L274-L435)) الحلقة الرئيسية لإزالة الضوضاء:

```python
def latents_from_embeddings(self, latents, scheduler_step_kwargs, 
                            conditioning_data, noise, seed, timesteps, ...):
    # 1. إضافة الضوضاء الأولية
    if noise is not None:
        latents = self.scheduler.add_noise(latents, noise, batched_init_timestep)
    
    # 2. تحسين الذاكرة
    self._adjust_memory_efficient_attention(latents)
    
    # 3. الحلقة التكرارية
    for i, t in enumerate(self.progress_bar(timesteps)):
        step_output = self.step(
            t=batched_t, latents=latents,
            conditioning_data=conditioning_data, ...
        )
        latents = step_output.prev_sample
    
    return latents
```

### 3.4. المُجدولات المتعددة (Schedulers)

يدعم المشروع أكثر من 15 مُجدولاً ([diffusers_pipeline.py:395-410](diffusers_pipeline.py#L395-L410)) تُدار عبر خريطة `SCHEDULER_MAP`، ومنها:
- `DDIMScheduler` - الانتشار غير العشوائي
- `DPMSolverMultistepScheduler` - محلل متعدد الخطوات
- `EulerDiscreteScheduler` - طريقة أويلر
- `LCMScheduler` - النموذج المتسارع

---

## رابعاً: أدوات إدارة الحزم الحديثة (uv و pnpm)

### 4.1. مُدير الحزم uv للبايثون

يُستخدم `uv` كمُدير حزم رئيسي للمشروع ([pyproject.toml:121-156](pyproject.toml#L121-L156)). الميّزات الرئيسية:

**آليات العزل:**

```toml
[tool.uv]
# منع تعارض OpenCV
override-dependencies = ["opencv-python; sys_platform=='never'"]
conflicts = [[{ extra = "cpu" }, { extra = "cuda" }, { extra = "rocm" }]]
index-strategy = "unsafe-best-match"
```

**مواقع التحميل المخصصة:**

```toml
[[tool.uv.index]]
name = "torch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "torch-cuda"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

**المقارنة مع أدوات تقليدية:**

| الميزة | uv | pip | pip-tools |
|---|---|---|---|
| **سرعة التثبيت** | 10-100x أسرع | عادي | عادي |
| **حل التعارضات** | خوارزمية ذكية | بسيط | جيد |
| **الذاكرة المؤقتة** | تلقائي + سريع | محدود | محدود |
| **عزل البيئة** | مدمج | يتطلب venv | يتطلب venv |
| **تثبيت متزامن** | نعم | لا | لا |

### 4.2. مُدير الحزم pnpm للواجهة الأمامية

يُستخدم `pnpm@10.12.4` كمُدير حزم وحيد لواجهة المستخدم ([package.json:161](package.json#L161)):

```json
"preinstall": "npx only-allow pnpm"
```

**آليات العزل في pnpm:**

1. **الرابط الرمزي (Symlinks):** يُنشئ pnpm مجلداً `.pnpm` يحتوي على جميع الحزم بversion محدد، ويُنشئ روابط رمزية للمشروع
2. **الذاكرة المؤقتة المشتركة:** جميع الحزم تُخزّن مرة واحدة في `.pnpm-store`
3. **العزل الكامل:** كل مشروع يرى فقط الحزم المُعرّفة في `package.json`

**المقارنة مع npm/yarn:**

| الميزة | pnpm | npm | yarn |
|---|---|---|---|
| **سرعة التثبيت** | أسرع 2x | عادي | أسرع قليلاً |
| **استهلاك التخزين** | أقل (شراكة) | أعلى (تكرار) | أعلى |
| **العزل** | ممتاز | ضعيف | متوسط |
| **الأمان** | stricter | عادي | عادي |
| **الدعم** | monorepo | عادي | monorepo |

---

## خامساً: ONNX Runtime و PatchMatch

### 5.1. ONNX Runtime: تسريع التنفيذ

يُقدّم ملف `invokeai/backend/onnx/onnx_runtime.py` ([onnx_runtime.py:20-100](onnx_runtime.py#L20-L100)) كلاس `IAIOnnxRuntimeModel`:

```python
class IAIOnnxRuntimeModel(RawModel):
    def __init__(self, model_path: str, provider: Optional[str]):
        self.path = model_path
        self.session = None
        self.provider = provider
```

**آلية العمل:**

1. **تحويل النموذج:** يتم تحويل نماذج PyTorch إلى تنسيق ONNX
2. **تحسين الرسم البياني:** ONNX يُحسّن الرسم البياني للتنفيذ
3. **تنفيذ مُسرّع:** ONNX Runtime يستخدم محركات تنفيذ متعددة:

| المزود | الوصف | الاستخدام |
|---|---|---|
| `CPUExecutionProvider` | CPU عادي | كل الأجهزة |
| `CUDAExecutionProvider` | NVIDIA GPU | CUDA |
| `DirectMLExecutionProvider` | Microsoft DirectML | Windows |
| `CoreMLExecutionProvider` | Apple CoreML | macOS |

**التحسينات على Windows Server:**

```python
# من onnx_runtime.py
from onnxruntime import InferenceSession, SessionOptions, get_available_providers

# تحسينات SessionOptions
session_options = SessionOptions()
session_options.intra_op_num_threads = 4  # عدد cores
session_options.inter_op_num_threads = 2  # عدد threads
session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
```

### 5.2. PatchMatch: خوارزمية ملء الفراغات

يُقدّم ملف `invokeai/backend/image_util/infill_methods/patchmatch.py` ([patchmatch.py:15-57](patchmatch.py#L15-L57)) كلاس `PatchMatch`:

```python
class PatchMatch:
    patch_match = None
    tried_load: bool = False

    @classmethod
    def _load_patch_match(cls):
        if cls.tried_load:
            return
        if get_config().patchmatch:
            from patchmatch import patch_match as pm
            if pm.patchmatch_available:
                cls.patch_match = pm

    @classmethod
    def inpaint(cls, image: Image.Image) -> Image.Image:
        np_image = np.array(image)
        mask = 255 - np_image[:, :, 3]
        infilled = cls.patch_match.inpaint(np_image[:, :, :3], mask, patch_size=3)
        return Image.fromarray(infilled, mode="RGB")
```

**الخوارزمية الرياضية:**

تعمل PatchMatch على مبدأ البحث عن أشباه التشابه في المناطق المحيطة بالمنطقة المفقودة:

$$\text{patch}(x, y) = \arg\min_{\text{patch}'} \| \text{patch}' - \text{neighborhood}(x, y) \|_2$$

**التحسينات على CPU:**

1. **تقليل حجم البلاط (Patch Size):** `patch_size=3` بدلاً من الأكبر
2. **البحثlocalObject:** البحث فقط في المناطق القريبة
3. **تحسين الذاكرة:** استخدام numpy بدلاً من PyTorch لعمليات المعالجة

---

## سادساً: تحليل التبعيات المتعددة

### 6.1. مكتبات الذكاء الاصطناعي

| المكتبة | الإصدار | الوظيفة | الاعتماد على |
|---|---|---|---|
| `torch` | ~2.7.0 | إطار التعلم العميق | - |
| `diffusers` | 0.37.0 | خطوط أنابيب الانتشار | torch |
| `transformers` | >=4.56.0 | نماذج اللغة والمعاينة | torch |
| `accelerate` | - | تسريع التدريب والاستدلال | torch |
| `safetensors` | - | تخزين آمن للأوزان | - |
| `einops` | - | إعادة ترتيب التنسورات | - |

### 6.2. مكتبات معالجة الصور

| المكتبة | الوظيفة |
|---|---|
| `pillow` | التعامل مع الصور |
| `opencv-contrib-python` | معالجة الصور المتقدمة |
| `torchvision` | تحويلات الصور |
| `mediapipe` | كشف الوجه والعودة |
| `spandrel` | تكبير الصور |

### 6.3. مكتبات الخادم

| المكتبة | الوظيفة |
|---|---|
| `fastapi` | بناء API |
| `uvicorn` | خادم ASGI |
| `python-socketio` | WebSocket |
| `pydantic` | التحقق من البيانات |

### 6.4. مكتبات الأمان

| المكتبة | الوظيفة |
|---|---|
| `python-jose` | JWT tokens |
| `passlib` | تشفير كلمات المرور |
| `bcrypt` | تجزئة كلمات المرور |
| `picklescan` | فحص الملفات الضارة |

---

## سابعاً: مخطط التبعيات الشامل

```
┌─────────────────────────────────────────────────────────────┐
│                    InvokeAI Application                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐    ┌─────────────────────────────┐    │
│  │   Frontend      │    │       Backend (API)          │    │
│  │   (React/TS)    │    │    (FastAPI/Python)          │    │
│  │                 │    │                              │    │
│  │  ┌───────────┐  │    │  ┌────────────────────────┐  │    │
│  │  │   Vite    │  │    │  │    FastAPI + Uvicorn   │  │    │
│  │  │  (Build)  │  │    │  │     (REST + WS)        │  │    │
│  │  └───────────┘  │    │  └────────────────────────┘  │    │
│  │  ┌───────────┐  │    │  ┌────────────────────────┐  │    │
│  │  │   pnpm    │  │    │  │   uv (Python)          │  │    │
│  │  │ (Packages)│  │    │  │    (Packages)           │  │    │
│  │  └───────────┘  │    │  └────────────────────────┘  │    │
│  └─────────────────┘    └─────────────────────────────┘    │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                    AI Engine (PyTorch)                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Stable Diffusion Pipeline               │    │
│  │                                                      │    │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────────────┐   │    │
│  │  │  VAE    │  │   CLIP   │  │      UNet        │   │    │
│  │  │ Encoder │  │  Text    │  │   (Denoising)    │   │    │
│  │  │ Decoder │  │ Encoder  │  │                  │   │    │
│  │  └─────────┘  └──────────┘  └──────────────────┘   │    │
│  │                                                      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                  Hardware Abstraction                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │   CPU    │    │   GPU    │    │   MPS    │              │
│  │ (x86/ARM)│    │ (CUDA)   │    │ (Apple)  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## ثامناً: استنتاجات أكاديمية

### 8.1. مزايا التصميم

1. **عزل الأجهزة:** يُوفّر المشروع طبقة تجريد كاملة عبر `TorchDevice` تُمكّن من التشغيل على أي جهاز دون تعديل الكود
2. **إدارة ذكية للذاكرة:** نظام التخزين المؤقت متعدد المستويات (RAM → VRAM → Disk) يمنع أخطاء Out of Memory
3. **توافق النماذج:** استخدام مكتبة Diffusers يضمن التوافق مع جميع نماذج Hugging Face
4. **تسريع متعدد:** دعم ONNX Runtime و xformers و torch-sdp لتحسين الأداء

### 8.2. تحديات تقنية

1. **استهلاك الذاكرة:** نماذج SD تتطلب 4-8 GB RAM على CPU
2. **بطء التوليد:** التوليد على CPU أبطأ بـ 20-40x من GPU
3. **تعقيد التبعيات:** عدد كبير من المكتبات يزيد من صيانة النظام
4. **تضارب الإصدارات:** بعض المكتبات لها قيود إصدارات صارمة

### 8.3. توصيات للتحسين

1. **تثبيت ONNX:** استخدام ONNX Runtime للنماذج الثابتة على CPU
2. **تحسين الذاكرة:** تطبيق quantization للنماذج الكبيرة
3. **تسريع CPU:** استخدام مكتبات Intel MKL أو OpenBLAS
4. **تخفيف الحمل:** تطبيق batch processing للمعالجة المتزامنة

---

## المراجع التقنية

1. PyTorch Documentation - https://pytorch.org/docs/stable/
2. Hugging Face Diffusers - https://huggingface.co/docs/diffusers/
3. ONNX Runtime - https://onnxruntime.ai/
4. uv Package Manager - https://github.com/astral-sh/uv
5. pnpm - https://pnpm.io/
6. PatchMatch Algorithm - Barnes et al., "PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing", 2009

---

*آخر تحديث: يونيو 2026*
*المؤلف: قسم أبحاث الذكاء الاصطناعي*
