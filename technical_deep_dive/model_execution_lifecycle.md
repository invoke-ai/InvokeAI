# المستند الثاني: دورة حياة تشغيل النموذج في الذاكرة
## Model Execution Lifecycle: From Disk to Display

```
invokeai/
├── docs/
│   └── technical_deep_dive/
│       └── model_execution_lifecycle.md  <-- هذا الملف
```

---

## ملخص البحث

يُقدّم هذا المستند تحليلاً تفصيلياً لدورة حياة تشغيل نموذج Stable Diffusion في الذاكرة، من لحظة كتابة المستخدم للنص إلى ظهور الصورة النهائية. يعتمد التحليل على فحص مباشر لأكواد المشروع مع شرح المعادلات الرياضية للانتشار (Diffusion) والتحويلات الكامنة.

---

## أولاً: نظرة عامة على دورة حياة النموذج

```
┌─────────────────────────────────────────────────────────────────┐
│                    دورة حياة النموذج في الذاكرة                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Model Loading (Disk → RAM)                                  │
│     └── قراءة ملفات النموذج من القرص الصلب إلى الذاكرة           │
│                                                                  │
│  2. Model Caching (RAM Management)                              │
│     └── إدارة التخزين المؤقت وتجنب Out of Memory               │
│                                                                  │
│  3. Text Encoding (Text → Embeddings)                           │
│     └── تحويل النص إلى مصفوفات رياضية عبر CLIP                 │
│                                                                  │
│  4. Denoising Loop (Noise → Latents)                            │
│     └── إزالة الضوضاء خطوة بخطوة عبر UNet                      │
│                                                                  │
│  5. VAE Decoding (Latents → Pixels)                             │
│     └── تحويل الفضاء الكامن إلى صورة نهائية                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ثانياً: آلية تحميل النموذج (Model Loading)

### 2.1. البنية الأساسية للكاش

يُقدّم ملف `invokeai/backend/model_manager/load/model_cache/model_cache.py` ([model_cache.py:106-207](model_cache.py#L106-L207)) كلاس `ModelCache` الذي يُمثّل نظام التخزين المؤقت متعدد المستويات:

```python
class ModelCache:
    def __init__(
        self,
        execution_device_working_mem_gb: float,
        enable_partial_loading: bool,
        keep_ram_copy_of_weights: bool,
        max_ram_cache_size_gb: float | None = None,
        max_vram_cache_size_gb: float | None = None,
        execution_device: torch.device | str = "cuda",
        storage_device: torch.device | str = "cpu",
    ):
        self._execution_device = torch.device(execution_device)
        self._storage_device = torch.device(storage_device)
        self._cached_models: Dict[str, CacheRecord] = {}
        self._cache_stack: List[str] = []
```

### 2.2. آليات حساب الحجم

**المعادلة الأساسية لحساب حجم النموذج:**

$$\text{Model Size} = \sum_{i=1}^{N} \left( \text{param}_i \times \text{dtype\_size} \right) + \text{overhead}$$

حيث:
- $\text{param}_i$: عدد المعلمات في الطبقة $i$
- $\text{dtype\_size}$: حجم النوع (4 bytes لـ float32، 2 bytes لـ float16)
- $\text{overhead}$: الحجم الإضافي للهيكلة

**حساب الحجم المتاح:**

$$\text{Available RAM} = \text{Total RAM} - \text{OS Reserved} - \text{Working Memory}$$

$$\text{Available VRAM} = \text{Total VRAM} - \text{Working Memory (GPU)}$$

### 2.3. خوارزمية إدارة الذاكرة

تُقدّم الدالة `_make_room_internal()` ([model_cache.py:350-365](model_cache.py#L350-L365)) خوارزمية ذكية لإدارة الذاكرة:

```python
def _make_room_internal(self, size_needed: int) -> None:
    """Ensure there is enough room in the cache for a model of the given size."""
    ram_available = self._get_ram_available()
    
    # إذا كان الحجم المتاح كافياً، لا نفعل شيئاً
    if ram_available >= size_needed:
        return
    
    # حساب المقدار المطلوب تحريره
    bytes_to_free = size_needed - ram_available
    
    # تحرير الذاكرة من النماذج الأقل استخداماً (LRU)
    self._offload_unlocked_models(bytes_to_free)
```

**خوارزمية LRU (Least Recently Used):**

$$\text{Priority}(m) = \frac{1}{\text{time\_since\_last\_use}(m) + 1}$$

حيث يتم تحميل النموذج ذو الأولوية الأدنى أولاً.

### 2.4. التخزين المؤقت متعدد المستويات

```
┌─────────────────────────────────────────────────────────────┐
│                 نموذج التخزين المؤقت المتعدد المستويات         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  المستوى 1: GPU (VRAM) - الأسرع                             │
│  ├── الحجم: 4-24 GB                                         │
│  ├── السرية: < 1ms                                          │
│  └── الاستخدام: النماذج النشطة فقط                          │
│                                                              │
│  المستوى 2: CPU (RAM) - متوسط                               │
│  ├── الحجم: 16-128 GB                                       │
│  ├── السرية: 10-100ms                                       │
│  └── الاستخدام: نسخة احتياطية + نماذج غير نشطة             │
│                                                              │
│  المستوى 3: Disk (SSD/HDD) - الأبطأ                        │
│  ├── الحجم: غير محدود                                       │
│  ├── السرية: 100ms - 1s                                     │
│  └── الاستخدام: النماذج غير المستخدمة                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.5. آليات النقل بين المستويات

```python
def _load_locked_model(self, cache_entry: CacheRecord, working_mem_bytes: Optional[int] = None) -> None:
    # 1. حساب مقدار الذاكرة المطلوبة
    model_vram_needed = model_total_bytes - model_cur_vram_bytes
    
    # 2. تحرير الذاكرة من النماذج غير المقفلة
    vram_bytes_freed = self._offload_unlocked_models(model_vram_needed, working_mem_bytes)
    
    # 3. نقل النموذج إلى GPU
    model_bytes_loaded = self._move_model_to_vram(cache_entry, vram_available + MB)
```

**معادلة الوقت المتوقع للنقل:**

$$T_{\text{transfer}} = T_{\text{disk} \to \text{RAM}} + T_{\text{RAM} \to \text{VRAM}}$$

$$T_{\text{disk} \to \text{RAM}} \approx \frac{\text{Model Size}}{\text{Disk Speed}}$$

$$T_{\text{RAM} \to \text{VRAM}} \approx \frac{\text{Model Size}}{\text{PCIe Bandwidth}}$$

---

## ثالثاً: مرحلة Text Encoding (Trembling of Text)

### 3.1. آلية عمل CLIP

يُقدّم ملف `invokeai/app/invocations/compel.py` ([compel.py:49-135](compel.py#L49-L135)) معالجة النص عبر مكتبة Compel:

```python
@invocation(
    "compel",
    title="Prompt - SD1.5",
    tags=["prompt", "compel"],
    category="prompt",
    version="1.2.1",
)
class CompelInvocation(BaseInvocation):
    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        # 1. تحميل النموذج
        text_encoder_info = context.models.load(self.clip.text_encoder)
        tokenizer_info = context.models.load(self.clip.tokenizer)
        
        # 2. تطبيق LoRA إذا وُجد
        with LayerPatcher.apply_smart_model_patches(...):
            # 3. إنشاء كائن Compel
            compel = Compel(
                tokenizer=patched_tokenizer,
                text_encoder=text_encoder,
                dtype_for_device_getter=TorchDevice.choose_torch_dtype,
                device=get_effective_device(text_encoder),
            )
            
            # 4. تحليل النص وإنشاء التثبيتات
            conjunction = Compel.parse_prompt_string(self.prompt)
            c, _options = compel.build_conditioning_tensor_for_conjunction(conjunction)
        
        return ConditioningOutput(conditioning=ConditioningField(...))
```

### 3.2. المعادلات الرياضية للـ Text Encoding

**المرحلة 1: Tokenization (تقسيم النص)**

$$\text{tokens} = \text{Tokenizer}(\text{text}) = [t_1, t_2, \ldots, t_n]$$

حيث كل $t_i$ هو رقم صحيح يمثل كلمة في قاموس CLIP.

**المرحلة 2: Embedding (التحويل إلى متجهات)**

$$\mathbf{E}_{\text{token}} = \text{TokenEmbed}(\text{tokens}) \in \mathbb{R}^{n \times d}$$

حيث:
- $n$: عدد الرموز (okens)
- $d$: بُعد المتجه (768 لـ SD 1.5، 1024 لـ SDXL)

**المرحلة 3: Positional Encoding (الترميز الموضعي)**

$$\mathbf{E}_{\text{pos}} = \text{PosEnc}([1, 2, \ldots, n]) \in \mathbb{R}^{n \times d}$$

**المرحلة 4: Transformer Processing (معالجة المحوّل)**

$$\mathbf{H} = \text{Transformer}(\mathbf{E}_{\text{token}} + \mathbf{E}_{\text{pos}}) \in \mathbb{R}^{n \times d}$$

**المرحلة 5: Projection (الإسقاط)**

$$\mathbf{E}_{\text{text}} = \text{Projection}(\mathbf{H}) \in \mathbb{R}^{n \times d}$$

### 3.3. معادلة Cross-Attention

يُستخدم النص في طبقة الانتباه المتقاطع داخل UNet:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

حيث:
- $Q = \mathbf{W}_Q \cdot \mathbf{H}_{\text{latent}}$ (استعلامات من الصورة)
- $K = \mathbf{W}_K \cdot \mathbf{E}_{\text{text}}$ (مفاتيح من النص)
- $V = \mathbf{W}_V \cdot \mathbf{E}_{\text{text}}$ (قيم من النص)

---

## رابعاً: حلقة إزالة الضوضاء (Denoising Loop)

### 4.1. المعادلة الأساسية للانتشار

معادلة الانتشار العكسي (Reverse Diffusion):

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}$$

حيث:
- $\mathbf{x}_t$: الحالة عند الخطوة $t$
- $\alpha_t$: جدول الضوضاء
- $\beta_t$: مقياس الضوضاء
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$: الضوضاء التراكمية
- $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$: توقع النموذج للضوضاء
- $\sigma_t$: الانحراف المعياري

### 4.2. كود الحلقة التكرارية

يُقدّم ملف `invokeai/backend/stable_diffusion/diffusion_backend.py` ([diffusion_backend.py:26-62](diffusion_backend.py#L26-L62)) الحلقة الرئيسية:

```python
def latents_from_embeddings(self, ctx: DenoiseContext, ext_manager: ExtensionsManager):
    # 1. إضافة الضوضاء الأولية
    ctx.latents = ctx.scheduler.add_noise(
        ctx.latents, ctx.inputs.noise, ctx.inputs.init_timestep.expand(batch_size)
    )
    
    # 2. الحلقة التكرارية
    for ctx.step_index, ctx.timestep in enumerate(tqdm(ctx.inputs.timesteps)):
        # استدعاء UNet للتنبؤ بالضوضاء
        ctx.step_output = self.step(ctx, ext_manager)
        
        # تحديث الحالة
        ctx.latents = ctx.step_output.prev_sample
    
    return ctx.latents
```

### 4.3. خطوة Denoising الواحدة

```python
@torch.inference_mode()
def step(self, ctx: DenoiseContext, ext_manager: ExtensionsManager) -> SchedulerOutput:
    # 1. تكبير مدخل النموذج
    ctx.latent_model_input = ctx.scheduler.scale_model_input(ctx.latents, ctx.timestep)
    
    # 2. استدعاء UNet للتنبؤ بالضوضاء
    ctx.noise_pred = self.run_unet(ctx, ext_manager, ConditioningMode.Both)
    
    # 3. دمج التنبؤات (CFG)
    ctx.noise_pred = self.combine_noise_preds(ctx)
    
    # 4. تطبيق خطوة المُجدول
    step_output = ctx.scheduler.step(
        ctx.noise_pred, ctx.timestep, ctx.latents, **ctx.inputs.scheduler_step_kwargs
    )
    
    return step_output
```

### 4.4. Classifier-Free Guidance (CFG)

معادلة CFG:

$$\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_{\text{uncond}} + w \cdot (\boldsymbol{\epsilon}_{\text{cond}} - \boldsymbol{\epsilon}_{\text{uncond}})$$

حيث:
- $\boldsymbol{\epsilon}_{\text{uncond}}$: توقع الضوضاء غير المشروط
- $\boldsymbol{\epsilon}_{\text{cond}}$: توقع الضوضاء المشروط
- $w$: مقياس التوجيه (Guidance Scale)

في الكود ([diffusion_backend.py:97-106](diffusion_backend.py#L97-L106)):

```python
@staticmethod
def combine_noise_preds(ctx: DenoiseContext) -> torch.Tensor:
    guidance_scale = ctx.inputs.conditioning_data.guidance_scale
    return ctx.negative_noise_pred + guidance_scale * (
        ctx.positive_noise_pred - ctx.negative_noise_pred
    )
```

### 4.5. المُجدولات المتعددة (Schedulers)

**جدول الضوضاء (Noise Schedule):**

$$\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$$

$$\beta_t = \beta_{\text{start}} + \frac{t}{T} (\beta_{\text{end}} - \beta_{\text{start}})$$

**أنواع المُجدولات المدعومة:**

| المُجدول | المعادلة | الاستخدام |
|---|---|---|
| DDIM | Deterministic sampling | جودة عالية |
| DPM++ 2M | Multi-step ODE solver | سرعة متوسطة |
| Euler | Euler method | أبسط طريقة |
| LCM | Latent Consistency Model | أسرع (4 خطوات) |

---

## خامساً: مرحلة VAE Decoding

### 5.1. بنية VAE (Variational Autoencoder)

```
┌─────────────────────────────────────────────────────────────┐
│                    بنية VAE                                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Encoder: Input Image (512x512x3) → Latent (64x64x4)       │
│  ├── Conv2d layers                                          │
│  ├── Downsample blocks                                      │
│  └── Mid block                                               │
│                                                              │
│  Latent Space: (B, 4, H/8, W/8)                            │
│  └── 4 قنوات تمثل الفضاء الكامن                             │
│                                                              │
│  Decoder: Latent (64x64x4) → Output Image (512x512x3)     │
│  ├── Upsample blocks                                        │
│  ├── Conv2d layers                                          │
│  └── Mid block                                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2. كود VAE Decoding

يُقدّم ملف `invokeai/app/invocations/latents_to_image.py` ([latents_to_image.py:51-112](latents_to_image.py#L51-L112)) عملية فك الترميز:

```python
@torch.no_grad()
def invoke(self, context: InvocationContext) -> ImageOutput:
    # 1. تحميل التنسورات
    latents = context.tensors.load(self.latents.latents_name)
    
    # 2. تحميل VAE
    vae_info = context.models.load(self.vae.vae)
    
    # 3. تحويل الدقة
    if self.fp32:
        vae.to(dtype=torch.float32)
        latents = latents.float()
    else:
        vae.to(dtype=torch.float16)
        latents = latents.half()
    
    # 4. فك الترميز
    with torch.inference_mode():
        latents = latents / vae.config.scaling_factor
        image = vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)  # denormalize
    
    # 5. تحويل إلى صورة
    np_image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = VaeImageProcessor.numpy_to_pil(np_image)[0]
    
    return ImageOutput.build(image_dto)
```

### 5.3. المعادلات الرياضية لـ VAE

**عملية الترميز (Encoding):**

$$\boldsymbol{\mu}, \boldsymbol{\sigma} = \text{Encoder}(\mathbf{x})$$

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$$

**عملية فك الترميز (Decoding):**

$$\hat{\mathbf{x}} = \text{Decoder}(\mathbf{z})$$

**خسارة VAE:**

$$\mathcal{L}_{\text{VAE}} = \underbrace{\| \mathbf{x} - \hat{\mathbf{x}} \|_2^2}_{\text{Reconstruction Loss}} + \underbrace{D_{\text{KL}}(\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2) \| \mathcal{N}(0, I))}_{\text{KL Divergence}}$$

### 5.4. تحسين الذاكرة في VAE

يُقدّم ملف `invokeai/backend/util/vae_working_memory.py` خوارزمية تقدير الذاكرة المطلوبة:

```python
def estimate_vae_working_memory_sd15_sdxl(
    operation: str,
    image_tensor: torch.Tensor,
    vae: AutoencoderKL | AutoencoderTiny,
    tile_size: int | None = None,
    fp32: bool = False,
) -> int:
    """تقدير الذاكرة المطلوبة لعملية VAE."""
    # حساب الذاكرة بناءً على حجم المدخلات ونوع الدقة
    bytes_per_element = 4 if fp32 else 2
    working_memory = image_tensor.numel() * bytes_per_element * 2  # 2x للإدخال/الإخراج
    return working_memory
```

**خوارزمية Tiling (التقسيم):**

$$\text{Image} = \bigcup_{i=1}^{N} \text{Tile}_i$$

حيث كل $\text{Tile}_i$ تُعالج بشكل مستقل لتقليل استهلاك الذاكرة.

---

## سادساً: ملخص دورة الحياة الكاملة

### 6.1. التسلسل الزمني

```
الوقت (ثانية)
│
├── 0.0 ──── Model Loading
│           ├── قراءة من القرص: ~2-5s
│           └── نقل إلى الذاكرة: ~0.5-1s
│
├── 2.5 ──── Text Encoding
│           ├── Tokenization: ~0.01s
│           ├── Transformer: ~0.1-0.5s
│           └── Projection: ~0.01s
│
├── 3.1 ──── Denoising Loop (25 خطوة)
│           ├── كل خطوة: ~2-10s (CPU)
│           ├── UNet forward: ~1-5s
│           ├── Scheduler step: ~0.1s
│           └── المجموع: ~50-250s
│
├── 53.1 ── VAE Decoding
│           ├── تحميل VAE: ~1-2s
│           ├── فك الترميز: ~2-5s
│           └── تحويل الصورة: ~0.1s
│
└── 60.2 ── الصورة النهائية
```

### 6.2. استهلاك الذاكرة

```
الذاكرة (GB)
│
├── 0 ──── 
│
├── 2 ──── Model Loading
│           ├── UNet: ~1.5 GB (fp16)
│           ├── CLIP: ~0.5 GB (fp16)
│           └── VAE: ~0.3 GB (fp16)
│
├── 4.3 ── Text Encoding
│           ├── Text Embeddings: ~0.1 GB
│           └── KV Cache: ~0.2 GB
│
├── 4.6 ── Denoising Loop
│           ├── Latents: ~0.1 GB
│           ├── Noise Prediction: ~0.1 GB
│           └── Intermediate: ~0.5 GB
│
├── 5.3 ── VAE Decoding
│           ├── Input Latent: ~0.1 GB
│           ├── Output Image: ~0.3 GB
│           └── Working Memory: ~0.5 GB
│
└── 6.2 ── النتيجة النهائية
```

---

## سابعاً: استراتيجيات تحسين الأداء على CPU

### 7.1. تحسين الذاكرة

| الاستراتيجية | الوصف | التأثير |
|---|---|---|
| **Partial Loading** | تحميل أجزاء من النموذج | تقليل الذاكرة |
| **Model Offloading** | نقل النماذج غير النشطة | توفير مساحة |
| **Tiled Decoding** | تقسيم VAE إلى أجزاء | تقليل الذروة |

### 7.2. تحسين السرعة

| الاستراتيجية | الوصف | التأثير |
|---|---|---|
| **ONNX Runtime** | تسريع التنفيذ | 2-5x أسرع |
| **Quantization** | تقليل الدقة | 1.5-2x أسرع |
| **Batch Processing** | معالجة متزامنة | 1.5-3x أسرع |

### 7.3. معادلة التكلفة الحسابية

$$\text{Cost}_{\text{CPU}} = \sum_{t=1}^{T} \left( \text{FLOPS}_{\text{UNet}}(t) \times \text{Time}_{\text{step}}(t) \right)$$

$$\text{Cost}_{\text{GPU}} = \frac{\text{Cost}_{\text{CPU}}}{\text{GPU Speedup Factor}}$$

حيث:
- $T$: عدد خطوات Denoising
- $\text{FLOPS}_{\text{UNet}}(t)$: عدد العمليات الحسابية في الخطوة $t$
- $\text{GPU Speedup Factor}$: معامل تسريع GPU (20-100x)

---

## ثامناً: استنتاجات أكاديمية

### 8.1. ملاحظات تقنية مهمة

1. **ذاكرة النموذج:** النموذج الأساسي (UNet) يستهلك حوالي 1.5 GB على CPU في وضع float16
2. **سرعة CPU:** التوليد على CPU أبطأ بـ 20-40x من GPU بسبب:
   - عدم وجود cores متعددة للتعلم العميق
   - بطء الذاكرة العشوائية مقارنة بـ VRAM
   - غياب التسريع المخصص (Tensor Cores)
3. **إدارة الذاكرة:** نظام الكاش متعدد المستويات يمنع أخطاء Out of Memory

### 8.2. توصيات للتحسين

1. **استخدام ONNX:** للنماذج الثابتة، ONNX Runtime أسرع بـ 2-5x
2. **Quantization:** تحويل float32 إلى int8 يُقلل الذاكرة بنسبة 50%
3. **Model Distillation:** استخدام نماذج مصغرة (如 LCM) لتقليل الخطوات
4. **CPU Optimization:** استخدام Intel MKL أو OpenBLAS لتسريع المصفوفات

### 8.3. جدول المقارنة

| المعلمة | CPU (float32) | CPU (float16) | GPU (float16) |
|---|---|---|---|
| **الذاكرة** | ~3 GB | ~1.5 GB | ~1.5 GB |
| **السرعة** | ~60s | ~50s | ~2s |
| **الدقة** | عالية | جيدة | جيدة |
| **التكلفة** | مجاني | مجاني | مرتفع |

---

## المراجع العلمية

1. Ho, J., et al. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
2. Rombach, R., et al. "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022.
3. Song, J., et al. "Denoising Diffusion Implicit Models." ICLR 2021.
4. Luo, C. "Understanding Diffusion Models: A Unified Perspective." arXiv 2022.
5. Esser, P., et al. "Taming Transformers for High-Resolution Image Synthesis." CVPR 2021.

---

*آخر تحديث: يونيو 2026*
*المؤلف: قسم أبحاث الذكاء الاصطناعي*
