# تحليل ملف denoise_latents.py

```
المسار المقترح للملف: docs/backend/denoise_latents.md
```

---

## اسم الملف ومساره

- **اسم الملف:** `denoise_latents.py`
- **المسار في المشروع:** `invokeai/app/invocations/denoise_latents.py`

---

## الوظيفة الأساسية للملف (High-level Purpose)

هذا الملف يُعرّف **عقدة Denoise Latents** وهي العقدة المسؤولة عن **عملية توليد الصور الفعلية** في Stable Diffusion. إنها تأخذ التمثيل الكامن (Latent Representation) وتقوم بـ Denoising لإنتاج الصورة النهائية. يمكن وصفها بأنها **قلب محرك التوليد** - حيث تحدث معجزة الذكاء الاصطناعي فعلياً.

---

## المكتبات والحزم المستخدمة (Dependencies)

| المكتبة | الغرض |
|---|---|
| `torch` | PyTorch |
| `diffusers` | مكتبة Stable Diffusion |
| `torchvision` | تحويلات الصور |
| `PIL` | معالجة الصور |
| `pydantic` | التحقق من البيانات |

---

## شرح المدخلات والمخرجات (Inputs/Outputs)

### المدخلات:
- `positive_prompt`: النص الإيجابي
- `negative_prompt`: النص السلبي
- `unet`: نموذج UNet
- `scheduler_info`: معلومات المُجدول
- `seed`: البذرة العشوائية
- `steps`: عدد خطوات Denoising
- `cfg_scale`: معامل التوجيه (Guidance Scale)
- `denoise_strength`: قوة Denoising
- `control`: بيانات ControlNet
- `ip_adapter`: بيانات IP-Adapter
- `t2i_adapter`: بيانات T2I-Adapter
- `mask`: قناع Inpainting
- `masked_latents`: الصور المقنعة

### المخرجات:
- `latents`: التنسورات المُعالجة
- `seed`: البذرة المستخدمة

---

## تفصيل الدوال والأساليب البرمجية (Functions & Methods)

### 1. `get_scheduler()` - تحميل المُجدول
```
دالة لتحميل وتكوين المُجدول مع تجاوزات خاصة.
```

### 2. `DenoiseLatentsInvocation` - العقدة الرئيسية
```
العقدة المسؤولة عن عملية Denoising.
```

#### `invoke()` - الدالة الرئيسية:
```
الدالة التي تُنفذ عملية التوليد بالكامل.
```

**الخطوات:**

1. **تحميل النموذج:** UNet + VAE + Text Encoder
2. **تجهيز النص:** Tokenize + Encode
3. **إعداد الـ Scheduler:** حسب الاختيار
4. **إعداد التوجيه:** cfg_scale + negative prompt
5. **إعداد ControlNet:** إن وُجد
6. **إعداد IP-Adapter:** إن وُجد
7. **إعداد T2I-Adapter:** إن وُجد
8. **إجراء Denoising:** عبر `StableDiffusionGeneratorPipeline`
9. **إعادة الترميز:** من Latent إلى RGB

### 3. المُجدولات المدعومة:
```python
SCHEDULER_MAP = {
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
    "deis": DEISMultistepScheduler,
    "dpm_solver_multistep": DPMSolverMultistepScheduler,
    "dpm_solver_singlestep": DPMSolverSinglestepScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "heun": HeunDiscreteScheduler,
    "kdpm2": KDPM2DiscreteScheduler,
    "kdpm2_a": KDPM2AncestralDiscreteScheduler,
    "lms": LMSDiscreteScheduler,
    "pndm": PNDMScheduler,
    "unipc": UniPCMultistepScheduler,
    "lcm": LCMScheduler,
    "tcd": TCDScheduler,
}
```

### 4. معالجة LoRA:
```
تطبيق طبقات LoRA على النموذج الأساسي.
```

### 5. معالجة FreeU:
```
تحسينات FreeU لجودة الصور.
```

---

## الأهمية في خط التوليد

```
User Prompt
     |
     v
Text-to-Latents (DenoiseLatentsInvocation)  <-- هذا الملف
     |
     v
Latents-to-Image (VAE Decode)
     |
     v
Final Image
```

هذه العقدة هي **الأكثر تعقيداً** في النظام بأكمله، وتتداخل مع جميع النماذج الفرعية.
