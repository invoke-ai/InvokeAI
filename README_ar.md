# دليل الوثائق الشامل - InvokeAI
## Comprehensive Documentation Guide

---

## 📚 نظرة عامة على هيكل الوثائق

يحتوي مجلد `docs/` على وثائق شاملة لمشروع InvokeAI، منظمة في أقسام رئيسية تغطي الجوانب التقنية والتجارية والعملياتية للمشروع.

### 🖼️ لقطة شاشة من التطبيق

![تطبيق InvokeAI](img/Screenshot%202026-06-01%20225222.png)

```
docs/
├── architecture/           # البنية المعمارية والتصميم
├── backend/               # شرح ملفات Backend الرئيسية
├── business/              # التحليل التجاري والتكاليف
├── code_explanations/     # شرح تفصيلي للأكواد المصدرية
├── frontend/              # وثائق واجهة المستخدم
├── plugins/               # الوثائق الخاصة بالمكونات الإضافية
├── scripts/               # وثائق السكريبتات
├── src/                   # وثائق الكود المصدري
├── technical_deep_dive/   # تحليل تقني معمق
└── ui-design/             # وثائق تصميم الواجهة
```

---

## 🏗️ القسم الأول: البنية المعمارية (Architecture)

### [system_design.md](architecture/system_design.md)
**المخطط العام لعمل المشروع**
- وصف شامل للبنية المعمارية الثلاثية (Frontend, API, Backend)
- شرح تدفق البيانات من النص إلى الصورة
- تفاصيل معمارية العقد (Node-Based Architecture)
- نظام إدارة النماذج وقاعدة البيانات
- نظام الأحداث والمصادقة
- تقنيات المكونات المختلفة

**المسارات المرتبطة:**
- ← [technical_deep_dive/dependencies_architecture.md](technical_deep_dive/dependencies_architecture.md)
- ← [technical_deep_dive/model_execution_lifecycle.md](technical_deep_dive/model_execution_lifecycle.md)
- ← [technical_deep_dive/network_communication.md](technical_deep_dive/network_communication.md)

### [prompts_guide.md](architecture/prompts_guide.md)
**مستند نصوص التوليد (Prompts Repository)**
- مجموعة من النصوص التوليدية الجاهزة
- إعدادات مُثلى لـ CPU
- نصوص لمشاهد داخلية وخارجية
- نصوص للمنتجات الفاخرة والمعمارية
- نصوص للطعام والأزياء
- قواعد كتابة النصوص الفعّالة

---

## 🔧 القسم الثاني: Backend (الخلفية)

### [api_app.md](backend/api_app.md)
**تحليل ملف api_app.py**
- النقطة المركزية لإنشاء وتكوين تطبيق FastAPI
- تسجيل مسارات API (16 مسار رئيسي)
- طبقات الوسيطات (Middleware)
- دورة حياة التطبيق
- نظام المصادقة JWT

**المسارات المرتبطة:**
- ← [code_explanations/invokeai/app/api_app.md](code_explanations/invokeai/app/api_app.md)
- ← [code_explanations/invokeai/app/api/dependencies.md](code_explanations/invokeai/app/api/dependencies.md)

### [dependencies.md](backend/dependencies.md)
**تحليل ملف dependencies.py**
- مُنسّق التبعيات الرئيسي
- تهيئة جميع الخدمات (25+ خدمة)
- إنشاء خدمة InvocationServices الموحدة
- المزودون الخارجيون للتوليد

### [baseinvocation.md](backend/baseinvocation.md)
**تحليل ملف baseinvocation.py**
- الفئة الأساسية للعقد (Base Invocation)
- نظام تسجيل العقد
- دورة حياة العقدة

### [config_default.md](backend/config_default.md)
**تحليل ملف config_default.py**
- الإعدادات الافتراضية للتطبيق
- تكوين الأجهزة والنماذج

### [denoise_latents.md](backend/denoise_latents.md)
**تحليل ملف denoise_latents.py**
- عملية إزالة الضوضاء
- حلقة Denoising

### [diffusers_pipeline.md](backend/diffusers_pipeline.md)
**تحليل ملف diffusers_pipeline.py**
- خط أنابيب Stable Diffusion
- إدارة الذاكرة الفعالة

### [invocation_services.md](backend/invocation_services.md)
**تحليل ملف invocation_services.py**
- خدمة InvocationServices الموحدة
- تجميع جميع الخدمات

### [run_app.md](backend/run_app.md)
**تحليل ملف run_app.py**
- نقطة دخول التطبيق
- تشغيل الخادم

### [session_queue_router.md](backend/session_queue_router.md)
**تحليل ملف session_queue_router.py**
- مسار طابور الجلسات
- إدارة الدفعات

---

## 💼 القسم الثالث: الأعمال التجارية (Business)

### [product_strategy.md](business/product_strategy.md)
**دراسة التشغيل والتكاليف والباقات**
- متطلبات التشغيل والموارد
- تحليل التكاليف التقديرية
- خطة الباقات والاشتراكات (Free, Pro, Business)
- المميزات والعيوب العامة
- تحليل SWOT
- التوصيات الاستراتيجية
- الإيرادات المتوقعة

**المسارات المرتبطة:**
- ← [architecture/system_design.md](architecture/system_design.md)
- ← [technical_deep_dive/dependencies_architecture.md](technical_deep_dive/dependencies_architecture.md)

---

## 🔬 القسم الرابع: التحليل التقني المعمق (Technical Deep Dive)

### [dependencies_architecture.md](technical_deep_dive/dependencies_architecture.md)
**المستند الأول: تحليل البنية التحتية والمكتبات الأساسية**
- إطار إدارة الحزم (uv و pnpm)
- دور PyTorch في العمليات الحسابية على CPU
- دور مكتبة Diffusers في إدارة خطوط الأنابيب
- ONNX Runtime و PatchMatch
- تحليل التبعيات المتعددة
- مخطط التبعيات الشامل

**المسارات المرتبطة:**
- ← [architecture/system_design.md](architecture/system_design.md)
- ← [backend/dependencies.md](backend/dependencies.md)

### [model_execution_lifecycle.md](technical_deep_dive/model_execution_lifecycle.md)
**المستند الثاني: دورة حياة تشغيل النموذج في الذاكرة**
- آلية تحميل النموذج (Model Loading)
- مرحلة Text Encoding
- حلقة إزالة الضوضاء (Denoising Loop)
- مرحلة VAE Decoding
- استراتيجيات تحسين الأداء على CPU
- استهلاك الذاكرة والسرعة

**المسارات المرتبطة:**
- ← [architecture/system_design.md](architecture/system_design.md)
- ← [backend/diffusers_pipeline.md](backend/diffusers_pipeline.md)

### [network_communication.md](technical_deep_dive/network_communication.md)
**المستند الثالث: بروتوكولات الاتصال وتدفق البيانات**
- REST API (FastAPI Backend)
- WebSocket (Socket.IO)
- HTTP Proxy (Vite Dev Server)
- أحداث التقدم Real-time
- إدارة الحالة (Redux Toolkit)
- الأمان والمصادقة

**المسارات المرتبطة:**
- ← [backend/api_app.md](backend/api_app.md)
- ← [code_explanations/invokeai/app/api/sockets.md](code_explanations/invokeai/app/api/sockets.md)

---

## 💻 القسم الخامس: شرح الأكواد المصدرية (Code Explanations)

### API Routes
- [dependencies.md](code_explanations/invokeai/app/api/dependencies.md) - تبعيات API
- [boards.md](code_explanations/invokeai/app/api/routers/boards.md) - مسار الألباب
- [download_queue.md](code_explanations/invokeai/app/api/routers/download_queue.md) - مسار طابور التنزيل
- [images.md](code_explanations/invokeai/app/api/routers/images.md) - مسار الصور
- [model_manager.md](code_explanations/invokeai/app/api/routers/model_manager.md) - مسار مدير النماذج
- [session_queue.md](code_explanations/invokeai/app/api/routers/session_queue.md) - مسار طابور الجلسات
- [sockets.md](code_explanations/invokeai/app/api/sockets.md) - خادم Socket.IO

### Invocations (العقد)
- [baseinvocation.md](code_explanations/invokeai/app/invocations/baseinvocation.md) - الفئة الأساسية للعقد
- [compel.md](code_explanations/invokeai/app/invocations/compel.md) - معالجة النصوص
- [controlnet.md](code_explanations/invokeai/app/invocations/controlnet.md) - عقدة ControlNet
- [denoise_latents.md](code_explanations/invokeai/app/invocations/denoise_latents.md) - إزالة الضوضاء
- [flux_denoise.md](code_explanations/invokeai/app/invocations/flux_denoise.md) - FLUX Denoise
- [flux_model_loader.md](code_explanations/invokeai/app/invocations/flux_model_loader.md) - تحميل نموذج FLUX
- [flux_text_encoder.md](code_explanations/invokeai/app/invocations/flux_text_encoder.md) - تشفير نص FLUX
- [latents_to_image.md](code_explanations/invokeai/app/invocations/latents_to_image.md) - تحويل الكامن إلى صورة
- [model.md](code_explanations/invokeai/app/invocations/model.md) - عقدة النموذج

### Services
- [board_image_records_sqlite.md](code_explanations/invokeai/app/services/board_image_records/board_image_records_sqlite.md) - سجلات الألباب
- [boards_default.md](code_explanations/invokeai/app/services/boards/boards_default.md) - خدمة الألباب
- [config_default.md](code_explanations/invokeai/app/services/config/config_default.md) - خدمة الإعدادات

---

## 🎨 القسم السادس: تصميم الواجهة (UI Design)

يحتوي هذا القسم على وثائق تصميم واجهة المستخدم، بما في ذلك:
- مبادئ التصميم
- نظام الألوان
- المكونات المرئية
- تجربة المستخدم

---

## 📊 خريطة التنقل السريع

### للمطورين الجدد:
1. ابدأ بـ [architecture/system_design.md](architecture/system_design.md) لفهم البنية العامة
2. انتقل إلى [business/product_strategy.md](business/product_strategy.md) لفهم الجانب التجاري
3. اقرأ [technical_deep_dive/dependencies_architecture.md](technical_deep_dive/dependencies_architecture.md) لفهم البنية التحتية

### للمطورين Backend:
1. ابدأ بـ [backend/api_app.md](backend/api_app.md)
2. انتقل إلى [backend/dependencies.md](backend/dependencies.md)
3. استكشف [code_explanations/invokeai/app/api/](code_explanations/invokeai/app/api/)

### للمطورين Frontend:
1. راجع [architecture/system_design.md](architecture/system_design.md) لفهم الاتصال
2. اقرأ [technical_deep_dive/network_communication.md](technical_deep_dive/network_communication.md)
3. استكشف مجلد [frontend/](frontend/)

### للمطورين AI/ML:
1. ابدأ بـ [technical_deep_dive/model_execution_lifecycle.md](technical_deep_dive/model_execution_lifecycle.md)
2. انتقل إلى [backend/diffusers_pipeline.md](backend/diffusers_pipeline.md)
3. استكشف [code_explanations/invokeai/app/invocations/](code_explanations/invokeai/app/invocations/)

---

## 🔗 روابط سريعة

- **البنية المعمارية:** [architecture/](architecture/)
- **الخلفية:** [backend/](backend/)
- **الأعمال:** [business/](business/)
- **التحليل التقني:** [technical_deep_dive/](technical_deep_dive/)
- **شرح الأكواد:** [code_explanations/](code_explanations/)
- **الواجهة:** [ui-design/](ui-design/)

---

## 📝 ملاحظات

- جميع الملفات مكتوبة باللغة العربية مع مصطلحات إنجليزية
- الروابط نسبية (relative links) للتنقل السهل
- المستندات محدثة حتى يونيو 2026
- للمساهمة في تحسين الوثائق، يرجى مراجعة [CONTRIBUTING.md](../CONTRIBUTING.md)

---

*آخر تحديث: يونيو 2026*
