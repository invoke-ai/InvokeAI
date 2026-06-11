# المخطط العام لعمل المشروع
## System Architecture & Workflow

```
invokeai/
├── docs/
│   └── architecture/
│       └── system_design.md  <-- هذا الملف
```

---

## ملخص عام

InvokeAI هو نظام متكامل لتوليد الصور بالذكاء الاصطناعي مبني على بنية **العقد (Node-Based Architecture)**. يتكون من ثلاثة طبقات رئيسية: واجهة المستخدم (Frontend)، وطبقة API، ومحرك التوليد (Backend). يُستخدم نموذج **Stable Diffusion** كمحرك أساسي مع دعم متعدد النماذج الأخرى.

---

## 1. المخطط الهيكلي العام

```
┌─────────────────────────────────────────────────────────────────┐
│                      واجهة المستخدم (Frontend)                   │
│                     React + TypeScript + Vite                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Parameter │  │  Prompt  │  │  Gallery │  │   Node       │   │
│  │  Panel    │  │  Editor  │  │          │  │   Editor     │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Model   │  │  Queue   │  │  Board   │  │   Workflow   │   │
│  │ Manager  │  │  Panel   │  │ Manager  │  │   Library    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                    Socket.IO + REST API
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    طبقة API (FastAPI Backend)                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Routers (18 مسار)                      │  │
│  │  auth │ models │ images │ boards │ queue │ workflows     │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                Services (25+ خدمة)                        │  │
│  │  SessionProcessor │ SessionQueue │ ModelManager          │  │
│  │  ImageService │ BoardService │ EventService              │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Invocations (115+ عقدة)                      │  │
│  │  DenoiseLatents │ TextToImage │ ControlNet │ LoRA        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                     InvocationServices
                              │
┌─────────────────────────────────────────────────────────────────┐
│                 محرك التوليد (AI Backend)                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Stable       │  │ FLUX Engine  │  │ Model Manager        │  │
│  │ Diffusion    │  │              │  │ (Load/Search/Cache)  │  │
│  └─────────────┘  └──────────────┘  └──────────────────────┘  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ ControlNet   │  │ IP-Adapter   │  │ Image Utils          │  │
│  └─────────────┘  └──────────────┘  └──────────────────────┘  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ LoRA/Patch  │  │ Quantization │  │ VAE Encode/Decode    │  │
│  └─────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              v               v               v
         ┌─────────┐   ┌──────────┐   ┌───────────┐
         │ SQLite  │   │  Models  │   │  Outputs  │
         │   DB    │   │  Folder  │   │  Folder   │
         └─────────┘   └──────────┘   └───────────┘
```

---

## 2. تدفق البيانات بالتفصيل

### 2.1. تدفق إنشاء صورة من النص إلى الصورة (Text-to-Image)

```
الخطوة 1: المستخدم يدخل النص
┌──────────────────────────────────────┐
│  Prompt: "A luxury minimalist room" │
│  Negative: "ugly, blurry"           │
│  Model: SD 1.5                      │
│  Steps: 30                          │
│  CFG Scale: 7                       │
│  Size: 512x512                      │
└──────────────────────────────────────┘
                │
                v
الخطوة 2: واجهة المستخدم ترسل الطلب
┌──────────────────────────────────────┐
│  POST /api/v1/queue/{id}/enqueue_batch │
│  Body: { graph: { nodes: [...] } }  │
└──────────────────────────────────────┘
                │
                v
الخطوة 3: طابور الجلسات يستقبل الطلب
┌──────────────────────────────────────┐
│  SessionQueue.enqueue_batch()        │
│  - إنشاء Batch جديد                 │
│  - إنشاء SessionQueueItem           │
│  - إضافته إلى الطابور              │
└──────────────────────────────────────┘
                │
                v
الخطوة 4: معالج الجلسات ينفذ
┌──────────────────────────────────────┐
│  SessionProcessor.process_next()     │
│  - أخذ العنصر التالي                │
│  - بناء GraphExecutionState         │
│  - تنفيذ العقد بالترتيب             │
└──────────────────────────────────────┘
                │
                v
الخطوة 5: العقد تُنفذ بالتسلسل
┌──────────────────────────────────────┐
│  1. model_loader (تحميل UNet+VAE)   │
│  2. text_encoder (ترميز النص)       │
│  3. noise (توليد ضوضاء عشوائية)     │
│  4. denoise_latents (Denoising)      │
│  5. latents_to_image (VAE Decode)    │
│  6. save_image (حفظ الصورة)         │
└──────────────────────────────────────┘
                │
                v
الخطوة 6: محرك التوليد يعالج
┌──────────────────────────────────────┐
│  StableDiffusionGeneratorPipeline:   │
│  - text_embeddings = CLIP(text)      │
│  - latent = zeros(1,4,64,64)         │
│  - for t in timesteps:               │
│      noise_pred = UNet(latent, t)    │
│      latent = scheduler.step(...)    │
│  - image = VAE.decode(latent)        │
└──────────────────────────────────────┘
                │
                v
الخطوة 7: الصورة تُحفظ وتُرسل
┌──────────────────────────────────────┐
│  - حفظ في outputs/images/           │
│  - إنشاء سجل في SQLite              │
│  - إرسال حدث عبر Socket.IO          │
│  - تحديث المعرض في الواجهة          │
└──────────────────────────────────────┘
```

### 2.2. تدفق معالجة الصورة (Image Processing)

```
Input Image → VAE Encode → Latent Space → Denoise → VAE Decode → Output Image
                                    ↑
                              ControlNet/IP-Adapter
```

### 2.3. تدفق Inpainting

```
Original Image + Mask → VAE Encode → Denoise (with mask guidance) → VAE Decode → Result
```

---

## 3. معمارية العقد (Node-Based Architecture)

### 3.1. مبدأ العمل

```
┌─────────┐     ┌─────────────┐     ┌─────────────┐
│ Text    │────▶│ Denoise     │────▶│ Latents     │
│ Encode  │     │ Latents     │     │ to Image    │
└─────────┘     └─────────────┘     └─────────────┘
                      │
                ┌─────┴─────┐
                │  Control  │
                │  Net      │
                └───────────┘
```

### 3.2. أنواع الاتصالات بين العقد

| النوع | الوصف |
|---|---|
| **Input Field** | حقل يأخذ قيمة من حقل آخر |
| **Connection** | اتصال مباشر بين حقلين |
| **Direct** | قيمة مباشرة من المستخدم |

### 3.3. دورة حياة العقدة

1. **التسجيل:** عبر الديكوريتر `@invocation`
2. **التصنيف:** Stable/Beta/Prototype
3. **التخزين المؤقت:** حسب `use_cache`
4. **التنفيذ:** عبر `invoke(context)`
5. **المخرجات:** عبر `BaseInvocationOutput`

---

## 4. نظام إدارة النماذج

```
┌─────────────────────────────────────────┐
│           Model Manager                  │
├─────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐            │
│  │ Search   │  │ Load     │            │
│  │ (Taxonomy)│  │ (Cache)  │            │
│  └──────────┘  └──────────┘            │
│  ┌──────────┐  ┌──────────┐            │
│  │ Install  │  │ Convert  │            │
│  │          │  │          │            │
│  └──────────┘  └──────────┘            │
└─────────────────────────────────────────┘
         │               │
    ┌────┴────┐     ┌────┴────┐
    │ Model   │     │ Model   │
    │ Configs │     │ Cache   │
    │ (SQL)   │     │ (Disk)  │
    └─────────┘     └─────────┘
```

### النماذج المدعومة:
- **Stable Diffusion 1.5/2.x**
- **SDXL (1.0, Turbo)**
- **SD 3.x**
- **FLUX (1.0, 2.0)**
- **ControlNet (Canny, Depth, OpenPose)**
- **IP-Adapter**
- **LoRA/DAN**
- **T2I-Adapter**

---

## 5. نظام قاعدة البيانات

```
SQLite Database (invokeai.db)
├── Images
├── Boards
├── Board Images
├── Model Records
├── Model Relationships
├── Session Queue
├── Workflow Records
├── Style Presets
├── Client State
├── Users
└── App Settings
```

### مكتبة ORM: SQLModel (Pydantic + SQLAlchemy)

---

## 6. نظام الأحداث (Event System)

```
Backend                          Frontend
   │                                │
   ├─── EventService ───────────────┤
   │    (FastAPI Events)            │
   │                                │
   ├─── Socket.IO ──────────────────┤
   │    (WebSocket)                 │
   │                                │
   └─── REST API ───────────────────┘
        (HTTP)                       │
                                     ├─── RTK Query
                                     ├─── React Components
                                     └─── Redux Store
```

### أنواع الأحداث الرئيسية:
- `invocation.started`
- `invocation.progress`
- `invocation.denoised`
- `invocation.complete`
- `invocation.error`
- `model.download.started`
- `model.download.progress`
- `model.download.complete`

---

## 7. نظام المصادقة (Authentication)

```
┌──────────────┐     ┌──────────────┐
│   Login      │────▶│  JWT Token   │
│   Request    │     │  Generation  │
└──────────────┘     └──────┬───────┘
                            │
                    ┌───────┴───────┐
                    │  Sliding      │
                    │  Window       │
                    │  Token        │
                    └───────┬───────┘
                            │
                    ┌───────┴───────┐
                    │  Token        │
                    │  Refresh      │
                    │  (on POST)    │
                    └───────────────┘
```

### الميزات:
- **Sliding Window Expiry:** تجديد التلقائي عند النشاط
- **Remember Me:** توقيت أطول (7 أيام)
- **Admin/User Roles:** صلاحيات متدرجة

---

## 8. نظام سير العمل (Workflows)

```
┌─────────────────────────────────────────┐
│           Workflow Editor                │
│  (Visual Node-Based Editor)             │
├─────────────────────────────────────────┤
│  ┌───────────┐     ┌───────────┐       │
│  │ Load      │────▶│ Process   │       │
│  │ Image     │     │ (Nodes)   │       │
│  └───────────┘     └─────┬─────┘       │
│                          │             │
│                    ┌─────▼─────┐       │
│                    │ Save      │       │
│                    │ Output    │       │
│                    └───────────┘       │
└─────────────────────────────────────────┘
         │
    ┌────┴────┐
    │ SQLite  │
    │ Storage │
    └─────────┘
```

### الميزات:
- **حفظ/تحميل:** سير العمل كملفات JSON
- **مشاركة:** تصدير واستيراد
- ** thumbnails:** صور مصغرة تلقائية
- **Version Control:** إصدارات متعددة

---

## 9. ملخص تقني

| المكون | التقنية |
|---|---|
| **Frontend** | React 18, TypeScript, Vite, Redux Toolkit |
| **Backend API** | FastAPI, Python 3.11+ |
| **AI Engine** | PyTorch, Diffusers, Transformers |
| **Database** | SQLite (via SQLModel) |
| **WebSocket** | Socket.IO |
| **File Storage** | Disk-based |
| **Model Storage** | Disk-based with caching |
| **Package Manager** | pnpm (Frontend), uv (Backend) |
| **Containerization** | Docker + Docker Compose |
