# تحليل ملف dependencies.py

```
المسار المقترح للملف: docs/backend/dependencies.md
```

---

## اسم الملف ومساره

- **اسم الملف:** `dependencies.py`
- **المسار في المشروع:** `invokeai/app/api/dependencies.py`

---

## الوظيفة الأساسية للملف (High-level Purpose)

هذا الملف هو **مُنسّق التبعيات الرئيسي (Main Dependency Coordinator)** الذي يُهيّئ جميع الخدمات ويُنشئ خدمة `InvocationServices` الموحدة. يعمل كـ **نقطة التحكم المركزية** لدورة حياة الخدمات من الإنشاء إلى الإغلاق. يمكن وصفه بأنه **المُهندس الذي يبني البنية التحتية الكاملة** للتطبيق.

---

## المكتبات والحزم المستخدمة (Dependencies)

| المكتبة | الغرض |
|---|---|
| `asyncio` | الحلقة غير المتزامنة |
| `torch` | PyTorch |
| جميع خدمات `invokeai.app.services.*` | جميع الخدمات الخلفية |

---

## شرح المدخلات والمخرجات (Inputs/Outputs)

### المدخلات:
- `config`: إعدادات التطبيق
- `event_handler_id`: معرّف معالج الأحداث
- `loop`: الحلقة غير المتزامنة
- `logger`: خدمة التسجيل

### المخرجات:
- `ApiDependencies.invoker`: مُنشئ الجلسات (Invoker) الجاهز للعمل

---

## تفصيل الدوال والأساليب البرمجية (Functions & Methods)

### 1. `check_internet()` - فحص الاتصال
```
دالة بسيطة للتحقق من الاتصال بالإنترنت عبر ping إلى huggingface.co.
```

### 2. `ApiDependencies` - فئة التبعيات

#### `initialize()` - دالة التهيئة (Static Method):
```
الدالة الرئيسية التي تُهيّء كل شيء.
```

**خطوات التهيئة بالترتيب:**

| # | الخطوة | الوصف |
|---|---|---|
| 1 | تهيئة ملفات الصور | `DiskImageFileStorage` |
| 2 | تهيئة قاعدة البيانات | `init_db()` - SQLite |
| 3 | تحميل JWT Secret | من قاعدة البيانات |
| 4 | تهيئة سجلات الألباب | `SqliteBoardImageRecordStorage` |
| 5 | تهيئة خدمة الألباب | `BoardImagesService` |
| 6 | تهيئة سجلات الصور | `SqliteImageRecordStorage` |
| 7 | تهيئة خدمة الصور | `ImageService` |
| 8 | تهيئة التخزين المؤقت | `MemoryInvocationCache` |
| 9 | تهيئة التسلسل | `ObjectSerializerDisk` للتنسورات والشرط |
| 10 | تهيئة طابور التنزيل | `DownloadQueueService` |
| 11 | تهيئة سجلات النماذج | `ModelRecordServiceSQL` |
| 12 | تهيئة مدير النماذج | `ModelManagerService` |
| 13 | تهيئة التوليد الخارجي | `ExternalGenerationService` مع 4 مزودين |
| 14 | تهيئة معالج الجلسات | `DefaultSessionProcessor` |
| 15 | تهيئة طابور الجلسات | `SqliteSessionQueue` |
| 16 | تهيئة سجلات سير العمل | `SqliteWorkflowRecordsStorage` |
| 17 | تهيئة سجلات الأنماط | `SqliteStylePresetRecordsStorage` |
| 18 | تهيئة إدارة المستخدمين | `UserService` |
| 19 | تجميع الخدمات | `InvocationServices(...)` |
| 20 | إنشاء Invoker | `Invoker(services)` |
| 21 | مزامنة النماذج الخارجية | `sync_configured_external_starter_models()` |
| 22 | تنظيف قاعدة البيانات | `db.clean()` |

#### `shutdown()` - دالة الإغلاق (Static Method):
```
إيقاف Invoker وجميع الخدمات المرتبطة به.
```

### 3. المزودون الخارجيون (External Providers):
```
4 مزودين للتوليد الخارجي:
1. AlibabaCloudProvider - خدمة علي بابا السحابية
2. GeminiProvider - نموذج Google Gemini
3. OpenAIProvider - نماذج OpenAI
4. SeedreamProvider - نموذج Seedream
```

---

## الأهمية في المعمارية

```
                    api_app.py
                        |
                        v
                ApiDependencies.initialize()
                        |
          ┌─────────────┼─────────────┐
          v             v             v
     Database    Services     Invoker
          |             |             |
          v             v             v
     SQLite      25+ Service    Session
     Tables      Classes       Processor
```

- ** initialise once:** يتم تهيئة كل شيء مرة واحدة عند بدء التشغيل
- **Singleton Pattern:** `ApiDependencies` هو singleton عبر السمة الثابتة
- **Graceful Shutdown:** إغلاق جميع الخدمات بسلاسة عند الإغلاق
