---
title: تكامل المزود الخارجي
---

يغطي هذا الدليل:

1. إضافة **نموذج خارجي** جديد (الأكثر شيوعًا؛ مزود موجود).
2. إضافة **مزود خارجي** جديد بالكامل (محول + تكوين + ربط واجهة المستخدم).

## 1) إضافة نموذج خارجي جديد (مزود موجود)

بالنسبة للنماذج المدعومة بالمزود (مثل OpenAI أو Gemini)، المصدر الأساسي هو
`invokeai/backend/model_manager/starter_models.py`.

### حقول النموذج المطلوبة

عرف `StarterModel` مع:

- `base=BaseModelType.External`
- `type=ModelType.ExternalImageGenerator`
- `format=ModelFormat.ExternalApi`
- `source="external://<provider_id>/<provider_model_id>"`
- `name`, `description`
- `capabilities=ExternalModelCapabilities(...)`
- اختياري `default_settings=ExternalApiModelDefaultSettings(...)`

مثال:

```python
new_external_model = StarterModel(
    name="Provider Model Name",
    base=BaseModelType.External,
    source="external://openai/my-model-id",
    description=(
        "Provider model (external API). "
        "Requires a configured OpenAI API key and may incur provider usage costs."
    ),
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img", "img2img", "inpaint"],
        supports_negative_prompt=False,
        supports_seed=False,
        supports_guidance=False,
        supports_steps=False,
        supports_reference_images=True,
        max_images_per_request=4,
    ),
    default_settings=ExternalApiModelDefaultSettings(
        width=1024,
        height=1024,
        num_images=1,
    ),
)
```

ثم ألحقه بـ `STARTER_MODELS`.

### نص الوصف المطلوب

يجب أن توضح أوصاف النماذج الخارجية الأولية بوضوح:

- مفتاح API مطلوب
- قد يترتب على الاستخدام تكاليف من جهة المزود

### يجب أن تكون القدرات دقيقة

تتحكم هذه العلامات مباشرة في رؤية واجهة المستخدم وحقول طلب الإرسال:

- `supports_negative_prompt`
- `supports_seed`
- `supports_guidance`
- `supports_steps`
- `supports_reference_images`

`supports_steps` مهم بشكل خاص: إذا كانت `False`، يتم إخفاء الخطوات لذلك النموذج ويتم إرسال `steps` كـ `null`.

### استقرار سلسلة المصدر

تتم مطابقة التجاوزات الأولية بواسطة `source` (`external://provider/model-id`). حافظ على استقرار هذا:

- تجاوزات القدرة/الإعدادات الافتراضية في وقت التشغيل تعتمد عليه
- اكتشاف التثبيت في واجهات برمجة تطبيقات النماذج الأولية يعتمد عليه

يفرض `STARTER_MODELS` قيم `source` فريدة مع تأكيد.

### ملاحظات سلوك التثبيت

- تتم إدارة النماذج الخارجية الأولية في إعداد **المزودين الخارجيين** (وليس علامة تبويب النماذج الأولية العادية).
- يتم تثبيت النماذج الخارجية الأولية تلقائيًا عند تكوين مزود.
- إزالة مفتاح API للمزود يزيل النماذج الخارجية المثبتة لذلك المزود.

## 2) بيانات الاعتماد والتكوين

يتم تخزين مفاتيح API للمزود الخارجي بشكل منفصل عن `invokeai.yaml`:

- الملف الافتراضي: `~/invokeai/api_keys.yaml`
- المسار المُحل: `<INVOKEAI_ROOT>/api_keys.yaml`

تبقى إعدادات المزود غير السرية (مثل تجاوزات عنوان URL الأساسي) في `invokeai.yaml`.

لا تزال متغيرات البيئة مدعومة، على سبيل المثال:

- `INVOKEAI_EXTERNAL_GEMINI_API_KEY`
- `INVOKEAI_EXTERNAL_OPENAI_API_KEY`

## 3) إضافة مزود جديد (فقط إذا لزم الأمر)

إذا كان نموذجك يستخدم مزودًا غير مدمج بالفعل:

1. أضف حقول التكوين في `invokeai/app/services/config/config_default.py`
   `external_<provider>_api_key` واختياري `external_<provider>_base_url`.
2. أضف تعيين حقل المزود في `invokeai/app/api/routers/app_info.py`
   (`EXTERNAL_PROVIDER_FIELDS`).
3. طبق محول المزود في `invokeai/app/services/external_generation/providers/`
   عن طريق وراثة `ExternalProvider`.
4. سجل المزود في `invokeai/app/api/dependencies.py` عند بناء
   `ExternalGenerationService`.
5. أضف إدخالات النموذج الأولي باستخدام `source="external://<provider>/<model-id>"`.
6. تعديل ترتيب واجهة المستخدم اختياريًا:
   `invokeai/frontend/web/src/features/modelManagerV2/subpanels/AddModelPanel/ExternalProviders/ExternalProvidersForm.tsx`
   (`PROVIDER_SORT_ORDER`).

## 4) التثبيت اليدوي الاختياري

يمكنك أيضًا تثبيت النماذج الخارجية مباشرة عبر:

`POST /api/v2/models/install?source=external://<provider_id>/<provider_model_id>`

إذا تم حذفه، يتم تعبئة `path` و `source` و `hash` تلقائيًا لتكوينات النموذج الخارجي.
عيّن القدرات بشكل متحفظ؛ خدمة التوليد الخارجي تفرض فحوصات القدرات في وقت التشغيل.
