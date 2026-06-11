# تحليل ملف package.json (Frontend)

```
المسار المقترح للملف: docs/frontend/package_json.md
```

---

## اسم الملف ومساره

- **اسم الملف:** `package.json`
- **المسار في المشروع:** `invokeai/frontend/web/package.json`

---

## الوظيفة الأساسية للملف (High-level Purpose)

هذا الملف هو **ملف تكوين Node.js الرئيسي** لواجهة المستخدم. يُعرّف التبعيات (Dependencies)، والأوامر (Scripts)، وإعدادات البناء. يمكن وصفه بأنه **الدليل التقني الشامل** لكل ما يتعلق بالبنية التحتية لواجهة المستخدم.

---

## معلومات المشروع

| الخاصية | القيمة |
|---|---|
| **الاسم** | `@invoke-ai/invoke-ai-ui` |
| **الإصدار** | `0.0.1` |
| **الخصوصية** | `true` (خاص) |
| **مُدير الحزم** | `pnpm@10.12.4` |

---

## أوامر التشغيل (Scripts)

| الأمر | الوصف |
|---|---|
| `dev` | تشغيل خادم التطوير Vite |
| `dev:host` | تشغيل مع إمكانية الوصول من الشبكة |
| `build` | البناء الكامل (lint + test + vite build) |
| `preview` | معاينة البناء |
| `lint` | تشغيل جميع فحوصات الجودة |
| `lint:prettier` | فحص التنسيق |
| `lint:eslint` | فحص الأخطاء البرمجية |
| `lint:knip` | فحص التبعيات غير المستخدمة |
| `lint:dpdm` | فحص الحلقات الدائرية |
| `lint:tsc` | فحص TypeScript |
| `fix` | إصلاح تلقائي |
| `test` | اختبارات Vitest مع المراقبة |
| `test:no-watch` | اختبارات بدون مراقبة |
| `storybook` | تشغيل Storybook |
| `typegen` | توليد الأنواع من OpenAPI |

---

## التبعيات الرئيسية (Dependencies)

### Core:
| الحزمة | الإصدار | الغرض |
|---|---|---|
| `react` | ^18.3.1 | مكتبة React |
| `react-dom` | ^18.3.1 | وصل React بـ DOM |
| `react-redux` | 9.2.0 | ربط Redux بـ React |
| `@reduxjs/toolkit` | 2.8.2 | إدارة الحالة |

### UI & Design:
| الحزمة | الغرض |
|---|---|
| `@invoke-ai/ui-library` | مكتبة UI المخصصة |
| `@xyflow/react` | محرر العقد المرئي |
| `framer-motion` | الحركات |
| `konva` | الرسم على Canvas |
| `cmdk` | لوحة الأوامر |

### Routing & State:
| الحزمة | الغرض |
|---|---|
| `react-router-dom` | التوجيه |
| `redux-remember` | حفظ الحالة持久化 |
| `redux-undo` | التراجع |
| `nanostores` | مخزن خفيف |

### Data & Validation:
| الحزمة | الغرض |
|---|---|
| `zod` | التحقق من البيانات |
| `react-hook-form` | إدارة النماذج |
| `jsondiffpatch` | مقارنة JSON |

### Utilities:
| الحزمة | الغرض |
|---|---|
| `es-toolkit` | أدوات JS |
| `nanoid` | توليد معرّفات فريدة |
| `uuid` | UUID |
| `i18next` | التدويل |

---

## تبعيات التطوير (DevDependencies)

| الحزمة | الغرض |
|---|---|
| `vite` | أدات البناء |
| `typescript` | TypeScript |
| `vitest` | إطار الاختبار |
| `eslint` | فحص الكود |
| `prettier` | تنسيق الكود |
| `storybook` | مكتبة المكونات |
| `@vitejs/plugin-react-swc` | دعم React مع SWC |
| `openapi-typescript` | توليد أنواع OpenAPI |

---

## إعدادات TypeScript

- **tsconfig.json** موجود
- **TypeScript 5.8.3**
- **Vite 7.0.5** كمُجمّع (Bundler)

---

## إضافات pnpm

```json
"preinstall": "npx only-allow pnpm"
```
- يُجبر استخدام `pnpm` فقط
- **لا يدعم npm أو yarn**

---

## إعدادات النشر

```json
"publishConfig": {
  "access": "restricted",
  "registry": "https://npm.pkg.github.com"
}
```
- يُنشر على GitHub Package Registry
- خاص (غير عام)
