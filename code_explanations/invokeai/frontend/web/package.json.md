# توثيق ملف: package.json

## مسار الملف الأصلي
```
invokeai/frontend/web/package.json
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/frontend/web/package.json.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **ملف تكوين المشروع** (Project Configuration) لواجهة المستخدم الأمامية في InvokeAI. وهو يحتوي على جميع التبعيات (Dependencies)، وأوامر النص (Scripts)، وإعدادات البناء.

---

## ثانياً: معلومات المشروع

```json
{
  "name": "@invoke-ai/invoke-ai-ui",
  "private": true,
  "version": "0.0.1"
}
```

---

## ثالثاً: أوامر النص (Scripts)

### أوامر التطوير
```json
{
  "dev": "vite dev",
  "dev:host": "vite dev --host"
}
```
- **dev**: تشغيل خادم التطوير المحلي.
- **dev:host**: تشغيل الخادم للوصول من أجهزة أخرى.

### أوامر البناء
```json
{
  "build": "pnpm run lint && vitest run && vite build",
  "preview": "vite preview"
}
```
- **build**: التحقق من الكود ثم بناء التطبيق.
- **preview**: معاينة التطبيق المبني.

### أوامر التحقق (Linting)
```json
{
  "lint": "concurrently -g -c red,green,yellow,blue,magenta pnpm:lint:*",
  "lint:knip": "knip --tags=-knipignore",
  "lint:dpdm": "dpdm --no-warning --no-tree --transform --exit-code circular:1 src/main.tsx",
  "lint:eslint": "eslint --max-warnings=0 .",
  "lint:prettier": "prettier --check .",
  "lint:tsc": "tsc --noEmit"
}
```
- **lint**: تشغيل جميع فحوصات الكود.
- **lint:knip**: فحص التبعيات غير المستخدمة.
- **lint:dpdm**: فحص الدورات في التبعيات.
- **lint:eslint**: فحص مشاكل الكود.
- **lint:prettier**: فحص التنسيق.
- **lint:tsc**: فحص TypeScript.

### أوامر الإصلاح
```json
{
  "fix": "eslint --fix . && prettier --log-level warn --write ."
}
```
- **fix**: إصلاح مشاكل الكود والتنسيق تلقائياً.

### أوامر الاختبار
```json
{
  "test": "vitest",
  "test:run": "vitest run",
  "test:ui": "vitest --coverage --ui",
  "test:no-watch": "vitest --no-watch"
}
```
- **test**: تشغيل اختبارات Vitest.
- **test:no-watch**: تشغيل الاختبارات بدون مراقبة.

---

## رابعاً: التبعيات الرئيسية (Dependencies)

### مكتبات React
```json
{
  "react": "^18.3.1",
  "react-dom": "^18.3.1",
  "react-redux": "9.2.0",
  "react-router-dom": "^7.12.0"
}
```

### مكتبات الحالة (State Management)
```json
{
  "@reduxjs/toolkit": "2.8.2",
  "nanostores": "^1.0.1",
  "@nanostores/react": "^1.0.0",
  "redux-remember": "^5.2.0",
  "redux-undo": "^1.1.0"
}
```

### مكتبات الواجهة (UI)
```json
{
  "@invoke-ai/ui-library": "github:invoke-ai/ui-library#v0.0.48",
  "chakra-react-select": "^4.9.2",
  "framer-motion": "^11.10.0",
  "react-icons": "^5.5.0"
}
```

### مكتبات الرسم (Drawing)
```json
{
  "@xyflow/react": "^12.8.2",
  "konva": "^9.3.22",
  "perfect-freehand": "^1.2.2"
}
```

### مكتبات الاتصال (Communication)
```json
{
  "socket.io-client": "^4.8.1"
}
```

### مكتبات التحقق (Validation)
```json
{
  "zod": "^4.0.10",
  "zod-validation-error": "^3.5.2"
}
```

### مكتبات الترجمة (Internationalization)
```json
{
  "i18next": "^25.3.2",
  "react-i18next": "^15.5.3",
  "i18next-http-backend": "^3.0.2"
}
```

---

## خامساً: التبعيات التطويرية (DevDependencies)

### TypeScript
```json
{
  "typescript": "^5.8.3"
}
```

### Vite
```json
{
  "vite": "^7.0.5",
  "@vitejs/plugin-react-swc": "^3.9.0"
}
```

### ESLint
```json
{
  "eslint": "^9.31.0",
  "@typescript-eslint/eslint-plugin": "^8.37.0",
  "@typescript-eslint/parser": "^8.37.0"
}
```

### Prettier
```json
{
  "prettier": "^3.5.3"
}
```

### Vitest
```json
{
  "vitest": "^3.1.2",
  "@vitest/coverage-v8": "^3.1.2"
}
```

---

## سادساً: إعدادات Node.js

```json
{
  "engines": {
    "pnpm": "10"
  },
  "packageManager": "pnpm@10.12.4"
}
```
- **pnpm**: إدارة الحزم المطلوبة.
- **packageManager**: إصدار pnpm المحدد.

---

## سابعاً: تقييم الكفاءة

### نقاط القوة
1. **استخدام pnpm**: أسرع من npm وyarn.
2. **TypeScript**: كود أكثر أماناً.
3. **Vite**: بناء سريع لتطبيقات React.
4. **ESLint + Prettier**: تنسيق وفحص موحد.
5. **Vitest**: اختبارات سريعة.

### نقاط الضعف
1. **عدد كبير من التبعيات**: قد يبطئ وقت البناء.
2. **التبعيات التجريبية**: بعض التبعيات في حالة beta.

---

## ثامناً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Frontend Tech Stack                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Build Tools                                                │
│  ├── Vite (Bundler)                                         │
│  ├── TypeScript (Type Safety)                               │
│  └── SWC (Fast Compilation)                                 │
│                                                             │
│  State Management                                           │
│  ├── Redux Toolkit (Global State)                           │
│  ├── Nanostores (Local State)                               │
│  └── Redux Remember (Persistence)                           │
│                                                             │
│  UI Framework                                               │
│  ├── React 18 (Component Model)                             │
│  ├── Chakra UI (Component Library)                          │
│  └── Framer Motion (Animations)                             │
│                                                             │
│  Communication                                              │
│  ├── Socket.IO (Real-time)                                  │
│  └── RTK Query (REST API)                                   │
│                                                             │
│  Testing                                                    │
│  ├── Vitest (Unit Tests)                                    │
│  └── ESLint (Code Quality)                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## تاسعاً: المراجع المرجعية
- [pnpm Documentation](https://pnpm.io/)
- [Vite Documentation](https://vitejs.dev/)
- [React 18](https://react.dev/)
- [Redux Toolkit](https://redux-toolkit.js.org/)
- [Vitest](https://vitest.dev/)
