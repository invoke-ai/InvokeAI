# توثيق ملف: gallerySlice.ts

## مسار الملف الأصلي
```
invokeai/frontend/web/src/features/gallery/store/gallerySlice.ts
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/frontend/web/src/features/gallery/store/gallerySlice.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **جزء المعرض** (Gallery Slice) في واجهة InvokeAI الأمامية. يدير حالة المعرض وعرض الصور.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 Redux Toolkit
```typescript
import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
```

### 2.2 مكتبات المشروع
```typescript
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject, uniq } from 'es-toolkit';
import { logout } from 'features/auth/store/authSlice';
import type { BoardRecordOrderBy } from 'services/api/types';
import { assert } from 'tsafe';
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 الحالة الأولية

```typescript
const getInitialState = (): GalleryState => ({
  selection: [],
  shouldAutoSwitch: true,
  autoAssignBoardOnClick: true,
  autoAddBoardId: 'none',
  galleryImageMinimumWidth: 90,
  alwaysShowImageSizeBadge: false,
  selectedBoardId: 'none',
  galleryView: 'images',
  boardSearchText: '',
  starredFirst: true,
  orderDir: 'DESC',
  searchTerm: '',
  imageToCompare: null,
  comparisonMode: 'slider',
  comparisonFit: 'fill',
  shouldShowArchivedBoards: false,
  showVirtualBoards: false,
  virtualBoardsSectionOpen: true,
  boardsListOrderBy: 'created_at',
  boardsListOrderDir: 'DESC',
});
```

### 3.2 الإجراءات (Reducers)

#### تحديد الصورة
```typescript
imageSelected: (state, action: PayloadAction<string | null>) => {
  const selectedItem = action.payload;
  if (!selectedItem) {
    state.selection = [];
  } else {
    state.selection = [selectedItem];
  }
},
```

#### تغيير التحديد
```typescript
selectionChanged: (state, action: PayloadAction<string[]>) => {
  state.selection = uniq(action.payload);
},
```

#### تحديد اللوحة
```typescript
boardIdSelected: (
  state,
  action: PayloadAction<{
    boardId: BoardId;
    select?: {
      selection: GalleryState['selection'];
      galleryView: GalleryState['galleryView'];
    };
  }>
) => {
  const { boardId, select } = action.payload;
  state.selectedBoardId = boardId;
  if (select) {
    state.selection = select.selection;
    state.galleryView = select.galleryView;
  }
},
```

#### تغيير عرض المعرض
```typescript
galleryViewChanged: (state, action: PayloadAction<GalleryView>) => {
  state.galleryView = action.payload;
},
```

#### تبديل وضع المقارنة
```typescript
comparisonModeCycled: (state) => {
  switch (state.comparisonMode) {
    case 'slider':
      state.comparisonMode = 'side-by-side';
      break;
    case 'side-by-side':
      state.comparisonMode = 'hover';
      break;
    case 'hover':
      state.comparisonMode = 'slider';
      break;
  }
},
```

#### تبديل الصور المقارنة
```typescript
comparedImagesSwapped: (state) => {
  if (state.imageToCompare) {
    const oldSelection = state.selection;
    state.selection = [state.imageToCompare];
    state.imageToCompare = oldSelection[0] ?? null;
  }
},
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التعامل مع التحديد الفارغ
```typescript
imageSelected: (state, action: PayloadAction<string | null>) => {
  const selectedItem = action.payload;
  if (!selectedItem) {
    state.selection = [];
  } else {
    state.selection = [selectedItem];
  }
},
```

### 4.2 التعامل مع اللوحات الافتراضية
```typescript
if (isVirtualBoardId(action.payload)) {
  return;
}
```

### 4.3 التعامل مع الإعدادات
```typescript
showVirtualBoardsChanged: (state, action: PayloadAction<boolean>) => {
  state.showVirtualBoards = action.payload;
  if (!action.payload && isVirtualBoardId(state.selectedBoardId)) {
    state.selectedBoardId = 'none';
    state.selection = [];
  }
},
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **تنظيم واضح**: فصل واضح للحالة والإجراءات.
2. **灵活性**: دعم أوضاع عرض متعددة.
3. **كفاءة الأداء**: استخدام uniq لإزالة التكرار.

### نقاط الضعف
1. **عدد كبير من الإجراءات**: قد يكون معقداً للصيانة.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Gallery Slice Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  GalleryState                                               │
│       │                                                     │
│       ├── selection: string[]                               │
│       ├── selectedBoardId: BoardId                          │
│       ├── galleryView: GalleryView                          │
│       ├── comparisonMode: ComparisonMode                    │
│       ├── orderDir: OrderDir                                │
│       └── ... (other properties)                            │
│       │                                                     │
│       ▼                                                     │
│  Reducers                                                   │
│       │                                                     │
│       ├── imageSelected()                                   │
│       ├── selectionChanged()                                │
│       ├── boardIdSelected()                                 │
│       ├── galleryViewChanged()                              │
│       ├── comparisonModeCycled()                            │
│       └── comparedImagesSwapped()                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [Redux Toolkit](https://redux-toolkit.js.org/)
- [React Flow](https://reactflow.dev/)
- [Gallery Component](https://en.wikipedia.org/wiki/Image_gallery)
