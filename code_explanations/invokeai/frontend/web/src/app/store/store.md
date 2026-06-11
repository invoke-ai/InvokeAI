# توثيق ملف: store.ts

## مسار الملف الأصلي
```
invokeai/frontend/web/src/app/store/store.ts
```

## مسار ملف التوثيق
```
docs/code_explanations/invokeai/frontend/web/src/app/store/store.md
```

---

## أولاً: نظرة عامة على الملف

يُمثّل هذا الملف **مخزن Redux** (Redux Store) الرئيسي في واجهة InvokeAI الأمامية. يدير حالة التطبيق باستخدام Redux Toolkit.

---

## ثانياً: تشريح المكتبات المستوردة

### 2.1 Redux Toolkit
```typescript
import type { ThunkDispatch, TypedStartListening, UnknownAction } from '@reduxjs/toolkit';
import { addListener, combineReducers, configureStore, createAction, createListenerMiddleware } from '@reduxjs/toolkit';
```

### 2.2 مكتبات المشروع
```typescript
import { logger } from 'app/logging/logger';
import { errorHandler } from 'app/store/enhancers/reduxRemember/errors';
import { addAdHocPostProcessingRequestedListener } from 'app/store/middleware/listenerMiddleware/listeners/addAdHocPostProcessingRequestedListener';
import { addAnyEnqueuedListener } from 'app/store/middleware/listenerMiddleware/listeners/anyEnqueued';
import { addAppStartedListener } from 'app/store/middleware/listenerMiddleware/listeners/appStarted';
import { addBatchEnqueuedListener } from 'app/store/middleware/listenerMiddleware/listeners/batchEnqueued';
import { addDeleteBoardAndImagesFulfilledListener } from 'app/store/middleware/listenerMiddleware/listeners/boardAndImagesDeleted';
import { addBoardIdSelectedListener } from 'app/store/middleware/listenerMiddleware/listeners/boardIdSelected';
import { addBulkDownloadListeners } from 'app/store/middleware/listenerMiddleware/listeners/bulkDownload';
import { addGetOpenAPISchemaListener } from 'app/store/middleware/listenerMiddleware/listeners/getOpenAPISchema';
import { addImageAddedToBoardFulfilledListener } from 'app/store/middleware/listenerMiddleware/listeners/imageAddedToBoard';
import { addImageRemovedFromBoardFulfilledListener } from 'app/store/middleware/listenerMiddleware/listeners/imageRemovedFromBoard';
import { addModelSelectedListener } from 'app/store/middleware/listenerMiddleware/listeners/modelSelected';
import { addModelsLoadedListener } from 'app/store/middleware/listenerMiddleware/listeners/modelsLoaded';
import { addSetDefaultSettingsListener } from 'app/store/middleware/listenerMiddleware/listeners/setDefaultSettings';
import { addSocketConnectedEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketConnected';
import { deepClone } from 'common/util/deepClone';
import { merge } from 'es-toolkit';
import { omit, pick } from 'es-toolkit/compat';
import { authSliceConfig } from 'features/auth/store/authSlice';
import { changeBoardModalSliceConfig } from 'features/changeBoardModal/store/slice';
import { canvasSettingsSliceConfig } from 'features/controlLayers/store/canvasSettingsSlice';
import { canvasSliceConfig } from 'features/controlLayers/store/canvasSlice';
import { canvasSessionSliceConfig } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { canvasTextSliceConfig } from 'features/controlLayers/store/canvasTextSlice';
import { canvasWorkflowIntegrationSliceConfig } from 'features/controlLayers/store/canvasWorkflowIntegrationSlice';
import { lorasSliceConfig } from 'features/controlLayers/store/lorasSlice';
import { paramsSliceConfig } from 'features/controlLayers/store/paramsSlice';
import { refImagesSliceConfig } from 'features/controlLayers/store/refImagesSlice';
import { dynamicPromptsSliceConfig } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { gallerySliceConfig } from 'features/gallery/store/gallerySlice';
import { modelManagerSliceConfig } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { nodesSliceConfig } from 'features/nodes/store/nodesSlice';
import { workflowLibrarySliceConfig } from 'features/nodes/store/workflowLibrarySlice';
import { workflowSettingsSliceConfig } from 'features/nodes/store/workflowSettingsSlice';
import { upscaleSliceConfig } from 'features/parameters/store/upscaleSlice';
import { queueSliceConfig } from 'features/queue/store/queueSlice';
import { stylePresetSliceConfig } from 'features/stylePresets/store/stylePresetSlice';
import { hotkeysSliceConfig } from 'features/system/store/hotkeysSlice';
import { systemSliceConfig } from 'features/system/store/systemSlice';
import { uiSliceConfig } from 'features/ui/store/uiSlice';
import { diff } from 'jsondiffpatch';
import type { SerializeFunction, UnserializeFunction } from 'redux-remember';
import { REMEMBER_REHYDRATED, rememberEnhancer, rememberReducer } from 'redux-remember';
import undoable, { newHistory } from 'redux-undo';
import { serializeError } from 'serialize-error';
import { api } from 'services/api';
import type { JsonObject } from 'type-fest';
```

---

## ثالثاً: المنطق البرمجي وتدفق الكود

### 3.1 تعريف الأجزاء (Slices)

```typescript
const SLICE_CONFIGS = {
  [authSliceConfig.slice.reducerPath]: authSliceConfig,
  [canvasSessionSliceConfig.slice.reducerPath]: canvasSessionSliceConfig,
  [canvasSettingsSliceConfig.slice.reducerPath]: canvasSettingsSliceConfig,
  [canvasTextSliceConfig.slice.reducerPath]: canvasTextSliceConfig,
  [canvasSliceConfig.slice.reducerPath]: canvasSliceConfig,
  [canvasWorkflowIntegrationSliceConfig.slice.reducerPath]: canvasWorkflowIntegrationSliceConfig,
  [changeBoardModalSliceConfig.slice.reducerPath]: changeBoardModalSliceConfig,
  [dynamicPromptsSliceConfig.slice.reducerPath]: dynamicPromptsSliceConfig,
  [gallerySliceConfig.slice.reducerPath]: gallerySliceConfig,
  [hotkeysSliceConfig.slice.reducerPath]: hotkeysSliceConfig,
  [lorasSliceConfig.slice.reducerPath]: lorasSliceConfig,
  [modelManagerSliceConfig.slice.reducerPath]: modelManagerSliceConfig,
  [nodesSliceConfig.slice.reducerPath]: nodesSliceConfig,
  [paramsSliceConfig.slice.reducerPath]: paramsSliceConfig,
  [queueSliceConfig.slice.reducerPath]: queueSliceConfig,
  [refImagesSliceConfig.slice.reducerPath]: refImagesSliceConfig,
  [stylePresetSliceConfig.slice.reducerPath]: stylePresetSliceConfig,
  [systemSliceConfig.slice.reducerPath]: systemSliceConfig,
  [uiSliceConfig.slice.reducerPath]: uiSliceConfig,
  [upscaleSliceConfig.slice.reducerPath]: upscaleSliceConfig,
  [workflowLibrarySliceConfig.slice.reducerPath]: workflowLibrarySliceConfig,
  [workflowSettingsSliceConfig.slice.reducerPath]: workflowSettingsSliceConfig,
};
```

### 3.2 تعريف المُخفضات (Reducers)

```typescript
const ALL_REDUCERS = {
  [api.reducerPath]: api.reducer,
  [authSliceConfig.slice.reducerPath]: authSliceConfig.slice.reducer,
  [canvasSessionSliceConfig.slice.reducerPath]: canvasSessionSliceConfig.slice.reducer,
  [canvasSettingsSliceConfig.slice.reducerPath]: canvasSettingsSliceConfig.slice.reducer,
  [canvasTextSliceConfig.slice.reducerPath]: canvasTextSliceConfig.slice.reducer,
  [canvasSliceConfig.slice.reducerPath]: undoable(
    canvasSliceConfig.slice.reducer,
    canvasSliceConfig.undoableConfig?.reduxUndoOptions
  ),
  [canvasWorkflowIntegrationSliceConfig.slice.reducerPath]: canvasWorkflowIntegrationSliceConfig.slice.reducer,
  [changeBoardModalSliceConfig.slice.reducerPath]: changeBoardModalSliceConfig.slice.reducer,
  [dynamicPromptsSliceConfig.slice.reducerPath]: dynamicPromptsSliceConfig.slice.reducer,
  [gallerySliceConfig.slice.reducerPath]: gallerySliceConfig.slice.reducer,
  [hotkeysSliceConfig.slice.reducerPath]: hotkeysSliceConfig.slice.reducer,
  [lorasSliceConfig.slice.reducerPath]: lorasSliceConfig.slice.reducer,
  [modelManagerSliceConfig.slice.reducerPath]: modelManagerSliceConfig.slice.reducer,
  [nodesSliceConfig.slice.reducerPath]: undoable(
    nodesSliceConfig.slice.reducer,
    nodesSliceConfig.undoableConfig?.reduxUndoOptions
  ),
  [paramsSliceConfig.slice.reducerPath]: paramsSliceConfig.slice.reducer,
  [queueSliceConfig.slice.reducerPath]: queueSliceConfig.slice.reducer,
  [refImagesSliceConfig.slice.reducerPath]: refImagesSliceConfig.slice.reducer,
  [stylePresetSliceConfig.slice.reducerPath]: stylePresetSliceConfig.slice.reducer,
  [systemSliceConfig.slice.reducerPath]: systemSliceConfig.slice.reducer,
  [uiSliceConfig.slice.reducerPath]: uiSliceConfig.slice.reducer,
  [upscaleSliceConfig.slice.reducerPath]: upscaleSliceConfig.slice.reducer,
  [workflowLibrarySliceConfig.slice.reducerPath]: workflowLibrarySliceConfig.slice.reducer,
  [workflowSettingsSliceConfig.slice.reducerPath]: workflowSettingsSliceConfig.slice.reducer,
};

const rootReducer = combineReducers(ALL_REDUCERS);
const rememberedRootReducer = rememberReducer(rootReducer);
```

### 3.3 وظيفة الإلغاء

```typescript
const unserialize: UnserializeFunction = (data, key) => {
  const sliceConfig = SLICE_CONFIGS[key as keyof typeof SLICE_CONFIGS];
  if (!sliceConfig?.persistConfig) {
    throw new Error(`No persist config for slice "${key}"`);
  }
  const { getInitialState, persistConfig, undoableConfig } = sliceConfig;
  let state;
  try {
    const initialState = getInitialState();
    const parsed = JSON.parse(data);

    const nonPersistedSubsetOfState = pick(initialState, persistConfig.persistDenylist ?? []);
    const stateToMigrate = merge(deepClone(parsed), nonPersistedSubsetOfState);

    const migrated = persistConfig.migrate(stateToMigrate);

    log.debug(
      `Restored state for slice "${key}" from persisted data (version: ${migrated._version ?? 'unknown'})`
    );

    state = migrated;
  } catch (e) {
    log.error(`Failed to restore state for slice "${key}"`, e);
    state = getInitialState();
  }

  if (undoableConfig) {
    state = undoable.reducer(state, newHistory());
  }

  return state;
};
```

---

## رابعاً: معالجة الأخطاء وحالات الحافة

### 4.1 التعامل مع أخطاء التسلسل
```typescript
const errorHandler: SerializeFunction = (data, key) => {
  try {
    return JSON.stringify(data);
  } catch (e) {
    log.error(`Failed to serialize state for slice "${key}"`, e);
    return JSON.stringify({});
  }
};
```

### 4.2 التعامل مع التحديثات
```typescript
const listenerMiddleware = createListenerMiddleware();
```

---

## خامساً: تقييم الكفاءة والأداء

### نقاط القوة
1. **تخزين مؤقت**: استخدام redux-remember للحفاظ على الحالة.
2. **灵活性**: دعم التراجع (undo/redo).
3. **كفاءة الأداء**: استخدام combineReducers.

### نقاط الضعف
1. **عدد كبير من الأجزاء**: قد يؤثر على الأداء.

---

## سادساً: مخطط التفاعل

```
┌─────────────────────────────────────────────────────────────┐
│              Redux Store Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  configureStore                                             │
│       │                                                     │
│       ├── rootReducer = combineReducers(ALL_REDUCERS)        │
│       │                                                     │
│       ├── rememberReducer(rootReducer)                      │
│       │                                                     │
│       └── rememberEnhancer(...)                             │
│       │                                                     │
│       ▼                                                     │
│  ALL_REDUCERS                                               │
│       │                                                     │
│       ├── api.reducer                                       │
│       ├── authSlice                                         │
│       ├── canvasSlice (undoable)                            │
│       ├── gallerySlice                                      │
│       ├── nodesSlice (undoable)                             │
│       ├── queueSlice                                        │
│       └── ... (20+ slices)                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## سابعاً: المراجع المرجعية
- [Redux Toolkit](https://redux-toolkit.js.org/)
- [Redux Remember](https://github.com/odinr/redux-remember)
- [Redux Undo](https://github.com/omnidan/redux-undo)
