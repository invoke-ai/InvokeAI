import type { ThunkDispatch, UnknownAction } from '@reduxjs/toolkit';
import {
  autoBatchEnhancer,
  combineReducers,
  configureStore,
} from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { idbKeyValDriver } from 'app/store/enhancers/reduxRemember/driver';
import { errorHandler } from 'app/store/enhancers/reduxRemember/errors';
import { canvasPersistDenylist } from 'features/canvas/store/canvasPersistDenylist';
import canvasReducer, {
  initialCanvasState,
  migrateCanvasState,
} from 'features/canvas/store/canvasSlice';
import changeBoardModalReducer from 'features/changeBoardModal/store/slice';
import { controlAdaptersPersistDenylist } from 'features/controlAdapters/store/controlAdaptersPersistDenylist';
import controlAdaptersReducer, {
  initialControlAdaptersState,
  migrateControlAdaptersState,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import deleteImageModalReducer from 'features/deleteImageModal/store/slice';
import { dynamicPromptsPersistDenylist } from 'features/dynamicPrompts/store/dynamicPromptsPersistDenylist';
import dynamicPromptsReducer, {
  initialDynamicPromptsState,
  migrateDynamicPromptsState,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { galleryPersistDenylist } from 'features/gallery/store/galleryPersistDenylist';
import galleryReducer, {
  initialGalleryState,
  migrateGalleryState,
} from 'features/gallery/store/gallerySlice';
import hrfReducer, {
  initialHRFState,
  migrateHRFState,
} from 'features/hrf/store/hrfSlice';
import loraReducer, {
  initialLoraState,
  migrateLoRAState,
} from 'features/lora/store/loraSlice';
import modelmanagerReducer, {
  initialModelManagerState,
  migrateModelManagerState,
} from 'features/modelManager/store/modelManagerSlice';
import { nodesPersistDenylist } from 'features/nodes/store/nodesPersistDenylist';
import nodesReducer, {
  initialNodesState,
  migrateNodesState,
} from 'features/nodes/store/nodesSlice';
import nodeTemplatesReducer from 'features/nodes/store/nodeTemplatesSlice';
import workflowReducer, {
  initialWorkflowState,
  migrateWorkflowState,
} from 'features/nodes/store/workflowSlice';
import { generationPersistDenylist } from 'features/parameters/store/generationPersistDenylist';
import generationReducer, {
  initialGenerationState,
  migrateGenerationState,
} from 'features/parameters/store/generationSlice';
import { postprocessingPersistDenylist } from 'features/parameters/store/postprocessingPersistDenylist';
import postprocessingReducer, {
  initialPostprocessingState,
  migratePostprocessingState,
} from 'features/parameters/store/postprocessingSlice';
import queueReducer from 'features/queue/store/queueSlice';
import sdxlReducer, {
  initialSDXLState,
  migrateSDXLState,
} from 'features/sdxl/store/sdxlSlice';
import configReducer from 'features/system/store/configSlice';
import { systemPersistDenylist } from 'features/system/store/systemPersistDenylist';
import systemReducer, {
  initialSystemState,
  migrateSystemState,
} from 'features/system/store/systemSlice';
import { uiPersistDenylist } from 'features/ui/store/uiPersistDenylist';
import uiReducer, {
  initialUIState,
  migrateUIState,
} from 'features/ui/store/uiSlice';
import { diff } from 'jsondiffpatch';
import { defaultsDeep, keys, omit, pick } from 'lodash-es';
import dynamicMiddlewares from 'redux-dynamic-middlewares';
import type { SerializeFunction, UnserializeFunction } from 'redux-remember';
import { rememberEnhancer, rememberReducer } from 'redux-remember';
import { serializeError } from 'serialize-error';
import { api } from 'services/api';
import { authToastMiddleware } from 'services/api/authToastMiddleware';
import type { JsonObject } from 'type-fest';

import { STORAGE_PREFIX } from './constants';
import { actionSanitizer } from './middleware/devtools/actionSanitizer';
import { actionsDenylist } from './middleware/devtools/actionsDenylist';
import { stateSanitizer } from './middleware/devtools/stateSanitizer';
import { listenerMiddleware } from './middleware/listenerMiddleware';
const allReducers = {
  canvas: canvasReducer,
  gallery: galleryReducer,
  generation: generationReducer,
  nodes: nodesReducer,
  nodeTemplates: nodeTemplatesReducer,
  postprocessing: postprocessingReducer,
  system: systemReducer,
  config: configReducer,
  ui: uiReducer,
  controlAdapters: controlAdaptersReducer,
  dynamicPrompts: dynamicPromptsReducer,
  deleteImageModal: deleteImageModalReducer,
  changeBoardModal: changeBoardModalReducer,
  lora: loraReducer,
  modelmanager: modelmanagerReducer,
  sdxl: sdxlReducer,
  queue: queueReducer,
  workflow: workflowReducer,
  hrf: hrfReducer,
  [api.reducerPath]: api.reducer,
};

const rootReducer = combineReducers(allReducers);

const rememberedRootReducer = rememberReducer(rootReducer);

const rememberedKeys = [
  'canvas',
  'gallery',
  'generation',
  'sdxl',
  'nodes',
  'workflow',
  'postprocessing',
  'system',
  'ui',
  'controlAdapters',
  'dynamicPrompts',
  'lora',
  'modelmanager',
  'hrf',
] satisfies (keyof typeof allReducers)[];

type SliceConfig = {
  /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
  initialState: any;
  /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
  migrate: (state: any) => any;
};

const sliceConfigs: {
  [key in (typeof rememberedKeys)[number]]: SliceConfig;
} = {
  canvas: { initialState: initialCanvasState, migrate: migrateCanvasState },
  gallery: { initialState: initialGalleryState, migrate: migrateGalleryState },
  generation: {
    initialState: initialGenerationState,
    migrate: migrateGenerationState,
  },
  nodes: { initialState: initialNodesState, migrate: migrateNodesState },
  postprocessing: {
    initialState: initialPostprocessingState,
    migrate: migratePostprocessingState,
  },
  system: { initialState: initialSystemState, migrate: migrateSystemState },
  workflow: {
    initialState: initialWorkflowState,
    migrate: migrateWorkflowState,
  },
  ui: { initialState: initialUIState, migrate: migrateUIState },
  controlAdapters: {
    initialState: initialControlAdaptersState,
    migrate: migrateControlAdaptersState,
  },
  dynamicPrompts: {
    initialState: initialDynamicPromptsState,
    migrate: migrateDynamicPromptsState,
  },
  sdxl: { initialState: initialSDXLState, migrate: migrateSDXLState },
  lora: { initialState: initialLoraState, migrate: migrateLoRAState },
  modelmanager: {
    initialState: initialModelManagerState,
    migrate: migrateModelManagerState,
  },
  hrf: { initialState: initialHRFState, migrate: migrateHRFState },
};

const unserialize: UnserializeFunction = (data, key) => {
  const log = logger('system');
  const config = sliceConfigs[key as keyof typeof sliceConfigs];
  if (!config) {
    throw new Error(`No unserialize config for slice "${key}"`);
  }
  try {
    const { initialState, migrate } = config;
    const parsed = JSON.parse(data);
    // strip out old keys
    const stripped = pick(parsed, keys(initialState));
    // run (additive) migrations
    const migrated = migrate(stripped);
    // merge in initial state as default values, covering any missing keys
    const transformed = defaultsDeep(migrated, initialState);

    log.debug(
      {
        persistedData: parsed,
        rehydratedData: transformed,
        diff: diff(parsed, transformed) as JsonObject, // this is always serializable
      },
      `Rehydrated slice "${key}"`
    );
    return transformed;
  } catch (err) {
    log.warn(
      { error: serializeError(err) },
      `Error rehydrating slice "${key}", falling back to default initial state`
    );
    return config.initialState;
  }
};

const serializationDenylist: {
  [key in (typeof rememberedKeys)[number]]?: string[];
} = {
  canvas: canvasPersistDenylist,
  gallery: galleryPersistDenylist,
  generation: generationPersistDenylist,
  nodes: nodesPersistDenylist,
  postprocessing: postprocessingPersistDenylist,
  system: systemPersistDenylist,
  ui: uiPersistDenylist,
  controlAdapters: controlAdaptersPersistDenylist,
  dynamicPrompts: dynamicPromptsPersistDenylist,
};

export const serialize: SerializeFunction = (data, key) => {
  const result = omit(
    data,
    serializationDenylist[key as keyof typeof serializationDenylist] ?? []
  );
  return JSON.stringify(result);
};

export const createStore = (uniqueStoreKey?: string, persist = true) =>
  configureStore({
    reducer: rememberedRootReducer,
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware({
        serializableCheck: false,
        immutableCheck: false,
      })
        .concat(api.middleware)
        .concat(dynamicMiddlewares)
        .concat(authToastMiddleware)
        .prepend(listenerMiddleware.middleware),
    enhancers: (getDefaultEnhancers) => {
      const _enhancers = getDefaultEnhancers().concat(autoBatchEnhancer());
      if (persist) {
        _enhancers.push(
          rememberEnhancer(idbKeyValDriver, rememberedKeys, {
            persistDebounce: 300,
            serialize,
            unserialize,
            prefix: uniqueStoreKey
              ? `${STORAGE_PREFIX}${uniqueStoreKey}-`
              : STORAGE_PREFIX,
            errorHandler,
          })
        );
      }
      return _enhancers;
    },
    devTools: {
      actionSanitizer,
      stateSanitizer,
      trace: true,
      predicate: (state, action) => {
        if (actionsDenylist.includes(action.type)) {
          return false;
        }
        return true;
      },
    },
  });

export type AppGetState = ReturnType<
  ReturnType<typeof createStore>['getState']
>;
export type RootState = ReturnType<ReturnType<typeof createStore>['getState']>;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AppThunkDispatch = ThunkDispatch<RootState, any, UnknownAction>;
export type AppDispatch = ReturnType<typeof createStore>['dispatch'];
