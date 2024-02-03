import type { ThunkDispatch, UnknownAction } from '@reduxjs/toolkit';
import { autoBatchEnhancer, combineReducers, configureStore } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { idbKeyValDriver } from 'app/store/enhancers/reduxRemember/driver';
import { errorHandler } from 'app/store/enhancers/reduxRemember/errors';
import canvasReducer, { canvasPersistConfig } from 'features/canvas/store/canvasSlice';
import changeBoardModalReducer from 'features/changeBoardModal/store/slice';
import controlAdaptersReducer, {
  controlAdaptersPersistConfig,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import deleteImageModalReducer from 'features/deleteImageModal/store/slice';
import dynamicPromptsReducer, { dynamicPromptsPersistConfig } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import galleryReducer, { galleryPersistConfig } from 'features/gallery/store/gallerySlice';
import hrfReducer, { hrfPersistConfig } from 'features/hrf/store/hrfSlice';
import loraReducer, { loraPersistConfig } from 'features/lora/store/loraSlice';
import modelmanagerReducer, { modelManagerPersistConfig } from 'features/modelManager/store/modelManagerSlice';
import nodesReducer, { nodesPersistConfig } from 'features/nodes/store/nodesSlice';
import nodeTemplatesReducer from 'features/nodes/store/nodeTemplatesSlice';
import workflowReducer, { workflowPersistConfig } from 'features/nodes/store/workflowSlice';
import generationReducer, { generationPersistConfig } from 'features/parameters/store/generationSlice';
import postprocessingReducer, { postprocessingPersistConfig } from 'features/parameters/store/postprocessingSlice';
import queueReducer from 'features/queue/store/queueSlice';
import sdxlReducer, { sdxlPersistConfig } from 'features/sdxl/store/sdxlSlice';
import configReducer from 'features/system/store/configSlice';
import systemReducer, { systemPersistConfig } from 'features/system/store/systemSlice';
import uiReducer, { uiPersistConfig } from 'features/ui/store/uiSlice';
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

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export type PersistConfig<T = any> = {
  /**
   * The name of the slice.
   */
  name: keyof typeof allReducers;
  /**
   * The initial state of the slice.
   */
  initialState: T;
  /**
   * Migrate the state to the current version during rehydration.
   * @param state The rehydrated state.
   * @returns A correctly-shaped state.
   */
  migrate: (state: unknown) => T;
  /**
   * Keys to omit from the persisted state.
   */
  persistDenylist: (keyof T)[];
};

const persistConfigs: { [key in keyof typeof allReducers]?: PersistConfig } = {
  [canvasPersistConfig.name]: canvasPersistConfig,
  [galleryPersistConfig.name]: galleryPersistConfig,
  [generationPersistConfig.name]: generationPersistConfig,
  [nodesPersistConfig.name]: nodesPersistConfig,
  [postprocessingPersistConfig.name]: postprocessingPersistConfig,
  [systemPersistConfig.name]: systemPersistConfig,
  [workflowPersistConfig.name]: workflowPersistConfig,
  [uiPersistConfig.name]: uiPersistConfig,
  [controlAdaptersPersistConfig.name]: controlAdaptersPersistConfig,
  [dynamicPromptsPersistConfig.name]: dynamicPromptsPersistConfig,
  [sdxlPersistConfig.name]: sdxlPersistConfig,
  [loraPersistConfig.name]: loraPersistConfig,
  [modelManagerPersistConfig.name]: modelManagerPersistConfig,
  [hrfPersistConfig.name]: hrfPersistConfig,
};

const unserialize: UnserializeFunction = (data, key) => {
  const log = logger('system');
  const config = persistConfigs[key as keyof typeof persistConfigs];
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
    log.warn({ error: serializeError(err) }, `Error rehydrating slice "${key}", falling back to default initial state`);
    return config.initialState;
  }
};

export const serialize: SerializeFunction = (data, key) => {
  const result = omit(data, persistConfigs[key as keyof typeof persistConfigs]?.persistDenylist ?? []);
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
          rememberEnhancer(idbKeyValDriver, keys(persistConfigs), {
            persistDebounce: 300,
            serialize,
            unserialize,
            prefix: uniqueStoreKey ? `${STORAGE_PREFIX}${uniqueStoreKey}-` : STORAGE_PREFIX,
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

export type AppGetState = ReturnType<ReturnType<typeof createStore>['getState']>;
export type RootState = ReturnType<ReturnType<typeof createStore>['getState']>;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AppThunkDispatch = ThunkDispatch<RootState, any, UnknownAction>;
export type AppDispatch = ReturnType<typeof createStore>['dispatch'];
