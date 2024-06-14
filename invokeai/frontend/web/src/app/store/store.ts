import type { ThunkDispatch, UnknownAction } from '@reduxjs/toolkit';
import { autoBatchEnhancer, combineReducers, configureStore } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { idbKeyValDriver } from 'app/store/enhancers/reduxRemember/driver';
import { errorHandler } from 'app/store/enhancers/reduxRemember/errors';
import type { JSONObject } from 'common/types';
import { canvasPersistConfig } from 'features/canvas/store/canvasSlice';
import { changeBoardModalSlice } from 'features/changeBoardModal/store/slice';
import {
  controlAdaptersV2PersistConfig,
  controlAdaptersV2Slice,
} from 'features/controlLayers/store/controlAdaptersSlice';
import { canvasV2PersistConfig, canvasV2Slice } from 'features/controlLayers/store/controlLayersSlice';
import { ipAdaptersPersistConfig, ipAdaptersSlice } from 'features/controlLayers/store/ipAdaptersSlice';
import { layersPersistConfig, layersSlice } from 'features/controlLayers/store/layersSlice';
import {
  regionalGuidancePersistConfig,
  regionalGuidanceSlice,
} from 'features/controlLayers/store/regionalGuidanceSlice';
import { deleteImageModalSlice } from 'features/deleteImageModal/store/slice';
import { dynamicPromptsPersistConfig, dynamicPromptsSlice } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { galleryPersistConfig, gallerySlice } from 'features/gallery/store/gallerySlice';
import { hrfPersistConfig, hrfSlice } from 'features/hrf/store/hrfSlice';
import { loraPersistConfig, loraSlice } from 'features/lora/store/loraSlice';
import { modelManagerV2PersistConfig, modelManagerV2Slice } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { nodesPersistConfig, nodesSlice, nodesUndoableConfig } from 'features/nodes/store/nodesSlice';
import { workflowSettingsPersistConfig, workflowSettingsSlice } from 'features/nodes/store/workflowSettingsSlice';
import { workflowPersistConfig, workflowSlice } from 'features/nodes/store/workflowSlice';
import { generationPersistConfig, generationSlice } from 'features/parameters/store/generationSlice';
import { upscalePersistConfig, upscaleSlice } from 'features/parameters/store/upscaleSlice';
import { queueSlice } from 'features/queue/store/queueSlice';
import { sdxlPersistConfig, sdxlSlice } from 'features/sdxl/store/sdxlSlice';
import { configSlice } from 'features/system/store/configSlice';
import { systemPersistConfig, systemSlice } from 'features/system/store/systemSlice';
import { uiPersistConfig, uiSlice } from 'features/ui/store/uiSlice';
import { diff } from 'jsondiffpatch';
import { defaultsDeep, keys, omit, pick } from 'lodash-es';
import dynamicMiddlewares from 'redux-dynamic-middlewares';
import type { SerializeFunction, UnserializeFunction } from 'redux-remember';
import { rememberEnhancer, rememberReducer } from 'redux-remember';
import undoable from 'redux-undo';
import { serializeError } from 'serialize-error';
import { api } from 'services/api';
import { authToastMiddleware } from 'services/api/authToastMiddleware';

import { STORAGE_PREFIX } from './constants';
import { actionSanitizer } from './middleware/devtools/actionSanitizer';
import { actionsDenylist } from './middleware/devtools/actionsDenylist';
import { stateSanitizer } from './middleware/devtools/stateSanitizer';
import { listenerMiddleware } from './middleware/listenerMiddleware';

const allReducers = {
  [api.reducerPath]: api.reducer,
  [gallerySlice.name]: gallerySlice.reducer,
  [generationSlice.name]: generationSlice.reducer,
  [nodesSlice.name]: undoable(nodesSlice.reducer, nodesUndoableConfig),
  [systemSlice.name]: systemSlice.reducer,
  [configSlice.name]: configSlice.reducer,
  [uiSlice.name]: uiSlice.reducer,
  [dynamicPromptsSlice.name]: dynamicPromptsSlice.reducer,
  [deleteImageModalSlice.name]: deleteImageModalSlice.reducer,
  [changeBoardModalSlice.name]: changeBoardModalSlice.reducer,
  [loraSlice.name]: loraSlice.reducer,
  [modelManagerV2Slice.name]: modelManagerV2Slice.reducer,
  [sdxlSlice.name]: sdxlSlice.reducer,
  [queueSlice.name]: queueSlice.reducer,
  [workflowSlice.name]: workflowSlice.reducer,
  [hrfSlice.name]: hrfSlice.reducer,
  [canvasV2Slice.name]: canvasV2Slice.reducer,
  [workflowSettingsSlice.name]: workflowSettingsSlice.reducer,
  [upscaleSlice.name]: upscaleSlice.reducer,
  [layersSlice.name]: layersSlice.reducer,
  [controlAdaptersV2Slice.name]: controlAdaptersV2Slice.reducer,
  [ipAdaptersSlice.name]: ipAdaptersSlice.reducer,
  [regionalGuidanceSlice.name]: regionalGuidanceSlice.reducer,
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
  [systemPersistConfig.name]: systemPersistConfig,
  [workflowPersistConfig.name]: workflowPersistConfig,
  [uiPersistConfig.name]: uiPersistConfig,
  [dynamicPromptsPersistConfig.name]: dynamicPromptsPersistConfig,
  [sdxlPersistConfig.name]: sdxlPersistConfig,
  [loraPersistConfig.name]: loraPersistConfig,
  [modelManagerV2PersistConfig.name]: modelManagerV2PersistConfig,
  [hrfPersistConfig.name]: hrfPersistConfig,
  [canvasV2PersistConfig.name]: canvasV2PersistConfig,
  [workflowSettingsPersistConfig.name]: workflowSettingsPersistConfig,
  [upscalePersistConfig.name]: upscalePersistConfig,
  [layersPersistConfig.name]: layersPersistConfig,
  [controlAdaptersV2PersistConfig.name]: controlAdaptersV2PersistConfig,
  [ipAdaptersPersistConfig.name]: ipAdaptersPersistConfig,
  [regionalGuidancePersistConfig.name]: regionalGuidancePersistConfig,
};

const unserialize: UnserializeFunction = (data, key) => {
  const log = logger('system');
  const persistConfig = persistConfigs[key as keyof typeof persistConfigs];
  if (!persistConfig) {
    throw new Error(`No persist config for slice "${key}"`);
  }
  try {
    const { initialState, migrate } = persistConfig;
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
        diff: diff(parsed, transformed) as JSONObject, // this is always serializable
      },
      `Rehydrated slice "${key}"`
    );
    return transformed;
  } catch (err) {
    log.warn({ error: serializeError(err) }, `Error rehydrating slice "${key}", falling back to default initial state`);
    return persistConfig.initialState;
  }
};

const serialize: SerializeFunction = (data, key) => {
  const persistConfig = persistConfigs[key as keyof typeof persistConfigs];
  if (!persistConfig) {
    throw new Error(`No persist config for slice "${key}"`);
  }
  // Heuristic to determine if the slice is undoable - could just hardcode it in the persistConfig
  const isUndoable = 'present' in data && 'past' in data && 'future' in data && '_latestUnfiltered' in data;
  const result = omit(isUndoable ? data.present : data, persistConfig.persistDenylist);
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

export type RootState = ReturnType<ReturnType<typeof createStore>['getState']>;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AppThunkDispatch = ThunkDispatch<RootState, any, UnknownAction>;
export type AppDispatch = ReturnType<typeof createStore>['dispatch'];
