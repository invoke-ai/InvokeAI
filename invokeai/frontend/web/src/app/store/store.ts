import type { ThunkDispatch, TypedStartListening, UnknownAction } from '@reduxjs/toolkit';
import { addListener, combineReducers, configureStore, createListenerMiddleware } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { errorHandler } from 'app/store/enhancers/reduxRemember/errors';
import { addAdHocPostProcessingRequestedListener } from 'app/store/middleware/listenerMiddleware/listeners/addAdHocPostProcessingRequestedListener';
import { addAnyEnqueuedListener } from 'app/store/middleware/listenerMiddleware/listeners/anyEnqueued';
import { addAppConfigReceivedListener } from 'app/store/middleware/listenerMiddleware/listeners/appConfigReceived';
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
import { keys, mergeWith, omit, pick } from 'es-toolkit/compat';
import { changeBoardModalSliceConfig } from 'features/changeBoardModal/store/slice';
import { canvasSettingsSliceConfig } from 'features/controlLayers/store/canvasSettingsSlice';
import { canvasSliceConfig } from 'features/controlLayers/store/canvasSlice';
import { canvasSessionSliceConfig } from 'features/controlLayers/store/canvasStagingAreaSlice';
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
import { configSliceConfig } from 'features/system/store/configSlice';
import { systemSliceConfig } from 'features/system/store/systemSlice';
import { uiSliceConfig } from 'features/ui/store/uiSlice';
import { diff } from 'jsondiffpatch';
import dynamicMiddlewares from 'redux-dynamic-middlewares';
import type { Driver, SerializeFunction, UnserializeFunction } from 'redux-remember';
import { rememberEnhancer, rememberReducer } from 'redux-remember';
import undoable, { newHistory } from 'redux-undo';
import { serializeError } from 'serialize-error';
import { api } from 'services/api';
import { authToastMiddleware } from 'services/api/authToastMiddleware';
import type { JsonObject } from 'type-fest';

import { actionSanitizer } from './middleware/devtools/actionSanitizer';
import { actionsDenylist } from './middleware/devtools/actionsDenylist';
import { stateSanitizer } from './middleware/devtools/stateSanitizer';
import { addArchivedOrDeletedBoardListener } from './middleware/listenerMiddleware/listeners/addArchivedOrDeletedBoardListener';
import { addImageUploadedFulfilledListener } from './middleware/listenerMiddleware/listeners/imageUploaded';

export const listenerMiddleware = createListenerMiddleware();

const log = logger('system');

// When adding a slice, add the config to the SLICE_CONFIGS object below, then add the reducer to ALL_REDUCERS.
const SLICE_CONFIGS = {
  [canvasSessionSliceConfig.slice.reducerPath]: canvasSessionSliceConfig,
  [canvasSettingsSliceConfig.slice.reducerPath]: canvasSettingsSliceConfig,
  [canvasSliceConfig.slice.reducerPath]: canvasSliceConfig,
  [changeBoardModalSliceConfig.slice.reducerPath]: changeBoardModalSliceConfig,
  [configSliceConfig.slice.reducerPath]: configSliceConfig,
  [dynamicPromptsSliceConfig.slice.reducerPath]: dynamicPromptsSliceConfig,
  [gallerySliceConfig.slice.reducerPath]: gallerySliceConfig,
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

// TS makes it really hard to dynamically create this object :/ so it's just hardcoded here.
// Remember to wrap undoable reducers in `undoable()`!
const ALL_REDUCERS = {
  [api.reducerPath]: api.reducer,
  [canvasSessionSliceConfig.slice.reducerPath]: canvasSessionSliceConfig.slice.reducer,
  [canvasSettingsSliceConfig.slice.reducerPath]: canvasSettingsSliceConfig.slice.reducer,
  // Undoable!
  [canvasSliceConfig.slice.reducerPath]: undoable(
    canvasSliceConfig.slice.reducer,
    canvasSliceConfig.undoableConfig?.reduxUndoOptions
  ),
  [changeBoardModalSliceConfig.slice.reducerPath]: changeBoardModalSliceConfig.slice.reducer,
  [configSliceConfig.slice.reducerPath]: configSliceConfig.slice.reducer,
  [dynamicPromptsSliceConfig.slice.reducerPath]: dynamicPromptsSliceConfig.slice.reducer,
  [gallerySliceConfig.slice.reducerPath]: gallerySliceConfig.slice.reducer,
  [lorasSliceConfig.slice.reducerPath]: lorasSliceConfig.slice.reducer,
  [modelManagerSliceConfig.slice.reducerPath]: modelManagerSliceConfig.slice.reducer,
  // Undoable!
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

const unserialize: UnserializeFunction = (data, key) => {
  const sliceConfig = SLICE_CONFIGS[key as keyof typeof SLICE_CONFIGS];
  if (!sliceConfig?.persistConfig) {
    throw new Error(`No persist config for slice "${key}"`);
  }
  const { getInitialState, persistConfig, undoableConfig } = sliceConfig;
  let state;
  try {
    const initialState = getInitialState();

    // strip out old keys
    const stripped = pick(deepClone(data), keys(initialState));
    /*
     * Merge in initial state as default values, covering any missing keys. You might be tempted to use _.defaultsDeep,
     * but that merges arrays by index and partial objects by key. Using an identity function as the customizer results
     * in behaviour like defaultsDeep, but doesn't overwrite any values that are not undefined in the migrated state.
     */
    const unPersistDenylisted = mergeWith(stripped, initialState, (objVal) => objVal);
    // run (additive) migrations
    const migrated = persistConfig.migrate(unPersistDenylisted);

    log.debug(
      {
        persistedData: data as JsonObject,
        rehydratedData: migrated as JsonObject,
        diff: diff(data, migrated) as JsonObject,
      },
      `Rehydrated slice "${key}"`
    );
    state = migrated;
  } catch (err) {
    log.warn(
      { error: serializeError(err as Error) },
      `Error rehydrating slice "${key}", falling back to default initial state`
    );
    state = getInitialState();
  }

  // Undoable slices must be wrapped in a history!
  if (undoableConfig) {
    return newHistory([], state, []);
  } else {
    return state;
  }
};

const serialize: SerializeFunction = (data, key) => {
  const sliceConfig = SLICE_CONFIGS[key as keyof typeof SLICE_CONFIGS];
  if (!sliceConfig?.persistConfig) {
    throw new Error(`No persist config for slice "${key}"`);
  }

  const result = omit(
    sliceConfig.undoableConfig ? data.present : data,
    sliceConfig.persistConfig.persistDenylist ?? []
  );

  return JSON.stringify(result);
};

const PERSISTED_KEYS = Object.values(SLICE_CONFIGS)
  .filter((sliceConfig) => !!sliceConfig.persistConfig)
  .map((sliceConfig) => sliceConfig.slice.reducerPath);

export const createStore = (reduxRememberOptions: { driver: Driver; persistThrottle: number }) =>
  configureStore({
    reducer: rememberedRootReducer,
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware({
        // serializableCheck: false,
        // immutableCheck: false,
        serializableCheck: import.meta.env.MODE === 'development',
        immutableCheck: import.meta.env.MODE === 'development',
      })
        .concat(api.middleware)
        .concat(dynamicMiddlewares)
        .concat(authToastMiddleware)
        // .concat(getDebugLoggerMiddleware())
        .prepend(listenerMiddleware.middleware),
    enhancers: (getDefaultEnhancers) => {
      const enhancers = getDefaultEnhancers();
      return enhancers.prepend(
        rememberEnhancer(reduxRememberOptions.driver, PERSISTED_KEYS, {
          persistThrottle: reduxRememberOptions.persistThrottle,
          serialize,
          unserialize,
          prefix: '',
          errorHandler,
        })
      );
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

export type AppStore = ReturnType<typeof createStore>;
export type RootState = ReturnType<AppStore['getState']>;
/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export type AppThunkDispatch = ThunkDispatch<RootState, any, UnknownAction>;
export type AppDispatch = ReturnType<typeof createStore>['dispatch'];
export type AppGetState = ReturnType<typeof createStore>['getState'];
export type AppStartListening = TypedStartListening<RootState, AppDispatch>;

export const addAppListener = addListener.withTypes<RootState, AppDispatch>();

const startAppListening = listenerMiddleware.startListening as AppStartListening;
addImageUploadedFulfilledListener(startAppListening);

// Image deleted
addDeleteBoardAndImagesFulfilledListener(startAppListening);

// User Invoked
addAnyEnqueuedListener(startAppListening);
addBatchEnqueuedListener(startAppListening);

// Socket.IO
addSocketConnectedEventListener(startAppListening);

// Gallery bulk download
addBulkDownloadListeners(startAppListening);

// Boards
addImageAddedToBoardFulfilledListener(startAppListening);
addImageRemovedFromBoardFulfilledListener(startAppListening);
addBoardIdSelectedListener(startAppListening);
addArchivedOrDeletedBoardListener(startAppListening);

// Node schemas
addGetOpenAPISchemaListener(startAppListening);

// Models
addModelSelectedListener(startAppListening);

// app startup
addAppStartedListener(startAppListening);
addModelsLoadedListener(startAppListening);
addAppConfigReceivedListener(startAppListening);

// Ad-hoc upscale workflwo
addAdHocPostProcessingRequestedListener(startAppListening);

addSetDefaultSettingsListener(startAppListening);
