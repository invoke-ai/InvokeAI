import type { ThunkDispatch, TypedStartListening, UnknownAction } from '@reduxjs/toolkit';
import { addListener, combineReducers, configureStore, createAction, createListenerMiddleware } from '@reduxjs/toolkit';
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
import { changeBoardModalSliceConfig } from 'features/changeBoardModal/store/slice';
import { canvasSliceConfig } from 'features/controlLayers/store/canvasSlice';
import { tabSliceConfig } from 'features/controlLayers/store/tabSlice';
import { dynamicPromptsSliceConfig } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { gallerySliceConfig } from 'features/gallery/store/gallerySlice';
import { modelManagerSliceConfig } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { nodesSliceConfig, undoableNodesSliceReducer } from 'features/nodes/store/nodesSlice';
import { workflowLibrarySliceConfig } from 'features/nodes/store/workflowLibrarySlice';
import { workflowSettingsSliceConfig } from 'features/nodes/store/workflowSettingsSlice';
import { upscaleSliceConfig } from 'features/parameters/store/upscaleSlice';
import { queueSliceConfig } from 'features/queue/store/queueSlice';
import { stylePresetSliceConfig } from 'features/stylePresets/store/stylePresetSlice';
import { systemSliceConfig } from 'features/system/store/systemSlice';
import { uiSliceConfig } from 'features/ui/store/uiSlice';
import { diff } from 'jsondiffpatch';
import type { SerializeFunction, UnserializeFunction } from 'redux-remember';
import { REMEMBER_REHYDRATED, rememberEnhancer, rememberReducer } from 'redux-remember';
import { serializeError } from 'serialize-error';
import { api } from 'services/api';
import type { JsonObject } from 'type-fest';

import { reduxRememberDriver } from './enhancers/reduxRemember/driver';
import { actionContextMiddleware } from './middleware/actionContextMiddleware';
import { actionSanitizer } from './middleware/devtools/actionSanitizer';
import { actionsDenylist } from './middleware/devtools/actionsDenylist';
import { stateSanitizer } from './middleware/devtools/stateSanitizer';
import { addArchivedOrDeletedBoardListener } from './middleware/listenerMiddleware/listeners/addArchivedOrDeletedBoardListener';
import { addImageUploadedFulfilledListener } from './middleware/listenerMiddleware/listeners/imageUploaded';

const listenerMiddleware = createListenerMiddleware();

const log = logger('system');

// When adding a slice, add the config to the SLICE_CONFIGS object below, then add the reducer to ALL_REDUCERS.
const SLICE_CONFIGS = {
  [canvasSliceConfig.slice.reducerPath]: canvasSliceConfig,
  [changeBoardModalSliceConfig.slice.reducerPath]: changeBoardModalSliceConfig,
  [dynamicPromptsSliceConfig.slice.reducerPath]: dynamicPromptsSliceConfig,
  [gallerySliceConfig.slice.reducerPath]: gallerySliceConfig,
  [tabSliceConfig.slice.reducerPath]: tabSliceConfig,
  [modelManagerSliceConfig.slice.reducerPath]: modelManagerSliceConfig,
  [nodesSliceConfig.slice.reducerPath]: nodesSliceConfig,
  [queueSliceConfig.slice.reducerPath]: queueSliceConfig,
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
  [canvasSliceConfig.slice.reducerPath]: canvasSliceConfig.slice.reducer,
  [changeBoardModalSliceConfig.slice.reducerPath]: changeBoardModalSliceConfig.slice.reducer,
  [dynamicPromptsSliceConfig.slice.reducerPath]: dynamicPromptsSliceConfig.slice.reducer,
  [gallerySliceConfig.slice.reducerPath]: gallerySliceConfig.slice.reducer,
  [tabSliceConfig.slice.reducerPath]: tabSliceConfig.slice.reducer,
  [modelManagerSliceConfig.slice.reducerPath]: modelManagerSliceConfig.slice.reducer,
  [nodesSliceConfig.slice.reducerPath]: undoableNodesSliceReducer,
  [queueSliceConfig.slice.reducerPath]: queueSliceConfig.slice.reducer,
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
  const { getInitialState, persistConfig } = sliceConfig;

  try {
    const parsedState = JSON.parse(data);
    const stateToMigrate = persistConfig.deserialize ? persistConfig.deserialize(parsedState) : parsedState;

    // Run migrations to bring old state up to date with the current version.
    const migrated = persistConfig.migrate(stateToMigrate);

    log.debug(
      {
        persistedData: parsedState as JsonObject,
        rehydratedData: migrated as JsonObject,
        diff: diff(data, migrated) as JsonObject,
      },
      `Rehydrated slice "${key}"`
    );

    return migrated;
  } catch (err) {
    log.warn(
      { error: serializeError(err as Error) },
      `Error rehydrating slice "${key}", falling back to default initial state`
    );
    const initialState = getInitialState();

    return persistConfig.deserialize ? persistConfig.deserialize(initialState) : initialState;
  }
};

const serialize: SerializeFunction = (data, key) => {
  const sliceConfig = SLICE_CONFIGS[key as keyof typeof SLICE_CONFIGS];
  if (!sliceConfig?.persistConfig) {
    throw new Error(`No persist config for slice "${key}"`);
  }

  const state = sliceConfig.persistConfig.serialize ? sliceConfig.persistConfig.serialize(data) : data;

  return JSON.stringify(state);
};

const PERSISTED_KEYS = Object.values(SLICE_CONFIGS)
  .filter((sliceConfig) => !!sliceConfig.persistConfig)
  .map((sliceConfig) => sliceConfig.slice.reducerPath);

export const createStore = (options?: { persist?: boolean; persistDebounce?: number; onRehydrated?: () => void }) => {
  const store = configureStore({
    reducer: rememberedRootReducer,
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware({
        // serializableCheck: false,
        // immutableCheck: false,
        serializableCheck: import.meta.env.MODE === 'development',
        immutableCheck: import.meta.env.MODE === 'development',
      })
        .concat(api.middleware)
        .concat(actionContextMiddleware)
        // .concat(getDebugLoggerMiddleware({ withDiff: true, withNextState: true }))
        .prepend(listenerMiddleware.middleware),
    enhancers: (getDefaultEnhancers) => {
      const enhancers = getDefaultEnhancers();
      if (options?.persist) {
        return enhancers.prepend(
          rememberEnhancer(reduxRememberDriver, PERSISTED_KEYS, {
            persistDebounce: options?.persistDebounce ?? 2000,
            serialize,
            unserialize,
            prefix: '',
            errorHandler,
          })
        );
      } else {
        return enhancers;
      }
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

  // Once-off listener to support waiting for rehydration before rendering the app
  startAppListening({
    actionCreator: createAction(REMEMBER_REHYDRATED),
    effect: (action, { unsubscribe }) => {
      unsubscribe();
      options?.onRehydrated?.();
    },
  });

  return store;
};

export type AppStore = ReturnType<typeof createStore>;
export type RootState = ReturnType<AppStore['getState']>;
/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export type AppThunkDispatch = ThunkDispatch<RootState, any, UnknownAction>;
export type AppDispatch = ReturnType<typeof createStore>['dispatch'];
export type AppGetState = ReturnType<typeof createStore>['getState'];
export type AppStartListening = TypedStartListening<RootState, AppDispatch>;

export const addAppListener = addListener.withTypes<RootState, AppDispatch>();

// To avoid circular dependencies, all listener middleware listeners are added here in the main store setup file.
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

// Ad-hoc upscale workflwo
addAdHocPostProcessingRequestedListener(startAppListening);

addSetDefaultSettingsListener(startAppListening);
