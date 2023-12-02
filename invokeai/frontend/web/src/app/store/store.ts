import {
  AnyAction,
  ThunkDispatch,
  autoBatchEnhancer,
  combineReducers,
  configureStore,
} from '@reduxjs/toolkit';
import canvasReducer from 'features/canvas/store/canvasSlice';
import changeBoardModalReducer from 'features/changeBoardModal/store/slice';
import controlAdaptersReducer from 'features/controlAdapters/store/controlAdaptersSlice';
import deleteImageModalReducer from 'features/deleteImageModal/store/slice';
import dynamicPromptsReducer from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import galleryReducer from 'features/gallery/store/gallerySlice';
import loraReducer from 'features/lora/store/loraSlice';
import modelmanagerReducer from 'features/modelManager/store/modelManagerSlice';
import nodesReducer from 'features/nodes/store/nodesSlice';
import workflowReducer from 'features/nodes/store/workflowSlice';
import generationReducer from 'features/parameters/store/generationSlice';
import postprocessingReducer from 'features/parameters/store/postprocessingSlice';
import queueReducer from 'features/queue/store/queueSlice';
import sdxlReducer from 'features/sdxl/store/sdxlSlice';
import configReducer from 'features/system/store/configSlice';
import systemReducer from 'features/system/store/systemSlice';
import hotkeysReducer from 'features/ui/store/hotkeysSlice';
import uiReducer from 'features/ui/store/uiSlice';
import { createStore as createIDBKeyValStore, get, set } from 'idb-keyval';
import dynamicMiddlewares from 'redux-dynamic-middlewares';
import { Driver, rememberEnhancer, rememberReducer } from 'redux-remember';
import { api } from 'services/api';
import { STORAGE_PREFIX } from './constants';
import { serialize } from './enhancers/reduxRemember/serialize';
import { unserialize } from './enhancers/reduxRemember/unserialize';
import { actionSanitizer } from './middleware/devtools/actionSanitizer';
import { actionsDenylist } from './middleware/devtools/actionsDenylist';
import { stateSanitizer } from './middleware/devtools/stateSanitizer';
import { listenerMiddleware } from './middleware/listenerMiddleware';

const allReducers = {
  canvas: canvasReducer,
  gallery: galleryReducer,
  generation: generationReducer,
  nodes: nodesReducer,
  postprocessing: postprocessingReducer,
  system: systemReducer,
  config: configReducer,
  ui: uiReducer,
  hotkeys: hotkeysReducer,
  controlAdapters: controlAdaptersReducer,
  dynamicPrompts: dynamicPromptsReducer,
  deleteImageModal: deleteImageModalReducer,
  changeBoardModal: changeBoardModalReducer,
  lora: loraReducer,
  modelmanager: modelmanagerReducer,
  sdxl: sdxlReducer,
  queue: queueReducer,
  workflow: workflowReducer,
  [api.reducerPath]: api.reducer,
};

const rootReducer = combineReducers(allReducers);

const rememberedRootReducer = rememberReducer(rootReducer);

const rememberedKeys: (keyof typeof allReducers)[] = [
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
];

// Create a custom idb-keyval store (just needed to customize the name)
export const idbKeyValStore = createIDBKeyValStore('invoke', 'invoke-store');

// Create redux-remember driver, wrapping idb-keyval
const idbKeyValDriver: Driver = {
  getItem: (key) => get(key, idbKeyValStore),
  setItem: (key, value) => set(key, value, idbKeyValStore),
};

export const createStore = (uniqueStoreKey?: string) =>
  configureStore({
    reducer: rememberedRootReducer,
    enhancers: (existingEnhancers) => {
      return existingEnhancers
        .concat(
          rememberEnhancer(idbKeyValDriver, rememberedKeys, {
            persistDebounce: 300,
            serialize,
            unserialize,
            prefix: uniqueStoreKey
              ? `${STORAGE_PREFIX}${uniqueStoreKey}-`
              : STORAGE_PREFIX,
          })
        )
        .concat(autoBatchEnhancer());
    },
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware({
        serializableCheck: false,
        immutableCheck: false,
      })
        .concat(api.middleware)
        .concat(dynamicMiddlewares)
        .prepend(listenerMiddleware.middleware),
    devTools: {
      actionSanitizer,
      stateSanitizer,
      trace: true,
      predicate: (state, action) => {
        // TODO: hook up to the log level param in system slice
        // manually type state, cannot type the arg
        // const typedState = state as ReturnType<typeof rootReducer>;

        // TODO: doing this breaks the rtk query devtools, commenting out for now
        // if (action.type.startsWith('api/')) {
        //   // don't log api actions, with manual cache updates they are extremely noisy
        //   return false;
        // }

        if (actionsDenylist.includes(action.type)) {
          // don't log other noisy actions
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
export type AppThunkDispatch = ThunkDispatch<RootState, any, AnyAction>;
export type AppDispatch = ReturnType<typeof createStore>['dispatch'];
export const stateSelector = (state: RootState) => state;
