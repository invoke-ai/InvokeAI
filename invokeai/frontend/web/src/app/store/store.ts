import type { ThunkDispatch, UnknownAction } from '@reduxjs/toolkit';
import {
  autoBatchEnhancer,
  combineReducers,
  configureStore,
} from '@reduxjs/toolkit';
import { idbKeyValDriver } from 'app/store/enhancers/reduxRemember/driver';
import { errorHandler } from 'app/store/enhancers/reduxRemember/errors';
import canvasReducer from 'features/canvas/store/canvasSlice';
import changeBoardModalReducer from 'features/changeBoardModal/store/slice';
import controlAdaptersReducer from 'features/controlAdapters/store/controlAdaptersSlice';
import deleteImageModalReducer from 'features/deleteImageModal/store/slice';
import dynamicPromptsReducer from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import galleryReducer from 'features/gallery/store/gallerySlice';
import hrfReducer from 'features/hrf/store/hrfSlice';
import loraReducer from 'features/lora/store/loraSlice';
import modelmanagerReducer from 'features/modelManager/store/modelManagerSlice';
import nodesReducer from 'features/nodes/store/nodesSlice';
import nodeTemplatesReducer from 'features/nodes/store/nodeTemplatesSlice';
import workflowReducer from 'features/nodes/store/workflowSlice';
import generationReducer from 'features/parameters/store/generationSlice';
import postprocessingReducer from 'features/parameters/store/postprocessingSlice';
import queueReducer from 'features/queue/store/queueSlice';
import sdxlReducer from 'features/sdxl/store/sdxlSlice';
import configReducer from 'features/system/store/configSlice';
import systemReducer from 'features/system/store/systemSlice';
import uiReducer from 'features/ui/store/uiSlice';
import dynamicMiddlewares from 'redux-dynamic-middlewares';
import { rememberEnhancer, rememberReducer } from 'redux-remember';
import { api } from 'services/api';
import { authToastMiddleware } from 'services/api/authToastMiddleware';

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
  'hrf',
];

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
