import {
  AnyAction,
  ThunkDispatch,
  combineReducers,
  configureStore,
} from '@reduxjs/toolkit';

import dynamicMiddlewares from 'redux-dynamic-middlewares';
import { rememberEnhancer, rememberReducer } from 'redux-remember';

import batchReducer from 'features/batch/store/batchSlice';
import canvasReducer from 'features/canvas/store/canvasSlice';
import controlNetReducer from 'features/controlNet/store/controlNetSlice';
import dynamicPromptsReducer from 'features/dynamicPrompts/store/slice';
import boardsReducer from 'features/gallery/store/boardSlice';
import galleryReducer from 'features/gallery/store/gallerySlice';
import imageDeletionReducer from 'features/imageDeletion/store/imageDeletionSlice';
import lightboxReducer from 'features/lightbox/store/lightboxSlice';
import loraReducer from 'features/lora/store/loraSlice';
import nodesReducer from 'features/nodes/store/nodesSlice';
import generationReducer from 'features/parameters/store/generationSlice';
import postprocessingReducer from 'features/parameters/store/postprocessingSlice';
import configReducer from 'features/system/store/configSlice';
import systemReducer from 'features/system/store/systemSlice';
import hotkeysReducer from 'features/ui/store/hotkeysSlice';
import uiReducer from 'features/ui/store/uiSlice';

import { listenerMiddleware } from './middleware/listenerMiddleware';

import { api } from 'services/api';
import { LOCALSTORAGE_PREFIX } from './constants';
import { serialize } from './enhancers/reduxRemember/serialize';
import { unserialize } from './enhancers/reduxRemember/unserialize';
import { actionSanitizer } from './middleware/devtools/actionSanitizer';
import { actionsDenylist } from './middleware/devtools/actionsDenylist';
import { stateSanitizer } from './middleware/devtools/stateSanitizer';

const allReducers = {
  canvas: canvasReducer,
  gallery: galleryReducer,
  generation: generationReducer,
  lightbox: lightboxReducer,
  nodes: nodesReducer,
  postprocessing: postprocessingReducer,
  system: systemReducer,
  config: configReducer,
  ui: uiReducer,
  hotkeys: hotkeysReducer,
  controlNet: controlNetReducer,
  boards: boardsReducer,
  dynamicPrompts: dynamicPromptsReducer,
  batch: batchReducer,
  imageDeletion: imageDeletionReducer,
  lora: loraReducer,
  [api.reducerPath]: api.reducer,
};

const rootReducer = combineReducers(allReducers);

const rememberedRootReducer = rememberReducer(rootReducer);

const rememberedKeys: (keyof typeof allReducers)[] = [
  'canvas',
  'gallery',
  'generation',
  'lightbox',
  'nodes',
  'postprocessing',
  'system',
  'ui',
  'controlNet',
  'dynamicPrompts',
  'batch',
  'lora',
  // 'boards',
  // 'hotkeys',
  // 'config',
];

export const store = configureStore({
  reducer: rememberedRootReducer,
  enhancers: [
    rememberEnhancer(window.localStorage, rememberedKeys, {
      persistDebounce: 300,
      serialize,
      unserialize,
      prefix: LOCALSTORAGE_PREFIX,
    }),
  ],
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      immutableCheck: false,
      serializableCheck: false,
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

      if (action.type.startsWith('api/')) {
        // don't log api actions, with manual cache updates they are extremely noisy
        return false;
      }

      if (actionsDenylist.includes(action.type)) {
        // don't log other noisy actions
        return false;
      }

      return true;
    },
  },
});

export type AppGetState = typeof store.getState;
export type RootState = ReturnType<typeof store.getState>;
export type AppThunkDispatch = ThunkDispatch<RootState, any, AnyAction>;
export type AppDispatch = typeof store.dispatch;
export const stateSelector = (state: RootState) => state;
