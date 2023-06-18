import {
  AnyAction,
  ThunkDispatch,
  combineReducers,
  configureStore,
} from '@reduxjs/toolkit';

import dynamicMiddlewares from 'redux-dynamic-middlewares';
import { rememberEnhancer, rememberReducer } from 'redux-remember';

import canvasReducer from 'features/canvas/store/canvasSlice';
import controlNetReducer from 'features/controlNet/store/controlNetSlice';
import galleryReducer from 'features/gallery/store/gallerySlice';
import imagesReducer from 'features/gallery/store/imagesSlice';
import lightboxReducer from 'features/lightbox/store/lightboxSlice';
import generationReducer from 'features/parameters/store/generationSlice';
import postprocessingReducer from 'features/parameters/store/postprocessingSlice';
import systemReducer from 'features/system/store/systemSlice';
// import sessionReducer from 'features/system/store/sessionSlice';
import nodesReducer from 'features/nodes/store/nodesSlice';
import configReducer from 'features/system/store/configSlice';
import hotkeysReducer from 'features/ui/store/hotkeysSlice';
import uiReducer from 'features/ui/store/uiSlice';

import { listenerMiddleware } from './middleware/listenerMiddleware';

import { actionSanitizer } from './middleware/devtools/actionSanitizer';
import { actionsDenylist } from './middleware/devtools/actionsDenylist';
import { stateSanitizer } from './middleware/devtools/stateSanitizer';

// Model Reducers
import sd1PipelineModelReducer from 'features/system/store/models/sd1PipelineModelSlice';
import sd2PipelineModelReducer from 'features/system/store/models/sd2PipelineModelSlice';

import { LOCALSTORAGE_PREFIX } from './constants';
import { serialize } from './enhancers/reduxRemember/serialize';
import { unserialize } from './enhancers/reduxRemember/unserialize';

const allReducers = {
  canvas: canvasReducer,
  gallery: galleryReducer,
  generation: generationReducer,
  lightbox: lightboxReducer,
  sd1pipelinemodels: sd1PipelineModelReducer,
  sd2pipelinemodels: sd2PipelineModelReducer,
  nodes: nodesReducer,
  postprocessing: postprocessingReducer,
  system: systemReducer,
  config: configReducer,
  ui: uiReducer,
  hotkeys: hotkeysReducer,
  images: imagesReducer,
  controlNet: controlNetReducer,
  // session: sessionReducer,
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
      .concat(dynamicMiddlewares)
      .prepend(listenerMiddleware.middleware),
  devTools: {
    actionsDenylist,
    actionSanitizer,
    stateSanitizer,
    trace: true,
  },
});

export type AppGetState = typeof store.getState;
export type RootState = ReturnType<typeof store.getState>;
export type AppThunkDispatch = ThunkDispatch<RootState, any, AnyAction>;
export type AppDispatch = typeof store.dispatch;
