import {
  AnyAction,
  ThunkDispatch,
  combineReducers,
  configureStore,
} from '@reduxjs/toolkit';

import { rememberReducer, rememberEnhancer } from 'redux-remember';
import dynamicMiddlewares from 'redux-dynamic-middlewares';

import canvasReducer from 'features/canvas/store/canvasSlice';
import galleryReducer from 'features/gallery/store/gallerySlice';
import imagesReducer from 'features/gallery/store/imagesSlice';
import lightboxReducer from 'features/lightbox/store/lightboxSlice';
import generationReducer from 'features/parameters/store/generationSlice';
import postprocessingReducer from 'features/parameters/store/postprocessingSlice';
import systemReducer from 'features/system/store/systemSlice';
// import sessionReducer from 'features/system/store/sessionSlice';
import configReducer from 'features/system/store/configSlice';
import uiReducer from 'features/ui/store/uiSlice';
import hotkeysReducer from 'features/ui/store/hotkeysSlice';
import modelsReducer from 'features/system/store/modelSlice';
import nodesReducer from 'features/nodes/store/nodesSlice';

import { listenerMiddleware } from './middleware/listenerMiddleware';

import { actionSanitizer } from './middleware/devtools/actionSanitizer';
import { stateSanitizer } from './middleware/devtools/stateSanitizer';
import { actionsDenylist } from './middleware/devtools/actionsDenylist';

import { serialize } from './enhancers/reduxRemember/serialize';
import { unserialize } from './enhancers/reduxRemember/unserialize';
import { LOCALSTORAGE_PREFIX } from './constants';

const allReducers = {
  canvas: canvasReducer,
  gallery: galleryReducer,
  generation: generationReducer,
  lightbox: lightboxReducer,
  models: modelsReducer,
  nodes: nodesReducer,
  postprocessing: postprocessingReducer,
  system: systemReducer,
  config: configReducer,
  ui: uiReducer,
  hotkeys: hotkeysReducer,
  images: imagesReducer,
  // session: sessionReducer,
};

const rootReducer = combineReducers(allReducers);

const rememberedRootReducer = rememberReducer(rootReducer);

const rememberedKeys: (keyof typeof allReducers)[] = [
  'canvas',
  'gallery',
  'generation',
  'lightbox',
  // 'models',
  'nodes',
  'postprocessing',
  'system',
  'ui',
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
