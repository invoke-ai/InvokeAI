import {
  AnyAction,
  ThunkDispatch,
  autoBatchEnhancer,
  combineReducers,
  configureStore,
} from '@reduxjs/toolkit';
import canvasReducer from 'features/canvas/store/canvasSlice';
import changeBoardModalReducer from 'features/changeBoardModal/store/slice';
import controlNetReducer from 'features/controlNet/store/controlNetSlice';
import deleteImageModalReducer from 'features/deleteImageModal/store/slice';
import dynamicPromptsReducer from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import galleryReducer from 'features/gallery/store/gallerySlice';
import loraReducer from 'features/lora/store/loraSlice';
import nodesReducer from 'features/nodes/store/nodesSlice';
import generationReducer from 'features/parameters/store/generationSlice';
import postprocessingReducer from 'features/parameters/store/postprocessingSlice';
import sdxlReducer from 'features/sdxl/store/sdxlSlice';
import configReducer from 'features/system/store/configSlice';
import systemReducer from 'features/system/store/systemSlice';
import modelmanagerReducer from 'features/ui/components/tabs/ModelManager/store/modelManagerSlice';
import hotkeysReducer from 'features/ui/store/hotkeysSlice';
import uiReducer from 'features/ui/store/uiSlice';
import dynamicMiddlewares from 'redux-dynamic-middlewares';
import { rememberEnhancer, rememberReducer } from 'redux-remember';
import { api } from 'services/api';
import { LOCALSTORAGE_PREFIX } from './constants';
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
  controlNet: controlNetReducer,
  dynamicPrompts: dynamicPromptsReducer,
  deleteImageModal: deleteImageModalReducer,
  changeBoardModal: changeBoardModalReducer,
  lora: loraReducer,
  modelmanager: modelmanagerReducer,
  sdxl: sdxlReducer,
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
  'postprocessing',
  'system',
  'ui',
  'controlNet',
  'dynamicPrompts',
  'lora',
  'modelmanager',
];

export const store = configureStore({
  reducer: rememberedRootReducer,
  enhancers: (existingEnhancers) => {
    return existingEnhancers
      .concat(
        rememberEnhancer(window.localStorage, rememberedKeys, {
          persistDebounce: 300,
          serialize,
          unserialize,
          prefix: LOCALSTORAGE_PREFIX,
        })
      )
      .concat(autoBatchEnhancer());
  },
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

export type AppGetState = typeof store.getState;
export type RootState = ReturnType<typeof store.getState>;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AppThunkDispatch = ThunkDispatch<RootState, any, AnyAction>;
export type AppDispatch = typeof store.dispatch;
export const stateSelector = (state: RootState) => state;
