import { combineReducers, configureStore } from '@reduxjs/toolkit';

import { persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage'; // defaults to localStorage for web
import dynamicMiddlewares from 'redux-dynamic-middlewares';
import { getPersistConfig } from 'redux-deep-persist';

import canvasReducer from 'features/canvas/store/canvasSlice';
import galleryReducer, {
  GalleryState,
} from 'features/gallery/store/gallerySlice';
import resultsReducer, {
  ResultsState,
} from 'features/gallery/store/resultsSlice';
import uploadsReducer, {
  UploadsState,
} from 'features/gallery/store/uploadsSlice';
import lightboxReducer, {
  LightboxState,
} from 'features/lightbox/store/lightboxSlice';
import generationReducer, {
  GenerationState,
} from 'features/parameters/store/generationSlice';
import postprocessingReducer, {
  PostprocessingState,
} from 'features/parameters/store/postprocessingSlice';
import systemReducer, { SystemState } from 'features/system/store/systemSlice';
import uiReducer from 'features/ui/store/uiSlice';
import modelsReducer, { ModelsState } from 'features/system/store/modelSlice';
import nodesReducer, { NodesState } from 'features/nodes/store/nodesSlice';

import { socketioMiddleware } from './socketio/middleware';
import { socketMiddleware } from 'services/events/middleware';
import { CanvasState } from 'features/canvas/store/canvasTypes';
import { UIState } from 'features/ui/store/uiTypes';

/**
 * redux-persist provides an easy and reliable way to persist state across reloads.
 *
 * While we definitely want generation parameters to be persisted, there are a number
 * of things we do *not* want to be persisted across reloads:
 *   - Gallery/selected image (user may add/delete images from disk between page loads)
 *   - Connection/processing status
 *   - Availability of external libraries like ESRGAN/GFPGAN
 *
 * These can be blacklisted in redux-persist.
 *
 * The necesssary nested persistors with blacklists are configured below.
 */

/**
 * Canvas slice persist blacklist
 */
const canvasBlacklist: (keyof CanvasState)[] = [
  'cursorPosition',
  'isCanvasInitialized',
  'doesCanvasNeedScaling',
];

canvasBlacklist.map((blacklistItem) => `canvas.${blacklistItem}`);

/**
 * System slice persist blacklist
 */
const systemBlacklist: (keyof SystemState)[] = [
  'currentIteration',
  'currentStatus',
  'currentStep',
  'isCancelable',
  'isConnected',
  'isESRGANAvailable',
  'isGFPGANAvailable',
  'isProcessing',
  'socketId',
  'totalIterations',
  'totalSteps',
  'openModel',
  'isCancelScheduled',
  'sessionId',
  'progressImage',
];

systemBlacklist.map((blacklistItem) => `system.${blacklistItem}`);

/**
 * Gallery slice persist blacklist
 */
const galleryBlacklist: (keyof GalleryState)[] = [
  'categories',
  'currentCategory',
  'currentImage',
  'currentImageUuid',
  'shouldAutoSwitchToNewImages',
  'intermediateImage',
];

galleryBlacklist.map((blacklistItem) => `gallery.${blacklistItem}`);

/**
 * Lightbox slice persist blacklist
 */
const lightboxBlacklist: (keyof LightboxState)[] = ['isLightboxOpen'];

lightboxBlacklist.map((blacklistItem) => `lightbox.${blacklistItem}`);

/**
 * Nodes slice persist blacklist
 */
const nodesBlacklist: (keyof NodesState)[] = ['schema', 'invocations'];

nodesBlacklist.map((blacklistItem) => `nodes.${blacklistItem}`);

/**
 * Generation slice persist blacklist
 */
const generationBlacklist: (keyof GenerationState)[] = [];

generationBlacklist.map((blacklistItem) => `generation.${blacklistItem}`);

/**
 * Postprocessing slice persist blacklist
 */
const postprocessingBlacklist: (keyof PostprocessingState)[] = [];

postprocessingBlacklist.map(
  (blacklistItem) => `postprocessing.${blacklistItem}`
);

/**
 * Results slice persist blacklist
 *
 * Currently blacklisting results slice entirely, see persist config below
 */
const resultsBlacklist: (keyof ResultsState)[] = [];

resultsBlacklist.map((blacklistItem) => `results.${blacklistItem}`);

/**
 * Uploads slice persist blacklist
 *
 * Currently blacklisting uploads slice entirely, see persist config below
 */
const uploadsBlacklist: (keyof UploadsState)[] = [];

uploadsBlacklist.map((blacklistItem) => `uploads.${blacklistItem}`);

/**
 * Models slice persist blacklist
 */
const modelsBlacklist: (keyof ModelsState)[] = ['entities', 'ids'];

modelsBlacklist.map((blacklistItem) => `models.${blacklistItem}`);

/**
 * UI slice persist blacklist
 */
const uiBlacklist: (keyof UIState)[] = [];

uiBlacklist.map((blacklistItem) => `ui.${blacklistItem}`);

const rootReducer = combineReducers({
  canvas: canvasReducer,
  gallery: galleryReducer,
  generation: generationReducer,
  lightbox: lightboxReducer,
  models: modelsReducer,
  nodes: nodesReducer,
  postprocessing: postprocessingReducer,
  results: resultsReducer,
  system: systemReducer,
  ui: uiReducer,
  uploads: uploadsReducer,
});

const rootPersistConfig = getPersistConfig({
  key: 'root',
  storage,
  rootReducer,
  blacklist: [
    ...canvasBlacklist,
    ...galleryBlacklist,
    ...generationBlacklist,
    ...lightboxBlacklist,
    ...modelsBlacklist,
    ...nodesBlacklist,
    ...postprocessingBlacklist,
    // ...resultsBlacklist,
    'results',
    ...systemBlacklist,
    ...uiBlacklist,
    // ...uploadsBlacklist,
    'uploads',
  ],
  debounce: 300,
});

const persistedReducer = persistReducer(rootPersistConfig, rootReducer);

// TODO: rip the old middleware out when nodes is complete
export function buildMiddleware() {
  if (import.meta.env.MODE === 'nodes' || import.meta.env.MODE === 'package') {
    return socketMiddleware();
  } else {
    return socketioMiddleware();
  }
}

export const store = configureStore({
  reducer: persistedReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      immutableCheck: false,
      serializableCheck: false,
    }).concat(dynamicMiddlewares),
  devTools: {
    // Uncommenting these very rapidly called actions makes the redux dev tools output much more readable
    actionsDenylist: [
      'canvas/setCursorPosition',
      'canvas/setStageCoordinates',
      'canvas/setStageScale',
      'canvas/setIsDrawing',
      'canvas/setBoundingBoxCoordinates',
      'canvas/setBoundingBoxDimensions',
      'canvas/setIsDrawing',
      'canvas/addPointToCurrentLine',
    ],
  },
});

export type AppGetState = typeof store.getState;
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
