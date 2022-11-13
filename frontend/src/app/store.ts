import { combineReducers, configureStore } from '@reduxjs/toolkit';
import { useDispatch, useSelector } from 'react-redux';
import type { TypedUseSelectorHook } from 'react-redux';

import { persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage'; // defaults to localStorage for web

import { getPersistConfig } from 'redux-deep-persist';

import optionsReducer from 'features/options/optionsSlice';
import galleryReducer from 'features/gallery/gallerySlice';
import systemReducer from 'features/system/systemSlice';
import canvasReducer from 'features/canvas/canvasSlice';

import { socketioMiddleware } from './socketio/middleware';

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

const genericCanvasBlacklist = ['cursorPosition'];

const inpaintingCanvasBlacklist = genericCanvasBlacklist.map(
  (blacklistItem) => `canvas.inpainting.${blacklistItem}`
);

const outpaintingCanvasBlacklist = genericCanvasBlacklist.map(
  (blacklistItem) => `canvas.outpainting.${blacklistItem}`
);

const systemBlacklist = [
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
].map((blacklistItem) => `system.${blacklistItem}`);

const galleryBlacklist = [
  'categories',
  'currentCategory',
  'currentImage',
  'currentImageUuid',
  'shouldAutoSwitchToNewImages',
  'shouldHoldGalleryOpen',
  'intermediateImage',
].map((blacklistItem) => `gallery.${blacklistItem}`);

const rootReducer = combineReducers({
  options: optionsReducer,
  gallery: galleryReducer,
  system: systemReducer,
  canvas: canvasReducer,
});

const rootPersistConfig = getPersistConfig({
  key: 'root',
  storage,
  rootReducer,
  blacklist: [
    ...inpaintingCanvasBlacklist,
    ...outpaintingCanvasBlacklist,
    ...systemBlacklist,
    ...galleryBlacklist,
  ],
  debounce: 300,
});

// console.log(rootPersistConfig)

const persistedReducer = persistReducer(rootPersistConfig, rootReducer);

// Continue with store setup
export const store = configureStore({
  reducer: persistedReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      immutableCheck: false,
      serializableCheck: false,
    }).concat(socketioMiddleware()),
  devTools: {
    actionsDenylist: [
      // 'canvas/setCursorPosition',
      // 'canvas/setStageCoordinates',
      // 'canvas/setStageScale',
      // 'canvas/setIsDrawing',
      // 'canvas/setBoundingBoxCoordinates',
      // 'canvas/setBoundingBoxDimensions',
      // 'canvas/setIsDrawing',
      // 'canvas/addPointToCurrentLine',
    ],
  },
});

export type AppGetState = typeof store.getState;
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Use throughout your app instead of plain `useDispatch` and `useSelector`
export const useAppDispatch: () => AppDispatch = useDispatch;
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
