import { combineReducers, configureStore } from '@reduxjs/toolkit';
import { useDispatch, useSelector } from 'react-redux';
import type { TypedUseSelectorHook } from 'react-redux';

import { persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage'; // defaults to localStorage for web

import optionsReducer, { OptionsState } from '../features/options/optionsSlice';
import galleryReducer, { GalleryState } from '../features/gallery/gallerySlice';
import inpaintingReducer, {
  InpaintingState,
} from '../features/tabs/Inpainting/inpaintingSlice';

import systemReducer, { SystemState } from '../features/system/systemSlice';
import { socketioMiddleware } from './socketio/middleware';
import autoMergeLevel2 from 'redux-persist/es/stateReconciler/autoMergeLevel2';
import { PersistPartial } from 'redux-persist/es/persistReducer';

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
 *
 * TODO: Do we blacklist initialImagePath? If the image is deleted from disk we get an
 * ugly 404. But if we blacklist it, then this is a valuable parameter that is lost
 * on reload. Need to figure out a good way to handle this.
 */

const rootPersistConfig = {
  key: 'root',
  storage,
  stateReconciler: autoMergeLevel2,
  blacklist: ['gallery', 'system', 'inpainting'],
};

const systemPersistConfig = {
  key: 'system',
  storage,
  stateReconciler: autoMergeLevel2,
  blacklist: [
    'isCancelable',
    'isConnected',
    'isProcessing',
    'currentStep',
    'socketId',
    'isESRGANAvailable',
    'isGFPGANAvailable',
    'currentStep',
    'totalSteps',
    'currentIteration',
    'totalIterations',
    'currentStatus',
  ],
};

const galleryPersistConfig = {
  key: 'gallery',
  storage,
  stateReconciler: autoMergeLevel2,
  whitelist: [
    'galleryWidth',
    'shouldPinGallery',
    'shouldShowGallery',
    'galleryScrollPosition',
    'galleryImageMinimumWidth',
    'galleryImageObjectFit',
  ],
};

const inpaintingPersistConfig = {
  key: 'inpainting',
  storage,
  stateReconciler: autoMergeLevel2,
  blacklist: ['pastLines', 'futuresLines', 'cursorPosition'],
};

const reducers = combineReducers({
  options: optionsReducer,
  gallery: persistReducer<GalleryState>(galleryPersistConfig, galleryReducer),
  system: persistReducer<SystemState>(systemPersistConfig, systemReducer),
  inpainting: persistReducer<InpaintingState>(
    inpaintingPersistConfig,
    inpaintingReducer
  ),
});

const persistedReducer = persistReducer<{
  options: OptionsState;
  gallery: GalleryState & PersistPartial;
  system: SystemState & PersistPartial;
  inpainting: InpaintingState & PersistPartial;
}>(rootPersistConfig, reducers);

// Continue with store setup
export const store = configureStore({
  reducer: persistedReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      // redux-persist sometimes needs to temporarily put a function in redux state, need to disable this check
      serializableCheck: false,
    }).concat(socketioMiddleware()),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Use throughout your app instead of plain `useDispatch` and `useSelector`
export const useAppDispatch: () => AppDispatch = useDispatch;
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
