import {
  createListenerMiddleware,
  addListener,
  ListenerEffect,
  AnyAction,
} from '@reduxjs/toolkit';
import type { TypedStartListening, TypedAddListener } from '@reduxjs/toolkit';

import type { RootState, AppDispatch } from '../../store';
import { initialImageSelected } from 'features/parameters/store/actions';
import { initialImageListener } from './listeners/initialImageListener';
import {
  imageResultReceivedListener,
  imageResultReceivedPrediate,
} from './listeners/invocationCompleteListener';
import { imageUploaded } from 'services/thunks/image';
import { imageUploadedListener } from './listeners/imageUploadedListener';

export const listenerMiddleware = createListenerMiddleware();

export type AppStartListening = TypedStartListening<RootState, AppDispatch>;

export const startAppListening =
  listenerMiddleware.startListening as AppStartListening;

export const addAppListener = addListener as TypedAddListener<
  RootState,
  AppDispatch
>;

export type AppListenerEffect = ListenerEffect<
  AnyAction,
  RootState,
  AppDispatch
>;

/**
 * Initial image selected
 */
startAppListening({
  actionCreator: initialImageSelected,
  effect: initialImageListener,
});

/**
 * Image Result received
 */
startAppListening({
  predicate: imageResultReceivedPrediate,
  effect: imageResultReceivedListener,
});

/**
 * Image Uploaded
 */
startAppListening({
  actionCreator: imageUploaded.fulfilled,
  effect: imageUploadedListener,
});
