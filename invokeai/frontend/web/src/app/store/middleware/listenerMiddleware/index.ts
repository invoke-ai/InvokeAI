import {
  createListenerMiddleware,
  addListener,
  ListenerEffect,
  AnyAction,
} from '@reduxjs/toolkit';
import type { TypedStartListening, TypedAddListener } from '@reduxjs/toolkit';

import type { RootState, AppDispatch } from '../../store';
import { addInitialImageSelectedListener } from './listeners/initialImageSelected';
import { addImageResultReceivedListener } from './listeners/invocationComplete';
import { addImageUploadedListener } from './listeners/imageUploaded';
import { addRequestedImageDeletionListener } from './listeners/imageDeleted';
import {
  addUserInvokedCanvasListener,
  addUserInvokedCreateListener,
  addUserInvokedNodesListener,
} from './listeners/userInvoked';
import { addCanvasGraphBuiltListener } from './listeners/canvasGraphBuilt';

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

addImageUploadedListener();
addInitialImageSelectedListener();
addImageResultReceivedListener();
addRequestedImageDeletionListener();
addUserInvokedCanvasListener();
addUserInvokedCreateListener();
addUserInvokedNodesListener();
// addCanvasGraphBuiltListener();
