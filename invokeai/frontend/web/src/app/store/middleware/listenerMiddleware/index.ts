import {
  createListenerMiddleware,
  addListener,
  ListenerEffect,
  AnyAction,
} from '@reduxjs/toolkit';
import type { TypedStartListening, TypedAddListener } from '@reduxjs/toolkit';

import type { RootState, AppDispatch } from '../../store';
import { addInitialImageSelectedListener } from './listeners/initialImageSelected';
import { addImageUploadedListener } from './listeners/imageUploaded';
import { addRequestedImageDeletionListener } from './listeners/imageDeleted';
import { addUserInvokedCanvasListener } from './listeners/userInvokedCanvas';
import { addUserInvokedNodesListener } from './listeners/userInvokedNodes';
import { addUserInvokedTextToImageListener } from './listeners/userInvokedTextToImage';
import { addUserInvokedImageToImageListener } from './listeners/userInvokedImageToImage';
import { addCanvasSavedToGalleryListener } from './listeners/canvasSavedToGallery';
import { addCanvasDownloadedAsImageListener } from './listeners/canvasDownloadedAsImage';
import { addCanvasCopiedToClipboardListener } from './listeners/canvasCopiedToClipboard';
import { addCanvasMergedListener } from './listeners/canvasMerged';
import { addGeneratorProgressListener } from './listeners/socketio/generatorProgress';
import { addGraphExecutionStateCompleteListener } from './listeners/socketio/graphExecutionStateComplete';
import { addInvocationCompleteListener } from './listeners/socketio/invocationComplete';
import { addInvocationErrorListener } from './listeners/socketio/invocationError';
import { addInvocationStartedListener } from './listeners/socketio/invocationStarted';
import { addSocketConnectedListener } from './listeners/socketio/socketConnected';
import { addSocketDisconnectedListener } from './listeners/socketio/socketDisconnected';
import { addSocketSubscribedListener } from './listeners/socketio/socketSubscribed';
import { addSocketUnsubscribedListener } from './listeners/socketio/socketUnsubscribed';
import { addSessionReadyToInvokeListener } from './listeners/sessionReadyToInvoke';
import { addImageMetadataReceivedListener } from './listeners/imageMetadataReceived';

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
addRequestedImageDeletionListener();
addImageMetadataReceivedListener();

addUserInvokedCanvasListener();
addUserInvokedNodesListener();
addUserInvokedTextToImageListener();
addUserInvokedImageToImageListener();
addSessionReadyToInvokeListener();

addCanvasSavedToGalleryListener();
addCanvasDownloadedAsImageListener();
addCanvasCopiedToClipboardListener();
addCanvasMergedListener();

// socketio

addGeneratorProgressListener();
addGraphExecutionStateCompleteListener();
addInvocationCompleteListener();
addInvocationErrorListener();
addInvocationStartedListener();
addSocketConnectedListener();
addSocketDisconnectedListener();
addSocketSubscribedListener();
addSocketUnsubscribedListener();
