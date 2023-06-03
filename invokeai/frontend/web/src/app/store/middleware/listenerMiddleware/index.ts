import {
  createListenerMiddleware,
  addListener,
  ListenerEffect,
  AnyAction,
} from '@reduxjs/toolkit';
import type { TypedStartListening, TypedAddListener } from '@reduxjs/toolkit';

import type { RootState, AppDispatch } from '../../store';
import { addInitialImageSelectedListener } from './listeners/initialImageSelected';
import {
  addImageUploadedFulfilledListener,
  addImageUploadedRejectedListener,
} from './listeners/imageUploaded';
import {
  addImageDeletedFulfilledListener,
  addImageDeletedPendingListener,
  addImageDeletedRejectedListener,
  addRequestedImageDeletionListener,
} from './listeners/imageDeleted';
import { addUserInvokedCanvasListener } from './listeners/userInvokedCanvas';
import { addUserInvokedNodesListener } from './listeners/userInvokedNodes';
import { addUserInvokedTextToImageListener } from './listeners/userInvokedTextToImage';
import { addUserInvokedImageToImageListener } from './listeners/userInvokedImageToImage';
import { addCanvasSavedToGalleryListener } from './listeners/canvasSavedToGallery';
import { addCanvasDownloadedAsImageListener } from './listeners/canvasDownloadedAsImage';
import { addCanvasCopiedToClipboardListener } from './listeners/canvasCopiedToClipboard';
import { addCanvasMergedListener } from './listeners/canvasMerged';
import { addGeneratorProgressEventListener as addGeneratorProgressListener } from './listeners/socketio/socketGeneratorProgress';
import { addGraphExecutionStateCompleteEventListener as addGraphExecutionStateCompleteListener } from './listeners/socketio/socketGraphExecutionStateComplete';
import { addInvocationCompleteEventListener as addInvocationCompleteListener } from './listeners/socketio/socketInvocationComplete';
import { addInvocationErrorEventListener as addInvocationErrorListener } from './listeners/socketio/socketInvocationError';
import { addInvocationStartedEventListener as addInvocationStartedListener } from './listeners/socketio/socketInvocationStarted';
import { addSocketConnectedEventListener as addSocketConnectedListener } from './listeners/socketio/socketConnected';
import { addSocketDisconnectedEventListener as addSocketDisconnectedListener } from './listeners/socketio/socketDisconnected';
import { addSocketSubscribedEventListener as addSocketSubscribedListener } from './listeners/socketio/socketSubscribed';
import { addSocketUnsubscribedEventListener as addSocketUnsubscribedListener } from './listeners/socketio/socketUnsubscribed';
import { addSessionReadyToInvokeListener } from './listeners/sessionReadyToInvoke';
import {
  addImageMetadataReceivedFulfilledListener,
  addImageMetadataReceivedRejectedListener,
} from './listeners/imageMetadataReceived';
import {
  addImageUrlsReceivedFulfilledListener,
  addImageUrlsReceivedRejectedListener,
} from './listeners/imageUrlsReceived';
import {
  addSessionCreatedFulfilledListener,
  addSessionCreatedPendingListener,
  addSessionCreatedRejectedListener,
} from './listeners/sessionCreated';
import {
  addSessionInvokedFulfilledListener,
  addSessionInvokedPendingListener,
  addSessionInvokedRejectedListener,
} from './listeners/sessionInvoked';
import {
  addSessionCanceledFulfilledListener,
  addSessionCanceledPendingListener,
  addSessionCanceledRejectedListener,
} from './listeners/sessionCanceled';
import {
  addImageUpdatedFulfilledListener,
  addImageUpdatedRejectedListener,
} from './listeners/imageUpdated';
import {
  addReceivedPageOfImagesFulfilledListener,
  addReceivedPageOfImagesRejectedListener,
} from './listeners/receivedPageOfImages';
import { addStagingAreaImageSavedListener } from './listeners/stagingAreaImageSaved';
import { addCommitStagingAreaImageListener } from './listeners/addCommitStagingAreaImageListener';
import { addImageCategoriesChangedListener } from './listeners/imageCategoriesChanged';

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

// Image uploaded
addImageUploadedFulfilledListener();
addImageUploadedRejectedListener();

// Image updated
addImageUpdatedFulfilledListener();
addImageUpdatedRejectedListener();

// Image selected
addInitialImageSelectedListener();

// Image deleted
addRequestedImageDeletionListener();
addImageDeletedPendingListener();
addImageDeletedFulfilledListener();
addImageDeletedRejectedListener();

// Image metadata
addImageMetadataReceivedFulfilledListener();
addImageMetadataReceivedRejectedListener();

// Image URLs
addImageUrlsReceivedFulfilledListener();
addImageUrlsReceivedRejectedListener();

// User Invoked
addUserInvokedCanvasListener();
addUserInvokedNodesListener();
addUserInvokedTextToImageListener();
addUserInvokedImageToImageListener();
addSessionReadyToInvokeListener();

// Canvas actions
addCanvasSavedToGalleryListener();
addCanvasDownloadedAsImageListener();
addCanvasCopiedToClipboardListener();
addCanvasMergedListener();
addStagingAreaImageSavedListener();
addCommitStagingAreaImageListener();

/**
 * Socket.IO Events - these handle SIO events directly and pass on internal application actions.
 * We don't handle SIO events in slices via `extraReducers` because some of these events shouldn't
 * actually be handled at all.
 *
 * For example, we don't want to respond to progress events for canceled sessions. To avoid
 * duplicating the logic to determine if an event should be responded to, we handle all of that
 * "is this session canceled?" logic in these listeners.
 *
 * The `socketGeneratorProgress` listener will then only dispatch the `appSocketGeneratorProgress`
 * action if it should be handled by the rest of the application. It is this `appSocketGeneratorProgress`
 * action that is handled by reducers in slices.
 */
addGeneratorProgressListener();
addGraphExecutionStateCompleteListener();
addInvocationCompleteListener();
addInvocationErrorListener();
addInvocationStartedListener();
addSocketConnectedListener();
addSocketDisconnectedListener();
addSocketSubscribedListener();
addSocketUnsubscribedListener();

// Session Created
addSessionCreatedPendingListener();
addSessionCreatedFulfilledListener();
addSessionCreatedRejectedListener();

// Session Invoked
addSessionInvokedPendingListener();
addSessionInvokedFulfilledListener();
addSessionInvokedRejectedListener();

// Session Canceled
addSessionCanceledPendingListener();
addSessionCanceledFulfilledListener();
addSessionCanceledRejectedListener();

// Fetching images
addReceivedPageOfImagesFulfilledListener();
addReceivedPageOfImagesRejectedListener();

// Gallery
addImageCategoriesChangedListener();
