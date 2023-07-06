import type { TypedAddListener, TypedStartListening } from '@reduxjs/toolkit';
import {
  AnyAction,
  ListenerEffect,
  addListener,
  createListenerMiddleware,
} from '@reduxjs/toolkit';

import type { AppDispatch, RootState } from '../../store';
import { addBoardListeners } from './listeners/addBoardListeners';
import { addCommitStagingAreaImageListener } from './listeners/addCommitStagingAreaImageListener';
import { addAppStartedListener } from './listeners/appStarted';
import { addBoardIdSelectedListener } from './listeners/boardIdSelected';
import { addRequestedBoardImageDeletionListener } from './listeners/boardImagesDeleted';
import { addCanvasCopiedToClipboardListener } from './listeners/canvasCopiedToClipboard';
import { addCanvasDownloadedAsImageListener } from './listeners/canvasDownloadedAsImage';
import { addCanvasMergedListener } from './listeners/canvasMerged';
import { addCanvasSavedToGalleryListener } from './listeners/canvasSavedToGallery';
import { addControlNetAutoProcessListener } from './listeners/controlNetAutoProcess';
import { addControlNetImageProcessedListener } from './listeners/controlNetImageProcessed';
import {
  addImageDeletedFulfilledListener,
  addImageDeletedPendingListener,
  addImageDeletedRejectedListener,
  addRequestedImageDeletionListener,
} from './listeners/imageDeleted';
import { addImageDroppedListener } from './listeners/imageDropped';
import {
  addImageMetadataReceivedFulfilledListener,
  addImageMetadataReceivedRejectedListener,
} from './listeners/imageMetadataReceived';
import { addImageToDeleteSelectedListener } from './listeners/imageToDeleteSelected';
import {
  addImageUpdatedFulfilledListener,
  addImageUpdatedRejectedListener,
} from './listeners/imageUpdated';
import {
  addImageUploadedFulfilledListener,
  addImageUploadedRejectedListener,
} from './listeners/imageUploaded';
import {
  addImageUrlsReceivedFulfilledListener,
  addImageUrlsReceivedRejectedListener,
} from './listeners/imageUrlsReceived';
import { addInitialImageSelectedListener } from './listeners/initialImageSelected';
import { addModelSelectedListener } from './listeners/modelSelected';
import { addReceivedOpenAPISchemaListener } from './listeners/receivedOpenAPISchema';
import {
  addReceivedPageOfImagesFulfilledListener,
  addReceivedPageOfImagesRejectedListener,
} from './listeners/receivedPageOfImages';
import { addSelectionAddedToBatchListener } from './listeners/selectionAddedToBatch';
import {
  addSessionCanceledFulfilledListener,
  addSessionCanceledPendingListener,
  addSessionCanceledRejectedListener,
} from './listeners/sessionCanceled';
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
import { addSessionReadyToInvokeListener } from './listeners/sessionReadyToInvoke';
import { addSocketConnectedEventListener as addSocketConnectedListener } from './listeners/socketio/socketConnected';
import { addSocketDisconnectedEventListener as addSocketDisconnectedListener } from './listeners/socketio/socketDisconnected';
import { addGeneratorProgressEventListener as addGeneratorProgressListener } from './listeners/socketio/socketGeneratorProgress';
import { addGraphExecutionStateCompleteEventListener as addGraphExecutionStateCompleteListener } from './listeners/socketio/socketGraphExecutionStateComplete';
import { addInvocationCompleteEventListener as addInvocationCompleteListener } from './listeners/socketio/socketInvocationComplete';
import { addInvocationErrorEventListener as addInvocationErrorListener } from './listeners/socketio/socketInvocationError';
import { addInvocationStartedEventListener as addInvocationStartedListener } from './listeners/socketio/socketInvocationStarted';
import { addSocketSubscribedEventListener as addSocketSubscribedListener } from './listeners/socketio/socketSubscribed';
import { addSocketUnsubscribedEventListener as addSocketUnsubscribedListener } from './listeners/socketio/socketUnsubscribed';
import { addStagingAreaImageSavedListener } from './listeners/stagingAreaImageSaved';
import { addUserInvokedCanvasListener } from './listeners/userInvokedCanvas';
import { addUserInvokedImageToImageListener } from './listeners/userInvokedImageToImage';
import { addUserInvokedNodesListener } from './listeners/userInvokedNodes';
import { addUserInvokedTextToImageListener } from './listeners/userInvokedTextToImage';

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
 * The RTK listener middleware is a lightweight alternative sagas/observables.
 *
 * Most side effect logic should live in a listener.
 */

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
addRequestedBoardImageDeletionListener();
addImageToDeleteSelectedListener();

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

// ControlNet
addControlNetImageProcessedListener();
addControlNetAutoProcessListener();

// Update image URLs on connect
// addUpdateImageUrlsOnConnectListener();

// Boards
addBoardListeners();
addBoardIdSelectedListener();

// Node schemas
addReceivedOpenAPISchemaListener();

// Batches
addSelectionAddedToBatchListener();

// DND
addImageDroppedListener();

// Models
addModelSelectedListener();

// app startup
addAppStartedListener();
