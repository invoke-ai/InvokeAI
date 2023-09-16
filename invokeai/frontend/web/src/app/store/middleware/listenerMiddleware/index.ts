import type { TypedAddListener, TypedStartListening } from '@reduxjs/toolkit';
import {
  AnyAction,
  ListenerEffect,
  addListener,
  createListenerMiddleware,
} from '@reduxjs/toolkit';

import type { AppDispatch, RootState } from '../../store';
import { addCommitStagingAreaImageListener } from './listeners/addCommitStagingAreaImageListener';
import { addFirstListImagesListener } from './listeners/addFirstListImagesListener.ts';
import { addAppConfigReceivedListener } from './listeners/appConfigReceived';
import { addAppStartedListener } from './listeners/appStarted';
import { addDeleteBoardAndImagesFulfilledListener } from './listeners/boardAndImagesDeleted';
import { addBoardIdSelectedListener } from './listeners/boardIdSelected';
import { addCanvasCopiedToClipboardListener } from './listeners/canvasCopiedToClipboard';
import { addCanvasDownloadedAsImageListener } from './listeners/canvasDownloadedAsImage';
import { addCanvasImageToControlNetListener } from './listeners/canvasImageToControlNet';
import { addCanvasMaskSavedToGalleryListener } from './listeners/canvasMaskSavedToGallery';
import { addCanvasMaskToControlNetListener } from './listeners/canvasMaskToControlNet';
import { addCanvasMergedListener } from './listeners/canvasMerged';
import { addCanvasSavedToGalleryListener } from './listeners/canvasSavedToGallery';
import { addControlNetAutoProcessListener } from './listeners/controlNetAutoProcess';
import { addControlNetImageProcessedListener } from './listeners/controlNetImageProcessed';
import {
  addImageAddedToBoardFulfilledListener,
  addImageAddedToBoardRejectedListener,
} from './listeners/imageAddedToBoard';
import {
  addImageDeletedFulfilledListener,
  addImageDeletedPendingListener,
  addImageDeletedRejectedListener,
  addRequestedMultipleImageDeletionListener,
  addRequestedSingleImageDeletionListener,
} from './listeners/imageDeleted';
import { addImageDroppedListener } from './listeners/imageDropped';
import {
  addImageRemovedFromBoardFulfilledListener,
  addImageRemovedFromBoardRejectedListener,
} from './listeners/imageRemovedFromBoard';
import { addImageToDeleteSelectedListener } from './listeners/imageToDeleteSelected';
import {
  addImageUploadedFulfilledListener,
  addImageUploadedRejectedListener,
} from './listeners/imageUploaded';
import { addImagesStarredListener } from './listeners/imagesStarred';
import { addImagesUnstarredListener } from './listeners/imagesUnstarred';
import { addInitialImageSelectedListener } from './listeners/initialImageSelected';
import { addModelSelectedListener } from './listeners/modelSelected';
import { addModelsLoadedListener } from './listeners/modelsLoaded';
import { addReceivedOpenAPISchemaListener } from './listeners/receivedOpenAPISchema';
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
import { addInvocationRetrievalErrorEventListener } from './listeners/socketio/socketInvocationRetrievalError';
import { addInvocationStartedEventListener as addInvocationStartedListener } from './listeners/socketio/socketInvocationStarted';
import { addModelLoadEventListener } from './listeners/socketio/socketModelLoad';
import { addSessionRetrievalErrorEventListener } from './listeners/socketio/socketSessionRetrievalError';
import { addSocketSubscribedEventListener as addSocketSubscribedListener } from './listeners/socketio/socketSubscribed';
import { addSocketUnsubscribedEventListener as addSocketUnsubscribedListener } from './listeners/socketio/socketUnsubscribed';
import { addStagingAreaImageSavedListener } from './listeners/stagingAreaImageSaved';
import { addTabChangedListener } from './listeners/tabChanged';
import { addUpscaleRequestedListener } from './listeners/upscaleRequested';
import { addUserInvokedCanvasListener } from './listeners/userInvokedCanvas';
import { addUserInvokedImageToImageListener } from './listeners/userInvokedImageToImage';
import { addUserInvokedNodesListener } from './listeners/userInvokedNodes';
import { addUserInvokedTextToImageListener } from './listeners/userInvokedTextToImage';
import { addWorkflowLoadedListener } from './listeners/workflowLoaded';

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

// Image selected
addInitialImageSelectedListener();

// Image deleted
addRequestedSingleImageDeletionListener();
addRequestedMultipleImageDeletionListener();
addImageDeletedPendingListener();
addImageDeletedFulfilledListener();
addImageDeletedRejectedListener();
addDeleteBoardAndImagesFulfilledListener();
addImageToDeleteSelectedListener();

// Image starred
addImagesStarredListener();
addImagesUnstarredListener();

// User Invoked
addUserInvokedCanvasListener();
addUserInvokedNodesListener();
addUserInvokedTextToImageListener();
addUserInvokedImageToImageListener();
addSessionReadyToInvokeListener();

// Canvas actions
addCanvasSavedToGalleryListener();
addCanvasMaskSavedToGalleryListener();
addCanvasImageToControlNetListener();
addCanvasMaskToControlNetListener();
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
addModelLoadEventListener();
addSessionRetrievalErrorEventListener();
addInvocationRetrievalErrorEventListener();

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

// ControlNet
addControlNetImageProcessedListener();
addControlNetAutoProcessListener();

// Boards
addImageAddedToBoardFulfilledListener();
addImageAddedToBoardRejectedListener();
addImageRemovedFromBoardFulfilledListener();
addImageRemovedFromBoardRejectedListener();
addBoardIdSelectedListener();

// Node schemas
addReceivedOpenAPISchemaListener();

// Workflows
addWorkflowLoadedListener();

// DND
addImageDroppedListener();

// Models
addModelSelectedListener();

// app startup
addAppStartedListener();
addModelsLoadedListener();
addAppConfigReceivedListener();
addFirstListImagesListener();

// Ad-hoc upscale workflwo
addUpscaleRequestedListener();

// Tab Change
addTabChangedListener();
