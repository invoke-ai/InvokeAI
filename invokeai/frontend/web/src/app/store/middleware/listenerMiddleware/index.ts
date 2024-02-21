import type { ListenerEffect, TypedAddListener, TypedStartListening, UnknownAction } from '@reduxjs/toolkit';
import { addListener, createListenerMiddleware } from '@reduxjs/toolkit';
import { addBulkDownloadListeners } from 'app/store/middleware/listenerMiddleware/listeners/bulkDownload';
import { addGalleryImageClickedListener } from 'app/store/middleware/listenerMiddleware/listeners/galleryImageClicked';
import type { AppDispatch, RootState } from 'app/store/store';

import { addCommitStagingAreaImageListener } from './listeners/addCommitStagingAreaImageListener';
import { addFirstListImagesListener } from './listeners/addFirstListImagesListener.ts';
import { addAnyEnqueuedListener } from './listeners/anyEnqueued';
import { addAppConfigReceivedListener } from './listeners/appConfigReceived';
import { addAppStartedListener } from './listeners/appStarted';
import { addBatchEnqueuedListener } from './listeners/batchEnqueued';
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
import { addEnqueueRequestedCanvasListener } from './listeners/enqueueRequestedCanvas';
import { addEnqueueRequestedLinear } from './listeners/enqueueRequestedLinear';
import { addEnqueueRequestedNodes } from './listeners/enqueueRequestedNodes';
import { addGetOpenAPISchemaListener } from './listeners/getOpenAPISchema';
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
import { addImagesStarredListener } from './listeners/imagesStarred';
import { addImagesUnstarredListener } from './listeners/imagesUnstarred';
import { addImageToDeleteSelectedListener } from './listeners/imageToDeleteSelected';
import { addImageUploadedFulfilledListener, addImageUploadedRejectedListener } from './listeners/imageUploaded';
import { addInitialImageSelectedListener } from './listeners/initialImageSelected';
import { addModelSelectedListener } from './listeners/modelSelected';
import { addModelsLoadedListener } from './listeners/modelsLoaded';
import { addDynamicPromptsListener } from './listeners/promptChanged';
import { addSocketConnectedEventListener as addSocketConnectedListener } from './listeners/socketio/socketConnected';
import { addSocketDisconnectedEventListener as addSocketDisconnectedListener } from './listeners/socketio/socketDisconnected';
import { addGeneratorProgressEventListener as addGeneratorProgressListener } from './listeners/socketio/socketGeneratorProgress';
import { addGraphExecutionStateCompleteEventListener as addGraphExecutionStateCompleteListener } from './listeners/socketio/socketGraphExecutionStateComplete';
import { addInvocationCompleteEventListener as addInvocationCompleteListener } from './listeners/socketio/socketInvocationComplete';
import { addInvocationErrorEventListener as addInvocationErrorListener } from './listeners/socketio/socketInvocationError';
import { addInvocationRetrievalErrorEventListener } from './listeners/socketio/socketInvocationRetrievalError';
import { addInvocationStartedEventListener as addInvocationStartedListener } from './listeners/socketio/socketInvocationStarted';
import { addModelInstallEventListener } from './listeners/socketio/socketModelInstall';
import { addModelLoadEventListener } from './listeners/socketio/socketModelLoad';
import { addSocketQueueItemStatusChangedEventListener } from './listeners/socketio/socketQueueItemStatusChanged';
import { addSessionRetrievalErrorEventListener } from './listeners/socketio/socketSessionRetrievalError';
import { addSocketSubscribedEventListener as addSocketSubscribedListener } from './listeners/socketio/socketSubscribed';
import { addSocketUnsubscribedEventListener as addSocketUnsubscribedListener } from './listeners/socketio/socketUnsubscribed';
import { addStagingAreaImageSavedListener } from './listeners/stagingAreaImageSaved';
import { addUpdateAllNodesRequestedListener } from './listeners/updateAllNodesRequested';
import { addUpscaleRequestedListener } from './listeners/upscaleRequested';
import { addWorkflowLoadRequestedListener } from './listeners/workflowLoadRequested';

export const listenerMiddleware = createListenerMiddleware();

export type AppStartListening = TypedStartListening<RootState, AppDispatch>;

export const startAppListening = listenerMiddleware.startListening as AppStartListening;

export const addAppListener = addListener as TypedAddListener<RootState, AppDispatch>;

export type AppListenerEffect = ListenerEffect<UnknownAction, RootState, AppDispatch>;

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

// Gallery
addGalleryImageClickedListener();

// User Invoked
addEnqueueRequestedCanvasListener();
addEnqueueRequestedNodes();
addEnqueueRequestedLinear();
addAnyEnqueuedListener();
addBatchEnqueuedListener();

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

// Socket.IO
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
addModelInstallEventListener();
addSessionRetrievalErrorEventListener();
addInvocationRetrievalErrorEventListener();
addSocketQueueItemStatusChangedEventListener();
addBulkDownloadListeners();

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
addGetOpenAPISchemaListener();

// Workflows
addWorkflowLoadRequestedListener();
addUpdateAllNodesRequestedListener();

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

// Dynamic prompts
addDynamicPromptsListener();
