import type { TypedStartListening } from '@reduxjs/toolkit';
import { createListenerMiddleware } from '@reduxjs/toolkit';
import { addCommitStagingAreaImageListener } from 'app/store/middleware/listenerMiddleware/listeners/addCommitStagingAreaImageListener';
import { addFirstListImagesListener } from 'app/store/middleware/listenerMiddleware/listeners/addFirstListImagesListener.ts';
import { addAnyEnqueuedListener } from 'app/store/middleware/listenerMiddleware/listeners/anyEnqueued';
import { addAppConfigReceivedListener } from 'app/store/middleware/listenerMiddleware/listeners/appConfigReceived';
import { addAppStartedListener } from 'app/store/middleware/listenerMiddleware/listeners/appStarted';
import { addBatchEnqueuedListener } from 'app/store/middleware/listenerMiddleware/listeners/batchEnqueued';
import { addDeleteBoardAndImagesFulfilledListener } from 'app/store/middleware/listenerMiddleware/listeners/boardAndImagesDeleted';
import { addBoardIdSelectedListener } from 'app/store/middleware/listenerMiddleware/listeners/boardIdSelected';
import { addBulkDownloadListeners } from 'app/store/middleware/listenerMiddleware/listeners/bulkDownload';
import { addCanvasCopiedToClipboardListener } from 'app/store/middleware/listenerMiddleware/listeners/canvasCopiedToClipboard';
import { addCanvasDownloadedAsImageListener } from 'app/store/middleware/listenerMiddleware/listeners/canvasDownloadedAsImage';
import { addCanvasImageToControlNetListener } from 'app/store/middleware/listenerMiddleware/listeners/canvasImageToControlNet';
import { addCanvasMaskSavedToGalleryListener } from 'app/store/middleware/listenerMiddleware/listeners/canvasMaskSavedToGallery';
import { addCanvasMaskToControlNetListener } from 'app/store/middleware/listenerMiddleware/listeners/canvasMaskToControlNet';
import { addCanvasMergedListener } from 'app/store/middleware/listenerMiddleware/listeners/canvasMerged';
import { addCanvasSavedToGalleryListener } from 'app/store/middleware/listenerMiddleware/listeners/canvasSavedToGallery';
import { addControlNetAutoProcessListener } from 'app/store/middleware/listenerMiddleware/listeners/controlNetAutoProcess';
import { addControlNetImageProcessedListener } from 'app/store/middleware/listenerMiddleware/listeners/controlNetImageProcessed';
import { addEnqueueRequestedCanvasListener } from 'app/store/middleware/listenerMiddleware/listeners/enqueueRequestedCanvas';
import { addEnqueueRequestedLinear } from 'app/store/middleware/listenerMiddleware/listeners/enqueueRequestedLinear';
import { addEnqueueRequestedNodes } from 'app/store/middleware/listenerMiddleware/listeners/enqueueRequestedNodes';
import { addGalleryImageClickedListener } from 'app/store/middleware/listenerMiddleware/listeners/galleryImageClicked';
import { addGetOpenAPISchemaListener } from 'app/store/middleware/listenerMiddleware/listeners/getOpenAPISchema';
import { addImageAddedToBoardFulfilledListener } from 'app/store/middleware/listenerMiddleware/listeners/imageAddedToBoard';
import { addRequestedSingleImageDeletionListener } from 'app/store/middleware/listenerMiddleware/listeners/imageDeleted';
import { addImageDroppedListener } from 'app/store/middleware/listenerMiddleware/listeners/imageDropped';
import { addImageRemovedFromBoardFulfilledListener } from 'app/store/middleware/listenerMiddleware/listeners/imageRemovedFromBoard';
import { addImagesStarredListener } from 'app/store/middleware/listenerMiddleware/listeners/imagesStarred';
import { addImagesUnstarredListener } from 'app/store/middleware/listenerMiddleware/listeners/imagesUnstarred';
import { addImageToDeleteSelectedListener } from 'app/store/middleware/listenerMiddleware/listeners/imageToDeleteSelected';
import { addImageUploadedFulfilledListener } from 'app/store/middleware/listenerMiddleware/listeners/imageUploaded';
import { addInitialImageSelectedListener } from 'app/store/middleware/listenerMiddleware/listeners/initialImageSelected';
import { addModelSelectedListener } from 'app/store/middleware/listenerMiddleware/listeners/modelSelected';
import { addModelsLoadedListener } from 'app/store/middleware/listenerMiddleware/listeners/modelsLoaded';
import { addDynamicPromptsListener } from 'app/store/middleware/listenerMiddleware/listeners/promptChanged';
import { addSocketConnectedEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketConnected';
import { addSocketDisconnectedEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketDisconnected';
import { addGeneratorProgressEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketGeneratorProgress';
import { addGraphExecutionStateCompleteEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketGraphExecutionStateComplete';
import { addInvocationCompleteEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketInvocationComplete';
import { addInvocationErrorEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketInvocationError';
import { addInvocationRetrievalErrorEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketInvocationRetrievalError';
import { addInvocationStartedEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketInvocationStarted';
import { addModelInstallEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketModelInstall';
import { addModelLoadEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketModelLoad';
import { addSocketQueueItemStatusChangedEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketQueueItemStatusChanged';
import { addSessionRetrievalErrorEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketSessionRetrievalError';
import { addSocketSubscribedEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketSubscribed';
import { addSocketUnsubscribedEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketUnsubscribed';
import { addStagingAreaImageSavedListener } from 'app/store/middleware/listenerMiddleware/listeners/stagingAreaImageSaved';
import { addUpdateAllNodesRequestedListener } from 'app/store/middleware/listenerMiddleware/listeners/updateAllNodesRequested';
import { addUpscaleRequestedListener } from 'app/store/middleware/listenerMiddleware/listeners/upscaleRequested';
import { addWorkflowLoadRequestedListener } from 'app/store/middleware/listenerMiddleware/listeners/workflowLoadRequested';
import type { AppDispatch, RootState } from 'app/store/store';

import { addPromptTriggerListChanged } from './listeners/promptTriggerListChanged';
import { addSetDefaultSettingsListener } from './listeners/setDefaultSettings';

export const listenerMiddleware = createListenerMiddleware();

export type AppStartListening = TypedStartListening<RootState, AppDispatch>;

const startAppListening = listenerMiddleware.startListening as AppStartListening;

/**
 * The RTK listener middleware is a lightweight alternative sagas/observables.
 *
 * Most side effect logic should live in a listener.
 */

// Image uploaded
addImageUploadedFulfilledListener(startAppListening);

// Image selected
addInitialImageSelectedListener(startAppListening);

// Image deleted
addRequestedSingleImageDeletionListener(startAppListening);
addDeleteBoardAndImagesFulfilledListener(startAppListening);
addImageToDeleteSelectedListener(startAppListening);

// Image starred
addImagesStarredListener(startAppListening);
addImagesUnstarredListener(startAppListening);

// Gallery
addGalleryImageClickedListener(startAppListening);

// User Invoked
addEnqueueRequestedCanvasListener(startAppListening);
addEnqueueRequestedNodes(startAppListening);
addEnqueueRequestedLinear(startAppListening);
addAnyEnqueuedListener(startAppListening);
addBatchEnqueuedListener(startAppListening);

// Canvas actions
addCanvasSavedToGalleryListener(startAppListening);
addCanvasMaskSavedToGalleryListener(startAppListening);
addCanvasImageToControlNetListener(startAppListening);
addCanvasMaskToControlNetListener(startAppListening);
addCanvasDownloadedAsImageListener(startAppListening);
addCanvasCopiedToClipboardListener(startAppListening);
addCanvasMergedListener(startAppListening);
addStagingAreaImageSavedListener(startAppListening);
addCommitStagingAreaImageListener(startAppListening);

// Socket.IO
addGeneratorProgressEventListener(startAppListening);
addGraphExecutionStateCompleteEventListener(startAppListening);
addInvocationCompleteEventListener(startAppListening);
addInvocationErrorEventListener(startAppListening);
addInvocationStartedEventListener(startAppListening);
addSocketConnectedEventListener(startAppListening);
addSocketDisconnectedEventListener(startAppListening);
addSocketSubscribedEventListener(startAppListening);
addSocketUnsubscribedEventListener(startAppListening);
addModelLoadEventListener(startAppListening);
addModelInstallEventListener(startAppListening);
addSessionRetrievalErrorEventListener(startAppListening);
addInvocationRetrievalErrorEventListener(startAppListening);
addSocketQueueItemStatusChangedEventListener(startAppListening);
addBulkDownloadListeners(startAppListening);

// ControlNet
addControlNetImageProcessedListener(startAppListening);
addControlNetAutoProcessListener(startAppListening);

// Boards
addImageAddedToBoardFulfilledListener(startAppListening);
addImageRemovedFromBoardFulfilledListener(startAppListening);
addBoardIdSelectedListener(startAppListening);

// Node schemas
addGetOpenAPISchemaListener(startAppListening);

// Workflows
addWorkflowLoadRequestedListener(startAppListening);
addUpdateAllNodesRequestedListener(startAppListening);

// DND
addImageDroppedListener(startAppListening);

// Models
addModelSelectedListener(startAppListening);

// app startup
addAppStartedListener(startAppListening);
addModelsLoadedListener(startAppListening);
addAppConfigReceivedListener(startAppListening);
addFirstListImagesListener(startAppListening);

// Ad-hoc upscale workflwo
addUpscaleRequestedListener(startAppListening);

// Prompts
addDynamicPromptsListener(startAppListening);
addPromptTriggerListChanged(startAppListening);

addSetDefaultSettingsListener(startAppListening);
