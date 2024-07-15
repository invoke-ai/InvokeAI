import type { TypedStartListening } from '@reduxjs/toolkit';
import { createListenerMiddleware } from '@reduxjs/toolkit';
import { addAdHocPostProcessingRequestedListener } from 'app/store/middleware/listenerMiddleware/listeners/addAdHocPostProcessingRequestedListener';
import { addStagingListeners } from 'app/store/middleware/listenerMiddleware/listeners/addCommitStagingAreaImageListener';
import { addAnyEnqueuedListener } from 'app/store/middleware/listenerMiddleware/listeners/anyEnqueued';
import { addAppConfigReceivedListener } from 'app/store/middleware/listenerMiddleware/listeners/appConfigReceived';
import { addAppStartedListener } from 'app/store/middleware/listenerMiddleware/listeners/appStarted';
import { addBatchEnqueuedListener } from 'app/store/middleware/listenerMiddleware/listeners/batchEnqueued';
import { addDeleteBoardAndImagesFulfilledListener } from 'app/store/middleware/listenerMiddleware/listeners/boardAndImagesDeleted';
import { addBoardIdSelectedListener } from 'app/store/middleware/listenerMiddleware/listeners/boardIdSelected';
import { addBulkDownloadListeners } from 'app/store/middleware/listenerMiddleware/listeners/bulkDownload';
import { addCanvasSessionRequestedListener } from 'app/store/middleware/listenerMiddleware/listeners/canvasSessionRequested';
import { addControlAdapterPreprocessor } from 'app/store/middleware/listenerMiddleware/listeners/controlAdapterPreprocessor';
import { addEnqueueRequestedLinear } from 'app/store/middleware/listenerMiddleware/listeners/enqueueRequestedLinear';
import { addEnqueueRequestedNodes } from 'app/store/middleware/listenerMiddleware/listeners/enqueueRequestedNodes';
import { addGalleryImageClickedListener } from 'app/store/middleware/listenerMiddleware/listeners/galleryImageClicked';
import { addGalleryOffsetChangedListener } from 'app/store/middleware/listenerMiddleware/listeners/galleryOffsetChanged';
import { addGetOpenAPISchemaListener } from 'app/store/middleware/listenerMiddleware/listeners/getOpenAPISchema';
import { addImageAddedToBoardFulfilledListener } from 'app/store/middleware/listenerMiddleware/listeners/imageAddedToBoard';
import { addImageDeletionListeners } from 'app/store/middleware/listenerMiddleware/listeners/imageDeletionListeners';
import { addImageDroppedListener } from 'app/store/middleware/listenerMiddleware/listeners/imageDropped';
import { addImageRemovedFromBoardFulfilledListener } from 'app/store/middleware/listenerMiddleware/listeners/imageRemovedFromBoard';
import { addImagesStarredListener } from 'app/store/middleware/listenerMiddleware/listeners/imagesStarred';
import { addImagesUnstarredListener } from 'app/store/middleware/listenerMiddleware/listeners/imagesUnstarred';
import { addImageToDeleteSelectedListener } from 'app/store/middleware/listenerMiddleware/listeners/imageToDeleteSelected';
import { addImageUploadedFulfilledListener } from 'app/store/middleware/listenerMiddleware/listeners/imageUploaded';
import { addModelSelectedListener } from 'app/store/middleware/listenerMiddleware/listeners/modelSelected';
import { addModelsLoadedListener } from 'app/store/middleware/listenerMiddleware/listeners/modelsLoaded';
import { addDynamicPromptsListener } from 'app/store/middleware/listenerMiddleware/listeners/promptChanged';
import { addSetDefaultSettingsListener } from 'app/store/middleware/listenerMiddleware/listeners/setDefaultSettings';
import { addSocketConnectedEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketConnected';
import { addSocketDisconnectedEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketDisconnected';
import { addGeneratorProgressEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketGeneratorProgress';
import { addInvocationCompleteEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketInvocationComplete';
import { addInvocationErrorEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketInvocationError';
import { addInvocationStartedEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketInvocationStarted';
import { addModelInstallEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketModelInstall';
import { addModelLoadEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketModelLoad';
import { addSocketQueueEventsListeners } from 'app/store/middleware/listenerMiddleware/listeners/socketio/socketQueueEvents';
import { addUpdateAllNodesRequestedListener } from 'app/store/middleware/listenerMiddleware/listeners/updateAllNodesRequested';
import { addWorkflowLoadRequestedListener } from 'app/store/middleware/listenerMiddleware/listeners/workflowLoadRequested';
import type { AppDispatch, RootState } from 'app/store/store';

import { addArchivedOrDeletedBoardListener } from './listeners/addArchivedOrDeletedBoardListener';
import { addEnqueueRequestedUpscale } from './listeners/enqueueRequestedUpscale';

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

// Image deleted
addImageDeletionListeners(startAppListening);
addDeleteBoardAndImagesFulfilledListener(startAppListening);
addImageToDeleteSelectedListener(startAppListening);

// Image starred
addImagesStarredListener(startAppListening);
addImagesUnstarredListener(startAppListening);

// Gallery
addGalleryImageClickedListener(startAppListening);
addGalleryOffsetChangedListener(startAppListening);

// User Invoked
addEnqueueRequestedNodes(startAppListening);
addEnqueueRequestedLinear(startAppListening);
addEnqueueRequestedUpscale(startAppListening);
addAnyEnqueuedListener(startAppListening);
addBatchEnqueuedListener(startAppListening);

// Canvas actions
// addCanvasSavedToGalleryListener(startAppListening);
// addCanvasMaskSavedToGalleryListener(startAppListening);
// addCanvasImageToControlNetListener(startAppListening);
// addCanvasMaskToControlNetListener(startAppListening);
// addCanvasDownloadedAsImageListener(startAppListening);
// addCanvasCopiedToClipboardListener(startAppListening);
// addCanvasMergedListener(startAppListening);
// addStagingAreaImageSavedListener(startAppListening);
// addCommitStagingAreaImageListener(startAppListening);
addStagingListeners(startAppListening);
addCanvasSessionRequestedListener(startAppListening);

// Socket.IO
addGeneratorProgressEventListener(startAppListening);
addInvocationCompleteEventListener(startAppListening);
addInvocationErrorEventListener(startAppListening);
addInvocationStartedEventListener(startAppListening);
addSocketConnectedEventListener(startAppListening);
addSocketDisconnectedEventListener(startAppListening);
addModelLoadEventListener(startAppListening);
addModelInstallEventListener(startAppListening);
addSocketQueueEventsListeners(startAppListening);
addBulkDownloadListeners(startAppListening);

// Boards
addImageAddedToBoardFulfilledListener(startAppListening);
addImageRemovedFromBoardFulfilledListener(startAppListening);
addBoardIdSelectedListener(startAppListening);
addArchivedOrDeletedBoardListener(startAppListening);

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

// Ad-hoc upscale workflwo
addAdHocPostProcessingRequestedListener(startAppListening);

// Prompts
addDynamicPromptsListener(startAppListening);

addSetDefaultSettingsListener(startAppListening);
addControlAdapterPreprocessor(startAppListening);
