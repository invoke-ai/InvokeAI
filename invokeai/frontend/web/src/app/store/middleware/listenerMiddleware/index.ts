import type { TypedStartListening } from '@reduxjs/toolkit';
import { addListener, createListenerMiddleware } from '@reduxjs/toolkit';
import { addAdHocPostProcessingRequestedListener } from 'app/store/middleware/listenerMiddleware/listeners/addAdHocPostProcessingRequestedListener';
import { addStagingListeners } from 'app/store/middleware/listenerMiddleware/listeners/addCommitStagingAreaImageListener';
import { addAnyEnqueuedListener } from 'app/store/middleware/listenerMiddleware/listeners/anyEnqueued';
import { addAppConfigReceivedListener } from 'app/store/middleware/listenerMiddleware/listeners/appConfigReceived';
import { addAppStartedListener } from 'app/store/middleware/listenerMiddleware/listeners/appStarted';
import { addBatchEnqueuedListener } from 'app/store/middleware/listenerMiddleware/listeners/batchEnqueued';
import { addDeleteBoardAndImagesFulfilledListener } from 'app/store/middleware/listenerMiddleware/listeners/boardAndImagesDeleted';
import { addBoardIdSelectedListener } from 'app/store/middleware/listenerMiddleware/listeners/boardIdSelected';
import { addBulkDownloadListeners } from 'app/store/middleware/listenerMiddleware/listeners/bulkDownload';
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
import { addSocketConnectedEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketConnected';
import { addUpdateAllNodesRequestedListener } from 'app/store/middleware/listenerMiddleware/listeners/updateAllNodesRequested';
import { addWorkflowLoadRequestedListener } from 'app/store/middleware/listenerMiddleware/listeners/workflowLoadRequested';
import type { AppDispatch, RootState } from 'app/store/store';

import { addArchivedOrDeletedBoardListener } from './listeners/addArchivedOrDeletedBoardListener';
import { addEnqueueRequestedUpscale } from './listeners/enqueueRequestedUpscale';

export const listenerMiddleware = createListenerMiddleware();

export type AppStartListening = TypedStartListening<RootState, AppDispatch>;

const startAppListening = listenerMiddleware.startListening as AppStartListening;

export const addAppListener = addListener.withTypes<RootState, AppDispatch>();

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
addStagingListeners(startAppListening);

// Socket.IO
addSocketConnectedEventListener(startAppListening);

// Gallery bulk download
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
