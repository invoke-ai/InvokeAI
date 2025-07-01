import type { TypedStartListening } from '@reduxjs/toolkit';
import { addListener, createListenerMiddleware } from '@reduxjs/toolkit';
import { addAdHocPostProcessingRequestedListener } from 'app/store/middleware/listenerMiddleware/listeners/addAdHocPostProcessingRequestedListener';
import { addAnyEnqueuedListener } from 'app/store/middleware/listenerMiddleware/listeners/anyEnqueued';
import { addAppConfigReceivedListener } from 'app/store/middleware/listenerMiddleware/listeners/appConfigReceived';
import { addAppStartedListener } from 'app/store/middleware/listenerMiddleware/listeners/appStarted';
import { addBatchEnqueuedListener } from 'app/store/middleware/listenerMiddleware/listeners/batchEnqueued';
import { addDeleteBoardAndImagesFulfilledListener } from 'app/store/middleware/listenerMiddleware/listeners/boardAndImagesDeleted';
import { addBoardIdSelectedListener } from 'app/store/middleware/listenerMiddleware/listeners/boardIdSelected';
import { addBulkDownloadListeners } from 'app/store/middleware/listenerMiddleware/listeners/bulkDownload';
import { addGetOpenAPISchemaListener } from 'app/store/middleware/listenerMiddleware/listeners/getOpenAPISchema';
import { addImageAddedToBoardFulfilledListener } from 'app/store/middleware/listenerMiddleware/listeners/imageAddedToBoard';
import { addImageRemovedFromBoardFulfilledListener } from 'app/store/middleware/listenerMiddleware/listeners/imageRemovedFromBoard';
import { addImageUploadedFulfilledListener } from 'app/store/middleware/listenerMiddleware/listeners/imageUploaded';
import { addModelSelectedListener } from 'app/store/middleware/listenerMiddleware/listeners/modelSelected';
import { addModelsLoadedListener } from 'app/store/middleware/listenerMiddleware/listeners/modelsLoaded';
import { addSetDefaultSettingsListener } from 'app/store/middleware/listenerMiddleware/listeners/setDefaultSettings';
import { addSocketConnectedEventListener } from 'app/store/middleware/listenerMiddleware/listeners/socketConnected';
import type { AppDispatch, RootState } from 'app/store/store';

import { addArchivedOrDeletedBoardListener } from './listeners/addArchivedOrDeletedBoardListener';
import { addPromptExpansionRequestedListener } from './listeners/addPromptExpansionRequestedListener';

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
addDeleteBoardAndImagesFulfilledListener(startAppListening);

// User Invoked
addAnyEnqueuedListener(startAppListening);
addBatchEnqueuedListener(startAppListening);

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

// Models
addModelSelectedListener(startAppListening);

// app startup
addAppStartedListener(startAppListening);
addModelsLoadedListener(startAppListening);
addAppConfigReceivedListener(startAppListening);

// Ad-hoc upscale workflwo
addAdHocPostProcessingRequestedListener(startAppListening);

addSetDefaultSettingsListener(startAppListening);

addPromptExpansionRequestedListener(startAppListening);
