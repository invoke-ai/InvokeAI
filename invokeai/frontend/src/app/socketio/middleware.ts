import { Middleware } from '@reduxjs/toolkit';

import makeSocketIOEmitters from './emitters';
import makeSocketIOListeners from './listeners';

import * as InvokeAI from 'app/invokeai';
import { io } from 'socket.io-client';

/**
 * Creates a socketio middleware to handle communication with server.
 *
 * Special `socketio/actionName` actions are created in actions.ts and
 * exported for use by the application, which treats them like any old
 * action, using `dispatch` to dispatch them.
 *
 * These actions are intercepted here, where `socketio.emit()` calls are
 * made on their behalf - see `emitters.ts`. The emitter functions
 * are the outbound communication to the server.
 *
 * Listeners are also established here - see `listeners.ts`. The listener
 * functions receive communication from the server and usually dispatch
 * some new action to handle whatever data was sent from the server.
 */
export const socketioMiddleware = () => {
  const { origin } = new URL(window.location.href);

  const socketio = io(origin, {
    timeout: 60000,
    path: `${window.location.pathname}socket.io`,
  });

  // immediately disconnect while working on nodes
  socketio.disconnect();

  let areListenersSet = false;

  const middleware: Middleware = (store) => (next) => (action) => {
    const {
      onConnect,
      onDisconnect,
      onError,
      onPostprocessingResult,
      onGenerationResult,
      onIntermediateResult,
      onProgressUpdate,
      onGalleryImages,
      onProcessingCanceled,
      onImageDeleted,
      onSystemConfig,
      onModelChanged,
      onFoundModels,
      onNewModelAdded,
      onModelDeleted,
      onModelChangeFailed,
      onTempFolderEmptied,
    } = makeSocketIOListeners(store);

    const {
      emitGenerateImage,
      emitRunESRGAN,
      emitRunFacetool,
      emitDeleteImage,
      emitRequestImages,
      emitRequestNewImages,
      emitCancelProcessing,
      emitRequestSystemConfig,
      emitSearchForModels,
      emitAddNewModel,
      emitDeleteModel,
      emitRequestModelChange,
      emitSaveStagingAreaImageToGallery,
      emitRequestEmptyTempFolder,
    } = makeSocketIOEmitters(store, socketio);

    /**
     * If this is the first time the middleware has been called (e.g. during store setup),
     * initialize all our socket.io listeners.
     */
    if (!areListenersSet) {
      socketio.on('connect', () => onConnect());

      socketio.on('disconnect', () => onDisconnect());

      socketio.on('error', (data: InvokeAI.ErrorResponse) => onError(data));

      socketio.on('generationResult', (data: InvokeAI.ImageResultResponse) =>
        onGenerationResult(data)
      );

      socketio.on(
        'postprocessingResult',
        (data: InvokeAI.ImageResultResponse) => onPostprocessingResult(data)
      );

      socketio.on('intermediateResult', (data: InvokeAI.ImageResultResponse) =>
        onIntermediateResult(data)
      );

      socketio.on('progressUpdate', (data: InvokeAI.SystemStatus) =>
        onProgressUpdate(data)
      );

      socketio.on('galleryImages', (data: InvokeAI.GalleryImagesResponse) =>
        onGalleryImages(data)
      );

      socketio.on('processingCanceled', () => {
        onProcessingCanceled();
      });

      socketio.on('imageDeleted', (data: InvokeAI.ImageDeletedResponse) => {
        onImageDeleted(data);
      });

      socketio.on('systemConfig', (data: InvokeAI.SystemConfig) => {
        onSystemConfig(data);
      });

      socketio.on('foundModels', (data: InvokeAI.FoundModelResponse) => {
        onFoundModels(data);
      });

      socketio.on('newModelAdded', (data: InvokeAI.ModelAddedResponse) => {
        onNewModelAdded(data);
      });

      socketio.on('modelDeleted', (data: InvokeAI.ModelDeletedResponse) => {
        onModelDeleted(data);
      });

      socketio.on('modelChanged', (data: InvokeAI.ModelChangeResponse) => {
        onModelChanged(data);
      });

      socketio.on('modelChangeFailed', (data: InvokeAI.ModelChangeResponse) => {
        onModelChangeFailed(data);
      });

      socketio.on('tempFolderEmptied', () => {
        onTempFolderEmptied();
      });

      areListenersSet = true;
    }

    /**
     * Handle redux actions caught by middleware.
     */
    switch (action.type) {
      case 'socketio/generateImage': {
        emitGenerateImage(action.payload);
        break;
      }

      case 'socketio/runESRGAN': {
        emitRunESRGAN(action.payload);
        break;
      }

      case 'socketio/runFacetool': {
        emitRunFacetool(action.payload);
        break;
      }

      case 'socketio/deleteImage': {
        emitDeleteImage(action.payload);
        break;
      }

      case 'socketio/requestImages': {
        emitRequestImages(action.payload);
        break;
      }

      case 'socketio/requestNewImages': {
        emitRequestNewImages(action.payload);
        break;
      }

      case 'socketio/cancelProcessing': {
        emitCancelProcessing();
        break;
      }

      case 'socketio/requestSystemConfig': {
        emitRequestSystemConfig();
        break;
      }

      case 'socketio/searchForModels': {
        emitSearchForModels(action.payload);
        break;
      }

      case 'socketio/addNewModel': {
        emitAddNewModel(action.payload);
        break;
      }

      case 'socketio/deleteModel': {
        emitDeleteModel(action.payload);
        break;
      }

      case 'socketio/requestModelChange': {
        emitRequestModelChange(action.payload);
        break;
      }

      case 'socketio/saveStagingAreaImageToGallery': {
        emitSaveStagingAreaImageToGallery(action.payload);
        break;
      }

      case 'socketio/requestEmptyTempFolder': {
        emitRequestEmptyTempFolder();
        break;
      }
    }

    next(action);
  };

  return middleware;
};
