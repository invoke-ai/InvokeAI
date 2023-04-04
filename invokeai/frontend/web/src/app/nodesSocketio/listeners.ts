import { MiddlewareAPI } from '@reduxjs/toolkit';
import dateFormat from 'dateformat';
import i18n from 'i18n';
import { v4 as uuidv4 } from 'uuid';

import {
  addLogEntry,
  errorOccurred,
  setCurrentStatus,
  setIsCancelable,
  setIsConnected,
  setIsProcessing,
  socketioConnected,
  socketioDisconnected,
} from 'features/system/store/systemSlice';

import {
  addImage,
  clearIntermediateImage,
  setIntermediateImage,
} from 'features/gallery/store/gallerySlice';

import type { AppDispatch, RootState } from 'app/store';
import {
  GeneratorProgressEvent,
  InvocationCompleteEvent,
  InvocationErrorEvent,
  InvocationStartedEvent,
} from 'services/events/types';
import {
  setProgress,
  setProgressImage,
  setSessionId,
  setStatus,
  STATUS,
} from 'services/apiSlice';
import { emitUnsubscribe, invocationComplete } from './actions';
import { resultAdded } from 'features/gallery/store/resultsSlice';
import {
  receivedResultImagesPage,
  receivedUploadImagesPage,
} from 'services/thunks/gallery';
import { deserializeImageField } from 'services/util/deserializeImageField';

/**
 * Returns an object containing listener callbacks
 */
const makeSocketIOListeners = (
  store: MiddlewareAPI<AppDispatch, RootState>
) => {
  const { dispatch, getState } = store;

  return {
    /**
     * Callback to run when we receive a 'connect' event.
     */
    onConnect: () => {
      try {
        dispatch(socketioConnected());

        // fetch more images, but only if we don't already have images
        if (!getState().results.ids.length) {
          dispatch(receivedResultImagesPage());
        }

        if (!getState().uploads.ids.length) {
          dispatch(receivedUploadImagesPage());
        }
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'disconnect' event.
     */
    onDisconnect: () => {
      try {
        dispatch(socketioDisconnected());
        dispatch(emitUnsubscribe(getState().api.sessionId));

        dispatch(
          addLogEntry({
            timestamp: dateFormat(new Date(), 'isoDateTime'),
            message: `Disconnected from server`,
            level: 'warning',
          })
        );
      } catch (e) {
        console.error(e);
      }
    },
    onInvocationStarted: (data: InvocationStartedEvent) => {
      console.log('invocation_started', data);
      dispatch(setStatus(STATUS.busy));
    },
    /**
     * Callback to run when we receive a 'generationResult' event.
     */
    onInvocationComplete: (data: InvocationCompleteEvent) => {
      console.log('invocation_complete', data);
      try {
        dispatch(invocationComplete({ data, timestamp: new Date() }));

        const sessionId = data.graph_execution_state_id;
        if (data.result.type === 'image') {
          // const resultImage = deserializeImageField(data.result.image);

          // dispatch(resultAdded(resultImage));
          // // need to update the type for this or figure out how to get these values
          // dispatch(
          //   addImage({
          //     category: 'result',
          //     image: {
          //       uuid: uuidv4(),
          //       url: resultImage.url,
          //       thumbnail: '',
          //       width: 512,
          //       height: 512,
          //       category: 'result',
          //       name: resultImage.name,
          //       mtime: new Date().getTime(),
          //     },
          //   })
          // );
          // dispatch(setIsProcessing(false));
          // dispatch(setIsCancelable(false));

          // dispatch(
          //   addLogEntry({
          //     timestamp: dateFormat(new Date(), 'isoDateTime'),
          //     message: `Generated: ${data.result.image.image_name}`,
          //   })
          // );
          dispatch(emitUnsubscribe(sessionId));
          // dispatch(setSessionId(''));
        }
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'progressUpdate' event.
     * TODO: Add additional progress phases
     */
    onGeneratorProgress: (data: GeneratorProgressEvent) => {
      try {
        console.log('generator_progress', data);
        dispatch(setProgress(data.step / data.total_steps));
        if (data.progress_image) {
          dispatch(
            setIntermediateImage({
              // need to update the type for this or figure out how to get these values
              category: 'result',
              uuid: uuidv4(),
              mtime: new Date().getTime(),
              url: data.progress_image.dataURL,
              thumbnail: '',
              ...data.progress_image,
            })
          );
        }
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'progressUpdate' event.
     */
    onInvocationError: (data: InvocationErrorEvent) => {
      const { error } = data;

      try {
        dispatch(
          addLogEntry({
            timestamp: dateFormat(new Date(), 'isoDateTime'),
            message: `Server error: ${error}`,
            level: 'error',
          })
        );
        dispatch(errorOccurred());
        dispatch(clearIntermediateImage());
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'galleryImages' event.
     */
  };
};

export default makeSocketIOListeners;
