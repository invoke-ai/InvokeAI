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
import { emitUnsubscribe } from './actions';
import { getGalleryImages } from 'services/thunks/extra';

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
        dispatch(setIsConnected(true));
        dispatch(setCurrentStatus(i18n.t('common.statusConnected')));
        dispatch(getGalleryImages({ count: 20 }));
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'disconnect' event.
     */
    onDisconnect: () => {
      try {
        dispatch(setIsConnected(false));
        dispatch(setCurrentStatus(i18n.t('common.statusDisconnected')));

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
        const sessionId = data.graph_execution_state_id;
        if (data.result.type === 'image') {
          const url = `api/v1/images/${data.result.image.image_type}/${data.result.image.image_name}`;

          // need to update the type for this or figure out how to get these values
          dispatch(
            addImage({
              category: 'result',
              image: {
                uuid: uuidv4(),
                url,
                thumbnail: '',
                width: 512,
                height: 512,
                category: 'result',
                name: data.result.image.image_name,
                mtime: new Date().getTime(),
              },
            })
          );
          dispatch(
            addLogEntry({
              timestamp: dateFormat(new Date(), 'isoDateTime'),
              message: `Generated: ${data.result.image.image_name}`,
            })
          );
          dispatch(setIsProcessing(false));
          dispatch(setIsCancelable(false));
          dispatch(emitUnsubscribe(sessionId));
          dispatch(setSessionId(null));
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
