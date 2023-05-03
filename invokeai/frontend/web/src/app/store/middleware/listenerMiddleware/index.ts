import {
  createListenerMiddleware,
  addListener,
  ListenerEffect,
  AnyAction,
} from '@reduxjs/toolkit';
import type { TypedStartListening, TypedAddListener } from '@reduxjs/toolkit';

import type { RootState, AppDispatch } from '../../store';
import { addInitialImageSelectedListener } from './listeners/initialImageSelected';
import { addImageResultReceivedListener } from './listeners/invocationComplete';
import { addImageUploadedListener } from './listeners/imageUploaded';
import { addRequestedImageDeletionListener } from './listeners/imageDeleted';
import {
  canvasGraphBuilt,
  sessionCreated,
  sessionInvoked,
} from 'services/thunks/session';
import { tabMap } from 'features/ui/store/tabMap';
import {
  canvasSessionIdChanged,
  stagingAreaInitialized,
} from 'features/canvas/store/canvasSlice';

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

addImageUploadedListener();
addInitialImageSelectedListener();
addImageResultReceivedListener();
addRequestedImageDeletionListener();

startAppListening({
  actionCreator: canvasGraphBuilt.fulfilled,
  effect: async (action, { dispatch, getState, condition, fork, take }) => {
    const [{ meta }] = await take(sessionInvoked.fulfilled.match);
    const { sessionId } = meta.arg;
    const state = getState();

    if (!state.canvas.layerState.stagingArea.boundingBox) {
      dispatch(
        stagingAreaInitialized({
          sessionId,
          boundingBox: {
            ...state.canvas.boundingBoxCoordinates,
            ...state.canvas.boundingBoxDimensions,
          },
        })
      );
    }

    dispatch(canvasSessionIdChanged(sessionId));
  },
});
