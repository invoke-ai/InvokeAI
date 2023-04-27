import { MiddlewareAPI } from '@reduxjs/toolkit';
import { AppDispatch, RootState } from 'app/store';
import { getTimestamp } from 'common/util/getTimestamp';
import { sessionCanceled } from 'services/thunks/session';
import { Socket } from 'socket.io-client';
import {
  generatorProgress,
  invocationComplete,
  invocationError,
  invocationStarted,
} from '../actions';
import { ClientToServerEvents, ServerToClientEvents } from '../types';

type SetEventListenersArg = {
  socket: Socket<ServerToClientEvents, ClientToServerEvents>;
  store: MiddlewareAPI<AppDispatch, RootState>;
};

export const setEventListeners = (arg: SetEventListenersArg) => {
  const { socket, store } = arg;
  const { dispatch, getState } = store;
  // Set up listeners for the present subscription
  socket.on('invocation_started', (data) => {
    dispatch(invocationStarted({ data, timestamp: getTimestamp() }));
  });

  socket.on('generator_progress', (data) => {
    dispatch(generatorProgress({ data, timestamp: getTimestamp() }));
  });

  socket.on('invocation_error', (data) => {
    dispatch(invocationError({ data, timestamp: getTimestamp() }));
  });

  socket.on('invocation_complete', (data) => {
    const sessionId = data.graph_execution_state_id;

    const { cancelType, isCancelScheduled } = getState().system;
    const { shouldFetchImages } = getState().config;

    // Handle scheduled cancelation
    if (cancelType === 'scheduled' && isCancelScheduled) {
      dispatch(sessionCanceled({ sessionId }));
    }

    dispatch(
      invocationComplete({
        data,
        timestamp: getTimestamp(),
        shouldFetchImages,
      })
    );
  });
};
