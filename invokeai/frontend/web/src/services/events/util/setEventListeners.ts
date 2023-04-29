import { MiddlewareAPI } from '@reduxjs/toolkit';
import { AppDispatch, RootState } from 'app/store/store';
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
import { Logger } from 'roarr';
import { JsonObject } from 'roarr/dist/types';

type SetEventListenersArg = {
  socket: Socket<ServerToClientEvents, ClientToServerEvents>;
  store: MiddlewareAPI<AppDispatch, RootState>;
  sessionLog: Logger<JsonObject>;
};

export const setEventListeners = (arg: SetEventListenersArg) => {
  const { socket, store, sessionLog } = arg;
  const { dispatch, getState } = store;
  // Set up listeners for the present subscription
  socket.on('invocation_started', (data) => {
    sessionLog.child({ data }).info('Invocation started');
    dispatch(invocationStarted({ data, timestamp: getTimestamp() }));
  });

  socket.on('generator_progress', (data) => {
    sessionLog.child({ data }).trace('Generator progress');
    dispatch(generatorProgress({ data, timestamp: getTimestamp() }));
  });

  socket.on('invocation_error', (data) => {
    sessionLog.child({ data }).error('Invocation error');
    dispatch(invocationError({ data, timestamp: getTimestamp() }));
  });

  socket.on('invocation_complete', (data) => {
    sessionLog.child({ data }).info('Invocation complete');
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
