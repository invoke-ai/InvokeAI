import { Middleware, MiddlewareAPI } from '@reduxjs/toolkit';
import { io } from 'socket.io-client';

import {
  GeneratorProgressEvent,
  InvocationCompleteEvent,
  InvocationErrorEvent,
  InvocationStartedEvent,
} from 'services/events/types';
import {
  generatorProgress,
  invocationComplete,
  invocationError,
  invocationStarted,
  socketioConnected,
  socketioDisconnected,
  socketioSubscribed,
} from './actions';
import {
  receivedResultImagesPage,
  receivedUploadImagesPage,
} from 'services/thunks/gallery';
import { AppDispatch, RootState } from 'app/store';

const socket_url = `ws://${window.location.host}`;

const socketio = io(socket_url, {
  timeout: 60000,
  path: '/ws/socket.io',
});

export const socketioMiddleware = () => {
  let areListenersSet = false;

  const middleware: Middleware =
    (store: MiddlewareAPI<AppDispatch, RootState>) => (next) => (action) => {
      const { dispatch, getState } = store;
      const timestamp = new Date();

      if (!areListenersSet) {
        socketio.on('connect', () => {
          dispatch(socketioConnected({ timestamp }));

          if (!getState().results.ids.length) {
            dispatch(receivedResultImagesPage());
          }

          if (!getState().uploads.ids.length) {
            dispatch(receivedUploadImagesPage());
          }
        });

        socketio.on('disconnect', () => {
          dispatch(socketioDisconnected({ timestamp }));
          socketio.removeAllListeners();
        });
      }

      areListenersSet = true;

      if (invocationComplete.match(action)) {
        socketio.emit('unsubscribe', {
          session: action.payload.data.graph_execution_state_id,
        });

        socketio.removeAllListeners();
      }

      if (socketioSubscribed.match(action)) {
        socketio.emit('subscribe', { session: action.payload.sessionId });

        socketio.on('invocation_started', (data: InvocationStartedEvent) => {
          dispatch(invocationStarted({ data, timestamp }));
        });

        socketio.on('generator_progress', (data: GeneratorProgressEvent) => {
          dispatch(generatorProgress({ data, timestamp }));
        });

        socketio.on('invocation_error', (data: InvocationErrorEvent) => {
          dispatch(invocationError({ data, timestamp }));
        });

        socketio.on('invocation_complete', (data: InvocationCompleteEvent) => {
          dispatch(invocationComplete({ data, timestamp }));
        });
      }

      next(action);
    };

  return middleware;
};
