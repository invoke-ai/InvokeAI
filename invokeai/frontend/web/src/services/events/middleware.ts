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
  socketConnected,
  socketDisconnected,
  socketSubscribed,
  socketUnsubscribed,
} from './actions';
import {
  receivedResultImagesPage,
  receivedUploadImagesPage,
} from 'services/thunks/gallery';
import { AppDispatch, RootState } from 'app/store';
import { getTimestamp } from 'common/util/getTimestamp';
import {
  invokeSession,
  isFulfilledCreateSession,
} from 'services/thunks/session';
import { OpenAPI } from 'services/api';

export const socketMiddleware = () => {
  let areListenersSet = false;

  let socket_url = `ws://${window.location.host}`;

  // if building in package mode, replace socket url with open api base url minus the http protocol
  if (import.meta.env.MODE === 'package' && OpenAPI.BASE) {
    //eslint-disable-next-line
    socket_url = OpenAPI.BASE.replace(/^https?\:\/\//i, '');
  }

  const socket = io(socket_url, {
    timeout: 60000,
    path: '/ws/socket.io',
    autoConnect: false, // achtung! removing this breaks the dynamic middleware
  });

  const middleware: Middleware =
    (store: MiddlewareAPI<AppDispatch, RootState>) => (next) => (action) => {
      const { dispatch, getState } = store;
      // Set listeners for `connect` and `disconnect` events once
      // Must happen in middleware to get access to `dispatch`

      if (!areListenersSet) {
        socket.on('connect', () => {
          dispatch(socketConnected({ timestamp: getTimestamp() }));

          // These thunks need to be dispatch in middleware; cannot handle in a reducer
          if (!getState().results.ids.length) {
            dispatch(receivedResultImagesPage());
          }

          if (!getState().uploads.ids.length) {
            dispatch(receivedUploadImagesPage());
          }
        });

        socket.on('disconnect', () => {
          dispatch(socketDisconnected({ timestamp: getTimestamp() }));
        });

        areListenersSet = true;

        // must manually connect
        socket.connect();
      }

      // Everything else only happens once we have created a session
      if (isFulfilledCreateSession(action)) {
        const sessionId = action.payload.id;

        // After a session is created, we immediately subscribe to events and then invoke the session
        socket.emit('subscribe', { session: sessionId });

        // Always dispatch the event actions for other consumers who want to know when we subscribed
        dispatch(
          socketSubscribed({
            sessionId,
            timestamp: getTimestamp(),
          })
        );

        // Set up listeners for the present subscription
        socket.on('invocation_started', (data: InvocationStartedEvent) => {
          dispatch(invocationStarted({ data, timestamp: getTimestamp() }));
        });

        socket.on('generator_progress', (data: GeneratorProgressEvent) => {
          dispatch(generatorProgress({ data, timestamp: getTimestamp() }));
        });

        socket.on('invocation_error', (data: InvocationErrorEvent) => {
          dispatch(invocationError({ data, timestamp: getTimestamp() }));
        });

        socket.on('invocation_complete', (data: InvocationCompleteEvent) => {
          const sessionId = data.graph_execution_state_id;

          // Unsubscribe when invocations complete
          socket.emit('unsubscribe', {
            session: sessionId,
          });

          dispatch(
            socketUnsubscribed({ sessionId, timestamp: getTimestamp() })
          );

          // Remove listeners for these events; we need to set them up fresh whenever we subscribe
          [
            'invocation_started',
            'generator_progress',
            'invocation_error',
            'invocation_complete',
          ].forEach((event) => socket.removeAllListeners(event));

          dispatch(invocationComplete({ data, timestamp: getTimestamp() }));
        });

        // Finally we actually invoke the session, starting processing
        dispatch(invokeSession({ sessionId }));
      }

      // Always pass the action on so other middleware and reducers can handle it
      next(action);
    };

  return middleware;
};
