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
  socketReset,
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
  sessionInvoked,
  isFulfilledSessionCreatedAction,
  sessionCanceled,
} from 'services/thunks/session';
import { OpenAPI } from 'services/api';
import { receivedModels } from 'services/thunks/model';

export const socketMiddleware = () => {
  let areListenersSet = false;

  let socket_url = `ws://${window.location.host}`;

  // if building in package mode, replace socket url with open api base url minus the http protocol
  if (import.meta.env.MODE === 'package') {
    if (OpenAPI.BASE) {
      //eslint-disable-next-line
      socket_url = OpenAPI.BASE.replace(/^https?\:\/\//i, '');
    }

    if (OpenAPI.TOKEN) {
      // TODO: handle providing jwt to socket.io
    }
  }

  const socket = io(socket_url, {
    timeout: 60000,
    path: '/ws/socket.io',
    autoConnect: false, // achtung! removing this breaks the dynamic middleware
  });

  const middleware: Middleware =
    (store: MiddlewareAPI<AppDispatch, RootState>) => (next) => (action) => {
      const { dispatch, getState } = store;

      // Nothing dispatches `socketReset` actions yet, so this is a noop, but including anyways
      if (socketReset.match(action)) {
        const { sessionId } = getState().system;

        if (sessionId) {
          socket.emit('unsubscribe', { sessionId });
          dispatch(
            socketUnsubscribed({ sessionId, timestamp: getTimestamp() })
          );
        }

        if (socket.connected) {
          socket.disconnect();
          dispatch(socketDisconnected({ timestamp: getTimestamp() }));
        }

        socket.removeAllListeners();
        areListenersSet = false;
      }

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

          if (!getState().models.modelList.length) {
            dispatch(receivedModels());
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
      if (isFulfilledSessionCreatedAction(action)) {
        const oldSessionId = getState().system.sessionId;
        const subscribedNodeIds = getState().system.subscribedNodeIds;

        // temp disable this
        const shouldHandleEvent = (id: string): boolean => true;

        // const shouldHandleEvent = (id: string): boolean => {
        //   if (subscribedNodeIds.length === 1 && subscribedNodeIds[0] === '*') {
        //     return true;
        //   }

        //   return subscribedNodeIds.includes(id);
        // };

        if (oldSessionId) {
          // Unsubscribe when invocations complete
          socket.emit('unsubscribe', {
            session: oldSessionId,
          });

          dispatch(
            socketUnsubscribed({
              sessionId: oldSessionId,
              timestamp: getTimestamp(),
            })
          );

          // Remove listeners for these events; we need to set them up fresh whenever we subscribe
          [
            'invocation_started',
            'generator_progress',
            'invocation_error',
            'invocation_complete',
          ].forEach((event) => socket.removeAllListeners(event));
        }

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
          if (shouldHandleEvent(data.source_id)) {
            dispatch(invocationStarted({ data, timestamp: getTimestamp() }));
          }
        });

        socket.on('generator_progress', (data: GeneratorProgressEvent) => {
          if (shouldHandleEvent(data.source_id)) {
            dispatch(generatorProgress({ data, timestamp: getTimestamp() }));
          }
        });

        socket.on('invocation_error', (data: InvocationErrorEvent) => {
          if (shouldHandleEvent(data.source_id)) {
            dispatch(invocationError({ data, timestamp: getTimestamp() }));
          }
        });

        socket.on('invocation_complete', (data: InvocationCompleteEvent) => {
          if (shouldHandleEvent(data.source_id)) {
            const sessionId = data.graph_execution_state_id;

            const { cancelType, isCancelScheduled } = getState().system;

            // Handle scheduled cancelation
            if (cancelType === 'scheduled' && isCancelScheduled) {
              dispatch(sessionCanceled({ sessionId }));
            }

            dispatch(invocationComplete({ data, timestamp: getTimestamp() }));
          }
        });

        // Finally we actually invoke the session, starting processing
        dispatch(sessionInvoked({ sessionId }));
      }

      // Always pass the action on so other middleware and reducers can handle it
      next(action);
    };

  return middleware;
};
