import { Middleware, MiddlewareAPI } from '@reduxjs/toolkit';
import { io, Socket } from 'socket.io-client';

import {
  ClientToServerEvents,
  ServerToClientEvents,
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
import { receivedOpenAPISchema } from 'services/thunks/schema';
import { isImageOutput } from 'services/types/guards';
import { imageReceived, thumbnailReceived } from 'services/thunks/image';

export const socketMiddleware = () => {
  let areListenersSet = false;

  let socketUrl = `ws://${window.location.host}`;

  const socketOptions: Parameters<typeof io>[0] = {
    timeout: 60000,
    path: '/ws/socket.io',
    autoConnect: false, // achtung! removing this breaks the dynamic middleware
  };

  // if building in package mode, replace socket url with open api base url minus the http protocol
  if (['nodes', 'package'].includes(import.meta.env.MODE)) {
    if (OpenAPI.BASE) {
      //eslint-disable-next-line
      socketUrl = OpenAPI.BASE.replace(/^https?\:\/\//i, '');
    }

    if (OpenAPI.TOKEN) {
      // TODO: handle providing jwt to socket.io
      socketOptions.auth = { token: OpenAPI.TOKEN };
    }
  }

  const socket: Socket<ServerToClientEvents, ClientToServerEvents> = io(
    socketUrl,
    socketOptions
  );

  const middleware: Middleware =
    (store: MiddlewareAPI<AppDispatch, RootState>) => (next) => (action) => {
      const { dispatch, getState } = store;

      // Nothing dispatches `socketReset` actions yet, so this is a noop, but including anyways
      if (socketReset.match(action)) {
        const { sessionId } = getState().system;

        if (sessionId) {
          socket.emit('unsubscribe', { session: sessionId });
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

          const { results, uploads, models, nodes, config } = getState();

          const { disabledTabs } = config;

          // These thunks need to be dispatch in middleware; cannot handle in a reducer
          if (!results.ids.length) {
            dispatch(receivedResultImagesPage());
          }

          if (!uploads.ids.length) {
            dispatch(receivedUploadImagesPage());
          }

          if (!models.ids.length) {
            dispatch(receivedModels());
          }

          if (!nodes.schema && !disabledTabs.includes('nodes')) {
            dispatch(receivedOpenAPISchema());
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

        // temp disable event subscription
        const shouldHandleEvent = (id: string): boolean => true;

        // const subscribedNodeIds = getState().system.subscribedNodeIds;
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

          const listenersToRemove: (keyof ServerToClientEvents)[] = [
            'invocation_started',
            'generator_progress',
            'invocation_error',
            'invocation_complete',
          ];

          // Remove listeners for these events; we need to set them up fresh whenever we subscribe
          listenersToRemove.forEach((event: keyof ServerToClientEvents) => {
            socket.removeAllListeners(event);
          });
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
        socket.on('invocation_started', (data) => {
          if (shouldHandleEvent(data.node.id)) {
            dispatch(invocationStarted({ data, timestamp: getTimestamp() }));
          }
        });

        socket.on('generator_progress', (data) => {
          if (shouldHandleEvent(data.node.id)) {
            dispatch(generatorProgress({ data, timestamp: getTimestamp() }));
          }
        });

        socket.on('invocation_error', (data) => {
          if (shouldHandleEvent(data.node.id)) {
            dispatch(invocationError({ data, timestamp: getTimestamp() }));
          }
        });

        socket.on('invocation_complete', (data) => {
          if (shouldHandleEvent(data.node.id)) {
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
          }
        });

        // Finally we actually invoke the session, starting processing
        dispatch(sessionInvoked({ sessionId }));
      }

      if (invocationComplete.match(action)) {
        const { config } = getState();

        if (config.shouldFetchImages) {
          const { result } = action.payload.data;
          if (isImageOutput(result)) {
            const imageName = result.image.image_name;
            const imageType = result.image.image_type;

            dispatch(imageReceived({ imageName, imageType }));
            dispatch(thumbnailReceived({ imageName, imageType }));
          }
        }
      }

      // Always pass the action on so other middleware and reducers can handle it
      next(action);
    };

  return middleware;
};
