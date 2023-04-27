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
import { AppDispatch, RootState } from 'app/store/store';
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
import { setEventListeners } from './util/setEventListeners';

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

      // Nothing dispatches `socketReset` actions yet
      // if (socketReset.match(action)) {
      //   const { sessionId } = getState().system;

      //   if (sessionId) {
      //     socket.emit('unsubscribe', { session: sessionId });
      //     dispatch(
      //       socketUnsubscribed({ sessionId, timestamp: getTimestamp() })
      //     );
      //   }

      //   if (socket.connected) {
      //     socket.disconnect();
      //     dispatch(socketDisconnected({ timestamp: getTimestamp() }));
      //   }

      //   socket.removeAllListeners();
      //   areListenersSet = false;
      // }

      // Set listeners for `connect` and `disconnect` events once
      // Must happen in middleware to get access to `dispatch`
      if (!areListenersSet) {
        socket.on('connect', () => {
          dispatch(socketConnected({ timestamp: getTimestamp() }));

          const { results, uploads, models, nodes, config, system } =
            getState();

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

          if (system.sessionId) {
            console.log(`Re-subscribing to session ${system.sessionId}`);
            socket.emit('subscribe', { session: system.sessionId });
            dispatch(
              socketSubscribed({
                sessionId: system.sessionId,
                timestamp: getTimestamp(),
              })
            );
            setEventListeners({ socket, store });
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

        socket.emit('subscribe', { session: sessionId });
        dispatch(
          socketSubscribed({
            sessionId: sessionId,
            timestamp: getTimestamp(),
          })
        );
        setEventListeners({ socket, store });

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
            dispatch(
              thumbnailReceived({
                thumbnailName: imageName,
                thumbnailType: imageType,
              })
            );
          }
        }
      }

      // Always pass the action on so other middleware and reducers can handle it
      next(action);
    };

  return middleware;
};
