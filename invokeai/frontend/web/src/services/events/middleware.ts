import { Middleware, MiddlewareAPI } from '@reduxjs/toolkit';
import { io, Socket } from 'socket.io-client';

import {
  ClientToServerEvents,
  ServerToClientEvents,
} from 'services/events/types';
import {
  invocationComplete,
  socketSubscribed,
  socketUnsubscribed,
} from './actions';
import { AppDispatch, RootState } from 'app/store/store';
import { getTimestamp } from 'common/util/getTimestamp';
import {
  sessionInvoked,
  isFulfilledSessionCreatedAction,
} from 'services/thunks/session';
import { OpenAPI } from 'services/api';
import { isImageOutput } from 'services/types/guards';
import { imageReceived, thumbnailReceived } from 'services/thunks/image';
import { setEventListeners } from 'services/events/util/setEventListeners';
import { log } from 'app/logging/useLogger';

const socketioLog = log.child({ namespace: 'socketio' });

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

    socketOptions.transports = ['websocket', 'polling'];
  }

  const socket: Socket<ServerToClientEvents, ClientToServerEvents> = io(
    socketUrl,
    socketOptions
  );

  const middleware: Middleware =
    (store: MiddlewareAPI<AppDispatch, RootState>) => (next) => (action) => {
      const { dispatch, getState } = store;

      // Set listeners for `connect` and `disconnect` events once
      // Must happen in middleware to get access to `dispatch`
      if (!areListenersSet) {
        setEventListeners({ store, socket, log: socketioLog });

        areListenersSet = true;

        socket.connect();
      }

      if (isFulfilledSessionCreatedAction(action)) {
        const sessionId = action.payload.id;
        const sessionLog = socketioLog.child({ sessionId });
        const oldSessionId = getState().system.sessionId;

        if (oldSessionId) {
          sessionLog.debug(
            { oldSessionId },
            `Unsubscribed from old session (${oldSessionId})`
          );

          socket.emit('unsubscribe', {
            session: oldSessionId,
          });

          dispatch(
            socketUnsubscribed({
              sessionId: oldSessionId,
              timestamp: getTimestamp(),
            })
          );
        }

        sessionLog.debug(`Subscribe to new session (${sessionId})`);

        socket.emit('subscribe', { session: sessionId });

        dispatch(
          socketSubscribed({
            sessionId: sessionId,
            timestamp: getTimestamp(),
          })
        );

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

      next(action);
    };

  return middleware;
};
