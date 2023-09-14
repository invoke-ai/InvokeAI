import { Middleware, MiddlewareAPI } from '@reduxjs/toolkit';
import { AppThunkDispatch, RootState } from 'app/store/store';
import { $authToken, $baseUrl } from 'services/api/client';
import {
  ClientToServerEvents,
  ServerToClientEvents,
} from 'services/events/types';
import { setEventListeners } from 'services/events/util/setEventListeners';
import { Socket, io } from 'socket.io-client';

export const socketMiddleware = () => {
  let areListenersSet = false;

  const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  let socketUrl = `${wsProtocol}://${window.location.host}`;

  const socketOptions: Parameters<typeof io>[0] = {
    timeout: 60000,
    path: '/ws/socket.io',
    autoConnect: false, // achtung! removing this breaks the dynamic middleware
  };

  // if building in package mode, replace socket url with open api base url minus the http protocol
  if (['nodes', 'package'].includes(import.meta.env.MODE)) {
    const baseUrl = $baseUrl.get();
    if (baseUrl) {
      //eslint-disable-next-line
      socketUrl = baseUrl.replace(/^https?\:\/\//i, '');
    }

    const authToken = $authToken.get();
    if (authToken) {
      // TODO: handle providing jwt to socket.io
      socketOptions.auth = { token: authToken };
    }

    socketOptions.transports = ['websocket', 'polling'];
  }

  const socket: Socket<ServerToClientEvents, ClientToServerEvents> = io(
    socketUrl,
    socketOptions
  );

  const middleware: Middleware =
    (storeApi: MiddlewareAPI<AppThunkDispatch, RootState>) =>
    (next) =>
    (action) => {
      // Set listeners for `connect` and `disconnect` events once
      // Must happen in middleware to get access to `dispatch`
      if (!areListenersSet) {
        setEventListeners({ storeApi, socket });

        areListenersSet = true;

        socket.connect();
      }

      next(action);
    };

  return middleware;
};
