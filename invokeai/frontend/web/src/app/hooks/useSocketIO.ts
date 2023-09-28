import { useStore } from '@nanostores/react';
import { $isDebugging } from 'app/store/nanostores/isDebugging';
import { useAppDispatch } from 'app/store/storeHooks';
import { MapStore, WritableAtom, atom, map } from 'nanostores';
import { useEffect } from 'react';
import { $authToken, $baseUrl } from 'services/api/client';
import {
  ClientToServerEvents,
  ServerToClientEvents,
} from 'services/events/types';
import { setEventListeners } from 'services/events/util/setEventListeners';
import { ManagerOptions, Socket, SocketOptions, io } from 'socket.io-client';

declare global {
  interface Window {
    $socketOptions?: MapStore<Partial<ManagerOptions & SocketOptions>>;
    $socketUrl?: WritableAtom<string>;
  }
}

const makeSocketOptions = (): Partial<ManagerOptions & SocketOptions> => {
  const socketOptions: Parameters<typeof io>[0] = {
    timeout: 60000,
    path: '/ws/socket.io',
    autoConnect: false, // achtung! removing this breaks the dynamic middleware
    forceNew: true,
  };

  // if building in package mode, replace socket url with open api base url minus the http protocol
  if (['nodes', 'package'].includes(import.meta.env.MODE)) {
    const authToken = $authToken.get();
    if (authToken) {
      // TODO: handle providing jwt to socket.io
      socketOptions.auth = { token: authToken };
    }

    socketOptions.transports = ['websocket', 'polling'];
  }

  return socketOptions;
};

const makeSocketUrl = (): string => {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  let socketUrl = `${wsProtocol}://${window.location.host}`;
  if (['nodes', 'package'].includes(import.meta.env.MODE)) {
    const baseUrl = $baseUrl.get();
    if (baseUrl) {
      //eslint-disable-next-line
      socketUrl = baseUrl.replace(/^https?\:\/\//i, '');
    }
  }
  return socketUrl;
};

const makeSocket = (): Socket<ServerToClientEvents, ClientToServerEvents> => {
  const socketOptions = makeSocketOptions();
  const socketUrl = $socketUrl.get();
  const socket: Socket<ServerToClientEvents, ClientToServerEvents> = io(
    socketUrl,
    { ...socketOptions, ...$socketOptions.get() }
  );
  return socket;
};

export const $socketOptions = map<Partial<ManagerOptions & SocketOptions>>({});
export const $socketUrl = atom<string>(makeSocketUrl());

export const useSocketIO = () => {
  const dispatch = useAppDispatch();
  const socketOptions = useStore($socketOptions);
  const socketUrl = useStore($socketUrl);
  const baseUrl = useStore($baseUrl);
  const authToken = useStore($authToken);

  useEffect(() => {
    const socket = makeSocket();
    setEventListeners({ dispatch, socket });
    socket.connect();

    if ($isDebugging.get()) {
      window.$socketOptions = $socketOptions;
      window.$socketUrl = $socketUrl;
      console.log('Socket initialized', socket);
    }

    return () => {
      if ($isDebugging.get()) {
        window.$socketOptions = undefined;
        window.$socketUrl = undefined;
        console.log('Socket teardown', socket);
      }
      socket.disconnect();
    };
  }, [dispatch, socketOptions, socketUrl, baseUrl, authToken]);
};
