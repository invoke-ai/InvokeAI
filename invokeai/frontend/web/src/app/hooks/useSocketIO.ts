import { useStore } from '@nanostores/react';
import { $authToken } from 'app/store/nanostores/authToken';
import { $baseUrl } from 'app/store/nanostores/baseUrl';
import { $isDebugging } from 'app/store/nanostores/isDebugging';
import { useAppStore } from 'app/store/nanostores/store';
import type { MapStore } from 'nanostores';
import { atom, map } from 'nanostores';
import { useEffect, useMemo } from 'react';
import { setEventListeners } from 'services/events/setEventListeners';
import type { ClientToServerEvents, ServerToClientEvents } from 'services/events/types';
import type { ManagerOptions, Socket, SocketOptions } from 'socket.io-client';
import { io } from 'socket.io-client';
import { assert } from 'tsafe';

// Inject socket options and url into window for debugging
declare global {
  interface Window {
    $socketOptions?: MapStore<Partial<ManagerOptions & SocketOptions>>;
  }
}

export type AppSocket = Socket<ServerToClientEvents, ClientToServerEvents>;

export const $socket = atom<AppSocket | null>(null);
export const getSocket = () => {
  const socket = $socket.get();
  assert(socket !== null, 'Socket is not initialized');
  return socket;
};
export const $socketOptions = map<Partial<ManagerOptions & SocketOptions>>({});

const $isSocketInitialized = atom<boolean>(false);
export const $isConnected = atom<boolean>(false);

/**
 * Initializes the socket.io connection and sets up event listeners.
 */
export const useSocketIO = () => {
  const { dispatch, getState } = useAppStore();
  const baseUrl = useStore($baseUrl);
  const authToken = useStore($authToken);
  const addlSocketOptions = useStore($socketOptions);

  const socketUrl = useMemo(() => {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    if (baseUrl) {
      return baseUrl.replace(/^https?:\/\//i, '');
    }

    return `${wsProtocol}://${window.location.host}`;
  }, [baseUrl]);

  const socketOptions = useMemo(() => {
    const options: Partial<ManagerOptions & SocketOptions> = {
      timeout: 60000,
      path: baseUrl ? '/ws/socket.io' : `${window.location.pathname}ws/socket.io`,
      autoConnect: false, // achtung! removing this breaks the dynamic middleware
      forceNew: true,
    };

    if (authToken) {
      options.auth = { token: authToken };
      options.transports = ['websocket', 'polling'];
    }

    return { ...options, ...addlSocketOptions };
  }, [authToken, addlSocketOptions, baseUrl]);

  useEffect(() => {
    if ($isSocketInitialized.get()) {
      // Singleton!
      return;
    }

    const socket: AppSocket = io(socketUrl, socketOptions);
    $socket.set(socket);
    setEventListeners({ socket, dispatch, getState, setIsConnected: $isConnected.set });
    socket.connect();

    if ($isDebugging.get() || import.meta.env.MODE === 'development') {
      window.$socketOptions = $socketOptions;
      // This is only enabled manually for debugging, console is allowed.
      /* eslint-disable-next-line no-console */
      console.log('Socket initialized', socket);
    }

    $isSocketInitialized.set(true);

    return () => {
      if ($isDebugging.get() || import.meta.env.MODE === 'development') {
        window.$socketOptions = undefined;
        // This is only enabled manually for debugging, console is allowed.
        /* eslint-disable-next-line no-console */
        console.log('Socket teardown', socket);
      }
      socket.disconnect();
      $isSocketInitialized.set(false);
    };
  }, [dispatch, getState, socketOptions, socketUrl]);
};
