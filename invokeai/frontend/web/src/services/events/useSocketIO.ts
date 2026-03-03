import { useAppStore } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import type { MapStore } from 'nanostores';
import { useEffect, useMemo } from 'react';
import { selectQueueStatus } from 'services/api/endpoints/queue';
import { setEventListeners } from 'services/events/setEventListeners';
import type { AppSocket } from 'services/events/types';
import type { ManagerOptions, SocketOptions } from 'socket.io-client';
import { io } from 'socket.io-client';

import { $isConnected, $lastProgressEvent, $socket } from './stores';

// Inject socket options and url into window for debugging
declare global {
  interface Window {
    $socketOptions?: MapStore<Partial<ManagerOptions & SocketOptions>>;
  }
}

/**
 * Initializes the socket.io connection and sets up event listeners.
 */
export const useSocketIO = () => {
  useAssertSingleton('useSocketIO');
  const store = useAppStore();

  const socketUrl = useMemo(() => {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${wsProtocol}://${window.location.host}`;
  }, []);

  const socketOptions = useMemo(() => {
    const token = localStorage.getItem('auth_token');
    const options: Partial<ManagerOptions & SocketOptions> = {
      timeout: 60000,
      path: '/ws/socket.io',
      autoConnect: false, // achtung! removing this breaks the dynamic middleware
      forceNew: true,
      auth: token ? { token } : undefined,
      extraHeaders: token
        ? {
            Authorization: `Bearer ${token}`,
          }
        : undefined,
    };

    return options;
  }, []);

  useEffect(() => {
    const socket: AppSocket = io(socketUrl, socketOptions);
    $socket.set(socket);

    setEventListeners({ socket, store, setIsConnected: $isConnected.set });

    socket.connect();

    if (import.meta.env.MODE === 'development') {
      // This is only enabled manually for debugging, console is allowed.
      /* eslint-disable-next-line no-console */
      console.log('Socket initialized', socket);
    }

    const unsubscribeQueueStatusListener = store.subscribe(() => {
      const queueStatusData = selectQueueStatus(store.getState()).data;
      if (!queueStatusData || queueStatusData.queue.in_progress === 0) {
        $lastProgressEvent.set(null);
      }
    });

    return () => {
      if (import.meta.env.MODE === 'development') {
        window.$socketOptions = undefined;
        // This is only enabled manually for debugging, console is allowed.
        /* eslint-disable-next-line no-console */
        console.log('Socket teardown', socket);
      }
      unsubscribeQueueStatusListener();
      socket.disconnect();
    };
  }, [socketOptions, socketUrl, store]);
};
