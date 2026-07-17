import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { getBasePath, getDeploymentBaseUrl } from 'common/util/baseUrl';
import { selectAuthToken, selectCurrentUser } from 'features/auth/store/authSlice';
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

  // In multiuser mode the socket must not connect until auth.user has hydrated from /me: the
  // event listeners classify every event's ownership against auth.user (see getEventScope), and
  // events received while it is still null would be misclassified as another user's — silently
  // dropping one-shot side effects (progress, node execution states, gallery auto-switch, the
  // failure toast) that never replay after hydration. No token means single-user mode (or a
  // stale token that ProtectedRoute has cleared), where every event is the client's own and the
  // socket can connect immediately.
  //
  // The token also feeds socketOptions, making it a dependency of the connect effect: an in-tab
  // logout or session expiry (which nulls the token) tears the authenticated socket down instead
  // of letting it keep the old user's room membership — and private events — until the next full
  // page reload.
  const token = useAppSelector(selectAuthToken);
  const currentUser = useAppSelector(selectCurrentUser);
  const isAuthHydrated = !token || currentUser !== null;

  const socketUrl = useMemo(() => {
    const base = new URL(getDeploymentBaseUrl());
    const wsProtocol = base.protocol === 'https:' ? 'wss' : 'ws';
    // Origin only - the sub-path prefix (if any) is passed via the socket.io `path` option below.
    return `${wsProtocol}://${base.host}`;
  }, []);

  // Derived from the redux token (hydrated synchronously from localStorage) rather than a
  // one-time localStorage read, so the socket always authenticates with the current session's
  // token and reconnects when it changes.
  const socketOptions = useMemo(() => {
    const options: Partial<ManagerOptions & SocketOptions> = {
      timeout: 60000,
      path: `${getBasePath()}/ws/socket.io`,
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
  }, [token]);

  useEffect(() => {
    if (!isAuthHydrated) {
      return;
    }
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
      $socket.set(null);
    };
  }, [isAuthHydrated, socketOptions, socketUrl, store]);
};
