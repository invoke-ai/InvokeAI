import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { getBasePath, getDeploymentBaseUrl } from 'common/util/baseUrl';
import { selectAuthSessionKey, selectAuthToken, selectCurrentUser } from 'features/auth/store/authSlice';
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
 * A server-initiated disconnect is terminal for socket.io: `Socket.ondisconnect` destroys the
 * manager subscriptions and sets `skipReconnect`, so the client never retries on its own. The
 * server uses that disconnect to act on authorization changes — a deactivated account, a
 * superseded token epoch (`_handle_user_access_changed` in `sockets.py`) — and expects a client
 * that still holds usable credentials to come back. Without a retry here this tab has no events
 * and no Invoke button until the page is reloaded, even after the change that caused it is
 * reverted.
 *
 * Bounded, because retrying is only ever right when the client is still welcome: a client that
 * is not gets a connect error, which socket.io does not retry, and the attempts stop there. The
 * count lives with the socket, so a session change (which builds a new one) starts it over.
 */
const MAX_SERVER_DISCONNECT_RECONNECTS = 5;
const SERVER_DISCONNECT_RECONNECT_DELAY_MS = 1000;

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
  // The session identity also feeds socketOptions, making it a dependency of the connect effect:
  // an in-tab logout, a session expiry (which nulls the token) or another account taking over the
  // tab tears the authenticated socket down instead of letting it keep the old user's room
  // membership — and private events — until the next full page reload.
  //
  // Identity, deliberately, and not the token itself. The sliding-window middleware mints a
  // replacement token on every mutating request, and the client commits one about once a minute
  // during any activity (TOKEN_REFRESH_THROTTLE_MS). Keying on the token's bytes meant every one
  // of those routine refreshes disconnected and rebuilt the live socket: listeners disposed, the
  // progress store cleared, and a visible glitch in the preview mid-generation, once a minute,
  // for a credential change that does not change who is connected or which rooms they belong to.
  const token = useAppSelector(selectAuthToken);
  const sessionKey = useAppSelector(selectAuthSessionKey);
  const currentUser = useAppSelector(selectCurrentUser);
  const isAuthHydrated = !token || currentUser !== null;

  const socketUrl = useMemo(() => {
    const base = new URL(getDeploymentBaseUrl());
    const wsProtocol = base.protocol === 'https:' ? 'wss' : 'ws';
    // Origin only - the sub-path prefix (if any) is passed via the socket.io `path` option below.
    return `${wsProtocol}://${base.host}`;
  }, []);

  // Derived from the redux session (hydrated synchronously from localStorage) rather than a
  // one-time localStorage read, so the socket always authenticates as the current session's user
  // and is rebuilt when that user changes.
  const socketOptions = useMemo(() => {
    const options: Partial<ManagerOptions & SocketOptions> = {
      timeout: 60000,
      path: `${getBasePath()}/ws/socket.io`,
      autoConnect: false, // achtung! removing this breaks the dynamic middleware
      forceNew: true,
      // A callback, so the token is read at each connection attempt instead of being baked into
      // the socket: this socket now outlives the token it was built with, and a reconnect — the
      // client's own retry, or one after the network drops — must present the token that is live
      // then, not the one that was live when the session started.
      //
      // This payload is the socket's only credential. It used to be backed by an `Authorization`
      // extra header, which `_handle_connect` falls back to when the payload carries no token —
      // but a header is fixed for the life of the manager, so it could only ever hold the token
      // that was live when the socket was built. That made it a way to authenticate a credential
      // the client had already discarded: log out in another tab while a reconnect is in flight
      // and the callback correctly sends `{}`, while the stale header would still have been
      // accepted. The payload is read live and cannot go stale, so the fallback only ever
      // weakened it.
      auth: sessionKey
        ? (cb: (data: object) => void) => {
            const currentToken = selectAuthToken(store.getState());
            cb(currentToken ? { token: currentToken } : {});
          }
        : undefined,
    };

    return options;
  }, [sessionKey, store]);

  useEffect(() => {
    if (!isAuthHydrated) {
      return;
    }
    const socket: AppSocket = io(socketUrl, socketOptions);
    $socket.set(socket);

    const disposeEventListeners = setEventListeners({ socket, store, setIsConnected: $isConnected.set });

    // See MAX_SERVER_DISCONNECT_RECONNECTS. Only `io server disconnect` is ours to retry —
    // socket.io reconnects transport-level drops itself, and `io client disconnect` is this
    // effect's own teardown.
    let serverDisconnects = 0;
    let reconnectTimeout: ReturnType<typeof setTimeout> | undefined;
    const reconnectAfterServerDisconnect = (reason: string) => {
      if (reason !== 'io server disconnect' || serverDisconnects >= MAX_SERVER_DISCONNECT_RECONNECTS) {
        return;
      }
      const delay = SERVER_DISCONNECT_RECONNECT_DELAY_MS * 2 ** serverDisconnects;
      serverDisconnects++;
      reconnectTimeout = setTimeout(() => {
        socket.connect();
      }, delay);
    };
    socket.on('disconnect', reconnectAfterServerDisconnect);

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
      // Before the socket goes: anything this session scheduled must not land in the next one.
      // A pending retry is exactly that — it would reconnect a socket this session has discarded,
      // with credentials that may no longer be the live ones.
      clearTimeout(reconnectTimeout);
      socket.off('disconnect', reconnectAfterServerDisconnect);
      disposeEventListeners();
      socket.disconnect();
      $socket.set(null);
    };
  }, [isAuthHydrated, socketOptions, socketUrl, store]);
};
