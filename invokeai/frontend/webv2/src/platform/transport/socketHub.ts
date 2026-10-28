import { createLogger } from '@platform/logging/logger';
import { io } from 'socket.io-client';

import type { BackendConnectionStatus } from './types';

import { setConnectionStatus } from './connectionStore';
import { getBackendSocketPath, getBackendSocketUrl, getHttpAuthToken, subscribeHttpCredential } from './http';

export interface BackendSocket {
  /** False once the server ended the session or refused the handshake; Socket.IO then never retries by itself. */
  readonly active: boolean;
  on(event: string, handler: (payload: never) => void): unknown;
  off(event: string, handler: (payload: never) => void): unknown;
  emit(event: string, payload: unknown): unknown;
  connect(): unknown;
  disconnect(): unknown;
}

export type ConnectionListener = (status: BackendConnectionStatus, error?: string) => void;

/** Own one authenticated socket; feature listeners stay outside the hub to preserve Launchpad bundle boundaries. */
export interface SocketHub {
  /** Idempotent: connects the single socket if one is not already live. */
  connect(): void;
  /** Tears down the socket (identity change / logout); a later `connect` rebuilds it. */
  disconnect(): void;
  /** Attach a raw socket listener; returns an unsubscribe. Survives socket recreation. */
  on(event: string, handler: (payload: never) => void): () => void;
  emit(event: string, payload: unknown): void;
  /** Subscribe to connection transitions; fires synchronously with current status on subscribe. */
  onConnectionChange(handler: ConnectionListener): () => void;
}

/** Supplies the handshake payload; Socket.IO calls it on every connect and reconnect attempt. */
export type SocketAuthenticator = (callback: (payload: { token?: string }) => void) => void;

/** The bearer credential the hub authenticates with. */
export interface SocketCredentialSource {
  getToken(): string | null;
  /** Notifies when the current identity's token is replaced in place. */
  subscribe(listener: () => void): () => void;
}

// The backend reads the handshake `auth.token` first. A static Authorization header would outlive token
// replacements, so none is sent.
const createDefaultSocket = (authenticate: SocketAuthenticator): BackendSocket =>
  // Narrow Socket.IO's generic off overload to the facade's supported calls.
  io(getBackendSocketUrl(), {
    auth: authenticate,
    autoConnect: false,
    path: getBackendSocketPath(),
    timeout: 60000,
  }) as unknown as BackendSocket;

const httpCredentialSource: SocketCredentialSource = {
  getToken: getHttpAuthToken,
  subscribe: subscribeHttpCredential,
};

const socketLogger = createLogger({ area: 'socket', namespace: 'transport' });

export const createSocketHub = (
  options: {
    createSocket?: (authenticate: SocketAuthenticator) => BackendSocket;
    credential?: SocketCredentialSource;
  } = {}
): SocketHub => {
  const createSocket = options.createSocket ?? createDefaultSocket;
  const credential = options.credential ?? httpCredentialSource;

  let socket: BackendSocket | null = null;
  /** The token the live socket last authenticated with. */
  let handshakeToken: string | null = null;
  let unsubscribeCredential: (() => void) | null = null;
  let socketLifecycleHandlers: {
    connect: (payload: never) => void;
    connectError: (payload: never) => void;
    disconnect: (payload: never) => void;
  } | null = null;
  let status: BackendConnectionStatus = 'connecting';
  let lastError: string | undefined;

  /** Registered consumer listeners, kept so they can be re-bound to a fresh socket. */
  const eventHandlers = new Map<string, Set<(payload: never) => void>>();
  const connectionListeners = new Set<ConnectionListener>();

  const publishStatus = (next: BackendConnectionStatus, error?: string): void => {
    const previous = status;
    const previousError = lastError;

    status = next;
    lastError = error;
    if (next === 'disconnected') {
      // Reconnect attempts repeat the same failure; keep one warning per outage and the retries as breadcrumbs.
      const isRepeat = previous === 'disconnected' && previousError === error;

      socketLogger[isRepeat ? 'debug' : 'warn']({
        context: { previous, reason: error },
        message: `Backend socket ${isRepeat ? 'reconnect failed' : 'disconnected'}${error ? `: ${error}` : ''}`,
        name: isRepeat ? 'socket.reconnect-failed' : 'socket.disconnected',
      });
    } else if (next === 'connected') {
      socketLogger.info({ context: { previous }, message: 'Backend socket connected', name: 'socket.connected' });
    } else {
      socketLogger.debug({ context: { previous }, message: 'Backend socket connecting', name: 'socket.connecting' });
    }
    setConnectionStatus(next, error);

    for (const listener of connectionListeners) {
      listener(next, error);
    }
  };

  /**
   * After an own password change the server closes sockets authenticated under the revoked token, and Socket.IO does
   * not retry a session the server ended. A replaced credential is the one reason to try again; an unchanged one would
   * only be refused again.
   */
  const resumeWithReplacedCredential = (): void => {
    if (socket && !socket.active && credential.getToken() !== handshakeToken) {
      socket.connect();
    }
  };

  const connect = (): void => {
    if (socket) {
      return;
    }

    const nextSocket: BackendSocket = createSocket((callback) => {
      const token = credential.getToken();

      if (socket === nextSocket) {
        handshakeToken = token;
      }
      callback(token === null ? {} : { token });
    });

    socket = nextSocket;
    handshakeToken = null;
    publishStatus('connecting');

    const onConnect = () => {
      publishStatus('connected');
      nextSocket.emit('subscribe_queue', { queue_id: 'default' });
    };
    const onConnectError = (error: { message: string }) => {
      publishStatus('disconnected', error.message);
      resumeWithReplacedCredential();
    };
    const onDisconnect = (reason: string) => {
      publishStatus('disconnected', reason);
      resumeWithReplacedCredential();
    };

    socketLifecycleHandlers = {
      connect: onConnect,
      connectError: onConnectError as (payload: never) => void,
      disconnect: onDisconnect as (payload: never) => void,
    };
    nextSocket.on('connect', onConnect);
    nextSocket.on('connect_error', socketLifecycleHandlers.connectError);
    nextSocket.on('disconnect', socketLifecycleHandlers.disconnect);

    // Re-bind consumer listeners so they survive a socket recreation.
    for (const [event, handlers] of eventHandlers) {
      for (const handler of handlers) {
        nextSocket.on(event, handler);
      }
    }

    unsubscribeCredential = credential.subscribe(resumeWithReplacedCredential);
    nextSocket.connect();
  };

  const disconnect = (): void => {
    const oldSocket = socket;
    const oldLifecycleHandlers = socketLifecycleHandlers;

    socket = null;
    socketLifecycleHandlers = null;
    unsubscribeCredential?.();
    unsubscribeCredential = null;

    if (oldSocket) {
      if (oldLifecycleHandlers) {
        oldSocket.off('connect', oldLifecycleHandlers.connect);
        oldSocket.off('connect_error', oldLifecycleHandlers.connectError);
        oldSocket.off('disconnect', oldLifecycleHandlers.disconnect);
      }

      for (const [event, handlers] of eventHandlers) {
        for (const handler of handlers) {
          oldSocket.off(event, handler);
        }
      }

      oldSocket.disconnect();
    }

    publishStatus('connecting');
  };

  const on = (event: string, handler: (payload: never) => void): (() => void) => {
    let handlers = eventHandlers.get(event);

    if (!handlers) {
      handlers = new Set();
      eventHandlers.set(event, handlers);
    }

    handlers.add(handler);
    socket?.on(event, handler);

    return () => {
      eventHandlers.get(event)?.delete(handler);
      socket?.off(event, handler);
    };
  };

  const emit = (event: string, payload: unknown): void => {
    socket?.emit(event, payload);
  };

  const onConnectionChange = (handler: ConnectionListener): (() => void) => {
    connectionListeners.add(handler);
    handler(status, lastError);

    return () => {
      connectionListeners.delete(handler);
    };
  };

  return { connect, disconnect, emit, on, onConnectionChange };
};

/** The app-wide socket hub singleton, connected by `SocketHubRuntime`. */
export const socketHub = createSocketHub();
