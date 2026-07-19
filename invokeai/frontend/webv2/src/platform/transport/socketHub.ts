import { io } from 'socket.io-client';

import type { BackendConnectionStatus } from './types';

import { setConnectionStatus } from './connectionStore';
import { getBackendSocketUrl, getHttpAuthToken } from './http';

const SOCKET_PATH = '/ws/socket.io';

/**
 * The minimal Socket.IO surface the hub uses; tests substitute a fake.
 */
export interface BackendSocket {
  on(event: string, handler: (payload: never) => void): unknown;
  off(event: string, handler: (payload: never) => void): unknown;
  emit(event: string, payload: unknown): unknown;
  connect(): unknown;
  disconnect(): unknown;
}

export type ConnectionListener = (status: BackendConnectionStatus, error?: string) => void;

/**
 * Owns the single backend socket for the whole authenticated app. It is only a
 * transport/status hub; feature runtimes attach their own listeners so admin
 * model code and editor queue code do not leak into the base Launchpad bundle.
 */
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

const createDefaultSocket = (): BackendSocket => {
  const token = getHttpAuthToken();

  // Socket.IO's generic `off` overload does not structurally match our minimal
  // facade; the socket satisfies the surface we actually use, so narrow it here.
  return io(getBackendSocketUrl(), {
    auth: token ? { token } : undefined,
    autoConnect: false,
    extraHeaders: token ? { Authorization: `Bearer ${token}` } : undefined,
    path: SOCKET_PATH,
    timeout: 60000,
  }) as unknown as BackendSocket;
};

export const createSocketHub = (options: { createSocket?: () => BackendSocket } = {}): SocketHub => {
  const createSocket = options.createSocket ?? createDefaultSocket;

  let socket: BackendSocket | null = null;
  let status: BackendConnectionStatus = 'connecting';
  let lastError: string | undefined;

  /** Registered consumer listeners, kept so they can be re-bound to a fresh socket. */
  const eventHandlers = new Map<string, Set<(payload: never) => void>>();
  const connectionListeners = new Set<ConnectionListener>();

  const publishStatus = (next: BackendConnectionStatus, error?: string): void => {
    status = next;
    lastError = error;
    setConnectionStatus(next, error);

    for (const listener of connectionListeners) {
      listener(next, error);
    }
  };

  const connect = (): void => {
    if (socket) {
      return;
    }

    const nextSocket = createSocket();

    socket = nextSocket;
    publishStatus('connecting');

    nextSocket.on('connect', () => {
      publishStatus('connected');
      nextSocket.emit('subscribe_queue', { queue_id: 'default' });
    });
    nextSocket.on('connect_error', (error: { message: string }) => {
      publishStatus('disconnected', error.message);
    });
    nextSocket.on('disconnect', (reason: string) => {
      publishStatus('disconnected', reason);
    });

    // Re-bind consumer listeners so they survive a socket recreation.
    for (const [event, handlers] of eventHandlers) {
      for (const handler of handlers) {
        nextSocket.on(event, handler);
      }
    }

    nextSocket.connect();
  };

  const disconnect = (): void => {
    socket?.disconnect();
    socket = null;
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
