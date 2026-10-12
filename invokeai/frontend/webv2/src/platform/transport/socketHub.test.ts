import { DEFAULT_LOGGING_CONFIG } from '@platform/logging/contracts';
import { configureLogging, getLogSnapshot, resetLogging } from '@platform/logging/logger';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { BackendSocket, SocketAuthenticator, SocketCredentialSource } from './socketHub';

import { getConnectionStatus } from './connectionStore';
import { createSocketHub } from './socketHub';

/** Mirrors Socket.IO: every connect attempt asks the authenticator, and a server-ended session stays inactive. */
class FakeSocket implements BackendSocket {
  active = false;
  /** False leaves each handshake pending until the test accepts or refuses it. */
  acceptsHandshakes = true;
  readonly emitted: { event: string; payload: unknown }[] = [];
  readonly handshakes: { token?: string }[] = [];
  private readonly handlers = new Map<string, Set<(payload: never) => void>>();

  constructor(private readonly authenticate?: SocketAuthenticator) {}

  on(event: string, handler: (payload: never) => void): void {
    let handlers = this.handlers.get(event);

    if (!handlers) {
      handlers = new Set();
      this.handlers.set(event, handlers);
    }

    handlers.add(handler);
  }

  off(event: string, handler: (payload: never) => void): void {
    this.handlers.get(event)?.delete(handler);
  }

  emit(event: string, payload: unknown): void {
    this.emitted.push({ event, payload });
  }

  connect(): void {
    this.active = true;
    this.authenticate?.((payload) => {
      this.handshakes.push(payload);
    });
    if (this.acceptsHandshakes) {
      this.fire('connect', undefined);
    }
  }

  /** The server refused the pending handshake; like a server disconnect, Socket.IO does not retry it. */
  refuseHandshake(): void {
    this.active = false;
    this.fire('connect_error', { message: 'Connection rejected by server' });
  }

  disconnect(): void {
    this.active = false;
    this.fire('disconnect', 'io client disconnect');
  }

  /** The server closed the session, e.g. after a password change revoked the token it authenticated with. */
  serverDisconnect(): void {
    this.active = false;
    this.fire('disconnect', 'io server disconnect');
  }

  fire(event: string, payload: unknown): void {
    for (const handler of this.handlers.get(event) ?? []) {
      (handler as (value: unknown) => void)(payload);
    }
  }
}

const createCredential = (initialToken: string | null) => {
  let token = initialToken;
  const listeners = new Set<() => void>();
  const source: SocketCredentialSource = {
    getToken: () => token,
    subscribe: (listener) => {
      listeners.add(listener);

      return () => listeners.delete(listener);
    },
  };

  return {
    listenerCount: () => listeners.size,
    replace: (nextToken: string) => {
      token = nextToken;
      for (const listener of listeners) {
        listener();
      }
    },
    source,
  };
};

describe('socketHub', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('connects, reports connected status, and subscribes to the queue', () => {
    const socket = new FakeSocket();
    const hub = createSocketHub({ createSocket: () => socket });
    const onChange = vi.fn();

    hub.onConnectionChange(onChange);
    onChange.mockClear();
    hub.connect();

    expect(getConnectionStatus().status).toBe('connected');
    expect(onChange).toHaveBeenNthCalledWith(1, 'connecting', undefined);
    expect(onChange).toHaveBeenLastCalledWith('connected', undefined);
    expect(socket.emitted).toContainEqual({ event: 'subscribe_queue', payload: { queue_id: 'default' } });
  });

  it('fires the current status synchronously on subscribe', () => {
    const socket = new FakeSocket();
    const hub = createSocketHub({ createSocket: () => socket });

    hub.connect();

    const late = vi.fn();

    hub.onConnectionChange(late);

    expect(late).toHaveBeenCalledWith('connected', undefined);
  });

  it('is idempotent — repeated connect keeps one socket', () => {
    let created = 0;
    const hub = createSocketHub({
      createSocket: () => {
        created += 1;

        return new FakeSocket();
      },
    });

    hub.connect();
    hub.connect();

    expect(created).toBe(1);
  });

  it('delivers consumer listeners and detaches them on unsubscribe', () => {
    const socket = new FakeSocket();
    const hub = createSocketHub({ createSocket: () => socket });

    hub.connect();

    const handler = vi.fn();
    const off = hub.on('queue_item_status_changed', handler);

    socket.fire('queue_item_status_changed', { item_id: 1 });
    expect(handler).toHaveBeenCalledTimes(1);

    off();
    socket.fire('queue_item_status_changed', { item_id: 2 });
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it('re-binds consumer listeners after a reconnect', () => {
    const sockets: FakeSocket[] = [];
    const hub = createSocketHub({
      createSocket: () => {
        const socket = new FakeSocket();

        sockets.push(socket);

        return socket;
      },
    });

    hub.connect();

    const handler = vi.fn();

    hub.on('queue_item_status_changed', handler);
    hub.disconnect();
    hub.connect();

    expect(sockets).toHaveLength(2);

    sockets[0]!.fire('queue_item_status_changed', { item_id: 0 });
    sockets[1]!.fire('queue_item_status_changed', { item_id: 1 });

    expect(handler).toHaveBeenCalledTimes(1);
  });

  it('reports a disconnect without interpreting domain events', () => {
    const socket = new FakeSocket();
    const hub = createSocketHub({ createSocket: () => socket });

    hub.connect();
    socket.fire('disconnect', 'transport close');

    expect(getConnectionStatus().status).toBe('disconnected');
  });

  it('records connection transitions with disconnects as warnings', () => {
    resetLogging();
    configureLogging({ ...DEFAULT_LOGGING_CONFIG, level: 'debug' });
    const socket = new FakeSocket();
    const hub = createSocketHub({ createSocket: () => socket });

    hub.connect();
    socket.fire('disconnect', 'transport close');
    socket.fire('connect_error', { message: 'xhr poll error' });
    socket.fire('connect_error', { message: 'xhr poll error' });

    expect(getLogSnapshot().entries).toMatchObject([
      { level: 'debug', message: 'Backend socket reconnect failed: xhr poll error', name: 'socket.reconnect-failed' },
      { level: 'warn', message: 'Backend socket disconnected: xhr poll error', name: 'socket.disconnected' },
      {
        context: { previous: 'connected', reason: 'transport close' },
        level: 'warn',
        message: 'Backend socket disconnected: transport close',
        name: 'socket.disconnected',
        source: { area: 'socket', namespace: 'transport' },
      },
      { level: 'info', name: 'socket.connected' },
      { level: 'debug', name: 'socket.connecting' },
    ]);
  });

  it('authenticates every connect attempt with the then-current token', () => {
    const credential = createCredential('token-a');
    let socket: FakeSocket | undefined;
    const hub = createSocketHub({
      createSocket: (authenticate) => (socket = new FakeSocket(authenticate)),
      credential: credential.source,
    });

    hub.connect();
    credential.replace('token-a-renewed');
    // A transport drop keeps the session active; Socket.IO's own retry asks for the payload again.
    socket!.fire('disconnect', 'transport close');
    socket!.connect();

    expect(socket!.handshakes).toEqual([{ token: 'token-a' }, { token: 'token-a-renewed' }]);
  });

  it('reconnects with the replacement token after the server ends the revoked session, in either order', () => {
    const credential = createCredential('token-a');
    const sockets: FakeSocket[] = [];
    const hub = createSocketHub({
      createSocket: (authenticate) => {
        const socket = new FakeSocket(authenticate);

        sockets.push(socket);

        return socket;
      },
      credential: credential.source,
    });

    hub.connect();
    const socket = sockets[0]!;

    // The server's disconnect can arrive before the password-change response delivers the replacement...
    socket.serverDisconnect();
    expect(getConnectionStatus().status).toBe('disconnected');
    credential.replace('token-b');
    expect(getConnectionStatus().status).toBe('connected');

    // ...or after it: the replacement alone does not drop a live socket, the server's disconnect does.
    credential.replace('token-c');
    expect(socket.handshakes).toEqual([{ token: 'token-a' }, { token: 'token-b' }]);
    socket.serverDisconnect();

    expect(sockets).toHaveLength(1);
    expect(socket.handshakes).toEqual([{ token: 'token-a' }, { token: 'token-b' }, { token: 'token-c' }]);
    expect(getConnectionStatus().status).toBe('connected');
  });

  it('stays down when the server ends a session whose credential has not changed', () => {
    const credential = createCredential('token-a');
    let socket: FakeSocket | undefined;
    const hub = createSocketHub({
      createSocket: (authenticate) => (socket = new FakeSocket(authenticate)),
      credential: credential.source,
    });

    hub.connect();
    socket!.serverDisconnect();

    expect(socket!.active).toBe(false);
    expect(getConnectionStatus().status).toBe('disconnected');

    hub.disconnect();
    // A hub torn down for an account transition ignores later replacements; the next account rebuilds it.
    expect(credential.listenerCount()).toBe(0);
    credential.replace('token-b');

    expect(socket!.handshakes).toEqual([{ token: 'token-a' }]);
  });

  it('retries a handshake the server refused for a token replaced while it was pending', () => {
    const credential = createCredential('token-a');
    const sockets: FakeSocket[] = [];
    const hub = createSocketHub({
      createSocket: (authenticate) => {
        const socket = new FakeSocket(authenticate);

        socket.acceptsHandshakes = false;
        sockets.push(socket);

        return socket;
      },
      credential: credential.source,
    });

    hub.connect();
    const socket = sockets[0]!;
    // The replacement lands while the old token's handshake is still in flight, so there is nothing to resume yet.
    credential.replace('token-b');
    expect(socket.handshakes).toEqual([{ token: 'token-a' }]);

    socket.refuseHandshake();
    expect(socket.handshakes).toEqual([{ token: 'token-a' }, { token: 'token-b' }]);

    // A refusal of the current token is final until the credential changes again.
    socket.refuseHandshake();
    expect(socket.handshakes).toHaveLength(2);
    expect(sockets).toHaveLength(1);
    hub.disconnect();
  });
});
