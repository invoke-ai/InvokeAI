import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { BackendSocket } from './socketHub';

import { getConnectionStatus } from './connectionStore';
import { createSocketHub } from './socketHub';

class FakeSocket implements BackendSocket {
  readonly emitted: { event: string; payload: unknown }[] = [];
  private readonly handlers = new Map<string, Set<(payload: never) => void>>();

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
    this.fire('connect', undefined);
  }

  disconnect(): void {
    this.fire('disconnect', 'io client disconnect');
  }

  fire(event: string, payload: unknown): void {
    for (const handler of this.handlers.get(event) ?? []) {
      (handler as (value: unknown) => void)(payload);
    }
  }
}

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
});
