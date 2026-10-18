import type { BackendConnectionStatus } from '@platform/transport/types';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { createGalleryRealtimeRuntime } from './realtimeRuntime';

describe('gallery realtime runtime', () => {
  const handlers = new Map<string, (payload: never) => void>();
  const detachers: Array<ReturnType<typeof vi.fn>> = [];
  let connectionHandler: ((status: BackendConnectionStatus) => void) | null = null;
  let initialStatus: BackendConnectionStatus = 'connected';
  const backend = {
    on: vi.fn((event: string, handler: (payload: never) => void) => {
      handlers.set(event, handler);
      const detach = vi.fn(() => handlers.delete(event));
      detachers.push(detach);
      return detach;
    }),
    onConnectionChange: vi.fn((handler: (status: BackendConnectionStatus) => void) => {
      connectionHandler = handler;
      handler(initialStatus);
      const detach = vi.fn(() => {
        connectionHandler = null;
      });
      detachers.push(detach);
      return detach;
    }),
  };
  const invalidate = vi.fn();
  let sessionUserId: string | null = 'alice';
  const getSessionUserId = () => sessionUserId;
  const onRevealRequest = vi.fn();
  const CLIENT_ORIGIN = 'this-client';
  const createRuntime = () =>
    createGalleryRealtimeRuntime({
      backend,
      getSessionUserId,
      invalidate,
      isClientQueueItemOrigin: (origin) => origin === CLIENT_ORIGIN,
      onRevealRequest,
    });
  const emitStatusChanged = (event: {
    destination?: string | null;
    item_id?: number;
    origin?: string | null;
    status?: string;
    user_id?: string;
  }): void =>
    handlers.get('queue_item_status_changed')?.({
      destination: null,
      item_id: 1,
      origin: 'chatbot',
      status: 'completed',
      user_id: 'alice',
      ...event,
    } as never);

  beforeEach(() => {
    vi.useFakeTimers();
    handlers.clear();
    detachers.length = 0;
    connectionHandler = null;
    initialStatus = 'connected';
    sessionUserId = 'alice';
    backend.on.mockClear();
    backend.onConnectionChange.mockClear();
    invalidate.mockReset();
    onRevealRequest.mockReset();
  });

  afterEach(() => vi.useRealTimers());

  it('coalesces a burst of upload events into one invalidation and detaches on dispose', () => {
    const runtime = createRuntime();

    runtime.start();
    runtime.start();
    handlers.get('image_uploaded')?.({ image_name: 'a.png' } as never);
    handlers.get('image_uploaded')?.({ image_name: 'b.png' } as never);
    handlers.get('video_uploaded')?.({ video_name: 'c.mp4' } as never);

    expect(backend.on).toHaveBeenCalledTimes(3);
    expect(backend.onConnectionChange).toHaveBeenCalledTimes(1);
    expect(invalidate).not.toHaveBeenCalled();

    vi.advanceTimersByTime(250);

    expect(invalidate).toHaveBeenCalledTimes(1);

    runtime.dispose();
    handlers.get('image_uploaded')?.({ image_name: 'd.png' } as never);
    vi.advanceTimersByTime(250);

    expect(detachers.every((detach) => detach.mock.calls.length === 1)).toBe(true);
    expect(invalidate).toHaveBeenCalledTimes(1);
  });

  it('holds passes a minimum interval apart under a stream of uploads and still runs a trailing pass', () => {
    const runtime = createRuntime();

    runtime.start();

    // A bulk drop completing one file every 300 ms for 3 s: each echo would
    // otherwise become its own pass, cancelling the page fetches under the user.
    for (let elapsed = 0; elapsed < 3000; elapsed += 300) {
      handlers.get('image_uploaded')?.({ image_name: `${elapsed}.png` } as never);
      vi.advanceTimersByTime(300);
    }

    const passesDuringStream = invalidate.mock.calls.length;

    expect(passesDuringStream).toBeGreaterThanOrEqual(2);
    expect(passesDuringStream).toBeLessThanOrEqual(4);

    vi.advanceTimersByTime(1000);

    // The last file's echo landed after the final pass and must not be lost.
    expect(invalidate).toHaveBeenCalledTimes(passesDuringStream + 1);
    runtime.dispose();
  });

  it('refetches after an outage but not on the boot-time connect', () => {
    initialStatus = 'connecting';
    const runtime = createRuntime();

    runtime.start();
    connectionHandler?.('connected');
    vi.advanceTimersByTime(250);

    expect(invalidate).not.toHaveBeenCalled();

    connectionHandler?.('disconnected');
    connectionHandler?.('connected');
    vi.advanceTimersByTime(250);

    expect(invalidate).toHaveBeenCalledTimes(1);
    runtime.dispose();
  });

  it('treats a connect after mounting on a downed backend as a reconnect', () => {
    initialStatus = 'disconnected';
    const runtime = createRuntime();

    runtime.start();
    connectionHandler?.('connected');
    vi.advanceTimersByTime(250);

    expect(invalidate).toHaveBeenCalledTimes(1);
    runtime.dispose();
  });

  it('drops a pending invalidation on dispose', () => {
    const runtime = createRuntime();

    runtime.start();
    handlers.get('image_uploaded')?.({ image_name: 'a.png' } as never);
    runtime.dispose();

    // Asserting the timer is gone, not just inert: the account-scope guard alone would
    // keep a disposed runtime's timer alive on every navigation away from the editor.
    expect(vi.getTimerCount()).toBe(0);

    vi.advanceTimersByTime(250);

    expect(invalidate).not.toHaveBeenCalled();
  });

  it('ignores socket callbacks after its account scope expires', () => {
    const runtime = createRuntime();

    runtime.start();
    accountLifecycle.invalidate();
    handlers.get('image_uploaded')?.({ image_name: 'a.png' } as never);
    vi.advanceTimersByTime(250);

    expect(invalidate).not.toHaveBeenCalled();
    runtime.dispose();
  });

  it("refreshes once for a burst of the session user's completions enqueued by other clients", () => {
    const runtime = createRuntime();

    runtime.start();
    emitStatusChanged({});
    emitStatusChanged({ origin: null });
    vi.advanceTimersByTime(250);

    expect(invalidate).toHaveBeenCalledTimes(1);
    runtime.dispose();
  });

  it('ignores completions that cannot add to this gallery or that Queue already refreshes for', () => {
    const runtime = createRuntime();
    const destination = 'gallery:reveal';

    runtime.start();
    // Another user's item, as an admin receives it, and as everyone else receives it.
    emitStatusChanged({ destination, user_id: 'bob' });
    emitStatusChanged({ destination: null, origin: null, user_id: 'redacted' });
    // An item this client enqueued, which Queue refreshes for in the tab tracking it.
    emitStatusChanged({ destination, origin: CLIENT_ORIGIN });

    for (const status of ['pending', 'in_progress', 'waiting', 'failed', 'canceled'] as const) {
      emitStatusChanged({ destination, status });
    }

    vi.advanceTimersByTime(1000);

    expect(invalidate).not.toHaveBeenCalled();
    expect(onRevealRequest).not.toHaveBeenCalled();
    runtime.dispose();
  });

  it('hands on only the completions whose batch asked to be revealed', () => {
    const runtime = createRuntime();

    runtime.start();
    emitStatusChanged({ destination: 'gallery:reveal', item_id: 7 });
    // Legacy web sends `gallery` on every workflow and upscale run, so it cannot opt in.
    emitStatusChanged({ destination: 'gallery', item_id: 8 });
    emitStatusChanged({ destination: 'canvas', item_id: 9 });
    emitStatusChanged({ item_id: 10 });

    expect(onRevealRequest.mock.calls).toEqual([[7]]);

    accountLifecycle.invalidate();
    emitStatusChanged({ destination: 'gallery:reveal', item_id: 11 });

    expect(onRevealRequest).toHaveBeenCalledTimes(1);
    runtime.dispose();
  });

  it("refreshes for the server's system user when a single-user session cannot fence by user", () => {
    sessionUserId = null;
    const runtime = createRuntime();

    runtime.start();
    emitStatusChanged({ user_id: 'system' });
    vi.advanceTimersByTime(250);

    expect(invalidate).toHaveBeenCalledTimes(1);
    runtime.dispose();
  });

  it('attributes each completion to the session user at the time it arrives', () => {
    sessionUserId = 'alice';
    const runtime = createRuntime();

    runtime.start();
    sessionUserId = 'bob';
    emitStatusChanged({ user_id: 'alice' });
    vi.advanceTimersByTime(1000);

    expect(invalidate).not.toHaveBeenCalled();

    emitStatusChanged({ user_id: 'bob' });
    vi.advanceTimersByTime(250);

    expect(invalidate).toHaveBeenCalledTimes(1);
    runtime.dispose();
  });
});
