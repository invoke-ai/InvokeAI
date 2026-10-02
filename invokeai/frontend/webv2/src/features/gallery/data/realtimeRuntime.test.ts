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

  beforeEach(() => {
    vi.useFakeTimers();
    handlers.clear();
    detachers.length = 0;
    connectionHandler = null;
    initialStatus = 'connected';
    backend.on.mockClear();
    backend.onConnectionChange.mockClear();
    invalidate.mockReset();
  });

  afterEach(() => vi.useRealTimers());

  it('coalesces a burst of upload events into one invalidation and detaches on dispose', () => {
    const runtime = createGalleryRealtimeRuntime({ backend, invalidate });

    runtime.start();
    runtime.start();
    handlers.get('image_uploaded')?.({ image_name: 'a.png' } as never);
    handlers.get('image_uploaded')?.({ image_name: 'b.png' } as never);
    handlers.get('video_uploaded')?.({ video_name: 'c.mp4' } as never);

    expect(backend.on).toHaveBeenCalledTimes(2);
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
    const runtime = createGalleryRealtimeRuntime({ backend, invalidate });

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
    const runtime = createGalleryRealtimeRuntime({ backend, invalidate });

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
    const runtime = createGalleryRealtimeRuntime({ backend, invalidate });

    runtime.start();
    connectionHandler?.('connected');
    vi.advanceTimersByTime(250);

    expect(invalidate).toHaveBeenCalledTimes(1);
    runtime.dispose();
  });

  it('drops a pending invalidation on dispose', () => {
    const runtime = createGalleryRealtimeRuntime({ backend, invalidate });

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
    const runtime = createGalleryRealtimeRuntime({ backend, invalidate });

    runtime.start();
    accountLifecycle.invalidate();
    handlers.get('image_uploaded')?.({ image_name: 'a.png' } as never);
    vi.advanceTimersByTime(250);

    expect(invalidate).not.toHaveBeenCalled();
    runtime.dispose();
  });
});
