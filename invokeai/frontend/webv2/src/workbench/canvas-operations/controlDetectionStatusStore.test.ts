import { describe, expect, it, vi } from 'vitest';

import { createControlDetectionStatusStore } from './controlDetectionStatusStore';

describe('controlDetectionStatusStore', () => {
  it('notifies only the changed layer and keeps snapshots stable between changes', () => {
    const store = createControlDetectionStatusStore();
    const first = vi.fn();
    const second = vi.fn();
    store.subscribe('a', first);
    store.subscribe('b', second);

    store.set('a', { detector: 'canny_edge_detection', status: 'processing' });
    const snapshot = store.get('a');
    store.set('a', { detector: 'canny_edge_detection', status: 'processing' });

    expect(first).toHaveBeenCalledOnce();
    expect(second).not.toHaveBeenCalled();
    expect(store.get('a')).toBe(snapshot);
    expect(store.get('b')).toEqual({ status: 'idle' });
  });

  it('forgets idle layers and clears every layer with a notification', () => {
    const store = createControlDetectionStatusStore();
    const listener = vi.fn();
    store.set('a', { status: 'queued' });
    store.set('b', { error: { code: 'process-failed', detail: 'boom' }, status: 'error' });
    store.subscribe('b', listener);

    store.set('a', { status: 'idle' });
    expect(store.layerIds()).toEqual(['b']);

    store.clear();
    expect(listener).toHaveBeenCalledOnce();
    expect(store.layerIds()).toEqual([]);
    expect(store.get('b')).toEqual({ status: 'idle' });
  });

  it('stops notifying after unsubscribe', () => {
    const store = createControlDetectionStatusStore();
    const listener = vi.fn();
    const unsubscribe = store.subscribe('a', listener);

    unsubscribe();
    store.set('a', { status: 'queued' });

    expect(listener).not.toHaveBeenCalled();
  });
});
