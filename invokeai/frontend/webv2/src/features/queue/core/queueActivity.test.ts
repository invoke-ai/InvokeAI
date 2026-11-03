import { describe, expect, it } from 'vitest';

import { getQueueActivity } from './queueActivity';

const read = (overrides: Partial<Parameters<typeof getQueueActivity>[0]>) =>
  getQueueActivity({ hasOpenWork: true, isConnected: true, isLoadingModels: false, isPaused: false, ...overrides });

describe('getQueueActivity', () => {
  it('reads offline over paused over a loading model over running, and idle without open work', () => {
    expect(read({ isConnected: false, isLoadingModels: true, isPaused: true })).toBe('offline');
    expect(read({ isLoadingModels: true, isPaused: true })).toBe('paused');
    expect(read({ isLoadingModels: true })).toBe('loading-model');
    expect(read({})).toBe('running');
    expect(read({ hasOpenWork: false, isConnected: false, isPaused: true })).toBe('idle');
  });
});
