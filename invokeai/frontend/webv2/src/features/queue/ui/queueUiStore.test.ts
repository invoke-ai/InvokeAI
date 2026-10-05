import { beforeEach, describe, expect, it } from 'vitest';

import { clearPendingQueueItemReveal, getPendingQueueItemReveal, requestQueueItemReveal } from './queueUiStore';

beforeEach(() => {
  const pending = getPendingQueueItemReveal();

  if (pending) {
    clearPendingQueueItemReveal(pending.requestId);
  }
});

describe('queue item reveal requests', () => {
  it('tokens repeated reveals of the same item independently', () => {
    requestQueueItemReveal(42);
    const first = getPendingQueueItemReveal();
    requestQueueItemReveal(42);
    const second = getPendingQueueItemReveal();

    expect(first?.itemId).toBe(42);
    expect(second?.itemId).toBe(42);
    expect(second?.requestId).toBeGreaterThan(first?.requestId ?? 0);
  });

  it('only clears the matching request', () => {
    requestQueueItemReveal(1);
    const first = getPendingQueueItemReveal();
    requestQueueItemReveal(2);
    const second = getPendingQueueItemReveal();

    clearPendingQueueItemReveal(first?.requestId ?? -1);
    expect(getPendingQueueItemReveal()).toEqual(second);

    clearPendingQueueItemReveal(second?.requestId ?? -1);
    expect(getPendingQueueItemReveal()).toBeNull();
  });
});
